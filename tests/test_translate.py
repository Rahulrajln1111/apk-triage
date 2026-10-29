"""tests for tools/translate.py - provider-agnostic LLM layer.

Every test drives a REAL localhost HTTP server (fake backend) so payload
shapes, retry behaviour and endpoint paths are verified against actual
wire traffic - not mocks of our own code. (Rule #9: never trust memory,
verify on the wire.)
"""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

import tools.translate as tr
from tools.translate import (
    TranslatorConfig,
    TranslateError,
    TranslateStats,
    chat,
    config_from_env,
    probe,
    split_methods,
    strip_think,
    translate_class,
    translate_tree,
)

OLLAMA_BODY = {"message": {"role": "assistant",
                           "content": "public class Foo {}"}}
OPENAI_BODY = {"choices": [{"message": {"role": "assistant",
                                        "content": "public class Foo {}"}}]}


# ============================================================================
# FAKE LLM SERVER (real sockets, no mocking of the code under test)
# ============================================================================

class FakeState:
    def __init__(self):
        self.lock = threading.Lock()
        self.responses = []   # [(status, body)] - last entry repeats
        self.requests = []    # [(path, payload, headers)]
        self.hits = 0


def _make_server(state: FakeState):
    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            # liveness endpoint (probe step 1: /api/tags for ollama)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"{}")

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(n) or b"{}")
            with state.lock:
                state.requests.append((self.path, payload,
                                       dict(self.headers)))
                i = min(state.hits, len(state.responses) - 1)
                status, body = state.responses[i]
                state.hits += 1
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


@pytest.fixture
def state():
    st = FakeState()
    st.srv = _make_server(st)
    yield st
    st.srv.shutdown()
    st.srv.server_close()


def ollama_cfg(state: FakeState, **kw) -> TranslatorConfig:
    kw.setdefault("retries", 0)
    return TranslatorConfig(
        provider="ollama",
        base_url=f"http://127.0.0.1:{state.srv.server_port}",
        model="test-model", **kw)


# ============================================================================
# WIRE FORMAT PER PROVIDER
# ============================================================================

def test_ollama_wire_format(state):
    state.responses = [(200, OLLAMA_BODY)]
    cfg = ollama_cfg(state)
    assert chat("sys", "user text", cfg) == "public class Foo {}"

    path, payload, headers = state.requests[0]
    assert path == "/api/chat"
    assert payload["model"] == "test-model"
    assert payload["stream"] is False
    assert payload["think"] is False                 # Qwen3 reasoning switch
    assert payload["options"]["num_ctx"] == cfg.num_ctx
    assert payload["options"]["num_predict"] == cfg.max_tokens
    sys_msg, user_msg = payload["messages"]
    assert sys_msg["role"] == "system"
    assert "/no_think" in sys_msg["content"]         # soft-switch appended
    assert user_msg == {"role": "user", "content": "user text"}
    assert "Authorization" not in headers            # no key sent by default


def test_llamacpp_wire_format(state):
    state.responses = [(200, OPENAI_BODY)]
    cfg = TranslatorConfig(
        provider="llamacpp",
        base_url=f"http://127.0.0.1:{state.srv.server_port}", model="m")
    assert chat("s", "u", cfg) == "public class Foo {}"

    path, payload, _ = state.requests[0]
    assert path == "/v1/chat/completions"
    assert payload["chat_template_kwargs"] == {"thinking": False}
    assert payload["max_tokens"] == cfg.max_tokens


def test_openai_api_key_header(state):
    state.responses = [(200, OPENAI_BODY)]
    cfg = TranslatorConfig(
        provider="openai",
        base_url=f"http://127.0.0.1:{state.srv.server_port}",
        model="m", api_key="sk-test")
    chat("s", "u", cfg)
    _, payload, headers = state.requests[0]
    assert headers.get("Authorization") == "Bearer sk-test"
    assert "chat_template_kwargs" not in payload     # plain OpenAI shape


def test_endpoint_shapes():
    c = TranslatorConfig(provider="ollama", base_url="http://x:11434/")
    assert c.endpoint() == "http://x:11434/api/chat"
    c = TranslatorConfig(provider="openai", base_url="http://x:8080")
    assert c.endpoint() == "http://x:8080/v1/chat/completions"


# ============================================================================
# RETRY / PROBE
# ============================================================================

def test_chat_retries_transient_errors(monkeypatch, state):
    monkeypatch.setattr(tr.time, "sleep", lambda s: None)  # keep test fast
    state.responses = [(500, {"error": "boom"}),
                       (500, {"error": "boom"}),
                       (200, OLLAMA_BODY)]
    cfg = ollama_cfg(state, retries=2)
    assert chat("s", "u", cfg) == "public class Foo {}"
    assert state.hits == 3


def test_chat_raises_after_final_failure(monkeypatch, state):
    monkeypatch.setattr(tr.time, "sleep", lambda s: None)
    state.responses = [(500, {"error": "boom"})]
    with pytest.raises(TranslateError):
        chat("s", "u", ollama_cfg(state, retries=1))
    assert state.hits == 2                           # 1 initial + 1 retry


def test_probe_up_and_down(state):
    state.responses = [(200, OLLAMA_BODY)]
    assert probe(ollama_cfg(state)) is True
    # server alive but the chat itself failing -> still 'not available'
    state.responses = [(500, {"error": "boom"})]
    state.hits = 0
    assert probe(ollama_cfg(state)) is False
    dead = TranslatorConfig(provider="ollama",
                            base_url="http://127.0.0.1:1",  # refused
                            model="m", timeout=2.0)
    assert probe(dead) is False


# ============================================================================
# PROMPT / RESPONSE HYGIENE
# ============================================================================

def test_strip_think_variants():
    text, stripped = strip_think("<think>reasoning</think>public class A {}")
    assert (text, stripped) == ("public class A {}", True)
    # unclosed <think> = truncated reasoning -> nothing usable
    text, stripped = strip_think("<think>truncated mid-thought")
    assert text == "" and stripped is True
    text, stripped = strip_think("public class A {}")
    assert (text, stripped) == ("public class A {}", False)


def test_strip_think_strips_fences_on_all_paths():
    # fences alone
    assert strip_think("```java\nclass A {}\n```")[0] == "class A {}"
    # think block AND fences: fences must go too (regression: early return
    # used to skip fence stripping when a think block was present)
    text, stripped = strip_think("<think>t</think>```java\nclass A {}\n```")
    assert (text, stripped) == ("class A {}", True)


def test_split_methods_respects_boundaries():
    smali = (".class public LFoo;\n.super Ljava/lang/Object;\n"
             ".method public a()V\n    return-void\n.end method\n\n"
             ".method public b()V\n    return-void\n.end method\n")
    chunks = split_methods(smali, 60)
    # a method body is never cut: every chunk starts at a boundary and the
    # .method count is preserved overall
    assert all(c.lstrip().startswith((".class", ".super", ".method"))
               for c in chunks)
    assert sum(c.count(".method ") for c in chunks) == 2
    # generous cap -> single chunk
    assert split_methods(smali, 10_000) == [smali]


# ============================================================================
# TRANSLATION (stats, think handling, failure tolerance)
# ============================================================================

def test_translate_class_strips_think_block(state):
    state.responses = [(200, {"message":
                              {"content": "<think>x</think>JAVA CODE"}})]
    stats = TranslateStats()
    out = translate_class(".method a()V\n.end method\n",
                          ollama_cfg(state), stats)
    assert out == "JAVA CODE"
    assert stats.ok == 1 and stats.error == 0
    assert stats.think_stripped == 1


def test_translate_class_empty_after_think_is_error(state):
    state.responses = [(200, {"message": {"content": "<think>oops"}})]
    stats = TranslateStats()
    out = translate_class(".method a()V\n.end method\n",
                          ollama_cfg(state), stats)
    assert stats.error == 1 and stats.think_stripped == 1
    assert "[APKAI] translation failed" in out       # logged, not fatal


def test_translate_class_mixed_success_and_failure(state):
    state.responses = [(200, OLLAMA_BODY),
                       (500, {"error": "x"}),
                       (200, OLLAMA_BODY)]
    smali = (".class public LFoo;\n"
             ".method public a()V\n    return-void\n.end method\n\n"
             ".method public b()V\n    return-void\n.end method\n")
    stats = TranslateStats()
    out = translate_class(smali, ollama_cfg(state, chunk_chars=40), stats)
    assert stats.ok == 2 and stats.error == 1
    assert "public class Foo {}" in out
    assert "[APKAI] translation failed" in out


def test_translate_tree_mirrors_and_resumes(tmp_path, state):
    state.responses = [(200, OLLAMA_BODY)]
    smali_dir = tmp_path / "smali"
    (smali_dir / "classes").mkdir(parents=True)
    (smali_dir / "classes2").mkdir(parents=True)
    body = ".method public a()V\n    return-void\n.end method\n"
    (smali_dir / "classes" / "a.smali").write_text(body)
    (smali_dir / "classes2" / "a.smali").write_text(body)
    out_dir = tmp_path / "java"

    cfg = ollama_cfg(state)
    stats = translate_tree(smali_dir, out_dir, cfg)
    assert stats.ok == 2 and stats.error == 0
    # paths MIRROR the smali tree (classes/ AND classes2/ both have a.smali)
    assert (out_dir / "classes" / "a.java").read_text() == "public class Foo {}"
    assert (out_dir / "classes2" / "a.java").exists()

    before = state.hits
    stats2 = translate_tree(smali_dir, out_dir, cfg)         # resume
    assert state.hits == before, "non-empty outputs must be skipped"
    assert stats2.ok == 2

    translate_tree(smali_dir, out_dir, cfg, force=True)      # redo
    assert state.hits == before + 2


# ============================================================================
# CONFIG
# ============================================================================

def test_config_from_env_and_overrides(monkeypatch):
    monkeypatch.setenv("APKAI_LLM_PROVIDER", "openai")
    monkeypatch.setenv("APKAI_LLM_URL", "https://api.example.com")
    monkeypatch.setenv("APKAI_LLM_MODEL", "gpt-x")
    monkeypatch.setenv("APKAI_LLM_API_KEY", "sk-env")
    cfg = config_from_env()
    assert (cfg.provider, cfg.base_url, cfg.model, cfg.api_key) == (
        "openai", "https://api.example.com", "gpt-x", "sk-env")

    # explicit kwargs beat env
    cfg2 = config_from_env(provider="ollama", model="local")
    assert cfg2.provider == "ollama" and cfg2.model == "local"

    with pytest.raises(ValueError):
        config_from_env(provider="brain")
