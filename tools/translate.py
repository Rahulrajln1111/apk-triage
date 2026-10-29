"""Smali -> Java translation via a local or remote LLM.

Provider-agnostic on purpose (user requirement): the same code drives

  ollama    -> http://127.0.0.1:11434/api/chat        (Ollama native chat)
  llamacpp  -> http://127.0.0.1:8080/v1/chat/completions
               (llama.cpp server, OpenAI-compatible + thinking knob)
  openai    -> any OpenAI-compatible /v1/chat/completions endpoint,
               api_key supported (Authorization: Bearer ...)

Configuration order: explicit args > APKAI_LLM_* environment > defaults.
Defaults target the local SmaliLLM setup (smalillm-8b on Ollama).

Hard-won batch rules (inherited from SmaliLLM-server/decompile.py):
  - output paths MIRROR the smali tree (a.smali exists in classes/ AND
    classes2/ of obfuscated APKs - flat naming silently overwrites files)
  - every request is retried with backoff; failures are logged, not fatal
  - Qwen3 may still emit a <think> block if the switch is ignored: it is
    stripped defensively and counted
  - classes are split on .method boundaries so no method is cut mid-body

stdlib only (urllib) - a forensic tool must not need pip installs, and the
fake-HTTP-server tests stay dependency-free.
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

# ============================================================================
# CONFIG
# ============================================================================

PROVIDER_OLLAMA = "ollama"
PROVIDER_LLAMACPP = "llamacpp"
PROVIDER_OPENAI = "openai"
PROVIDERS = (PROVIDER_OLLAMA, PROVIDER_LLAMACPP, PROVIDER_OPENAI)

DEFAULT_SYSTEM_PROMPT = (
    "Decompile following smali code to java code. Output only Java code, "
    "no explanation. Preserve method names, string constants and control "
    "flow. If a chunk is truncated, decompile what is visible without "
    "inventing APIs."
)


@dataclass
class TranslatorConfig:
    provider: str = PROVIDER_OLLAMA
    base_url: str = "http://127.0.0.1:11434"
    model: str = "smalillm-8b"
    api_key: str | None = None       # OpenAI-compatible endpoints
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    temperature: float = 0.1
    max_tokens: int = 2048           # num_predict / max_tokens
    num_ctx: int = 8192              # Ollama context window
    seed: int = 42
    timeout: float = 300.0
    retries: int = 2
    chunk_chars: int = 12000

    def endpoint(self) -> str:
        base = self.base_url.rstrip("/")
        if self.provider == PROVIDER_OLLAMA:
            return base + "/api/chat"
        return base + "/v1/chat/completions"


def config_from_env(**overrides) -> TranslatorConfig:
    """env: APKAI_LLM_PROVIDER / APKAI_LLM_URL / APKAI_LLM_MODEL /
    APKAI_LLM_API_KEY. Explicit keyword args win over env."""
    env = os.environ.get
    cfg = TranslatorConfig(
        provider=(overrides.get("provider") or env("APKAI_LLM_PROVIDER")
                  or PROVIDER_OLLAMA),
        base_url=(overrides.get("base_url") or env("APKAI_LLM_URL")
                  or "http://127.0.0.1:11434"),
        model=(overrides.get("model") or env("APKAI_LLM_MODEL")
               or "smalillm-8b"),
        api_key=(overrides.get("api_key") or env("APKAI_LLM_API_KEY") or None),
    )
    for k in ("temperature", "max_tokens", "num_ctx", "timeout", "retries",
              "chunk_chars"):
        if overrides.get(k) is not None:
            setattr(cfg, k, overrides[k])
    if cfg.provider not in PROVIDERS:
        raise ValueError(
            f"unknown provider {cfg.provider!r} (want one of {PROVIDERS})")
    return cfg


# ============================================================================
# PROMPT / RESPONSE HYGIENE
# ============================================================================

THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
THINK_OPEN_RE = re.compile(r"<think>.*", re.DOTALL)
# models love wrapping code in markdown fences; java files should not
# start with ```java
FENCE_OPEN_RE = re.compile(r"^\s*```[a-zA-Z0-9_+-]*\s*\n")
FENCE_CLOSE_RE = re.compile(r"\n?```\s*$")


def strip_think(text: str) -> tuple[str, bool]:
    """Remove reasoning blocks and markdown fences. Returns
    (cleaned, think_stripped?).

    An UNCLOSED <think> means generation was truncated mid-reasoning -
    nothing usable follows it, so the output is dropped (empty string)."""
    cleaned = THINK_RE.sub("", text)
    think_stripped = cleaned != text
    if not think_stripped and THINK_OPEN_RE.match(text):
        return "", True
    # fences can wrap the code with or without a think block - strip on
    # every path (early return here silently kept them once; test caught it)
    cleaned = FENCE_OPEN_RE.sub("", cleaned.strip())
    cleaned = FENCE_CLOSE_RE.sub("", cleaned)
    return cleaned.strip(), think_stripped


def split_methods(smali: str, max_chars: int) -> list[str]:
    """Split a smali class into method-sized chunks under max_chars.

    Splits on .method boundaries so a method is never cut mid-body; the
    only chunks that exceed the cap are single methods larger than it."""
    parts = re.split(r"(?=\.method\s)", smali)
    chunks: list[str] = []
    buf = ""
    for part in parts:
        if buf and len(buf) + len(part) > max_chars:
            chunks.append(buf)
            buf = part
        else:
            buf += part
    if buf:
        chunks.append(buf)
    return [c for c in chunks if c.strip()] or [smali[:max_chars]]


# ============================================================================
# PROVIDERS
# ============================================================================

class TranslateError(RuntimeError):
    pass


def _post_json(url: str, payload: dict, cfg: TranslatorConfig) -> dict:
    headers = {"Content-Type": "application/json"}
    if cfg.api_key:
        headers["Authorization"] = f"Bearer {cfg.api_key}"
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=cfg.timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _payload_ollama(system: str, user: str, cfg: TranslatorConfig) -> dict:
    # "think": False is Ollama's Qwen3 reasoning switch; the trailing
    # /no_think soft switch is the documented belt-and-braces fallback.
    system_txt = system if "/no_think" in system else system + " /no_think"
    return {
        "model": cfg.model,
        "messages": [
            {"role": "system", "content": system_txt},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "think": False,
        "options": {
            "num_ctx": cfg.num_ctx,
            "num_predict": cfg.max_tokens,
            "temperature": cfg.temperature,
            "seed": cfg.seed,
            "repeat_penalty": 1.05,
        },
    }


def _payload_openai(system: str, user: str, cfg: TranslatorConfig) -> dict:
    payload = {
        "model": cfg.model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "temperature": cfg.temperature,
        "max_tokens": cfg.max_tokens,
    }
    if cfg.provider == PROVIDER_LLAMACPP:
        # llama.cpp server's thinking-mode knob (Qwen3 chat template).
        payload["chat_template_kwargs"] = {"thinking": False}
    return payload


def chat(system: str, user: str, cfg: TranslatorConfig) -> str:
    """One chat completion against cfg's provider. Retries on transient
    errors; raises TranslateError with the last reason on final failure."""
    url = cfg.endpoint()
    payload = (_payload_ollama(system, user, cfg)
               if cfg.provider == PROVIDER_OLLAMA
               else _payload_openai(system, user, cfg))

    last_err: Exception | None = None
    for attempt in range(cfg.retries + 1):
        if attempt:
            time.sleep(min(2 ** attempt, 8))     # 2s, 4s, ... capped
        try:
            data = _post_json(url, payload, cfg)
            if cfg.provider == PROVIDER_OLLAMA:
                return data["message"]["content"]
            return data["choices"][0]["message"]["content"]
        except (urllib.error.HTTPError, urllib.error.URLError,
                TimeoutError, OSError, KeyError, IndexError,
                json.JSONDecodeError) as exc:
            last_err = exc
    raise TranslateError(f"{cfg.provider} {url}: {last_err!r}")


def _server_alive(cfg: TranslatorConfig, timeout: float = 3.0) -> bool:
    """Cheap liveness GET: /api/tags (ollama), /health (llamacpp),
    /v1/models (openai). Never raises."""
    base = cfg.base_url.rstrip("/")
    path = {PROVIDER_OLLAMA: "/api/tags",
            PROVIDER_LLAMACPP: "/health",
            PROVIDER_OPENAI: "/v1/models"}[cfg.provider]
    try:
        with urllib.request.urlopen(base + path, timeout=timeout) as r:
            return 200 <= r.status < 500      # even 404 means 'server there'
    except Exception:
        return False


def probe(cfg: TranslatorConfig, timeout: float = 120.0) -> bool:
    """Availability check, in two steps. Never raises.

    1. server alive (3s) - catches 'nothing listening'
    2. 1-token chat with a GENEROUS timeout - a cold model must load
       into RAM first (an 8B Q4 takes tens of seconds), which a 5s
       timeout wrongly reports as 'endpoint unreachable'."""
    if not _server_alive(cfg):
        return False
    tiny = TranslatorConfig(**{**cfg.__dict__, "max_tokens": 1,
                               "timeout": timeout, "retries": 0})
    try:
        chat("Reply with OK.", "ping", tiny)
        return True
    except Exception:
        return False


# ============================================================================
# BATCH TRANSLATION
# ============================================================================

@dataclass
class TranslateStats:
    ok: int = 0
    error: int = 0
    think_stripped: int = 0
    fences_stripped: int = 0
    chunks: int = 0
    errors: list = field(default_factory=list)

    def as_line(self) -> str:
        return (f"ok={self.ok} error={self.error} "
                f"think_stripped={self.think_stripped}")


def translate_class(smali: str, cfg: TranslatorConfig,
                    stats: TranslateStats | None = None) -> str:
    """Translate one smali class body to Java (chunked on methods)."""
    stats = stats or TranslateStats()
    chunks = split_methods(smali, cfg.chunk_chars)
    out: list[str] = []
    for chunk in chunks:
        stats.chunks += 1
        try:
            raw = chat(cfg.system_prompt, chunk, cfg)
            text, stripped = strip_think(raw)
            if stripped:
                stats.think_stripped += 1
            if not text:
                raise TranslateError("empty content (truncated thinking?)")
            stats.fences_stripped += 1 if text != raw.strip() else 0
            out.append(text)
            stats.ok += 1
        except Exception as exc:              # logged, not fatal (batch rule)
            stats.error += 1
            stats.errors.append(str(exc))
            out.append(f"/* [APKAI] translation failed: {exc} */\n"
                       + chunk.split("\n", 1)[0][:200])
    return "\n\n".join(out)


def translate_tree(smali_dir: Path, out_dir: Path, cfg: TranslatorConfig,
                   force: bool = False) -> TranslateStats:
    """Walk a baksmali-style tree; mirror it under out_dir. Existing
    non-empty outputs are skipped unless force (resume rule)."""
    stats = TranslateStats()
    smali_dir, out_dir = Path(smali_dir), Path(out_dir)
    files = sorted(p for p in smali_dir.rglob("*.smali") if p.is_file())
    for path in files:
        rel = path.relative_to(smali_dir)
        dst = out_dir / rel.with_suffix(".java")
        if dst.exists() and dst.stat().st_size > 0 and not force:
            stats.ok += 1                     # resumed
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        java = translate_class(path.read_text(errors="replace"), cfg, stats)
        dst.write_text(java)
    return stats


# ============================================================================
# CLI
# ============================================================================

def _main(argv: list[str] | None = None) -> int:
    import argparse
    p = argparse.ArgumentParser(
        description="Translate smali to Java via an LLM "
                    "(ollama | llamacpp | openai-compatible).")
    p.add_argument("--smali-file", type=Path)
    p.add_argument("--smali-dir", type=Path)
    p.add_argument("--out", type=Path, help="output .java for --smali-file")
    p.add_argument("--out-dir", type=Path, help="output root for --smali-dir")
    p.add_argument("--provider", choices=PROVIDERS)
    p.add_argument("--base-url")
    p.add_argument("--model")
    p.add_argument("--api-key")
    p.add_argument("--force", action="store_true")
    args = p.parse_args(argv)

    cfg = config_from_env(
        provider=args.provider, base_url=args.base_url,
        model=args.model, api_key=args.api_key)
    print(f"[translate] provider={cfg.provider} model={cfg.model} "
          f"url={cfg.base_url}")

    if not probe(cfg):
        print("[translate] ERROR: model endpoint unreachable - "
              "start the server first (SmaliLLM-server/scripts/).")
        return 2

    if args.smali_file:
        java = translate_class(args.smali_file.read_text(errors="replace"),
                               cfg)
        out = args.out or args.smali_file.with_suffix(".java")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(java)
        print(f"[translate] wrote {out}")
        return 0

    if args.smali_dir and args.out_dir:
        stats = translate_tree(args.smali_dir, args.out_dir, cfg,
                               force=args.force)
        print(f"[translate] {stats.as_line()}")
        return 0 if stats.error == 0 else 1

    p.error("need --smali-file/--out or --smali-dir/--out-dir")
    return 1


if __name__ == "__main__":
    raise SystemExit(_main())
