"""tests for tools/api_report.py - sensitive-API reachability.

Synthetic DEX wire-encoded like the smali tests: onCreate -> helper ->
sneaky -> SmsManager.sendTextMessage must be flagged as REACHABLE; an
identical call from a never-called method must land in UNREACHABLE; a
cleartext http:// constant in the entry method must be flagged.
"""
import struct

from tests.test_smali import TClass, TMethod, build_dex
from tools.api_report import _classify_call, _classify_const, build_api_report
from tools.dex import parse_dex_bytes
from tools.smali import build_call_graph

# strings (indices matter below)
STRINGS = ["", "V", "LFoo;", "LBar;", "onCreate", "()V", "helper", "sneaky",
           "sendTextMessage", "Landroid/telephony/SmsManager;",
           "deadCaller", "http://evil.example/payload"]
# types: [0]=V, [1]=LFoo;, [2]=SmsManager  (values are string indices:
# 1="V", 2="LFoo;", 9="Landroid/telephony/SmsManager;")
TYPES = [1, 2, 9]
# protos: ()V
PROTOS = [(1, 0, None)]
# methods: LFoo;->{onCreate,helper,sneaky,deadCaller}, SmsManager->sendTextMessage
METHODS = [(1, 0, 4), (1, 0, 6), (1, 0, 7), (2, 0, 8), (1, 0, 10)]

INVOKE = 0x106E          # invoke-virtual {v0}: A=1 (bits 12-15), op 0x6E


def _invoke(method_idx: int) -> bytes:
    return struct.pack("<3H", INVOKE, method_idx, 0x0000) + \
        struct.pack("<H", 0x000E)          # + return-void


def api_dex():
    insns_onCreate = struct.pack("<2H", 0x001A, 11) + _invoke(1)  # const-string + ->helper
    insns_helper = _invoke(2)                                     # ->sneaky
    insns_sneaky = _invoke(3)                                     # ->sendTextMessage
    insns_dead = _invoke(3)                                       # ->sendTextMessage (dead)
    dex = parse_dex_bytes(build_dex(
        strings=STRINGS, types=TYPES, protos=PROTOS, methods=METHODS,
        classes=[TClass(type_idx=1, methods=[           # LFoo;
            TMethod(0, 0x9, insns_onCreate),
            TMethod(1, 0x9, insns_helper),
            TMethod(2, 0x9, insns_sneaky),
            TMethod(4, 0x9, insns_dead),
        ])]))
    return dex


SMS_HIT = ("LFoo;->sneaky()V: "
           "Landroid/telephony/SmsManager;->sendTextMessage()V")
SMS_DEAD = ("LFoo;->deadCaller()V: "
            "Landroid/telephony/SmsManager;->sendTextMessage()V")


def test_classify_tables():
    # calls
    assert _classify_call(
        "Landroid/telephony/SmsManager;->sendTextMessage(...)V") == "sms"
    assert _classify_call("Ljava/lang/Runtime;->exec(...)V") == "shell"
    assert _classify_call(
        "Ldalvik/system/DexClassLoader;->loadClass(...)") == "dynamic-code"
    assert _classify_call("Ljava/lang/String;->length()I") is None
    # constants: long needles case-insensitive, short needles case-SENSITIVE
    assert _classify_const("HTTP://EVIL") == "cleartext-http-url"
    assert _classify_const("/system/xbin/su") == "su-path"   # beats bare su
    assert _classify_const("su") == "su"
    assert _classify_const("issue") is None      # 'su' must not hit 'issue'
    assert _classify_const("DexFile") is None    # 'dex' is case-sensitive
    assert _classify_const("classes.dex") == "dex-string"


def test_reachability_and_dead_code():
    dex = api_dex()
    graph = build_call_graph(dex)
    rep = build_api_report(dex, graph)

    assert rep.total_methods == 4            # all coded methods in the dex
    # multi-hop: onCreate -> helper -> sneaky -> sendTextMessage
    assert rep.reachable_sensitive.get("sms") == [SMS_HIT]
    # identical call from a dead method stays out of the reachable bucket
    assert rep.unreachable_sensitive.get("sms") == [SMS_DEAD]

    by_key = {f.key: f for f in rep.findings}
    assert by_key["LFoo;->sneaky()V"].reachable is True
    assert by_key["LFoo;->deadCaller()V"].reachable is False
    # entry method flagged for its cleartext http constant
    entry = by_key["LFoo;->onCreate()V"]
    assert entry.reachable is True
    assert "cleartext-http-url" in entry.constants

    line = rep.summary_line()
    assert "1 reachable" in line and "1 unreachable" in line


def test_findings_sorted_reachable_first():
    dex = api_dex()
    rep = build_api_report(dex, build_call_graph(dex))
    reach = [f.reachable for f in rep.findings if f.calls]
    assert reach == sorted(reach, reverse=True)
