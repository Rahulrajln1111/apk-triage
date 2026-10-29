"""Sensitive-API reachability report: entry points -> risky calls.

Triage view for malware analysis: for each method in the DEX we record
sensitive API invocations and suspicious string constants, then BFS from
the entry points (manifest components + framework-callback overrides) to
mark what is actually REACHABLE from code the OS can invoke.

Honest scope: this is substring matching over resolved call signatures +
constants, plus graph reachability. It is NOT a taint analysis - a hit
means "reachable code touches this API", not "data flows to it". Every
finding cites the exact method that made the call.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import re

# ============================================================================
# PATTERN TABLES (substring match, case-sensitive, on resolved signatures)
# ============================================================================

# category -> substrings looked for in the CALLED method signature
SENSITIVE_CALLS: dict[str, tuple[str, ...]] = {
    "sms":      ("Landroid/telephony/SmsManager;->send",
                 "Landroid/telephony/SmsManager;->divide"),
    "phone":    ("Landroid/telephony/TelephonyManager;->getDeviceId",
                 "Landroid/telephony/TelephonyManager;->getImei",
                 "Landroid/telephony/TelephonyManager;->getSubscriberId",
                 "Landroid/telephony/TelephonyManager;->getLine1Number",
                 "Landroid/telephony/TelephonyManager;->getSimSerialNumber",
                 "Landroid/telephony/TelephonyManager;->listen"),
    "location": ("Landroid/location/LocationManager;->getLastKnownLocation",
                 "Landroid/location/LocationManager;->requestLocationUpdates",
                 "Lcom/google/android/gms/location/"),
    "crypto":   ("Ljavax/crypto/Cipher;->getInstance",
                 "Ljavax/crypto/Mac;->getInstance",
                 "Ljavax/crypto/spec/SecretKeySpec;-><init>",
                 "Ljavax/crypto/spec/IvParameterSpec;-><init>"),
    "http":     ("Ljava/net/URL;-><init>",
                 "Ljava/net/HttpURLConnection;->",
                 "Lokhttp3/OkHttpClient;->newCall",
                 "Lokhttp3/Request$Builder;->url",
                 "Lorg/apache/http/"),
    "shell":    ("Ljava/lang/Runtime;->exec",
                 "Ljava/lang/ProcessBuilder;-><init>"),
    "reflection": ("Ljava/lang/Class;->forName",
                   "Ljava/lang/reflect/Method;->invoke",
                   "Ldalvik/system/DexClassLoader;-><init>",
                   "Ldalvik/system/PathClassLoader;-><init>"),
    "dynamic-code": ("Ldalvik/system/DexClassLoader;->loadClass",
                     "Ldalvik/system/InMemoryDexClassLoader;-><init>"),
    "persistence": ("Landroid/content/SharedPreferences;->getString",
                    "Landroid/content/ContentResolver;->query",
                    "Landroid/database/sqlite/SQLiteDatabase;->execSQL"),
    "device-intel": ("Landroid/os/Build;->",
                     "Landroid/provider/Settings$Secure;->ANDROID_ID",
                     "Ljava/net/NetworkInterface;->getHardwareAddress"),
}

# suspicious STRING CONSTANTS. Needles >3 chars match case-insensitively;
# short ones ('su', 'dex') are CASE-SENSITIVE substrings - otherwise
# 'because' or 'console' would flag 'su'. Specific paths come before
# bare 'su' so '/system/xbin/su' reports the stronger label.
SUSPICIOUS_CONSTANTS: tuple[tuple[str, str], ...] = (
    ("http://", "cleartext-http-url"),
    ("/data/data/", "private-dir-path"),
    ("/system/bin/", "system-binary-path"),
    ("/system/xbin/", "su-path"),
    ("getDeviceId", "imei-reflection"),
    ("base64,", "base64-blob"),
    ("su", "su"),
    ("dex", "dex-string"),
)


@dataclass
class MethodFinding:
    key: str                    # Lcls;->name(sig)
    reachable: bool = False
    calls: dict[str, list[str]] = field(default_factory=dict)   # cat -> signature
    constants: dict[str, list[str]] = field(default_factory=dict)  # hit -> value


@dataclass
class ApiReport:
    findings: list[MethodFinding] = field(default_factory=list)
    reachable_sensitive: dict[str, list[str]] = field(default_factory=dict)
    unreachable_sensitive: dict[str, list[str]] = field(default_factory=dict)
    total_methods: int = 0

    def summary_line(self) -> str:
        r = sum(len(v) for v in self.reachable_sensitive.values())
        u = sum(len(v) for v in self.unreachable_sensitive.values())
        return (f"{self.total_methods} methods, "
                f"{r} reachable sensitive calls, {u} unreachable")


# ============================================================================
# ANALYSIS
# ============================================================================

def _classify_call(sig: str) -> str | None:
    for cat, needles in SENSITIVE_CALLS.items():
        for n in needles:
            if n in sig:
                return cat
    return None


def _classify_const(value: str) -> str | None:
    for needle, label in SUSPICIOUS_CONSTANTS:
        if len(needle) <= 3:
            # short needle ('su', 'dex'): delimiter-bounded, CASE-SENSITIVE
            # - plain substring would flag 'issue' for 'su'
            if re.search(rf"(?<![A-Za-z0-9]){re.escape(needle)}"
                         rf"(?![A-Za-z0-9])", value):
                return label
        elif needle.lower() in value.lower():
            return label
    return None


def build_api_report(dex, graph) -> ApiReport:
    """Walk every method once (calls + string constants), then BFS the call
    graph from entry points to mark reachability.

    dex : tools.dex.DexFile (parsed)
    graph: tools.smali.CallGraph from build_call_graph() - supplies the
           per-method resolved call signatures and the entry-point set.

    Scope note: the graph covers ONE dex file. Callees living in other
    DEX files of a multi-dex APK are visible as signatures but their
    bodies are not followed (they count as graph-external).
    """
    from tools.smali import (decode_method_insns, method_signature,
                             proto_signature)

    report = ApiReport()
    findings: dict[str, MethodFinding] = {}

    # pass 1a: sensitive CALLS come from the graph's resolved call lists
    # (same signature strings used for graph edges - keys line up)
    for cls in graph.classes.values():
        for mi in cls.methods.values():
            report.total_methods += 1
            fnd = MethodFinding(key=mi.key, reachable=False)
            for sig in mi.calls:
                cat = _classify_call(sig)
                if cat:
                    fnd.calls.setdefault(cat, []).append(sig)
            if fnd.calls:
                findings[mi.key] = fnd

    # pass 1b: suspicious STRING CONSTANTS need the raw insns
    for cdef in dex.class_defs:
        for m in cdef.direct_methods + cdef.virtual_methods:
            if not (0 <= m.method_idx < len(dex.method_ids)) or not m.code_off:
                continue
            mid = dex.method_ids[m.method_idx]
            key = f"{dex.type_desc(mid.class_idx)}->{dex.string(mid.name_idx)}" \
                  f"{proto_signature(dex, mid.proto_idx)}"
            code = dex.code_for(m)
            if code is None:
                continue
            try:
                insns, _payloads = decode_method_insns(dex.data, code)
            except Exception:
                continue
            for insn in insns:
                if insn.kind == "string" and insn.index is not None:
                    text = dex.string(insn.index)
                    if not text:
                        continue
                    label = _classify_const(text)
                    if not label:
                        continue
                    fnd = findings.setdefault(key,
                                              MethodFinding(key=key))
                    fnd.constants.setdefault(label, []).append(text)

    # pass 2: BFS from entry points over the per-method call edges.
    # graph.classes[cls].methods[key].calls holds resolved callee keys for
    # internal targets (same key format) - reuse instead of re-decoding.
    adj: dict[str, list[str]] = {}
    for cls in graph.classes.values():
        for mi in cls.methods.values():
            adj.setdefault(mi.key, []).extend(mi.calls)

    seen: set[str] = set()
    queue = deque()
    for cls in graph.classes.values():
        for mi in cls.methods.values():
            if mi.entry:
                queue.append(mi.key)
    while queue:
        key = queue.popleft()
        if key in seen:
            continue
        seen.add(key)
        for nxt in adj.get(key, ()):
            if nxt not in seen:
                queue.append(nxt)

    for key, fnd in findings.items():
        fnd.reachable = key in seen
        for cat, sigs in fnd.calls.items():
            bucket = (report.reachable_sensitive if fnd.reachable
                      else report.unreachable_sensitive)
            bucket.setdefault(cat, []).append(f"{key}: {sigs[0]}")
        report.findings.append(fnd)

    report.findings.sort(key=lambda f: (not f.reachable, f.key))
    return report


def print_api_report(report: ApiReport, top: int = 15) -> None:
    print(f"Methods scanned    : {report.total_methods:,}")
    r = sum(len(v) for v in report.reachable_sensitive.values())
    u = sum(len(v) for v in report.unreachable_sensitive.values())
    print(f"Sensitive calls    : {r} reachable from entry points, "
          f"{u} unreachable (reflection/JNI may still reach those)")

    for title, bucket in (("REACHABLE SENSITIVE CALLS",
                           report.reachable_sensitive),
                          ("UNREACHABLE (dead-code) SENSITIVE CALLS",
                           report.unreachable_sensitive)):
        if not bucket:
            continue
        print()
        print("-" * 70)
        print(title)
        print("-" * 70)
        for cat in sorted(bucket):
            hits = bucket[cat]
            print(f"  [{cat}] {len(hits)} hit(s)")
            for line in hits[:top]:
                print(f"    {line}")
            if len(hits) > top:
                print(f"    ... +{len(hits) - top} more")

    const_hits = [f for f in report.findings if f.constants and f.reachable]
    if const_hits:
        print()
        print("-" * 70)
        print(f"REACHABLE SUSPICIOUS CONSTANTS ({len(const_hits)} methods)")
        print("-" * 70)
        for f in const_hits[:top]:
            labels = ", ".join(sorted(f.constants))
            sample = next(iter(f.constants.values()))[0][:40]
            print(f"  {f.key}\n    [{labels}] {sample!r}")
