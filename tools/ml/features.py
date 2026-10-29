"""Versioned APK -> numeric feature row (one row per APK).

Rules that keep this honest:
  * one column per scalar; no nested structures in the matrix;
  * a missing artifact (no resources.arsc, unparseable DEX) becomes NaN,
    NEVER 0 - "absent" and "present but zero" are different forensic facts,
    and HistGradientBoosting/XGBoost handle NaN natively;
  * every block is isolated: a hostile APK degrades its own block instead of
    aborting the corpus scan;
  * bump FEATURE_VERSION whenever a column is added/renamed. The training
    artifact stores it and refuses a mismatched matrix.

The block prefix (zip/axml/arsc/dex/graph) is what the ablation study in
tools/ml/train.py slices on, so keep the naming scheme flat.
"""
from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from pathlib import Path

FEATURE_VERSION = "apkai-ml-1"

# ---------------------------------------------------------------------------
# COLUMN GROUPS
# ---------------------------------------------------------------------------

ZIP_FIELDS = (
    "zip.total_entries",
    "zip.total_size_log",
    "zip.max_entry_size_log",
    "zip.overall_ratio_log",
    "zip.max_ratio_log",
    "zip.entries_over_100x",
    "zip.entries_over_1000x",
    "zip.dex_count",
    "zip.dex_ratio",
    "zip.zero_padded_dex_count",
    "zip.sequential_dex_score",
    "zip.png_ratio",
    "zip.xml_ratio",
    "zip.native_lib_count",
    "zip.native_lib_bytes_log",
    "zip.asset_count",
    "zip.asset_bytes_log",
    "zip.directory_entries_ratio",
    "zip.max_depth",
    "zip.depth_entropy",
    "zip.distinct_top_dirs",
    "zip.path_traversal",
    "zip.absolute_path",
    "zip.duplicate_names",
    "zip.null_byte_names",
    "zip.symlink",
    "zip.encrypted",
    "zip.zip64_extra",
    "zip.nested_archive",
    "zip.data_descriptor_ratio",
    "zip.filename_family_score",
)

AXML_FIELDS = (
    "axml.string_count",
    "axml.resource_id_count",
    "axml.permission_count",
    "axml.dangerous_permission_count",
    "axml.custom_permission_count",
    "axml.component_count",
    "axml.activity_count",
    "axml.service_count",
    "axml.receiver_count",
    "axml.provider_count",
    "axml.exported_component_count",
    "axml.intent_filter_count",
    "axml.suspicious_attribute_count",
    "axml.split_count",
    "axml.min_sdk",
    "axml.target_sdk",
    "axml.allow_backup",
    "axml.debuggable",
    "axml.cleartext_traffic",
    "axml.extract_native_libs",
    "axml.flag_vpn_service",
    "axml.flag_accessibility_service",
    "axml.flag_device_admin",
    "axml.flag_boot_receiver",
    "axml.flag_install_packages",
    "axml.flag_query_all_packages",
)

ARSC_FIELDS = (
    "arsc.table_bytes_log",
    "arsc.package_count",
    "arsc.resource_entries",
    "arsc.config_variants",
    "arsc.entries_per_config",
    "arsc.type_count",
    "arsc.string_values",
    "arsc.reference_values",
    "arsc.attribute_refs",
    "arsc.complex_bags",
    "arsc.compact_entries",
    "arsc.public_entries",
    "arsc.weak_entries",
    "arsc.sparse_types",
    "arsc.offset16_types",
    "arsc.anomalies",
    "arsc.null_chunks",
    "arsc.library_packages",
    "arsc.scramble_score",
    "arsc.scramble_ratio",
    "arsc.rebuild_tool",
)

DEX_FIELDS = (
    "dex.entry_count",
    "dex.real_count",
    "dex.fake_count",
    "dex.fake_ratio",
    "dex.parse_failures",
    "dex.zero_padded_ratio",
    "dex.strings",
    "dex.types",
    "dex.protos",
    "dex.fields",
    "dex.methods",
    "dex.classes",
    "dex.classes_max_per_dex",
    "dex.classes_per_dex_mean",
    "dex.methods_per_class",
    "dex.checksum_bad",
    "dex.signature_bad",
    "dex.anomaly_count",
    "dex.misaligned_methods",
    "dex.oversized_payload_methods",
    "dex.anomaly_per_class",
    "dex.insn_total_log",
    "dex.op_entropy",
    "dex.op_overrun_ratio",
)

# opcode fractions: same opcodes Drebin-era work leans on, plus the invoke mix
OPCODE_FRACTIONS = (
    "invoke-virtual",
    "invoke-static",
    "invoke-direct",
    "invoke-interface",
    "const-string",
    "const-string/jumbo",
    "iget-object",
    "sget-object",
    "new-instance",
    "new-array",
    "filled-new-array",
    "check-cast",
    "if-eqz",
    "goto",
    "move-result-object",
    "return-void",
)

SENSITIVE_CATEGORIES = (
    "sms",
    "phone",
    "location",
    "crypto",
    "http",
    "shell",
    "reflection",
    "dynamic-code",
    "persistence",
    "device-intel",
)

CONSTANT_LABELS = (
    "cleartext-http-url",
    "private-dir-path",
    "system-binary-path",
    "su-path",
    "imei-reflection",
    "base64-blob",
    "su",
    "dex-string",
)


def _safe(name: str) -> str:
    return re.sub(r"[^0-9a-zA-Z]+", "_", name).strip("_").lower()


def _op_field(opcode_name: str) -> str:
    return "dex.op_" + _safe(opcode_name)


GRAPH_FIELDS = (
    "graph.classes",
    "graph.methods",
    "graph.internal_edges",
    "graph.external_targets",
    "graph.dead_candidates",
    "graph.dead_ratio",
    "graph.entry_points",
    "graph.edges_per_class",
    "graph.external_per_method",
    "graph.reachable_sensitive_total",
    "graph.unreachable_sensitive_total",
    "graph.sensitive_ratio",
    "graph.methods_scanned",
) + tuple(f"graph.reach_{_safe(c)}" for c in SENSITIVE_CATEGORIES) \
  + tuple(f"graph.const_{_safe(l)}" for l in CONSTANT_LABELS)

DEX_FIELDS = DEX_FIELDS + tuple(_op_field(op) for op in OPCODE_FRACTIONS)

FIELDS: tuple[str, ...] = (
    ZIP_FIELDS + AXML_FIELDS + ARSC_FIELDS + DEX_FIELDS + GRAPH_FIELDS
)

BLOCKS: tuple[str, ...] = ("zip", "axml", "arsc", "dex", "graph")


def block_of(name: str) -> str:
    """Feature block a column belongs to (zip/axml/arsc/dex/graph)."""
    return name.split(".", 1)[0]

# Android permissions that get an app into a review queue. Suffix match:
# manufacturers ship their own dialects (com.android..., android.permission...).
DANGEROUS_PERMISSIONS = frozenset({
    "SEND_SMS", "RECEIVE_SMS", "READ_SMS", "RECEIVE_MMS", "RECEIVE_WAP_PUSH",
    "CALL_PHONE", "READ_PHONE_STATE", "READ_PHONE_NUMBERS", "READ_CALL_LOG",
    "WRITE_CALL_LOG", "PROCESS_OUTGOING_CALLS", "ANSWER_PHONE_CALLS",
    "ACCESS_FINE_LOCATION", "ACCESS_COARSE_LOCATION",
    "ACCESS_BACKGROUND_LOCATION", "CAMERA", "RECORD_AUDIO",
    "READ_CONTACTS", "WRITE_CONTACTS", "GET_ACCOUNTS", "READ_CALENDAR",
    "WRITE_CALENDAR", "READ_EXTERNAL_STORAGE", "WRITE_EXTERNAL_STORAGE",
    "MANAGE_EXTERNAL_STORAGE", "BIND_ACCESSIBILITY_SERVICE",
    "SYSTEM_ALERT_WINDOW", "REQUEST_INSTALL_PACKAGES", "QUERY_ALL_PACKAGES",
    "PACKAGE_USAGE_STATS", "BODY_SENSORS", "ACTIVITY_RECOGNITION",
    "BIND_VPN_SERVICE", "BIND_DEVICE_ADMIN", "RECEIVE_BOOT_COMPLETED",
    "SMS_FINANCIAL_TRANSACTIONS", "MANAGE_ACCOUNTS", "USE_BIOMETRIC",
})


# ---------------------------------------------------------------------------
# SMALL NUMERIC HELPERS
# ---------------------------------------------------------------------------

def _num(value) -> float:
    if value is None:
        return math.nan
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    try:
        out = float(value)
    except (TypeError, ValueError):
        return math.nan
    if math.isnan(out) or math.isinf(out):
        return math.nan
    return out


def _log1p(value) -> float:
    out = _num(value)
    if math.isnan(out):
        return math.nan
    return math.log1p(max(out, 0.0))


def _ratio(numerator: float, denominator: float) -> float:
    top, bottom = _num(numerator), _num(denominator)
    if math.isnan(top) or math.isnan(bottom) or bottom == 0:
        return math.nan
    return top / bottom


def _put(row: dict, name: str, value) -> None:
    if name not in row:                      # fail loudly while developing
        raise KeyError(f"unknown feature column: {name}")
    row[name] = _num(value)


def _as_bool(value) -> float:
    if value is None:
        return math.nan
    text = str(value).strip().lower()
    if text in ("true", "1", "yes"):
        return 1.0
    if text in ("false", "0", "no"):
        return 0.0
    return math.nan


def _as_int(value) -> float:
    if value is None:
        return math.nan
    text = str(value).strip()
    try:
        return float(int(text, 0))
    except (TypeError, ValueError):
        return math.nan


def _total_from_anomalies(anomalies, prefix: str) -> int:
    """Pull 'total <prefix> ...: N' out of the DEX anomaly strings."""
    for item in anomalies:
        if item.startswith(prefix):
            tail = item.rsplit(":", 1)[-1].strip()
            if tail.isdigit():
                return int(tail)
    return 0


# ---------------------------------------------------------------------------
# BLOCK 1 - ZIP
# ---------------------------------------------------------------------------

def _zip_block(apk_path: Path, row: dict, meta: dict, ctx: dict) -> None:
    from tools.features import extract_features
    from tools.report import analyze_features
    from tools.zip_audit import audit_zip

    try:
        audit = audit_zip(apk_path)
    except Exception as exc:                  # corrupt archive: whole block NaN
        meta["errors"].append(f"zip:{type(exc).__name__}:{exc}")
        return

    entries = audit.entries
    if not entries and audit.issues:
        # audit_zip() reports a structurally broken archive instead of
        # raising; that must leave the whole block NaN, not a row of zeros.
        meta["errors"].append(f"zip:{audit.issues[0]}")
        return
    feats = extract_features(entries)
    total = max(1, feats.total_entries)

    libs = [e for e in entries
            if e.normalized_name.startswith("lib/")
            and e.normalized_name.endswith(".so")]
    assets = [e for e in entries if e.normalized_name.startswith("assets/")]
    dirs = [e for e in entries
            if e.is_directory or e.normalized_name.endswith("/")]

    _put(row, "zip.total_entries", feats.total_entries)
    _put(row, "zip.total_size_log", _log1p(feats.total_size))
    _put(row, "zip.max_entry_size_log",
         _log1p(max((e.size for e in entries), default=0)))
    _put(row, "zip.overall_ratio_log", _log1p(feats.overall_ratio))
    _put(row, "zip.max_ratio_log", _log1p(feats.max_ratio))
    _put(row, "zip.entries_over_100x",
         sum(1 for e in entries if e.ratio >= 100))
    _put(row, "zip.entries_over_1000x",
         sum(1 for e in entries if e.ratio >= 1000))
    _put(row, "zip.dex_count", feats.dex_count)
    _put(row, "zip.dex_ratio", _ratio(feats.dex_count, total))
    _put(row, "zip.zero_padded_dex_count",
         sum(1 for e in entries
             if re.match(r"classes0\d*\.dex$", e.name, re.IGNORECASE)))
    _put(row, "zip.sequential_dex_score", feats.sequential_dex_score)
    _put(row, "zip.png_ratio", _ratio(feats.png_count, total))
    _put(row, "zip.xml_ratio", _ratio(feats.xml_count, total))
    _put(row, "zip.native_lib_count", len(libs))
    _put(row, "zip.native_lib_bytes_log",
         _log1p(sum(e.size for e in libs)))
    _put(row, "zip.asset_count", len(assets))
    _put(row, "zip.asset_bytes_log", _log1p(sum(e.size for e in assets)))
    _put(row, "zip.directory_entries_ratio", _ratio(len(dirs), total))
    _put(row, "zip.max_depth", feats.max_depth)
    _put(row, "zip.depth_entropy", feats.directory_depth_entropy)
    _put(row, "zip.distinct_top_dirs", len({
        e.normalized_name.split("/", 1)[0] for e in entries
        if "/" in e.normalized_name}))
    _put(row, "zip.path_traversal", feats.path_traversal_count)
    _put(row, "zip.absolute_path", feats.absolute_path_count)
    _put(row, "zip.duplicate_names", feats.duplicate_name_count)
    _put(row, "zip.null_byte_names", feats.null_byte_name_count)
    _put(row, "zip.symlink", feats.symlink_count)
    _put(row, "zip.encrypted", feats.encrypted_entry_count)
    _put(row, "zip.zip64_extra", feats.zip64_present)
    _put(row, "zip.nested_archive", feats.nested_archive_count)
    _put(row, "zip.data_descriptor_ratio",
         _ratio(feats.data_descriptor_count, total))
    _put(row, "zip.filename_family_score", feats.filename_family_score)

    # Baseline for the "did the model beat the hand-tuned score?" comparison.
    meta["rule_score"] = analyze_features(feats).score
    meta["archive_ok"] = True


# ---------------------------------------------------------------------------
# BLOCK 2 - AXML MANIFEST
# ---------------------------------------------------------------------------

COMPONENT_TAGS = ("activity", "activity-alias", "service", "receiver",
                  "provider")


def _iter_elements(node):
    stack = [node]
    while stack:
        current = stack.pop()
        yield current
        stack.extend(getattr(current, "children", ()) or ())


def _attr(node, *names):
    for attribute in getattr(node, "attributes", ()) or ():
        if getattr(attribute, "name", None) in names:
            return getattr(attribute, "value", None)
    return None


def _axml_block(apk_path: Path, row: dict, meta: dict, ctx: dict) -> None:
    from tools.manifest import BinaryXmlError, parse_manifest

    try:
        _header, pool, resource_map, root = parse_manifest(apk_path)
    except (BinaryXmlError, Exception) as exc:
        meta["errors"].append(f"axml:{type(exc).__name__}:{exc}")
        return
    if root is None:
        meta["errors"].append("axml:no-root")
        return

    nodes = list(_iter_elements(root))
    tags = Counter(getattr(n, "tag", None) for n in nodes)
    permissions = [
        _attr(n, "name") for n in nodes
        if getattr(n, "tag", None) in ("uses-permission",
                                       "uses-permission-sdk-23")
    ]
    permissions = [p for p in permissions if p]
    permission_suffixes = {str(p).rsplit(".", 1)[-1] for p in permissions}

    try:
        from tools.manifest import is_valid_xml_name
    except ImportError:                      # pragma: no cover - older manifest.py
        is_valid_xml_name = lambda _n: True  # noqa: E731

    suspicious = sum(
        1 for n in nodes for a in (getattr(n, "attributes", ()) or ())
        if not is_valid_xml_name(getattr(a, "name", "") or "")
    )

    exported = 0
    for node in nodes:
        if getattr(node, "tag", None) in COMPONENT_TAGS:
            if _as_bool(_attr(node, "exported")) == 1.0:
                exported += 1

    uses_sdk = next((n for n in nodes if getattr(n, "tag", None) == "uses-sdk"),
                    None)
    application = next(
        (n for n in nodes if getattr(n, "tag", None) == "application"), None)

    try:
        from tools.smali import manifest_components
        ctx["components"] = manifest_components(root)
    except Exception:
        ctx["components"] = []

    _put(row, "axml.string_count", len(getattr(pool, "strings", ()) or ()))
    _put(row, "axml.resource_id_count",
         len(getattr(resource_map, "ids", ()) or ()))
    _put(row, "axml.permission_count", len(permissions))
    _put(row, "axml.dangerous_permission_count",
         len(permission_suffixes & DANGEROUS_PERMISSIONS))
    _put(row, "axml.custom_permission_count", tags.get("permission", 0))
    _put(row, "axml.component_count",
         sum(tags.get(tag, 0) for tag in COMPONENT_TAGS))
    _put(row, "axml.activity_count",
         tags.get("activity", 0) + tags.get("activity-alias", 0))
    _put(row, "axml.service_count", tags.get("service", 0))
    _put(row, "axml.receiver_count", tags.get("receiver", 0))
    _put(row, "axml.provider_count", tags.get("provider", 0))
    _put(row, "axml.exported_component_count", exported)
    _put(row, "axml.intent_filter_count", tags.get("intent-filter", 0))
    _put(row, "axml.suspicious_attribute_count", suspicious)
    _put(row, "axml.split_count",
         tags.get("uses-split", 0)
         + (1 if _as_bool(_attr(application, "isSplitRequired")
                          if application is not None else None) == 1.0 else 0))
    _put(row, "axml.min_sdk",
         _as_int(_attr(uses_sdk, "minSdkVersion")) if uses_sdk else math.nan)
    _put(row, "axml.target_sdk",
         _as_int(_attr(uses_sdk, "targetSdkVersion")) if uses_sdk else math.nan)
    _put(row, "axml.allow_backup",
         _as_bool(_attr(application, "allowBackup")) if application else math.nan)
    _put(row, "axml.debuggable",
         _as_bool(_attr(application, "debuggable")) if application else math.nan)
    _put(row, "axml.cleartext_traffic",
         _as_bool(_attr(application, "usesCleartextTraffic"))
         if application else math.nan)
    _put(row, "axml.extract_native_libs",
         _as_bool(_attr(application, "extractNativeLibs"))
         if application else math.nan)

    filenames = " ".join(
        str(getattr(n, "tag", "")) + " " + " ".join(
            str(getattr(a, "value", "")) for a in
            (getattr(n, "attributes", ()) or ()))
        for n in nodes
    )
    _put(row, "axml.flag_vpn_service",
         1.0 if "BIND_VPN_SERVICE" in filenames
         or "android.net.VpnService" in filenames else 0.0)
    _put(row, "axml.flag_accessibility_service",
         1.0 if "BIND_ACCESSIBILITY_SERVICE" in filenames else 0.0)
    _put(row, "axml.flag_device_admin",
         1.0 if "BIND_DEVICE_ADMIN" in filenames else 0.0)
    _put(row, "axml.flag_boot_receiver",
         1.0 if ("RECEIVE_BOOT_COMPLETED" in filenames
                 and tags.get("receiver", 0)) else 0.0)
    _put(row, "axml.flag_install_packages",
         1.0 if "REQUEST_INSTALL_PACKAGES" in filenames else 0.0)
    _put(row, "axml.flag_query_all_packages",
         1.0 if "QUERY_ALL_PACKAGES" in filenames else 0.0)


# ---------------------------------------------------------------------------
# BLOCK 3 - RESOURCES.ARSC
# ---------------------------------------------------------------------------

def _arsc_block(apk_path: Path, row: dict, meta: dict, ctx: dict) -> None:
    from tools.features import extract_arsc_features
    from tools.resource_table import (ResourceTableError,
                                      inspect_resource_entry,
                                      parse_resource_table)

    try:
        info = inspect_resource_entry(apk_path)
        _put(row, "arsc.table_bytes_log", _log1p(info.uncompressed_size))
    except (ResourceTableError, Exception):
        pass                                  # absent arsc is a valid fact

    try:
        table = parse_resource_table(apk_path)
    except (ResourceTableError, Exception) as exc:
        meta["errors"].append(f"arsc:{type(exc).__name__}:{exc}")
        return

    f = extract_arsc_features(table)
    _put(row, "arsc.package_count", f.package_count)
    _put(row, "arsc.resource_entries", f.resource_entries)
    _put(row, "arsc.config_variants", f.config_variants)
    _put(row, "arsc.entries_per_config",
         _ratio(f.resource_entries, f.config_variants))
    _put(row, "arsc.type_count", f.type_count)
    _put(row, "arsc.string_values", f.string_value_count)
    _put(row, "arsc.reference_values", f.reference_value_count)
    _put(row, "arsc.attribute_refs", f.attribute_ref_count)
    _put(row, "arsc.complex_bags", f.complex_bag_count)
    _put(row, "arsc.compact_entries", f.compact_entry_count)
    _put(row, "arsc.public_entries", f.public_entry_count)
    _put(row, "arsc.weak_entries", f.weak_entry_count)
    _put(row, "arsc.sparse_types", f.sparse_type_count)
    _put(row, "arsc.offset16_types", f.offset16_type_count)
    _put(row, "arsc.anomalies", f.anomaly_count)
    _put(row, "arsc.null_chunks", f.null_chunk_count)
    _put(row, "arsc.library_packages", f.library_package_count)
    _put(row, "arsc.scramble_score", f.scramble_score)
    _put(row, "arsc.scramble_ratio",
         _ratio(f.scramble_score, f.string_value_count))
    _put(row, "arsc.rebuild_tool", 1.0 if f.rebuild_fingerprints else 0.0)


# ---------------------------------------------------------------------------
# BLOCK 4 - DEX (headers always, opcodes only when deep)
# ---------------------------------------------------------------------------

def _dex_block(apk_path: Path, row: dict, meta: dict, ctx: dict,
               deep: bool, deep_dex_limit: int) -> None:
    from tools.dex import (DexError, discover_dex_entries,
                           walk_instructions_histogram, parse_dex_from_apk,
                           sniff_dex_magic)

    try:
        names = discover_dex_entries(apk_path)
    except Exception as exc:
        meta["errors"].append(f"dex:{type(exc).__name__}:{exc}")
        return

    real, fake = [], []
    for name in names:
        try:
            (real if sniff_dex_magic(apk_path, name) else fake).append(name)
        except (DexError, ValueError):
            fake.append(name)

    parsed: list[tuple[str, object]] = []
    failures = 0
    for name in real:
        try:
            dex = parse_dex_from_apk(apk_path, name)
        except (DexError, ValueError):
            failures += 1
            continue
        parsed.append((name, dex))
        meta["dex_hashes"].append(hashlib.sha256(dex.data).hexdigest()[:16])

    _put(row, "dex.entry_count", len(names))
    _put(row, "dex.real_count", len(parsed))
    _put(row, "dex.fake_count", len(fake))
    _put(row, "dex.fake_ratio", _ratio(len(fake), len(names)))
    _put(row, "dex.parse_failures", failures)
    _put(row, "dex.zero_padded_ratio",
         _ratio(sum(1 for n in names
                    if re.match(r"classes0\d*\.dex$", n, re.IGNORECASE)),
                len(names)))

    if parsed:
        classes = [len(d.class_defs) for _n, d in parsed]
        methods = [len(d.method_ids) for _n, d in parsed]
        _put(row, "dex.strings", sum(len(d.string_offsets) for _n, d in parsed))
        _put(row, "dex.types", sum(len(d.type_ids) for _n, d in parsed))
        _put(row, "dex.protos", sum(len(d.proto_ids) for _n, d in parsed))
        _put(row, "dex.fields", sum(len(d.field_ids) for _n, d in parsed))
        _put(row, "dex.methods", sum(methods))
        _put(row, "dex.classes", sum(classes))
        _put(row, "dex.classes_max_per_dex", max(classes, default=0))
        _put(row, "dex.classes_per_dex_mean",
             _ratio(sum(classes), len(classes)))
        _put(row, "dex.methods_per_class",
             _ratio(sum(methods), sum(classes)))
        _put(row, "dex.checksum_bad",
             sum(1 for _n, d in parsed if not d.checksum_ok))
        _put(row, "dex.signature_bad",
             sum(1 for _n, d in parsed if not d.signature_ok))

        anomalies = [a for _n, d in parsed for a in d.anomalies]
        misaligned = sum(_total_from_anomalies(d.anomalies,
                                               "total misaligned methods:")
                         for _n, d in parsed)
        oversized = sum(_total_from_anomalies(
            d.anomalies, "total oversized payload methods:") for _n, d in parsed)
        _put(row, "dex.anomaly_count", len(anomalies))
        _put(row, "dex.misaligned_methods", misaligned)
        _put(row, "dex.oversized_payload_methods", oversized)
        _put(row, "dex.anomaly_per_class", _ratio(len(anomalies),
                                                  sum(classes)))
    else:
        for name in ("dex.strings", "dex.types", "dex.protos", "dex.fields",
                     "dex.methods", "dex.classes", "dex.classes_max_per_dex",
                     "dex.classes_per_dex_mean", "dex.methods_per_class",
                     "dex.checksum_bad", "dex.signature_bad",
                     "dex.anomaly_count", "dex.misaligned_methods",
                     "dex.oversized_payload_methods", "dex.anomaly_per_class"):
            _put(row, name, math.nan)

    if not deep or not parsed:
        return

    # ---- opcode distribution of the primary DEX -------------------------
    name, dex = next(((n, d) for n, d in parsed if n == "classes.dex"),
                     parsed[0])
    histogram: Counter = Counter()
    for _cdef, _method, code in dex.iter_code_items():
        try:
            histogram.update(walk_instructions_histogram(dex.data, code))
        except Exception:
            continue
    ctx["deep_dex"] = dex
    ctx["deep_graph_ready"] = True

    from tools.smali import OPCODE_NAMES

    by_name: Counter = Counter()
    total_units = 0
    for key, count in histogram.items():
        if isinstance(key, int):
            by_name[OPCODE_NAMES.get(key, f"op_{key:#04x}")] += count
            total_units += count

    overrun = histogram.get("<overrun>", 0)
    _put(row, "dex.insn_total_log", _log1p(total_units))
    _put(row, "dex.op_overrun_ratio", _ratio(overrun, total_units))

    if total_units:
        entropy = 0.0
        for count in by_name.values():
            p = count / total_units
            entropy -= p * math.log2(p)
        _put(row, "dex.op_entropy", entropy)
    for opcode in OPCODE_FRACTIONS:
        _put(row, _op_field(opcode),
             _ratio(by_name.get(opcode, 0), total_units))


# ---------------------------------------------------------------------------
# BLOCK 5 - CALL GRAPH + SENSITIVE API REACHABILITY (deep only)
# ---------------------------------------------------------------------------

def _graph_block(apk_path: Path, row: dict, meta: dict, ctx: dict) -> None:
    dex = ctx.get("deep_dex")
    if dex is None:
        return

    from tools.api_report import build_api_report
    from tools.smali import build_call_graph

    try:
        graph = build_call_graph(dex, components=ctx.get("components") or None)
        stats = graph.stats()
    except Exception as exc:
        meta["errors"].append(f"graph:{type(exc).__name__}:{exc}")
        return

    classes = stats.get("classes", 0) or 0
    methods = stats.get("methods", 0) or 0
    _put(row, "graph.classes", classes)
    _put(row, "graph.methods", methods)
    _put(row, "graph.internal_edges", stats.get("internal_edges", 0))
    _put(row, "graph.external_targets", stats.get("external_targets", 0))
    _put(row, "graph.dead_candidates", stats.get("dead_candidates", 0))
    _put(row, "graph.dead_ratio", _ratio(stats.get("dead_candidates", 0),
                                         methods))
    _put(row, "graph.entry_points", len(graph.entry_points()))
    _put(row, "graph.edges_per_class",
         _ratio(stats.get("internal_edges", 0), classes))
    _put(row, "graph.external_per_method",
         _ratio(stats.get("external_targets", 0), methods))

    try:
        api = build_api_report(dex, graph)
    except Exception as exc:
        meta["errors"].append(f"api:{type(exc).__name__}:{exc}")
        return

    reachable = sum(len(v) for v in api.reachable_sensitive.values())
    unreachable = sum(len(v) for v in api.unreachable_sensitive.values())
    for category in SENSITIVE_CATEGORIES:
        _put(row, f"graph.reach_{_safe(category)}",
             len(api.reachable_sensitive.get(category, ())))
    _put(row, "graph.reachable_sensitive_total", reachable)
    _put(row, "graph.unreachable_sensitive_total", unreachable)
    _put(row, "graph.sensitive_ratio",
         _ratio(reachable, reachable + unreachable))
    _put(row, "graph.methods_scanned", api.total_methods)

    counted = Counter()
    for finding in api.findings:
        for label in finding.constants:
            counted[label] += len(finding.constants[label])
    for label in CONSTANT_LABELS:
        _put(row, f"graph.const_{_safe(label)}", counted.get(label, 0))


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

def extract_apk_features(apk_path: str | Path, deep: bool = True,
                         deep_dex_limit: int = 1
                         ) -> tuple[dict[str, float], dict]:
    """APK -> (feature_row, meta).

    deep=False skips the opcode histogram and the call-graph/API pass, which
    is where most of the runtime goes (measured on sample2.apk: ~13 s deep vs
    ~2 s header-only). meta is NOT fed to the model.
    """
    apk_path = Path(apk_path)
    row = {name: math.nan for name in FIELDS}
    meta: dict = {
        "apk": apk_path.name,
        "feature_version": FEATURE_VERSION,
        "deep": bool(deep),
        "errors": [],
        "dex_hashes": [],
        "dex_set_hash": "",
        "rule_score": None,
        "archive_ok": False,
    }
    ctx: dict = {}

    _zip_block(apk_path, row, meta, ctx)
    _axml_block(apk_path, row, meta, ctx)
    _arsc_block(apk_path, row, meta, ctx)
    _dex_block(apk_path, row, meta, ctx, deep=deep,
               deep_dex_limit=deep_dex_limit)
    if deep:
        _graph_block(apk_path, row, meta, ctx)

    hashes = sorted(meta["dex_hashes"])
    meta["dex_set_hash"] = (
        hashlib.sha256("|".join(hashes).encode()).hexdigest()[:16]
                            if hashes else "")
    return row, meta
