import sys
from pathlib import Path

from tools.binary import (
    read_head,
    read_tail,
    has_zip_signature,
    find_eocd,
    read_eocd,
)

from tools.zip_audit import (
    audit_zip,
    parse_zip,
)

from tools.features import (
    extract_features,
)

from tools.report import (
    analyze_features,
    print_report,
)

from tools.manifest import (
    parse_manifest,
    print_tree,
    print_anomalies,
    BinaryXmlError,
)

from tools.resource_table import (
    ResourceTableError,
    parse_resource_table,
    print_resource_table,
    print_anomalies as print_arsc_anomalies,
    inspect_resource_entry,
)

from tools.features import (
    extract_arsc_features,
    print_arsc_features,
)

from tools.dex import (
    DexError,
    discover_dex_entries,
    parse_dex_from_apk,
    print_dex_report,
    sniff_dex_magic,
)

from tools.smali import (
    build_call_graph,
    build_resource_context,
    export_smali_tree,
    manifest_components,
    print_call_graph_report,
    render_class,
)
from tools.api_report import build_api_report, print_api_report
from tools.translate import TranslatorConfig, config_from_env, probe as probe_llm


# ============================================================================
# PATHS
# ============================================================================

BASE_DIR = Path(__file__).parent
APK_PATH = BASE_DIR / "apks" / "sample.apk"

# Phase 5 output root: smali trees + call-graph artifacts + (optional)
# LLM-translated Java. Out/ is disposable, never committed.
OUT_DIR = BASE_DIR / "out"

# LLM settings come from env/defaults at startup (see tools/translate.py);
# the endpoint itself is only probed when translation is requested.
LLM_CONFIG: TranslatorConfig = config_from_env()
RESOURCE_CONTEXT: dict[int, str] = {}

# Translation is opt-in via --translate (Phase 5 spec: smali analysis is
# offline-first; the LLM pass needs a live model endpoint).
LLM_TRANSLATE = "--translate" in sys.argv

# Full smali export (every class of every real DEX) is opt-in via
# --export-smali: the default stage renders only the first classes.
EXPORT_SMALI = "--export-smali" in sys.argv

# Optional ML triage pass: needs models/apkai_triage.joblib from
# `python3 -m tools.ml.train`; the core stages never depend on it.
ML_SCORE = "--ml" in sys.argv
ML_MODEL_PATH = BASE_DIR / "models" / "apkai_triage.joblib"


# ============================================================================
# BINARY VERIFICATION
# ============================================================================

def run_binary_verification():
    # Bounded reads only: the whole archive is never loaded into memory.
    head = read_head(APK_PATH, 4)
    tail, tail_offset = read_tail(APK_PATH)

    print("=" * 70)
    print("BINARY VERIFICATION")
    print("=" * 70)

    print(f"File Size : {APK_PATH.stat().st_size:,} bytes")
    print(f"ZIP Magic : {has_zip_signature(head)}")

    offset = find_eocd(tail)
    print(f"EOCD Offset: {tail_offset + offset if offset != -1 else -1}")

    fields = read_eocd(tail)

    print(f"Signature      : {fields['signature']}")
    print(f"Total Entries  : {fields['total_entries']}")
    print(f"CD Size        : {fields['cd_size']}")
    print(f"CD Offset      : {fields['cd_offset']}")
    print(f"Comment Length : {fields['comment_len']}")
    print(f"ZIP64 Sentinel : {fields['is_zip64']}")

    print()


# ============================================================================
# ZIP METADATA / FORENSIC AUDIT
# ============================================================================

def run_zip_audit():
    # audit_zip() is the pass that flags duplicates/symlinks/encryption; going
    # straight to parse_zip() left those features permanently at zero.
    audit = audit_zip(APK_PATH)

    if not audit.entries and audit.issues:
        print("=" * 70)
        print("ZIP FEATURES")
        print("=" * 70)
        print()
        print("[ZIP ERROR] unreadable archive")
        for issue in audit.issues:
            print(f"  {issue}")
        print()
        return

    features = extract_features(audit.entries)

    report = analyze_features(features)

    print("=" * 70)
    print("ZIP FEATURES")
    print("=" * 70)

    print()

    print_report(report)

    print()


# ============================================================================
# MANIFEST / AXML ANALYSIS
# ============================================================================

def run_manifest_analysis():
    print("=" * 70)
    print("ANDROID MANIFEST / AXML ANALYSIS")
    print("=" * 70)

    try:
        # ------------------------------------------------------------
        # High-level parser
        #
        # APK
        #  ↓
        # AndroidManifest.xml
        #  ↓
        # Binary AXML parser
        #  ↓
        # string pool
        # resource map
        # namespaces
        # attributes
        # XML tree
        # ------------------------------------------------------------

        (
            xml_header,
            pool,
            resource_map,
            root,
        ) = parse_manifest(APK_PATH)

    except BinaryXmlError as exc:

        print()
        print("[AXML ERROR]")
        print(exc)
        print()

        return

    # ----------------------------------------------------------------
    # AXML HEADER
    # ----------------------------------------------------------------

    print()
    print("=" * 60)
    print("BINARY AXML HEADER")
    print("=" * 60)

    print(f"Chunk Type  : 0x{xml_header.chunk_type:04X}")
    print(f"Header Size : {xml_header.header_size}")
    print(f"File Size   : {xml_header.file_size}")

    # ----------------------------------------------------------------
    # STRING POOL
    # ----------------------------------------------------------------

    print()
    print("=" * 60)
    print("STRING POOL")
    print("=" * 60)

    print(f"String Count : {len(pool.strings)}")
    print(f"Style Count  : {pool.header.style_count}")

    utf8 = bool(pool.header.flags & (1 << 8))

    print(f"Encoding     : {'UTF-8' if utf8 else 'UTF-16'}")

    # ----------------------------------------------------------------
    # FIRST 20 STRINGS
    # ----------------------------------------------------------------

    print()
    print("=" * 60)
    print("FIRST 20 STRINGS")
    print("=" * 60)

    print()
    print("Index | Offset | String")
    print("-" * 70)

    count = min(20, len(pool.strings))

    for i in range(count):

        print(
            f"{i:5d} | "
            f"{pool.offsets[i]:6d} | "
            f"{pool.strings[i]!r}"
        )

    # ----------------------------------------------------------------
    # RESOURCE MAP
    # ----------------------------------------------------------------

    print()
    print("=" * 60)
    print("RESOURCE MAP")
    print("=" * 60)

    print(f"Resource IDs : {len(resource_map.ids)}")

    if resource_map.ids:

        print()
        print("Index | Resource ID")
        print("-" * 40)

        for i, resource_id in enumerate(resource_map.ids[:20]):

            print(
                f"{i:5d} | "
                f"0x{resource_id:08X}"
            )

    else:

        print("No resource map entries.")

    # ----------------------------------------------------------------
    # ROOT MANIFEST
    # ----------------------------------------------------------------

    print()
    print("=" * 60)
    print("ROOT ELEMENT")
    print("=" * 60)

    if root is None:

        print("No root XML element found.")

        return

    print(f"Tag  : {root.tag}")
    print(f"Line : {root.line}")

    if root.prefix:
        print(f"Prefix : {root.prefix}")

    if root.namespace_uri:
        print(f"Namespace : {root.namespace_uri}")

    # ----------------------------------------------------------------
    # ROOT ATTRIBUTES
    # ----------------------------------------------------------------

    print()
    print("=" * 60)
    print("ROOT MANIFEST ATTRIBUTES")
    print("=" * 60)

    if not root.attributes:

        print("No attributes.")

    else:

        for attribute in root.attributes:

            prefix = ""

            if attribute.prefix:
                prefix = f"{attribute.prefix}:"

            name = attribute.name or "<unknown>"

            value = attribute.value

            if value is None:
                value = "<none>"

            print(
                f"{prefix}{name:35} = {value}"
            )

            # Show resource ID when available.
            if attribute.resource_id is not None:

                print(
                    f"{'':4}"
                    f"resource_id = "
                    f"0x{attribute.resource_id:08X}"
                )

    # ----------------------------------------------------------------
    # COMPLETE XML TREE
    # ----------------------------------------------------------------

    print()
    print("=" * 60)
    print("XML TREE")
    print("=" * 60)

    print_tree(root)
    print_anomalies(root)

    print()


# ============================================================================
# RESOURCE TABLE / ARSC ANALYSIS
# ============================================================================

def run_resource_table_analysis():
    print("=" * 70)
    print("RESOURCE TABLE / RESOURCES.ARSC ANALYSIS")
    print("=" * 70)

    try:
        info = inspect_resource_entry(APK_PATH)
    except ResourceTableError as exc:
        print()
        print("[ARSC ERROR]")
        print(exc)
        print()
        return

    print()
    print(f"Entry        : {info.name}")
    print(f"Compression  : {info.compression_type} "
          f"({'stored' if info.compression_type == 0 else 'deflated'})")
    print(f"Size         : {info.uncompressed_size:,} bytes "
          f"(compressed {info.compressed_size:,})")

    try:
        table = parse_resource_table(APK_PATH)
    except ResourceTableError as exc:
        print()
        print("[ARSC ERROR]")
        print(exc)
        print()
        return

    print()
    print_resource_table(table, max_entries_per_type=6)
    print_arsc_anomalies(table)

    features = extract_arsc_features(table)
    print_arsc_features(features)

    print()


# ============================================================================
# DEX ANALYSIS
# ============================================================================

def run_dex_analysis(max_full_reports: int = 3):
    print("=" * 70)
    print("DEX ANALYSIS")
    print("=" * 70)

    try:
        names = discover_dex_entries(APK_PATH)
    except ValueError as exc:
        print(f"[DEX ERROR] {exc}")
        return

    if not names:
        print("No classes*.dex entries found in APK.")
        return

    print(f"DEX entries: {len(names)}")

    # ------------------------------------------------------------------
    # Forensic signal: mass DEX splitting.
    # Android's 64K method limit leads legit apps to a handful of DEX
    # files. Hundreds of tiny DEX files (especially zero-padded names
    # like classes001.dex) indicate a packer/scatter tool deliberately
    # fragmenting code to defeat static analysis.
    # ------------------------------------------------------------------

    if len(names) > 20:
        print(
            f"[!] ANOMALY: {len(names)} DEX files - mass DEX splitting "
            "(packing/evasion indicator)"
        )
    zero_padded = [n for n in names if n.startswith("classes0")]
    if zero_padded:
        print(
            f"[!] ANOMALY: {len(zero_padded)} zero-padded DEX names "
            "(non-standard toolchain)"
        )

    total_methods = 0
    total_classes = 0
    failed = 0

    # ------------------------------------------------------------------
    # Sniff every entry before parsing: zero-padded tarpit "DEX" files
    # must be classified as FAKE, not as parse failures. The bounded
    # reader caps the sniff at 64 KB, so bombs cannot expand.
    # ------------------------------------------------------------------

    real_dex: list[str] = []
    fake_dex: list[str] = []
    for name in names:
        try:
            is_dex = sniff_dex_magic(APK_PATH, name)
        except (DexError, ValueError):
            is_dex = False
        (real_dex if is_dex else fake_dex).append(name)

    if fake_dex:
        declared_bytes = 0
        try:
            declared = {e.name: e.size for e in parse_zip(APK_PATH)}
            declared_bytes = sum(declared.get(n, 0) for n in fake_dex)
        except ValueError:
            pass
        print(
            f"[!] ANOMALY: {len(fake_dex)} FAKE DEX entries - no dex magic "
            f"({declared_bytes / (1024 * 1024):.0f} MB declared content). "
            "Padding/tarpit disguised as code to waste analysis time."
        )

    # Anomaly detail was only ever printed for the first few DEX files; keep
    # the ones from the rest so operators can actually act on them.
    unprinted_anomalies: list[tuple[str, list[str]]] = []

    for position, name in enumerate(real_dex):
        try:
            dex = parse_dex_from_apk(APK_PATH, name)
        except DexError as exc:
            print(f"[DEX ERROR] {name}: {exc}")
            failed += 1
            continue

        total_methods += len(dex.method_ids)
        total_classes += len(dex.class_defs)

        if position >= max_full_reports:
            checksum = "OK" if dex.checksum_ok else "BAD"
            signature = "OK" if dex.signature_ok else "BAD"
            print(
                f"  {name:22} classes={len(dex.class_defs):5} "
                f"methods={len(dex.method_ids):6} "
                f"checksum={checksum} sha1={signature} "
                f"anomalies={len(dex.anomalies)}"
            )
            if dex.anomalies:
                unprinted_anomalies.append((name, dex.anomalies))
            continue

        print()
        print(f"--- {name} " + "-" * 55)
        print_dex_report(dex, max_classes=10)

        print()
        refs = dex.api_references()
        print(f"Distinct API references: {len(refs)}")
        for ref, count in refs.most_common(8):
            print(f"  {count:5}x  {ref}")

    if unprinted_anomalies:
        print()
        print(f"DEX ANOMALIES (files beyond the first {max_full_reports} "
              "full reports)")
        print("-" * 70)
        for name, items in unprinted_anomalies:
            for item in items[:5]:
                print(f"  {name}: {item}")
            if len(items) > 5:
                print(f"  {name}: ... +{len(items) - 5} more")

    print()
    print("DEX AGGREGATE")
    print("-" * 70)
    print(
        f"DEX files          : {len(real_dex)} real, "
        f"{len(fake_dex)} fake/tarpit ({failed} failed to parse)"
    )
    print(f"Total classes      : {total_classes:,}")
    print(f"Total method refs  : {total_methods:,}")
    print()


# ============================================================================
# MAIN
# ============================================================================

def run_smali_analysis(max_full_reports: int = 3, translate: bool = False,
                       export: bool = False):
    print("=" * 70)
    print("SMALI / CALL GRAPH ANALYSIS")
    print("=" * 70)

    global RESOURCE_CONTEXT
    try:
        dex = parse_dex_from_apk(APK_PATH, "classes.dex")
    except (DexError, ValueError) as exc:
        print(f"[SMALI ERROR] {exc}")
        return

    # lazy: 0x7f id -> '@pkg:type/name' for const rendering (LLM context)
    if not RESOURCE_CONTEXT:
        try:
            RESOURCE_CONTEXT = build_resource_context(APK_PATH)
        except Exception as exc:
            print(f"[SMALI WARN] no resource context ({exc})")

    # ---- manifest entry points ------------------------------------------
    components: list[tuple[str, str]] = []
    try:
        _xml, _pool, _rmap, root = parse_manifest(APK_PATH)
        components = manifest_components(root)
    except BinaryXmlError as exc:
        print(f"[SMALI WARN] manifest unreadable, entry points generic: {exc}")

    # ---- call graph ------------------------------------------------------
    graph = build_call_graph(dex, components=components)
    print_call_graph_report(graph, top=10)

    # ---- sensitive-API reachability (entry points -> risky calls) --------
    import io
    from contextlib import redirect_stdout
    api = build_api_report(dex, graph)
    buf = io.StringIO()
    with redirect_stdout(buf):
        print_api_report(api)
    text = buf.getvalue()
    print(text)
    (OUT_DIR / "api_report.txt").write_text(text)

    # ---- artifacts -------------------------------------------------------
    smali_dir = OUT_DIR / "smali"
    smali_dir.mkdir(parents=True, exist_ok=True)
    classes: list[tuple[str, str]] = []
    for cdef in dex.class_defs:
        name = dex.type_desc(cdef.class_idx) or f"<class@{cdef.class_idx}>"
        try:
            classes.append((name, render_class(
                dex, cdef, resource_context=RESOURCE_CONTEXT)))
        except Exception as exc:      # hostile class: log, keep going
            print(f"[SMALI WARN] render failed for {name}: "
                  f"{type(exc).__name__}: {exc}")

    for name, text in classes[:max_full_reports]:
        out = smali_dir / f"{name.replace('/', '_').lstrip('L').rstrip(';')}.smali"
        out.write_text(text)
    print(f"\n[SMALI] rendered {len(classes)} classes, "
          f"wrote first {min(max_full_reports, len(classes))} to {smali_dir}")

    (OUT_DIR / "callgraph.mmd").write_text(graph.to_mermaid())
    (OUT_DIR / "callgraph.dot").write_text(graph.to_dot())

    # ---- full export: every class of every real DEX ----------------------
    if export:
        print("\n[export] rendering ALL classes of ALL real DEX files "
              f"-> {OUT_DIR / 'smali_full'}")
        stats = export_smali_tree(
            APK_PATH, OUT_DIR / "smali_full",
            resource_context=RESOURCE_CONTEXT or None)
        if stats["fake_dex"]:
            print(f"[export] skipped {len(stats['fake_dex'])} fake/tarpit "
                  "DEX entries")
        for line in stats["failures"][:10]:
            print(f"[export FAIL] {line}")
        print(f"[export] {stats['dex_files']} DEX files, "
              f"{stats['written']:,} classes written, "
              f"{stats['failed']} failed")

    # ---- optional: LLM smali -> java -------------------------------------
    if translate:
        if not probe_llm(LLM_CONFIG):
            print("[translate] skipped: no LLM endpoint (start "
                  "SmaliLLM-server or set APKAI_LLM_URL)")
            return
        from tools.translate import translate_class, TranslateStats

        # Entry-point classes can live in ANY of the APK's DEX files
        # (sample.apk ships 16 real ones) - index them all before
        # declaring a class missing.
        wanted = [n for _t, n in components[:10]]
        if not wanted:
            wanted = [dex.type_desc(c.class_idx)
                      for c in dex.class_defs[:10]]

        try:
            all_names = discover_dex_entries(APK_PATH)
        except ValueError:
            all_names = ["classes.dex"]
        dex_names = []
        for n in all_names:
            try:
                if sniff_dex_magic(APK_PATH, n):
                    dex_names.append(n)
            except (DexError, ValueError):
                pass

        locations: dict[str, tuple] = {}
        for name in dex_names:
            try:
                dx = parse_dex_from_apk(APK_PATH, name)
            except (DexError, ValueError):
                continue
            for cdef in dx.class_defs:
                desc = dx.type_desc(cdef.class_idx)
                if desc in wanted and desc not in locations:
                    locations[desc] = (dx, cdef)

        if not locations:
            print("[translate] WARN: none of the target classes found "
                  "in any DEX")
            return

        stats = TranslateStats()
        java_dir = OUT_DIR / "java"
        java_dir.mkdir(parents=True, exist_ok=True)
        print(f"[translate] translating {len(locations)}/{len(wanted)} "
              "target classes -> out/java/ ...")
        for desc, (dx, cdef) in locations.items():
            try:
                smali = render_class(dx, cdef,
                                     resource_context=RESOURCE_CONTEXT)
            except Exception as exc:
                print(f"[translate] render failed for {desc}: "
                      f"{type(exc).__name__}: {exc}")
                continue
            java = translate_class(smali, LLM_CONFIG, stats)
            out = java_dir / (desc.strip("L;").replace("/", "_") + ".java")
            out.write_text(java)
        print(f"[translate] {stats.as_line()} -> {java_dir}")


# ============================================================================
# ML TRIAGE (optional)
# ============================================================================

def run_ml_triage():
    print()
    print("=" * 70)
    print("Stage 7: ML triage (calibrated model + feature attribution)")
    print("=" * 70)
    print()

    if not ML_MODEL_PATH.exists():
        print(f"[ML SKIP] no artifact at {ML_MODEL_PATH}")
        print("          train one: python3 -m tools.ml.train "
              "--dataset data/dataset.jsonl")
        print()
        return

    try:
        from tools.ml.features import extract_apk_features
        from tools.ml.model import ArtifactVersionError, TriageModel
    except ImportError as exc:
        print(f"[ML SKIP] sklearn/numpy not installed ({exc})")
        print()
        return

    try:
        model = TriageModel.load(ML_MODEL_PATH)
    except ArtifactVersionError as exc:
        print(f"[ML ERROR] {exc}")
        print()
        return

    features, meta = extract_apk_features(APK_PATH, deep=True)
    result = model.explain(features, top=8)

    print(f"Rule score (heuristic) : {meta.get('rule_score')}/100")
    for line in result.as_lines():
        print(line)
    if meta["errors"]:
        print(f"[ML NOTE] {len(meta['errors'])} feature block(s) degraded: "
              f"{meta['errors'][:2]}")
    print()


def main():

    print()
    print("=" * 70)
    print("APKAI")
    print("Android APK Forensic Analyzer")
    print("=" * 70)
    print()

    if not APK_PATH.exists():

        print("[ERROR]")
        print(f"APK not found: {APK_PATH}")
        print()

        return

    # ------------------------------------------------------------
    # Stage 1: Binary verification
    # ------------------------------------------------------------

    run_binary_verification()

    # ------------------------------------------------------------
    # Stage 2: ZIP forensic analysis
    # ------------------------------------------------------------

    run_zip_audit()

    # ------------------------------------------------------------
    # Stage 3: Android Binary XML analysis
    # ------------------------------------------------------------

    run_manifest_analysis()

    # ------------------------------------------------------------
    # Stage 4: resources.arsc analysis
    # ------------------------------------------------------------

    run_resource_table_analysis()

    # ------------------------------------------------------------
    # Stage 5: DEX analysis
    # ------------------------------------------------------------

    run_dex_analysis()

    # ------------------------------------------------------------
    # Stage 6: Smali reconstruction + call graph (+ optional LLM pass)
    # ------------------------------------------------------------

    run_smali_analysis(translate=LLM_TRANSLATE, export=EXPORT_SMALI)

    # ------------------------------------------------------------
    # Stage 7: ML triage (only with --ml and a trained artifact)
    # ------------------------------------------------------------

    if ML_SCORE:
        run_ml_triage()


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()