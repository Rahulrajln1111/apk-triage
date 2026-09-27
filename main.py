from pathlib import Path

from tools.binary import (
    read_bytes,
    has_zip_signature,
    find_eocd,
    read_eocd,
)

from tools.zip_audit import (
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


# ============================================================================
# PATHS
# ============================================================================

BASE_DIR = Path(__file__).parent
APK_PATH = BASE_DIR / "apks" / "sample.apk"


# ============================================================================
# BINARY VERIFICATION
# ============================================================================

def run_binary_verification():
    raw = read_bytes(APK_PATH)

    print("=" * 70)
    print("BINARY VERIFICATION")
    print("=" * 70)

    print(f"File Size : {len(raw):,} bytes")
    print(f"ZIP Magic : {has_zip_signature(raw)}")

    offset = find_eocd(raw)
    print(f"EOCD Offset: {offset}")

    fields = read_eocd(raw)

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
    entries = parse_zip(APK_PATH)

    features = extract_features(entries)

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
# MAIN
# ============================================================================

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


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()