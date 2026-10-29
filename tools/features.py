from dataclasses import dataclass, field
from tools.zip_audit import ZipEntry
import re
from pathlib import PurePosixPath
from collections import Counter, defaultdict
from math import log2


@dataclass
class ZipFeatures:
    total_entries: int
    total_size: int
    total_compressed: int

    dex_count: int
    png_count: int
    xml_count: int

    overall_ratio: float
    max_ratio: float

    sequential_dex_score: int
    absolute_path_count: int
    filename_family_score: int

    directory_depth_entropy: float
    max_depth: int
    path_traversal_count: int
    duplicate_name_count: int = 0
    null_byte_name_count: int = 0
    symlink_count: int = 0
    encrypted_entry_count: int = 0
    zip64_present: int = 0
    data_descriptor_count: int = 0
    nested_archive_count: int = 0


def file_depth(entry: ZipEntry) -> int:
    path = PurePosixPath(entry.normalized_name or entry.name)
    return max(0, len(path.parts) - 1)


def directory_depth_entropy(entries):
    if not entries:
        return 0.0

    depths = [file_depth(e) for e in entries]
    freq = Counter(depths)
    total = len(depths)

    entropy = 0.0
    for count in freq.values():
        p = count / total
        entropy -= p * log2(p)
    return entropy


def max_depth(entries):
    if not entries:
        return 0
    return max(file_depth(e) for e in entries)


def filename_family_score(entries):
    pattern = re.compile(r"^([A-Za-z_]+)(\d*)\.([A-Za-z0-9]+)$")
    families = defaultdict(set)

    for e in entries:
        filename = e.name.split("/")[-1]
        match = pattern.match(filename)
        if not match:
            continue

        prefix = match.group(1)
        number = match.group(2)
        ext = match.group(3)
        key = (prefix, ext)

        if number == "":
            families[key].add(1)
        else:
            families[key].add(int(number))

    best = 0
    for numbers in families.values():
        current = 1
        while current in numbers:
            current += 1
        best = max(best, current - 1)
    return best


def path_traversal_count(entries):
    return sum(1 for e in entries if e.is_traversal)


def absolute_path_count(entries):
    return sum(1 for e in entries if e.is_absolute)


def count_extension(entries: list[ZipEntry], ext: str) -> int:
    return sum(1 for e in entries if e.name.lower().endswith(ext))


def overall_ratio(entries: list[ZipEntry]) -> float:
    total_original = sum(e.size for e in entries)
    total_compressed = sum(e.compressed for e in entries)
    if total_compressed == 0:
        return 0.0
    return total_original / total_compressed


def max_ratio(entries: list[ZipEntry]) -> float:
    if not entries:
        return 0.0
    return max(e.ratio for e in entries)


def sequential_dex_score(entries: list[ZipEntry]) -> int:
    pattern = re.compile(r"^classes(\d*)\.dex$")
    numbers = set()

    for e in entries:
        filename = e.name.split("/")[-1]
        match = pattern.match(filename)
        if match:
            number = match.group(1)
            if number == "":
                numbers.add(1)
            else:
                numbers.add(int(number))

    score = 1
    while score in numbers:
        score += 1
    return score - 1


def duplicate_name_count(entries: list[ZipEntry]) -> int:
    """Number of entries that repeat an earlier entry name.

    Derived from the names themselves: parse_zip() (unlike audit_zip()) does
    not set the per-entry duplicate_name flag, so trusting that flag silently
    reported zero duplicates for every archive read through parse_zip().
    """
    counts = Counter((e.normalized_name or e.name).rstrip("/") for e in entries)
    return sum(count - 1 for count in counts.values() if count > 1)


def extract_features(entries: list[ZipEntry]) -> ZipFeatures:
    duplicates = duplicate_name_count(entries)

    return ZipFeatures(
        total_entries=len(entries),
        total_size=sum(e.size for e in entries),
        total_compressed=sum(e.compressed for e in entries),
        dex_count=count_extension(entries, ".dex"),
        png_count=count_extension(entries, ".png"),
        xml_count=count_extension(entries, ".xml"),
        overall_ratio=overall_ratio(entries),
        max_ratio=max_ratio(entries),
        sequential_dex_score=sequential_dex_score(entries),
        absolute_path_count=absolute_path_count(entries),
        path_traversal_count=path_traversal_count(entries),
        filename_family_score=filename_family_score(entries),
        directory_depth_entropy=directory_depth_entropy(entries),
        max_depth=max_depth(entries),
        duplicate_name_count=duplicates,
        null_byte_name_count=sum(1 for e in entries if e.is_null_name),
        symlink_count=sum(1 for e in entries if e.is_symlink),
        encrypted_entry_count=sum(1 for e in entries if e.is_encrypted),
        zip64_present=sum(1 for e in entries if e.has_zip64_extra),
        data_descriptor_count=sum(1 for e in entries if e.has_data_descriptor),
        nested_archive_count=sum(1 for e in entries if e.is_nested_archive),
    )


# ============================================================================
# RESOURCE TABLE (arsc) FEATURES  ->  Phase 9 ML groundwork
# ============================================================================

@dataclass
class ArscFeatures:
    """Compact numeric description of resources.arsc for the ML pipeline."""
    package_count: int = 0
    resource_entries: int = 0
    config_variants: int = 0
    type_count: int = 0
    string_value_count: int = 0
    reference_value_count: int = 0
    attribute_ref_count: int = 0
    complex_bag_count: int = 0
    compact_entry_count: int = 0
    public_entry_count: int = 0
    weak_entry_count: int = 0
    sparse_type_count: int = 0
    offset16_type_count: int = 0
    anomaly_count: int = 0
    null_chunk_count: int = 0
    library_package_count: int = 0
    scramble_score: int = 0
    rebuild_fingerprints: list[str] = field(default_factory=list)


def extract_arsc_features(table) -> ArscFeatures:
    """Derive ML features from a parsed ResourceTable.

    The scramble score counts entries whose STRING value looks like a mangled
    filesystem path (obfuscators rewrite resource values with junk paths to
    break naive tooling). Legitimate apps rarely store paths as values.
    """
    features = ArscFeatures()
    features.package_count = len(table.packages)
    features.anomaly_count = len(table.anomalies)
    features.null_chunk_count = sum(
        1 for u in table.unknown_chunks if u.chunk_type == 0x0000
    )

    path_like = re.compile(
        r"(res/|kotlin/|META-INF/|classes\.dex|AndroidManifest\.xml|"
        r"resources\.arsc|\\\\|//)",
        re.IGNORECASE,
    )

    for pkg in table.packages:
        features.library_package_count += sum(
            len(lib.entries) for lib in pkg.libraries
        )
        for t in pkg.types:
            if t.is_sparse:
                features.sparse_type_count += 1
            if t.uses_offset16:
                features.offset16_type_count += 1

            for entry in t.entries.values():
                features.resource_entries += 1
                if entry.is_public:
                    features.public_entry_count += 1
                if entry.is_weak:
                    features.weak_entry_count += 1
                if entry.is_compact:
                    features.compact_entry_count += 1
                if entry.is_complex:
                    features.complex_bag_count += 1

                values = []
                if entry.value is not None:
                    values.append(entry.value)
                values.extend(item.value for item in entry.bag)

                for value in values:
                    if value.data_type == 0x03:  # TYPE_STRING
                        features.string_value_count += 1
                        text = table.global_pool.get(value.data)
                        if text and path_like.search(text):
                            features.scramble_score += 1
                    elif value.data_type == 0x01:  # TYPE_REFERENCE
                        features.reference_value_count += 1
                    elif value.data_type == 0x02:  # TYPE_ATTRIBUTE
                        features.attribute_ref_count += 1

    features.config_variants = len({
        (p.package_id, t.type_id, t.config.qualifier)
        for p in table.packages for t in p.types
    })
    features.type_count = len({
        name for p in table.packages
        for t in p.types
        if (name := p.type_name(t.type_id))
    })

    # Repackaging-tool fingerprints observed in the wild. The printable
    # payload renders UTF-16 with '.' between characters, so normalize by
    # stripping dots before matching.
    fingerprints = []
    for unknown in table.unknown_chunks:
        text = unknown.printable.replace(".", "").lower()
        if "reandroid" in text:
            fingerprints.append("ARSClib (REAndroid)")
        if "apktool" in text:
            fingerprints.append("Apktool")
    features.rebuild_fingerprints = fingerprints

    return features


def print_arsc_features(f: ArscFeatures) -> None:
    print()
    print("=" * 70)
    print("ARSC ML FEATURES")
    print("=" * 70)
    for key, value in (
        ("packages", f.package_count),
        ("resource_entries", f.resource_entries),
        ("config_variants", f.config_variants),
        ("types", f.type_count),
        ("string_values", f.string_value_count),
        ("reference_values", f.reference_value_count),
        ("attribute_refs", f.attribute_ref_count),
        ("complex_bags", f.complex_bag_count),
        ("compact_entries", f.compact_entry_count),
        ("public_entries", f.public_entry_count),
        ("weak_entries", f.weak_entry_count),
        ("sparse_types", f.sparse_type_count),
        ("offset16_types", f.offset16_type_count),
        ("anomalies", f.anomaly_count),
        ("null_chunks", f.null_chunk_count),
        ("library_packages", f.library_package_count),
        ("scramble_score", f.scramble_score),
    ):
        print(f"{key:20}: {value}")
    if f.rebuild_fingerprints:
        print(f"{'rebuild_tools':20}: {', '.join(sorted(set(f.rebuild_fingerprints)))}")