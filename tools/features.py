from dataclasses import dataclass
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


def extract_features(entries: list[ZipEntry]) -> ZipFeatures:
    duplicate_name_count = sum(
        1 for e in entries
        if e.duplicate_name
    )

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
        duplicate_name_count=duplicate_name_count,
        null_byte_name_count=sum(1 for e in entries if e.is_null_name),
        symlink_count=sum(1 for e in entries if e.is_symlink),
        encrypted_entry_count=sum(1 for e in entries if e.is_encrypted),
        zip64_present=sum(1 for e in entries if e.has_zip64_extra),
        data_descriptor_count=sum(1 for e in entries if e.has_data_descriptor),
        nested_archive_count=sum(1 for e in entries if e.is_nested_archive),
    )