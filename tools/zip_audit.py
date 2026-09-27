from __future__ import annotations

import posixpath
import stat
import struct
import zlib
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath


@dataclass
class ZipEntry:
    name: str
    size: int
    compressed: int
    crc: int
    mode: int = 0
    flag_bits: int = 0
    compression_method: int = 0
    is_directory: bool = False
    is_symlink: bool = False
    is_traversal: bool = False
    is_absolute: bool = False
    is_encrypted: bool = False
    has_data_descriptor: bool = False
    is_null_name: bool = False
    is_nested_archive: bool = False
    has_zip64_extra: bool = False
    normalized_name: str = ""
    duplicate_name: bool = False

    @property
    def ratio(self) -> float:
        """Original size / Compressed size"""
        if self.compressed == 0:
            return 0.0
        return self.size / self.compressed


@dataclass
class ZipAuditResult:
    entries: list[ZipEntry]
    has_path_traversal: bool = False
    has_absolute_path: bool = False
    duplicate_name_count: int = 0
    symlink_count: int = 0
    encrypted_entry_count: int = 0
    null_byte_name_count: int = 0
    nested_archive_count: int = 0
    zip64_entry_count: int = 0
    data_descriptor_count: int = 0
    absolute_path_count: int = 0
    issues: list[str] = field(default_factory=list)


# -----------------------------
# Convert ZipInfo -> ZipEntry
# -----------------------------
def _normalize_archive_name(name: str) -> str:
    if not name:
        return ""
    normalized = name.replace("\\", "/")
    while "//" in normalized:
        normalized = normalized.replace("//", "/")
    return normalized


def _canonicalize_archive_path(name: str) -> str:
    normalized = _normalize_archive_name(name)
    if not normalized:
        return ""
    return posixpath.normpath(normalized)


def _is_absolute_path(name: str) -> bool:
    normalized = _normalize_archive_name(name)
    if normalized.startswith("/"):
        return True
    if len(normalized) >= 3 and normalized[1] == ":" and normalized[2] == "/":
        return True
    return False


def _is_traversal_path(name: str) -> bool:
    normalized = _canonicalize_archive_path(name)
    if not normalized or normalized == ".":
        return False
    return ".." in normalized.split("/") or normalized.startswith("../") or normalized == ".."


def _has_zip64_extra(extra: bytes) -> bool:
    if not extra:
        return False
    idx = 0
    while idx + 4 <= len(extra):
        header_id, data_size = struct.unpack_from("<HH", extra, idx)
        idx += 4
        if idx + data_size > len(extra):
            break
        if header_id == 0x0001:
            return True
        idx += data_size
    return False


def _is_nested_archive(name: str) -> bool:
    lowered = name.lower()
    suffixes = (
        ".zip",
        ".apk",
        ".aar",
        ".jar",
        ".war",
        ".ear",
        ".whl",
        ".xpi",
        ".nupkg",
        ".docx",
        ".xlsx",
        ".pptx",
    )
    return lowered.endswith(suffixes)


def to_entry(info: zipfile.ZipInfo) -> ZipEntry:
    raw_name = info.filename
    normalized_name = _normalize_archive_name(raw_name)
    mode = (info.external_attr >> 16) & 0xFFFF
    is_symlink = bool(mode) and stat.S_ISLNK(mode)
    is_directory = info.is_dir()
    parts = PurePosixPath(_canonicalize_archive_path(normalized_name)).parts if normalized_name else ()
    flag_bits = getattr(info, "flag_bits", 0)

    return ZipEntry(
        name=raw_name,
        size=info.file_size,
        compressed=info.compress_size,
        crc=info.CRC,
        mode=mode,
        flag_bits=flag_bits,
        compression_method=info.compress_type,
        is_directory=is_directory,
        is_symlink=is_symlink,
        is_traversal=_is_traversal_path(raw_name),
        is_absolute=_is_absolute_path(raw_name),
        is_encrypted=bool(flag_bits & 0x1),
        has_data_descriptor=bool(flag_bits & 0x8),
        is_null_name="\x00" in raw_name,
        is_nested_archive=_is_nested_archive(raw_name),
        has_zip64_extra=_has_zip64_extra(info.extra or b""),
        normalized_name=normalized_name,
        duplicate_name=False,
    )


# -----------------------------
# Parse metadata only
# -----------------------------
def _validate_eocd_metadata(apk_path: Path) -> None:
    raw = apk_path.read_bytes()
    if len(raw) < 22:
        raise ValueError("invalid ZIP archive: file too short to contain EOCD")

    eocd_offset = raw.rfind(b"PK\x05\x06")
    if eocd_offset == -1:
        raise ValueError("invalid ZIP archive: EOCD not found")

    if len(raw) - eocd_offset < 22:
        raise ValueError("invalid ZIP archive: truncated EOCD record")

    _, disk_no, cd_disk_no, disk_entries, total_entries, cd_size, cd_offset, comment_len = struct.unpack_from(
        "<4sHHHHIIH", raw, eocd_offset
    )

    if disk_no != 0 or cd_disk_no != 0:
        raise ValueError("invalid ZIP archive: multi-disk archive is not supported")
    if disk_entries != total_entries:
        raise ValueError(
            "invalid ZIP archive: central directory entry counts disagree "
            f"({disk_entries} on disk vs {total_entries} total)"
        )
    if cd_size == 0xFFFFFFFF or cd_offset == 0xFFFFFFFF or total_entries == 0xFFFF or disk_entries == 0xFFFF:
        raise ValueError("invalid ZIP archive: ZIP64 metadata requires full ZIP64 parsing support")
    if cd_offset + cd_size > len(raw):
        raise ValueError(
            "invalid ZIP archive: central directory offset/size exceeds file length "
            f"(offset={cd_offset}, size={cd_size}, file_size={len(raw)})"
        )
    if comment_len != (len(raw) - (eocd_offset + 22)):
        raise ValueError(
            "invalid ZIP archive: EOCD comment length does not match trailing bytes "
            f"(comment_len={comment_len}, trailing={len(raw) - (eocd_offset + 22)})"
        )


def parse_zip(apk_path: Path) -> list[ZipEntry]:
    entries: list[ZipEntry] = []

    try:
        _validate_eocd_metadata(apk_path)
        with zipfile.ZipFile(apk_path, "r") as z:
            for info in z.infolist():
                try:
                    with z.open(info, "r") as fh:
                        data = fh.read()
                except (zipfile.BadZipFile, RuntimeError, ValueError) as exc:
                    raise ValueError(f"archive integrity failure for {info.filename}: {exc}") from exc

                crc = zlib.crc32(data) & 0xFFFFFFFF
                if info.CRC != crc:
                    raise ValueError(f"CRC mismatch for {info.filename}: expected {info.CRC:#x}, got {crc:#x}")
                entries.append(to_entry(info))
    except ValueError:
        raise
    except zipfile.BadZipFile as exc:
        raise ValueError(f"invalid ZIP archive: {exc}") from exc

    return entries


# -----------------------------
# Security-aware ZIP audit
# -----------------------------
def audit_zip(apk_path: Path) -> ZipAuditResult:
    try:
        entries = parse_zip(apk_path)
    except ValueError as exc:
        entries = []
        issues = [str(exc)]
        return ZipAuditResult(entries=entries, issues=issues)

    name_counts = Counter(e.normalized_name.rstrip("/") for e in entries)
    duplicate_names = {name for name, count in name_counts.items() if count > 1}
    for entry in entries:
        entry.duplicate_name = entry.normalized_name.rstrip("/") in duplicate_names

    duplicate_name_count = sum(count - 1 for count in name_counts.values() if count > 1)

    has_path_traversal = any(e.is_traversal for e in entries)
    has_absolute_path = any(e.is_absolute for e in entries)
    symlink_count = sum(1 for e in entries if e.is_symlink)
    encrypted_entry_count = sum(1 for e in entries if e.is_encrypted)
    null_byte_name_count = sum(1 for e in entries if e.is_null_name)
    nested_archive_count = sum(1 for e in entries if e.is_nested_archive)
    zip64_entry_count = sum(1 for e in entries if e.has_zip64_extra)
    data_descriptor_count = sum(1 for e in entries if e.has_data_descriptor)
    absolute_path_count = sum(1 for e in entries if e.is_absolute)

    issues: list[str] = []
    if has_path_traversal:
        issues.append("path traversal detected")
    if has_absolute_path:
        issues.append("absolute path detected")
    if duplicate_name_count:
        issues.append("duplicate normalized names detected")
    if symlink_count:
        issues.append("symlink entry detected")
    if encrypted_entry_count:
        issues.append("encrypted entries detected")
    if null_byte_name_count:
        issues.append("null-byte names detected")
    if nested_archive_count:
        issues.append("nested archive candidates detected")
    if zip64_entry_count:
        issues.append("ZIP64 extra field detected")
    if data_descriptor_count:
        issues.append("data descriptor entries detected")

    return ZipAuditResult(
        entries=entries,
        has_path_traversal=has_path_traversal,
        has_absolute_path=has_absolute_path,
        duplicate_name_count=duplicate_name_count,
        symlink_count=symlink_count,
        encrypted_entry_count=encrypted_entry_count,
        null_byte_name_count=null_byte_name_count,
        nested_archive_count=nested_archive_count,
        zip64_entry_count=zip64_entry_count,
        data_descriptor_count=data_descriptor_count,
        absolute_path_count=absolute_path_count,
        issues=issues,
    )


# -----------------------------
# Print metadata report
# -----------------------------
def inspect_zip(apk_path: Path):
    entries = parse_zip(apk_path)
    audit = audit_zip(apk_path)

    total_original = sum(e.size for e in entries)
    total_compressed = sum(e.compressed for e in entries)

    overall_ratio = total_original / total_compressed if total_compressed else 0

    print("=" * 70)
    print("ZIP METADATA REPORT")
    print("=" * 70)

    print(f"APK                 : {apk_path.name}")
    print(f"Entries             : {len(entries)}")
    print(f"Original Size       : {total_original:,} bytes")
    print(f"Compressed Size     : {total_compressed:,} bytes")
    print(f"Overall Ratio       : {overall_ratio:.2f}x")
    print(f"Traversal Entries   : {sum(1 for e in entries if e.is_traversal)}")
    print(f"Absolute Paths      : {sum(1 for e in entries if e.is_absolute)}")
    print(f"Duplicate Names     : {audit.duplicate_name_count}")
    print(f"Symlinks            : {sum(1 for e in entries if e.is_symlink)}")
    print(f"Encrypted Entries   : {audit.encrypted_entry_count}")
    print(f"Null-Byte Names     : {audit.null_byte_name_count}")
    print(f"Nested Archives     : {audit.nested_archive_count}")
    print(f"ZIP64 Entries       : {audit.zip64_entry_count}")
    print(f"Data Descriptor     : {audit.data_descriptor_count}")

    if audit.issues:
        print("Issues              : " + "; ".join(audit.issues))

    print("-" * 70)

    for e in entries:
        print(
            f"{e.name:<45}"
            f"{e.size:>10,} B"
            f"   {e.ratio:>6.2f}x"
        )

    return entries