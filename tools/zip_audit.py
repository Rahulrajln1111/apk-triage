from __future__ import annotations

import posixpath
import stat
import struct
import zlib
import zipfile
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath


# ============================================================================
# BOUNDED ENTRY EXTRACTION (zip-bomb defense)
# ============================================================================
#
# APKAI never decompresses an archive wholesale. A hostile APK can carry
# entries whose declared size is small but whose DEFLATE stream expands to
# gigabytes (a classic zip bomb). Extraction therefore happens in three
# phases:
#
#   1. CHEAP PRE-CHECK (metadata only, no decompression)
#        - the entry must exist exactly once
#        - declared uncompressed size must be within the caller's limit
#        - declared ratio must be within the limit
#        - encrypted / unknown-compression entries are refused before any
#          decompression is attempted
#
#   2. LOCAL HEADER CROSS-CHECK (reads raw bytes, no decompression)
#        - walk to the local file header (LFH) ourselves
#        - LFH and Central Directory must agree on the essential fields
#          (name, sizes, method, CRC). A disagreement is a classic
#          parser-differential trick and is surfaced as an integrity error.
#
#   3. BOUNDED STREAM (the only place decompression happens)
#        - inflate in small chunks, enforcing the output cap on every chunk
#        - abort the moment the cap is exceeded, so a bomb cannot expand
#          past the limit even if its header lies
#        - CRC32 is accumulated over the bounded output
#
# Everything in this module decompresses AT MOST one entry, and only when
# explicitly asked for by name.
# ============================================================================

SAFE_READ_CHUNK = 256 * 1024
SAFE_READ_DEFAULT_LIMIT = 64 * 1024 * 1024
SAFE_READ_DEFAULT_MAX_RATIO = 1000.0


class ZipReadLimitError(ValueError):
    """Raised when an entry would exceed the configured extraction limits."""


class ZipEntryError(ValueError):
    """Raised when a specific entry is structurally inconsistent."""


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
# Bounded single-entry reader
# -----------------------------

def _walk_zip64_extra(extra: bytes) -> dict[int, int]:
    """Return a map of ZIP64 extended-information fields keyed by field id."""
    fields: dict[int, int] = {}
    idx = 0
    while idx + 4 <= len(extra):
        header_id, data_size = struct.unpack_from("<HH", extra, idx)
        idx += 4
        if idx + data_size > len(extra):
            break
        payload = extra[idx:idx + data_size]
        if header_id == 0x0001:
            pos = 0
            for name in ("usize", "csize", "offset", "disk"):
                if name == "disk":
                    if pos + 4 <= len(payload):
                        fields[3] = struct.unpack_from("<I", payload, pos)[0]
                    break
                if pos + 8 <= len(payload):
                    fields[{"usize": 0, "csize": 1, "offset": 2}[name]] = struct.unpack_from(
                        "<Q", payload, pos
                    )[0]
                pos += 8
        idx += data_size
    return fields


def _effective_entry_sizes(info: zipfile.ZipInfo) -> tuple[int, int]:
    """Resolve (uncompressed, compressed) sizes, honoring ZIP64 extras."""
    usize = info.file_size
    csize = info.compress_size
    if usize == 0xFFFFFFFF or csize == 0xFFFFFFFF:
        z64 = _walk_zip64_extra(info.extra or b"")
        if usize == 0xFFFFFFFF and 0 in z64:
            usize = z64[0]
        if csize == 0xFFFFFFFF and 1 in z64:
            csize = z64[1]
    return usize, csize


def _entry_select(apk_path: Path, entry_name: str) -> tuple[zipfile.ZipInfo, zipfile.ZipFile]:
    """Open the archive and locate exactly one entry by exact name.

    Returns (info, zip_handle). The caller owns (and must close) the handle.
    """
    z = zipfile.ZipFile(apk_path, "r")
    try:
        infos = [i for i in z.infolist() if i.filename == entry_name]
        if not infos:
            raise ZipEntryError(f"entry not found in APK: {entry_name}")
        if len(infos) > 1:
            # Duplicate entries are themselves a forensic anomaly; for
            # extraction we refuse to guess which one is "real".
            raise ZipEntryError(
                f"duplicate entries for {entry_name!r}: refusing ambiguous read"
            )
        return infos[0], z
    except Exception:
        z.close()
        raise


def _precheck_entry(
    info: zipfile.ZipInfo,
    limit: int,
    max_ratio: float | None,
    prefix_only: bool = False,
) -> tuple[int, int]:
    """Phase 1: refuse clearly hostile declarations before decompression.

    max_ratio=None skips the ratio check - only safe in combination with a
    small absolute limit (the cap itself bounds any expansion).

    prefix_only=True means the caller wants at most `limit` OUTPUT bytes and
    does not care about the declared size (magic sniffing). The output cap
    still applies; only the declared-size pre-check is skipped.
    """
    usize, csize = _effective_entry_sizes(info)

    if info.flag_bits & 0x1:
        raise ZipEntryError(f"entry is encrypted: {info.filename}")

    method = info.compress_type
    if method not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
        raise ZipEntryError(
            f"unsupported compression method {method} for {info.filename}"
        )

    if not prefix_only and usize > limit:
        raise ZipReadLimitError(
            f"entry {info.filename!r} declares {usize:,} uncompressed bytes "
            f"(limit {limit:,})"
        )

    if max_ratio is not None and csize > 0 and usize / csize > max_ratio:
        raise ZipReadLimitError(
            f"entry {info.filename!r} declares expansion ratio "
            f"{usize / csize:.1f}x (limit {max_ratio:.1f}x)"
        )

    return usize, csize


def _locate_local_header(raw: bytes, window_start: int, info: zipfile.ZipInfo, csize: int, file_len: int) -> int:
    """Phase 2: locate the local file header; return the absolute data offset.

    The central directory claims the LFH lives at info.header_offset. We read
    only a small window around it (never the payload) and require the local
    signature at that exact position. The local header's extra field can have
    a different length than the central directory's copy, so the data offset
    must be recomputed from the LFH's own name/extra lengths.
    """
    pos_rel = info.header_offset - window_start
    if (
        pos_rel < 0
        or pos_rel + 30 > len(raw)
        or raw[pos_rel:pos_rel + 4] != b"PK\x03\x04"
    ):
        raise ZipEntryError(
            f"local file header not at central-directory offset {info.header_offset} "
            "(central directory is inconsistent)"
        )

    try:
        (
            _sig,
            _ver,
            flags,
            method,
            _time,
            _date,
            crc,
            comp_size,
            uncomp_size,
            name_len,
            extra_len,
        ) = struct.unpack_from("<IHHHHHIIIHH", raw, pos_rel)
    except struct.error as exc:
        raise ZipEntryError(f"truncated local file header at {info.header_offset}") from exc

    data_offset = window_start + pos_rel + 30 + name_len + extra_len

    # Cross-check the fields that must agree between LFH and central dir.
    # Data-descriptor archives legitimately store zero sizes/CRC in the LFH,
    # so those comparisons are skipped when bit 3 is set.
    has_descriptor = bool(flags & 0x8)
    disagreements = []
    if method != info.compress_type:
        disagreements.append(f"method {method} != {info.compress_type}")
    if not has_descriptor:
        if name_len != len(info.filename.encode("utf-8")):
            disagreements.append(f"name length {name_len}")
        if crc != info.CRC:
            disagreements.append(f"crc 0x{crc:08X} != 0x{info.CRC:08X}")
        if comp_size != csize:
            disagreements.append(f"compressed size {comp_size} != {csize}")
        if uncomp_size != info.file_size:
            disagreements.append(f"uncompressed size {uncomp_size} != {info.file_size}")

    if disagreements:
        raise ZipEntryError(
            f"local header / central directory mismatch for {info.filename!r}: "
            + ", ".join(disagreements)
        )

    if data_offset + csize > file_len:
        raise ZipEntryError(
            f"entry data for {info.filename!r} extends past end of file"
        )

    return data_offset


@contextmanager
def open_entry_stream(
    apk_path: Path,
    entry_name: str,
    max_uncompressed: int = SAFE_READ_DEFAULT_LIMIT,
    max_ratio: float | None = SAFE_READ_DEFAULT_MAX_RATIO,
    verify_crc: bool = True,
    prefix_only: bool = False,
):
    """Yield a bounded, one-shot binary stream over a single archive entry.

    The stream decompresses incrementally and raises ZipReadLimitError the
    moment the uncompressed output would exceed *max_uncompressed*, no matter
    what the entry headers claim. Nothing is ever written to disk.
    """
    info, z = _entry_select(apk_path, entry_name)
    try:
        usize, csize = _precheck_entry(
            info, max_uncompressed, max_ratio, prefix_only=prefix_only
        )

        # In prefix mode the OUTPUT cap is the caller's byte budget, not the
        # entry's declared size; readers stop (instead of raising) at it.
        stream_limit = max_uncompressed if prefix_only else usize
        stop_at_limit = prefix_only

        # Phase 2 needs raw bytes around the local file header ONLY. We never
        # read the payload here - that happens chunk-by-chunk in the reader.
        window_start = max(0, info.header_offset - 64)
        window_end = min(info.header_offset + 4096, _file_size(apk_path))
        with open(apk_path, "rb") as fh:
            fh.seek(window_start)
            window = fh.read(window_end - window_start)

        data_offset = _locate_local_header(
            window, window_start, info, csize, _file_size(apk_path)
        )

        def _raw_chunks():
            remaining = csize
            with open(apk_path, "rb") as fh:
                fh.seek(data_offset)
                while remaining > 0:
                    chunk = fh.read(min(SAFE_READ_CHUNK, remaining))
                    if not chunk:
                        raise ZipEntryError(
                            f"unexpected end of file while reading {info.filename!r}"
                        )
                    remaining -= len(chunk)
                    yield chunk

        if info.compress_type == zipfile.ZIP_STORED:
            yield _BoundedStream(
                _RawReader(_raw_chunks(), stream_limit, stop_at_limit),
                stream_limit, verify_crc, info.CRC, "stored",
            )
        else:
            yield _BoundedStream(
                _InflateReader(_raw_chunks(), stream_limit, stop_at_limit),
                stream_limit, verify_crc, info.CRC, "deflate",
            )
    finally:
        z.close()


class _RawReader:
    def __init__(self, chunks, limit: int, stop_at_limit: bool = False):
        self._chunks = chunks
        self._produced = 0
        self._limit = limit
        self._stop_at_limit = stop_at_limit

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            size = self._limit + 1
        out = bytearray()
        while len(out) < size:
            try:
                chunk = next(self._chunks)
            except StopIteration:
                break
            out.extend(chunk)
            if len(out) > self._limit:
                if self._stop_at_limit:
                    del out[self._limit:]
                    break
                raise ZipReadLimitError("stored stream exceeded output limit")
        return bytes(out)


class _InflateReader:
    """Incremental DEFLATE reader with a hard output cap."""

    def __init__(self, chunks, limit: int, stop_at_limit: bool = False):
        self._chunks = chunks
        self._dec = zlib.decompressobj(-15)
        self._buffer = bytearray()
        self._consumed_input = False
        self._limit = limit
        self._stop_at_limit = stop_at_limit

    def _refill(self) -> bool:
        if self._consumed_input:
            return False

        try:
            raw = next(self._chunks)
        except StopIteration:
            self._consumed_input = True
            return False

        # Feed the decompressor in bounded steps. max_out allows ONE byte
        # beyond the cap so that an overflow (a real bomb) is DETECTABLE;
        # reaching the cap exactly is normal completion, not an error.
        max_out = self._limit - len(self._buffer) + 1
        pending = raw
        while pending:
            out = self._dec.decompress(
                pending, min(SAFE_READ_CHUNK, max(1, max_out))
            )
            pending = self._dec.unconsumed_tail
            self._buffer.extend(out)
            max_out = self._limit - len(self._buffer) + 1

            if len(self._buffer) > self._limit:
                if self._stop_at_limit:
                    del self._buffer[self._limit:]
                    self._consumed_input = True
                    return True
                raise ZipReadLimitError("inflated output exceeded limit")

            if self._stop_at_limit and len(self._buffer) == self._limit:
                # Prefix read satisfied; drop the decompressor's tail input.
                self._consumed_input = True
                return True

            if not out and not pending:
                break
        return True

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            size = self._limit + 1
        while len(self._buffer) < size and self._refill():
            pass
        out = bytes(self._buffer[:size])
        del self._buffer[:size]
        return out


class _BoundedStream:
    """File-like wrapper enforcing the output cap and CRC verification."""

    def __init__(self, reader, declared_size: int, verify_crc: bool, expected_crc: int, kind: str):
        self._reader = reader
        self._declared = declared_size
        self._verify_crc = verify_crc
        self._expected_crc = expected_crc
        self._kind = kind
        self._consumed = 0
        self._crc = 0
        self._done = False

    def read(self, size: int = -1) -> bytes:
        data = self._reader.read(size)
        self._consumed += len(data)
        if self._consumed > self._declared + 4096:
            raise ZipReadLimitError(
                f"entry expanded past its declared size ({self._declared:,} bytes)"
            )
        if self._verify_crc and data:
            self._crc = zlib.crc32(data, self._crc)
        if not data:
            self._finish()
        return data

    def read_all(self) -> bytes:
        parts = []
        while True:
            part = self.read(SAFE_READ_CHUNK)
            if not part:
                break
            parts.append(part)
        return b"".join(parts)

    def _finish(self) -> None:
        if self._done:
            return
        self._done = True
        if self._verify_crc and self._crc != self._expected_crc:
            raise ZipEntryError(
                f"CRC mismatch ({self._kind}): expected 0x{self._expected_crc:08X}, "
                f"got 0x{self._crc:08X}"
            )

    def close(self) -> None:
        self._finish()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def _file_size(path: Path) -> int:
    return path.stat().st_size


def read_entry_bytes(
    apk_path: Path,
    entry_name: str,
    max_uncompressed: int = SAFE_READ_DEFAULT_LIMIT,
    max_ratio: float | None = SAFE_READ_DEFAULT_MAX_RATIO,
    verify_crc: bool = True,
    prefix_only: bool = False,
) -> bytes:
    """Decompress exactly one entry into memory, bounded and CRC-verified.

    max_ratio=None disables the ratio pre-check; prefix_only=True reads at
    most max_uncompressed OUTPUT bytes regardless of the declared size
    (magic sniffing). The output cap always applies.
    """
    with open_entry_stream(
        apk_path,
        entry_name,
        max_uncompressed=max_uncompressed,
        max_ratio=max_ratio,
        verify_crc=verify_crc,
        prefix_only=prefix_only,
    ) as stream:
        return stream.read_all()


# -----------------------------
# Parse metadata only
# -----------------------------

# The EOCD record is 22 bytes and is followed by at most 65,535 bytes of
# archive comment - so the last 65,557 bytes of a file always contain it.
# Reading only that tail keeps metadata parsing O(1) in archive size; loading
# the whole APK (as this used to do) hands a hostile multi-GB file an OOM.
EOCD_MAX_COMMENT = 0xFFFF
EOCD_WINDOW = 22 + EOCD_MAX_COMMENT


def read_eocd_window(apk_path: Path) -> tuple[int, int, bytes]:
    """Read only the archive tail that can contain the EOCD record.

    Returns (file_size, window_offset, window). Never reads more than
    EOCD_WINDOW bytes regardless of the archive size.
    """
    size = _file_size(apk_path)
    start = max(0, size - EOCD_WINDOW)
    with open(apk_path, "rb") as fh:
        fh.seek(start)
        window = fh.read(size - start)
    return size, start, window


def _validate_eocd_metadata(apk_path: Path) -> None:
    file_size, window_start, raw = read_eocd_window(apk_path)
    if file_size < 22:
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
    if cd_offset + cd_size > file_size:
        raise ValueError(
            "invalid ZIP archive: central directory offset/size exceeds file length "
            f"(offset={cd_offset}, size={cd_size}, file_size={file_size})"
        )
    trailing = file_size - (window_start + eocd_offset + 22)
    if comment_len != trailing:
        raise ValueError(
            "invalid ZIP archive: EOCD comment length does not match trailing bytes "
            f"(comment_len={comment_len}, trailing={trailing})"
        )


def parse_zip(apk_path: Path, decompress: bool = False) -> list[ZipEntry]:
    """Parse archive metadata.

    By default this is a METADATA-ONLY pass: no entry is ever decompressed,
    which makes it safe to run against hostile archives (zip bombs, parser
    differentials) before deciding which entries deserve a bounded read via
    read_entry_bytes().

    With decompress=True every entry is CRC-verified through the bounded
    reader; archives with huge payloads will be rejected by the default
    limits, so this mode is meant for small/test archives only.
    """
    entries: list[ZipEntry] = []

    try:
        _validate_eocd_metadata(apk_path)
        with zipfile.ZipFile(apk_path, "r") as z:
            for info in z.infolist():
                if decompress:
                    try:
                        with z.open(info, "r") as fh:
                            data = fh.read()
                    except (zipfile.BadZipFile, RuntimeError, ValueError) as exc:
                        raise ValueError(
                            f"archive integrity failure for {info.filename}: {exc}"
                        ) from exc

                    crc = zlib.crc32(data) & 0xFFFFFFFF
                    if info.CRC != crc:
                        raise ValueError(
                            f"CRC mismatch for {info.filename}: expected {info.CRC:#x}, got {crc:#x}"
                        )
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