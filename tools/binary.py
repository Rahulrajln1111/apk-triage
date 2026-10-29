from pathlib import Path
import struct

LOCAL_SIGNATURE = b"PK\x03\x04"
EOCD_SIGNATURE = b"PK\x05\x06"
ZIP64_EOCD_SIGNATURE = b"PK\x06\x06"
CENTRAL_DIRECTORY_SIGNATURE = b"PK\x01\x02"

# EOCD record (22 bytes) + its maximum comment (65,535) - the largest tail
# that can still contain the end-of-central-directory record.
EOCD_WINDOW = 22 + 0xFFFF


def read_head(path: Path, size: int = 4) -> bytes:
    """Read at most *size* bytes from the start of a file (magic probe)."""
    with open(path, "rb") as f:
        return f.read(size)


def read_tail(path: Path, max_bytes: int = EOCD_WINDOW) -> tuple[bytes, int]:
    """Read at most the last *max_bytes* bytes of a file.

    Returns (data, file_offset): add file_offset to any position inside data
    to get the absolute offset. Never loads the whole archive, so a hostile
    multi-GB file cannot exhaust memory before a single entry is parsed.
    """
    size = path.stat().st_size
    start = max(0, size - max_bytes)
    with open(path, "rb") as f:
        f.seek(start)
        return f.read(size - start), start


def read_eocd(data: bytes):
    offset = data.rfind(EOCD_SIGNATURE)
    if offset == -1:
        raise ValueError("EOCD not found")

    if len(data) - offset < 22:
        raise ValueError("Truncated EOCD record")

    eocd = data[offset:offset + 22]
    fields = struct.unpack("<4sHHHHIIH", eocd)
    signature, disk_no, cd_disk_no, disk_entries, total_entries, cd_size, cd_offset, comment_len = fields

    return {
        "signature": signature,
        "disk_no": disk_no,
        "cd_disk_no": cd_disk_no,
        "disk_entries": disk_entries,
        "total_entries": total_entries,
        "cd_size": cd_size,
        "cd_offset": cd_offset,
        "comment_len": comment_len,
        "offset": offset,
        "is_zip64": cd_size == 0xFFFFFFFF or cd_offset == 0xFFFFFFFF or total_entries == 0xFFFF or disk_entries == 0xFFFF,
    }


def has_zip_signature(data: bytes) -> bool:
    return data[:4] == LOCAL_SIGNATURE


def find_eocd(data: bytes) -> int:
    """Return the byte offset of EOCD. Returns -1 if not found."""
    return data.rfind(EOCD_SIGNATURE)


def parse_eocd(data: bytes):
    """Compatibility wrapper for the EOCD parse used by the project."""
    return read_eocd(data)


def validate_eocd_consistency(data: bytes) -> list[str]:
    issues = []
    try:
        eocd = read_eocd(data)
    except ValueError:
        return ["EOCD missing or truncated"]

    if eocd["disk_no"] != 0 or eocd["cd_disk_no"] != 0:
        issues.append("multi-disk archive detected")
    if eocd["is_zip64"]:
        issues.append("ZIP64 EOCD sentinel values detected")
    if eocd["comment_len"] > 0:
        # The comment is present but we do not fully parse the trailing data here.
        issues.append("archive comment present")
    if eocd["cd_offset"] > len(data) or eocd["cd_size"] > len(data):
        issues.append("central directory offset/size exceeds file length")

    return issues