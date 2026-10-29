import stat
import struct
import zipfile

import pytest

from tools.binary import read_tail
from tools.features import extract_features
from tools.report import analyze_features
from tools.zip_audit import (
    EOCD_WINDOW,
    ZipReadLimitError,
    audit_zip,
    parse_zip,
    read_entry_bytes,
    read_eocd_window,
)


def _write_zip(path, items):
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, data, mode in items:
            info = zipfile.ZipInfo(name)
            info.compress_type = zipfile.ZIP_DEFLATED
            if mode is not None:
                info.external_attr = (mode & 0xFFFF) << 16
            zf.writestr(info, data)


def test_parse_zip_detects_traversal_and_symlink(tmp_path):
    apk = tmp_path / "archive.zip"
    _write_zip(
        apk,
        [
            ("../evil.txt", "pwned", None),
            ("safe/ok.txt", "ok", None),
            ("link", "target", stat.S_IFLNK | 0o777),
        ],
    )

    entries = parse_zip(apk)
    assert any(e.is_traversal for e in entries)
    assert any(e.is_symlink for e in entries)


def test_traversal_check_does_not_false_positive_on_filename_suffix(tmp_path):
    apk = tmp_path / "archive.zip"
    _write_zip(apk, [("safe/..png", "blob", None), ("safe/..\\file.txt", "blob", None)])

    entries = parse_zip(apk)
    assert not any(e.is_traversal for e in entries)


def test_data_descriptor_is_not_a_risk_signal(tmp_path):
    apk = tmp_path / "data.desc.zip"
    _write_zip(apk, [("safe/one.txt", "one", None), ("safe/two.txt", "two", None)])

    entries = parse_zip(apk)
    assert all(not e.is_traversal for e in entries)
    assert sum(1 for e in entries if e.has_data_descriptor) == 0


def test_bounded_reader_detects_crc_mismatch(tmp_path):
    apk = tmp_path / "corrupt.zip"
    with zipfile.ZipFile(apk, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("safe.txt", b"hello world")

    raw = apk.read_bytes()
    payload_offset = raw.find(b"hello world")
    assert payload_offset != -1
    raw = raw[:payload_offset] + b"X" + raw[payload_offset + 1 :]
    apk.write_bytes(raw)

    with pytest.raises(ValueError):
        read_entry_bytes(apk, "safe.txt")


def test_bounded_reader_enforces_size_limit(tmp_path):
    apk = tmp_path / "big.zip"
    with zipfile.ZipFile(apk, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("blob", b"\x00" * (4 * 1024 * 1024))

    with pytest.raises(ZipReadLimitError):
        read_entry_bytes(apk, "blob", max_uncompressed=1024)


def test_bounded_reader_refuses_duplicate_entries(tmp_path):
    apk = tmp_path / "dup.zip"
    with zipfile.ZipFile(apk, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("twice.txt", b"one")
        zf.writestr("twice.txt", b"two")

    with pytest.raises(ValueError):
        read_entry_bytes(apk, "twice.txt")


def test_bounded_reader_rejects_oversized_declared_size(tmp_path):
    apk = tmp_path / "liar.zip"
    with zipfile.ZipFile(apk, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("payload", b"tiny")

    # Lie about the uncompressed size in both CD and LFH so the pre-check
    # fires before any decompression.
    raw = bytearray(apk.read_bytes())
    lfh = raw.find(b"PK\x03\x04")
    struct.pack_into("<I", raw, lfh + 22, 0x7FFFFFFF)  # uncompressed size
    cd = raw.find(b"PK\x01\x02")
    struct.pack_into("<I", raw, cd + 24, 0x7FFFFFFF)
    apk.write_bytes(raw)

    with pytest.raises(ZipReadLimitError):
        read_entry_bytes(apk, "payload", max_uncompressed=1024)


def test_parse_zip_is_metadata_only_by_default(tmp_path):
    apk = tmp_path / "meta.zip"
    with zipfile.ZipFile(apk, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("a.txt", b"A" * 1000)

    # Corrupt the stored payload; metadata parsing must not care.
    raw = bytearray(apk.read_bytes())
    idx = raw.find(b"AAAA")
    assert idx != -1
    raw[idx] = 0x42
    apk.write_bytes(raw)

    entries = parse_zip(apk)
    assert len(entries) == 1
    assert entries[0].name == "a.txt"


def test_parse_zip_detects_invalid_central_directory_metadata(tmp_path):
    apk = tmp_path / "structural.zip"
    with zipfile.ZipFile(apk, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("safe.txt", b"hello world")

    raw = bytearray(apk.read_bytes())
    eocd_offset = raw.rfind(b"PK\x05\x06")
    assert eocd_offset != -1
    # Corrupt the total entry count so the archive claims a different central directory layout.
    raw[eocd_offset + 10:eocd_offset + 12] = struct.pack("<H", 99)
    apk.write_bytes(raw)

    try:
        parse_zip(apk)
    except ValueError:
        return

    raise AssertionError("central directory metadata inconsistency was not detected")


def test_duplicate_entries_are_counted_through_parse_zip(tmp_path):
    # Regression: parse_zip() does not set the per-entry duplicate_name flag,
    # so features built from it reported zero duplicates and the finding
    # could never fire in the main pipeline.
    apk = tmp_path / "dup-feature.zip"
    _write_zip(apk, [("safe/one.txt", "one", None),
                     ("safe/one.txt", "two", None)])

    features = extract_features(parse_zip(apk))
    assert features.duplicate_name_count == 1        # one *extra* entry
    titles = [f.title for f in analyze_features(features).findings]
    assert "Duplicate normalized names" in titles


def test_eocd_window_is_bounded_and_handles_max_comment(tmp_path):
    # Regression: EOCD validation used to read the whole archive into memory.
    # The last 65,557 bytes always suffice, even with a maximal ZIP comment.
    apk = tmp_path / "comment.zip"
    with zipfile.ZipFile(apk, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("blob", b"\x00" * (2 * 1024 * 1024))
        zf.comment = b"x" * 65535

    size, start, window = read_eocd_window(apk)
    assert size == apk.stat().st_size
    assert len(window) <= EOCD_WINDOW
    assert start == size - EOCD_WINDOW
    assert len(parse_zip(apk)) == 1                  # comment still validated

    tail, offset = read_tail(apk)
    assert offset + len(tail) == size
    assert tail == window


def test_audit_zip_reports_structural_issues(tmp_path):
    apk = tmp_path / "bad.zip"
    _write_zip(
        apk,
        [
            ("../x.txt", "boom", None),
            ("safe/one.txt", "one", None),
            ("safe/one.txt", "duplicate", None),
            ("/abs.txt", "abs", None),
        ],
    )

    result = audit_zip(apk)
    assert result.has_path_traversal
    assert result.has_absolute_path
    assert result.duplicate_name_count >= 1
    assert result.issues
