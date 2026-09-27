import stat
import struct
import zipfile

from tools.zip_audit import audit_zip, parse_zip


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


def test_parse_zip_detects_crc_mismatch(tmp_path):
    apk = tmp_path / "corrupt.zip"
    with zipfile.ZipFile(apk, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("safe.txt", b"hello world")

    raw = apk.read_bytes()
    payload_offset = raw.find(b"hello world")
    assert payload_offset != -1
    raw = raw[:payload_offset] + b"X" + raw[payload_offset + 1 :]
    apk.write_bytes(raw)

    try:
        parse_zip(apk)
    except ValueError:
        return

    raise AssertionError("CRC mismatch was not detected")


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
