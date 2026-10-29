"""
Tests for the resources.arsc parser.

These tests build synthetic resource tables byte-by-byte from the AOSP
structure definitions, then verify the parser decodes them exactly.
Building the binary ourselves is the best way to catch offset mistakes.
"""

import struct
import zipfile

import pytest

from tools.resource_table import (
    ENTRY_FLAG_COMPACT,
    ENTRY_FLAG_COMPLEX,
    ENTRY_FLAG_PUBLIC,
    PLURAL_KEYS,
    ResourceTableError,
    TYPE_INT_DEC,
    TYPE_REFERENCE,
    TYPE_STRING,
    parse_resource_table,
    parse_resource_table_bytes,
    print_anomalies,
    print_resource_table,
)


# ============================================================================
# SYNTHETIC TABLE BUILDER
# ============================================================================

def build_string_pool(strings: list[str], utf8: bool = False) -> bytes:
    """Build a ResStringPool chunk (UTF-16 by default)."""
    if utf8:
        flags = 0x100
        blobs = []
        for s in strings:
            raw = s.encode("utf-8")
            utf16_len = len(s.encode("utf-16-le")) // 2
            blobs.append(_utf8_len_prefix(utf16_len) + _utf8_len_prefix(len(raw)) + raw + b"\x00")
    else:
        flags = 0
        blobs = []
        for s in strings:
            raw = s.encode("utf-16-le")
            assert len(s) < 0x8000
            blobs.append(struct.pack("<H", len(s)) + raw + b"\x00\x00")

    strings_start = 28 + 4 * len(strings)
    data = b"".join(blobs)
    header = struct.pack(
        "<HHIIIIII",
        0x0001,                    # type
        28,                        # headerSize
        strings_start + len(data), # chunkSize
        len(strings),
        0,                         # styleCount
        flags,
        strings_start,
        0,                         # stylesStart
    )
    header += struct.pack("<I", 0) * len(strings)  # placeholder, replaced below

    offsets = []
    running = 0
    for blob in blobs:
        offsets.append(running)
        running += len(blob)
    offset_array = b"".join(struct.pack("<I", o) for o in offsets)

    return header[:28] + offset_array + data


def _utf8_len_prefix(value: int) -> bytes:
    if value < 0x80:
        return bytes([value])
    return bytes([0x80 | (value >> 8), value & 0xFF])


def build_default_config() -> bytes:
    """A 28-byte default ResTable_config (all 'any')."""
    return struct.pack("<I", 28) + b"\x00" * 24


def build_simple_entry(key_index: int, data_type: int, data: int,
                       flags: int = 0) -> bytes:
    return (
        struct.pack("<HHI", 8, flags, key_index)
        + struct.pack("<HBB I", 8, 0, data_type, data)
    )


def build_compact_entry(key_index: int, data_type: int, data: int,
                        flags: int = 0) -> bytes:
    """ResTable_entry::Compact: key:u16 | flags:u16 | data:u32 (type>>24)."""
    f = flags | ENTRY_FLAG_COMPACT
    word4 = ((data_type & 0xFF) << 24) | (data & 0xFFFFFF)
    return struct.pack("<HHI", key_index, f, word4)


def build_bag_entry(key_index: int, items: list[tuple[int, int, int]],
                    parent: int = 0, flags: int = ENTRY_FLAG_COMPLEX) -> bytes:
    body = struct.pack("<HHIII", 16, flags, key_index, parent, len(items))
    for ident, data_type, data in items:
        body += struct.pack("<IHBB I", ident, 8, 0, data_type, data)
    return body


def build_typespec(type_id: int, entry_count: int) -> bytes:
    return (
        struct.pack("<HHI", 0x0202, 16, 16 + 4 * entry_count)
        + struct.pack("<BBHI", type_id, 0, 0, entry_count)
        + b"\x00" * (4 * entry_count)
    )


def build_type_chunk(
    type_id: int,
    entry_count: int,
    entries: dict[int, bytes],
    *,
    sparse: bool = False,
    offset16: bool = False,
    config: bytes | None = None,
) -> bytes:
    """Build a ResTable_type. `entries` maps entry index -> raw entry bytes."""
    config = config if config is not None else build_default_config()
    header_size = 20 + len(config)

    # For sparse types the offset array holds one u32 per PRESENT entry and
    # the entryCount field carries that same count (AOSP semantics).
    declared_count = len(entries) if sparse else entry_count

    # Lay entries out contiguously in index order.
    ordered = sorted(entries)
    entry_blobs = {}
    running = 0
    for idx in ordered:
        entry_blobs[idx] = running
        running += len(entries[idx])
    entries_data = b"".join(entries[idx] for idx in ordered)
    entries_start = header_size + declared_count * (4 if sparse else (2 if offset16 else 4))
    chunk_size = entries_start + len(entries_data)

    flags_byte = (0x01 if sparse else 0) | (0x02 if offset16 else 0)
    head = struct.pack("<HHI", 0x0201, header_size, chunk_size)
    head += struct.pack("<BBH", type_id, flags_byte, 0)
    head += struct.pack("<II", declared_count, entries_start)
    head += config

    offsets = b""
    if sparse:
        # ResTable_sparseTypeEntry = union { u32; struct { u16 idx; u16 offset_div_4; } }
        # Little-endian => idx occupies the LOW 16 bits.
        for idx in ordered:
            offsets += struct.pack("<I", idx | ((entry_blobs[idx] // 4) << 16))
    elif offset16:
        packed = []
        for i in range(entry_count):
            if i in entry_blobs:
                assert entry_blobs[i] % 4 == 0
                packed.append(struct.pack("<H", entry_blobs[i] // 4))
            else:
                packed.append(struct.pack("<H", 0xFFFF))
        offsets = b"".join(packed)
    else:
        packed = []
        for i in range(entry_count):
            packed.append(
                struct.pack("<I", entry_blobs.get(i, 0xFFFFFFFF))
            )
        offsets = b"".join(packed)

    return head + offsets + entries_data


def build_library_chunk(entries: list[tuple[int, str]]) -> bytes:
    body = struct.pack("<HHII", 0x0203, 12, 12 + 260 * len(entries), len(entries))
    for package_id, name in entries:
        raw = name.encode("utf-16-le")[:254]
        raw += b"\x00" * (256 - len(raw))
        body += struct.pack("<I", package_id) + raw
    return body


def build_package(
    package_id: int,
    name: str,
    type_strings: list[str],
    key_strings: list[str],
    children: list[bytes],
) -> bytes:
    type_pool = build_string_pool(type_strings)
    key_pool = build_string_pool(key_strings)

    type_strings_off = 288
    key_strings_off = 288 + len(type_pool)

    body = type_pool + key_pool + b"".join(children)
    chunk_size = 288 + len(body)

    header = struct.pack("<HHI", 0x0200, 288, chunk_size)
    header += struct.pack("<I", package_id)
    raw_name = name.encode("utf-16-le")[:254]
    raw_name += b"\x00" * (256 - len(raw_name))
    header += raw_name
    header += struct.pack("<IIIII", type_strings_off, len(type_strings),
                          key_strings_off, len(key_strings), 0)

    return header + body


def build_table(
    global_strings: list[str],
    package: bytes,
    *,
    junk_chunks: list[bytes] | None = None,
    package_count: int | None = None,
) -> bytes:
    junk_chunks = junk_chunks or []
    body = b"".join(junk_chunks) + build_string_pool(global_strings) + package
    header = struct.pack(
        "<HHII", 0x0002, 12, 12 + len(body),
        len(junk_chunks) + 1 if package_count is None else package_count,
    )
    return header + body


def default_table(
    *,
    entries_for_drawable: dict[int, bytes] | None = None,
    entries_for_layout: dict[int, bytes] | None = None,
    drawable_flags: dict = {},
    children_extra: list[bytes] | None = None,
    junk_chunks: list[bytes] | None = None,
    global_strings: list[str] | None = None,
    package_count: int | None = None,
):
    global_strings = global_strings if global_strings is not None else [
        "hello world", "one!", "com.example.payload",
    ]
    entries_for_drawable = entries_for_drawable if entries_for_drawable is not None else {
        0: build_simple_entry(0, TYPE_STRING, 0, ENTRY_FLAG_PUBLIC),
    }
    entries_for_layout = entries_for_layout if entries_for_layout is not None else {
        0: build_simple_entry(1, TYPE_INT_DEC, 42),
    }

    children = [
        build_typespec(1, max(entries_for_drawable) + 1),
        build_type_chunk(1, max(entries_for_drawable) + 1,
                         entries_for_drawable, **drawable_flags),
        build_typespec(2, max(entries_for_layout) + 1),
        build_type_chunk(2, max(entries_for_layout) + 1, entries_for_layout),
    ] + (children_extra or [])

    package = build_package(
        0x7F, "com.example.test",
        ["drawable", "layout"],
        ["icon", "main", "app_name"],
        children,
    )
    return build_table(global_strings, package,
                       junk_chunks=junk_chunks, package_count=package_count)


# ============================================================================
# TESTS
# ============================================================================

def test_simple_string_entry_roundtrip():
    table = parse_resource_table_bytes(default_table())
    assert table.header.package_count == 1
    assert table.global_pool.strings == ["hello world", "one!", "com.example.payload"]

    pkg = table.packages[0]
    assert pkg.package_id == 0x7F
    assert pkg.name == "com.example.test"
    assert pkg.type_name(1) == "drawable"
    assert pkg.type_name(2) == "layout"

    entry = pkg.types[0].entries[0]
    assert entry.key_name == "icon"
    assert entry.is_public
    assert entry.value.data_type == TYPE_STRING

    rendered = _render(table, pkg.types[0], entry)
    assert "'hello world'" in rendered


def test_reference_resolution_uses_name():
    # drawable/icon references layout/main (0x7F020000).
    table = parse_resource_table_bytes(default_table())
    pkg = table.packages[0]
    t = pkg.types[0]
    entry = t.entries[0]
    assert entry.value.data_type == TYPE_STRING  # sanity

    entries = {0: build_simple_entry(0, TYPE_REFERENCE, 0x7F020000)}
    table = parse_resource_table_bytes(default_table(entries_for_drawable=entries))
    pkg = table.packages[0]
    rendered = _render(table, pkg.types[0], pkg.types[0].entries[0])
    assert "@com.example.test:layout/main" in rendered


def test_complex_bag_values_and_plural_labels():
    # idents are the well-known plural bag keys themselves (not their labels).
    items = [
        (0x01000004, TYPE_STRING, 0),   # other -> "hello world"
        (0x01000006, TYPE_STRING, 1),   # one   -> "one!"
    ]
    entries = {0: build_bag_entry(2, items, parent=0x7F020000)}
    table = parse_resource_table_bytes(default_table(entries_for_drawable=entries))
    pkg = table.packages[0]
    entry = pkg.types[0].entries[0]

    assert entry.is_complex
    assert entry.parent_ref == 0x7F020000
    assert len(entry.bag) == 2
    assert entry.bag[0].ident_label == "plural:other"
    assert entry.bag[1].ident_label == "plural:one"

    rendered = _render(table, pkg.types[0], entry)
    assert "parent=" in rendered
    assert "'hello world'" in rendered
    assert "'one!'" in rendered


def test_sparse_and_offset16_type():
    entries = {5: build_simple_entry(0, TYPE_INT_DEC, 7)}
    table = parse_resource_table_bytes(
        default_table(entries_for_drawable=entries,
                      drawable_flags={"sparse": True, "offset16": True})
    )
    pkg = table.packages[0]
    t = pkg.types[0]
    assert t.is_sparse and t.uses_offset16
    assert 5 in t.entries
    assert t.entries[5].value.data == 7
    # Entry 0 must not exist even though the compact layout skipped it.
    assert 0 not in t.entries


def test_compact_entry_decoding():
    entries = {0: build_compact_entry(0, TYPE_INT_DEC, 1234)}
    table = parse_resource_table_bytes(default_table(entries_for_layout=entries))
    pkg = table.packages[0]
    entry = pkg.types[1].entries[0]
    assert entry.is_compact
    assert entry.value.data_type == TYPE_INT_DEC
    assert entry.value.data == 1234
    assert entry.key_index == 0


def test_null_chunk_is_reported_as_anomaly():
    junk = struct.pack("<HHI", 0x0000, 8, 8 + 20) + b"https://evil.example\x00\x00\x00\x00"
    table = parse_resource_table_bytes(default_table(junk_chunks=[junk]))
    assert any("RES_NULL" in a for a in table.anomalies)
    assert any("evil.example" in u.printable for u in table.unknown_chunks)


def test_package_count_mismatch_is_reported():
    table = parse_resource_table_bytes(default_table(package_count=3))
    assert any("packageCount" in a for a in table.anomalies)


def test_library_chunk_resolution():
    lib = build_library_chunk([(0x02, "com.android.support")])
    table = parse_resource_table_bytes(default_table(children_extra=[lib]))
    resolver = table.resolver()
    assert resolver.package_label(0x02) == "com.android.support"
    # The android framework package id is always labeled 'android'.
    assert resolver.package_label(0x01) == "android"


def test_invalid_table_type_rejected():
    blob = struct.pack("<HHII", 0x9999, 12, 12, 0)
    with pytest.raises(ResourceTableError):
        parse_resource_table_bytes(blob)


def test_config_density_qualifier():
    config = bytearray(build_default_config())
    struct.pack_into("<H", config, 14, 480)  # density at offset 14
    table = parse_resource_table_bytes(
        default_table(drawable_flags={"config": bytes(config)})
    )
    t = table.packages[0].types[0]
    assert "480dpi" in t.config.qualifier


def test_zip_bomb_guard_rejects_explosive_arsc(tmp_path):
    # 10 MB of zeros compresses to ~10 KB -> ratio far above the 200x limit.
    apk = tmp_path / "bomb.apk"
    with zipfile.ZipFile(apk, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("resources.arsc", b"\x00" * (10 * 1024 * 1024))

    with pytest.raises(ResourceTableError, match="rejected"):
        parse_resource_table(apk)


def test_print_report_runs(capsys):
    table = parse_resource_table_bytes(default_table())
    print_resource_table(table)
    print_anomalies(table)
    out = capsys.readouterr().out
    assert "drawable/icon" in out
    assert "0x7F010000" in out


# ============================================================================
# HELPERS
# ============================================================================

def _render(table, type_chunk, entry) -> str:
    from tools.resource_table import format_entry_value
    return format_entry_value(entry, table, table.global_pool)
