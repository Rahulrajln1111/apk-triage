"""
Tests for the DEX parser.

The synthetic DEX is built byte-by-byte from the AOSP specification - the
same technique that caught struct-layout bugs in the arsc parser. If our
understanding of the format were wrong, these tests could not pass.
"""

import hashlib
import struct
import zlib
from pathlib import Path

import pytest

from tools.dex import (
    NO_INDEX,
    DexError,
    decode_mutf8,
    parse_dex_bytes,
    parse_dex_from_apk,
    sleb128,
    uleb128,
    uleb128p1,
)


# ============================================================================
# SYNTHETIC DEX BUILDER
# ============================================================================

def uleb(value: int) -> bytes:
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def mutf8(s: str) -> bytes:
    out = bytearray()
    for ch in s:
        cp = ord(ch)
        if cp == 0:
            out += b"\xc0\x80"
        elif cp < 0x80:
            out.append(cp)
        elif cp < 0x800:
            out.append(0xC0 | (cp >> 6))
            out.append(0x80 | (cp & 0x3F))
        else:
            out.append(0xE0 | (cp >> 12))
            out.append(0x80 | ((cp >> 6) & 0x3F))
            out.append(0x80 | (cp & 0x3F))
    return bytes(out)


STRINGS = ["", "V", "LFoo;", "main", "()V", "LBar;", "go"]


def build_minimal_dex() -> bytes:
    """One class LFoo; with one public static method main()V containing:
    nop; const-string v0, '()V'; invoke-static {v0}, LBar;->go()V; return-void
    """
    # ---- section content -------------------------------------------------
    string_data = b"".join(
        uleb(len(s)) + mutf8(s) + b"\x00" for s in STRINGS
    )
    string_ids = b"".join(struct.pack("<I", 0) for _ in STRINGS)  # patched later
    type_ids = struct.pack("<III", 1, 2, 5)          # V, LFoo;, LBar;
    proto_ids = struct.pack("<III", 1, 0, 0)         # shorty V, ret V, no params
    method_ids = struct.pack("<HHI", 1, 0, 3) + struct.pack("<HHI", 2, 0, 6)
    class_defs = struct.pack(
        "<IIIIIIII",
        1,              # class_idx -> LFoo;
        0x1,            # ACC_PUBLIC
        NO_INDEX,       # superclass
        0,              # interfaces_off
        NO_INDEX,       # source_file_idx
        0,              # annotations_off
        0,              # class_data_off (patched)
        0,              # static_values_off
    )

    # NOTE: Dalvik opcodes live in the LOW byte of the little-endian code
    # unit. const-string is 0x001A (op=0x1A, AA=v0), NOT 0x1A00.
    insns = struct.pack(
        "<7H",
        0x0000,   # nop
        0x001A,   # const-string vAA=0
        0x0004,   #   string@4 -> "()V"
        0x0071,   # invoke-static {v0}
        0x0001,   #   method@1 -> LBar;->go()V
        0x0000,   #   (args word)
        0x000E,   # return-void
    )

    # ---- layout ----------------------------------------------------------
    cursor = 112
    string_ids_off = cursor;      cursor += len(string_ids)          # 28
    type_ids_off = cursor;        cursor += len(type_ids)            # 12
    proto_ids_off = cursor;       cursor += len(proto_ids)           # 12
    method_ids_off = cursor;      cursor += len(method_ids)          # 16
    class_defs_off = cursor;      cursor += len(class_defs)          # 32
    string_data_off = cursor;     cursor += len(string_data)
    class_data_off = cursor;      cursor += 5   # 4x uleb counts + 3 ulebs below
    class_data = (
        uleb(0) + uleb(0) + uleb(1) + uleb(0)      # sizes: 0f,0i,1d,0v
        + uleb(0) + uleb(0x9) + uleb(0)            # method diff 0, flags, code_off patched
    )
    assert len(class_data) <= 8
    cursor += len(class_data)
    code_off = (cursor + 3) & ~3; cursor = code_off + 16 + len(insns)
    map_off = (cursor + 3) & ~3

    map_items = [
        (0x0000, 1, 0),
        (0x0001, len(STRINGS), string_ids_off),
        (0x0002, 3, type_ids_off),
        (0x0003, 1, proto_ids_off),
        (0x0005, 2, method_ids_off),
        (0x0006, 1, class_defs_off),
        (0x2002, len(STRINGS), string_data_off),
        (0x2000, 1, class_data_off),
        (0x2001, 1, code_off),
        (0x1000, 10, map_off),
    ]
    map_list = struct.pack("<I", len(map_items)) + b"".join(
        struct.pack("<HHII", t, 0, n, o) for t, n, o in map_items
    )
    file_size = map_off + 4 + len(map_list)

    # string_id -> string_data offsets
    string_ids = b"".join(
        struct.pack("<I", string_data_off + off)
        for off in [p for p in (
            sum(len(uleb(len(s))) + len(mutf8(s)) + 1 for s in STRINGS[:i])
            for i in range(len(STRINGS))
        )]
    )

    class_data = (
        uleb(0) + uleb(0) + uleb(1) + uleb(0)
        + uleb(0) + uleb(0x9) + uleb(code_off)
    )

    # ---- assemble --------------------------------------------------------
    buf = bytearray(file_size)
    buf[0:8] = b"dex\n035\x00"

    def put(off: int, blob: bytes) -> None:
        buf[off:off + len(blob)] = blob

    put(string_ids_off, string_ids)
    put(type_ids_off, type_ids)
    put(proto_ids_off, proto_ids)
    put(method_ids_off, method_ids)
    put(class_defs_off, class_defs)
    put(string_data_off, string_data)
    put(class_data_off, class_data)
    put(code_off, struct.pack(
        "<HHHHII", 1, 0, 0, 0, 0, len(insns) // 2
    ))
    put(code_off + 16, insns)
    put(map_off, map_list)

    # class_def[0].class_data_off lives at byte 24 of the 32-byte item.
    struct.pack_into("<I", buf, class_defs_off + 24, class_data_off)

    # ---- header ----------------------------------------------------------
    struct.pack_into("<I", buf, 32, file_size)
    struct.pack_into("<I", buf, 36, 112)
    struct.pack_into("<I", buf, 40, 0x12345678)
    struct.pack_into("<II", buf, 44, 0, 0)          # link
    struct.pack_into("<I", buf, 52, map_off)
    struct.pack_into("<II", buf, 56, len(STRINGS), string_ids_off)
    struct.pack_into("<II", buf, 64, 3, type_ids_off)
    struct.pack_into("<II", buf, 72, 1, proto_ids_off)
    struct.pack_into("<II", buf, 80, 0, 0)          # fields
    struct.pack_into("<II", buf, 88, 2, method_ids_off)
    struct.pack_into("<II", buf, 96, 1, class_defs_off)
    struct.pack_into("<II", buf, 104, file_size - string_data_off, string_data_off)

    # Spec order matters: the SHA-1 signature covers bytes 32..end, and the
    # adler32 checksum covers bytes 12..end (i.e. INCLUDING the signature).
    buf[12:32] = hashlib.sha1(bytes(buf[32:])).digest()
    struct.pack_into("<I", buf, 8, zlib.adler32(bytes(buf[12:])) & 0xFFFFFFFF)

    return bytes(buf)


# ============================================================================
# PRIMITIVE TESTS (vectors straight from the AOSP spec tables)
# ============================================================================

def test_uleb128_spec_vectors():
    assert uleb128(b"\x00", 0) == (0, 1)
    assert uleb128(b"\x01", 0) == (1, 1)
    assert uleb128(b"\x7f", 0) == (127, 1)
    assert uleb128(b"\x80\x7f", 0) == (16256, 2)


def test_sleb128_spec_vectors():
    assert sleb128(b"\x00", 0) == (0, 1)
    assert sleb128(b"\x01", 0) == (1, 1)
    assert sleb128(b"\x7f", 0) == (-1, 1)
    assert sleb128(b"\x80\x7f", 0) == (-128, 2)


def test_uleb128p1_no_index():
    # NO_INDEX (-1) is the single-byte encoding 0x00.
    assert uleb128p1(b"\x00", 0) == (-1, 1)
    assert uleb128p1(b"\x7f", 0) == (126, 1)


def test_mutf8_decoding():
    assert decode_mutf8(b"hello", 0, 5) == "hello"
    assert decode_mutf8(b"\xc0\x80", 0, 2) == "\x00"        # encoded NUL
    assert decode_mutf8("€".encode("utf-8"), 0, 3) == "€"
    assert decode_mutf8(b"\xff", 0, 1) == "\ufffd"          # malformed -> U+FFFD


# ============================================================================
# SYNTHETIC DEX TESTS
# ============================================================================

def test_synthetic_header_and_integrity():
    dex = parse_dex_bytes(build_minimal_dex())
    assert dex.checksum_ok and dex.signature_ok
    assert dex.header.version == "035"
    assert dex.stats() == {
        "version": "035", "strings": 7, "types": 3, "protos": 1,
        "fields": 0, "methods": 2, "classes": 1,
        "checksum_ok": True, "signature_ok": True, "anomalies": 0,
    }
    assert dex.anomalies == []


def test_resolution_chain():
    dex = parse_dex_bytes(build_minimal_dex())
    assert dex.string(3) == "main"
    assert dex.type_desc(1) == "LFoo;"
    assert dex.type_desc(2) == "LBar;"
    assert dex.method_str(0) == "LFoo;->main() -> V"
    assert dex.method_str(1) == "LBar;->go() -> V"


def test_class_def_and_code_item():
    dex = parse_dex_bytes(build_minimal_dex())
    cdef = dex.class_defs[0]
    assert dex.type_desc(cdef.class_idx) == "LFoo;"
    assert cdef.access_flags == 0x1
    assert len(cdef.direct_methods) == 1
    method = cdef.direct_methods[0]
    assert method.access_flags == 0x9  # public + static

    code = dex.code_for(method)
    assert code is not None
    assert code.registers_size == 1
    assert code.insns_size == 7
    assert code.tries_size == 0


def test_histogram_api_refs_and_string_constants():
    dex = parse_dex_bytes(build_minimal_dex())
    histogram = dex.opcode_histogram()
    assert histogram[0x00] == 1   # nop
    assert histogram[0x1A] == 1   # const-string
    assert histogram[0x71] == 1   # invoke-static
    assert histogram[0x0E] == 1   # return-void

    refs = dex.api_references()
    assert refs == {"LBar;->go() -> V": 1}

    assert "()V" in dex.string_constants()


def test_checksum_mismatch_is_detected():
    blob = bytearray(build_minimal_dex())
    blob[-20] ^= 0xFF  # corrupt a byte inside the map list
    dex = parse_dex_bytes(bytes(blob))
    assert not dex.checksum_ok
    assert any("adler32" in a for a in dex.anomalies)


def test_trailing_data_is_detected():
    blob = build_minimal_dex() + b"STAGEDEX"  # appended hidden data
    dex = parse_dex_bytes(blob)
    assert any("appended after declared file_size" in a for a in dex.anomalies)


def test_bad_magic_rejected():
    with pytest.raises(DexError):
        parse_dex_bytes(b"NOPE" + b"\x00" * 200)


# ============================================================================
# REAL APK VALIDATION
# ============================================================================

APK_PATH = Path(__file__).parent.parent / "apks" / "sample.apk"


@pytest.mark.skipif(not APK_PATH.exists(), reason="sample.apk not present")
def test_real_apk_classes_dex():
    dex = parse_dex_from_apk(APK_PATH, "classes.dex")

    assert dex.checksum_ok, "adler32 must match on a genuine file"
    assert dex.signature_ok, "sha1 must match on a genuine file"
    assert len(dex.class_defs) == 142

    # The self-validating instruction walk must be clean on all code items:
    # this is the acceptance test for the opcode width table.
    assert not any("misaligned" in a for a in dex.anomalies), dex.anomalies

    histogram = dex.opcode_histogram()
    assert sum(histogram.values()) > 1000
    assert 0x71 in histogram or 0x6E in histogram  # invoke-* present

    refs = dex.api_references()
    assert len(refs) > 100

    strings = dex.string_constants()
    assert any("http" in s.lower() for s in strings)


@pytest.mark.skipif(not APK_PATH.exists(), reason="sample.apk not present")
def test_multi_dex_discovery():
    from tools.dex import discover_dex_entries
    names = discover_dex_entries(APK_PATH)
    assert "classes.dex" in names
    assert all(n.startswith("classes") for n in names)
