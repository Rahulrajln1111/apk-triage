"""
DEX (Dalvik Executable) forensic parser.

Parses .dex files directly from the AOSP "Dalvik executable format"
specification - no external libraries. Designed for APKAI Phase 4:

- header parsing with checksum (adler32) + SHA-1 signature verification
- uleb128 / sleb128 / uleb128p1 and MUTF-8 string decoding
- string_ids (lazy decode + cache), type_ids, proto_ids, field_ids, method_ids
- class_defs -> class_data_item -> encoded_field / encoded_method
- code_item parsing including try/catch handler structures
- map_list parsing with cross-checks against the header
- linear instruction walk with the Dalvik opcode width table
  (self-validating: every method must walk to exactly insns_size)
- opcode histograms and API-reference extraction (Phase 8/9 features)
- anomalies instead of hard failures: hostile DEX must still yield data

Memory discipline: string data is decoded on demand and cached; instruction
bytes are never copied - only walked in place.
"""

from __future__ import annotations

import hashlib
import re
import struct
import zlib
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from tools.zip_audit import ZipEntryError, ZipReadLimitError, read_entry_bytes


# ============================================================================
# CONSTANTS (AOSP dex-format spec)
# ============================================================================

DEX_MAGIC_PREFIX = b"dex\n"
ENDIAN_CONSTANT = 0x12345678
REVERSE_ENDIAN_CONSTANT = 0x78563412
NO_INDEX = 0xFFFFFFFF

HEADER_SIZE_V35_TO_V40 = 0x70
HEADER_SIZE_V41 = 0x78

# Access flags (classes / fields / methods).
ACC_PUBLIC = 0x1
ACC_PRIVATE = 0x2
ACC_PROTECTED = 0x4
ACC_STATIC = 0x8
ACC_FINAL = 0x10
ACC_SYNCHRONIZED = 0x20
ACC_VOLATILE = 0x40
ACC_BRIDGE = 0x40
ACC_TRANSIENT = 0x80
ACC_VARARGS = 0x80
ACC_NATIVE = 0x100
ACC_INTERFACE = 0x200
ACC_ABSTRACT = 0x400
ACC_STRICT = 0x800
ACC_SYNTHETIC = 0x1000
ACC_ANNOTATION = 0x2000
ACC_ENUM = 0x4000
ACC_CONSTRUCTOR = 0x10000
ACC_DECLARED_SYNCHRONIZED = 0x20000

# map_item type codes.
TYPE_HEADER_ITEM = 0x0000
TYPE_STRING_ID_ITEM = 0x0001
TYPE_TYPE_ID_ITEM = 0x0002
TYPE_PROTO_ID_ITEM = 0x0003
TYPE_FIELD_ID_ITEM = 0x0004
TYPE_METHOD_ID_ITEM = 0x0005
TYPE_CLASS_DEF_ITEM = 0x0006
TYPE_CALL_SITE_ID_ITEM = 0x0007
TYPE_METHOD_HANDLE_ITEM = 0x0008
TYPE_MAP_LIST = 0x1000
TYPE_TYPE_LIST = 0x1001
TYPE_ANNOTATION_SET_REF_LIST = 0x1002
TYPE_ANNOTATION_SET_ITEM = 0x1003
TYPE_CLASS_DATA_ITEM = 0x2000
TYPE_CODE_ITEM = 0x2001
TYPE_STRING_DATA_ITEM = 0x2002
TYPE_DEBUG_INFO_ITEM = 0x2003
TYPE_ANNOTATION_ITEM = 0x2004
TYPE_ENCODED_ARRAY_ITEM = 0x2005
TYPE_ANNOTATIONS_DIRECTORY_ITEM = 0x2006
TYPE_HIDDENAPI_CLASS_DATA_ITEM = 0xF000

MAP_TYPE_NAMES = {
    0x0000: "header_item",
    0x0001: "string_id_item",
    0x0002: "type_id_item",
    0x0003: "proto_id_item",
    0x0004: "field_id_item",
    0x0005: "method_id_item",
    0x0006: "class_def_item",
    0x0007: "call_site_id_item",
    0x0008: "method_handle_item",
    0x1000: "map_list",
    0x1001: "type_list",
    0x1002: "annotation_set_ref_list",
    0x1003: "annotation_set_item",
    0x2000: "class_data_item",
    0x2001: "code_item",
    0x2002: "string_data_item",
    0x2003: "debug_info_item",
    0x2004: "annotation_item",
    0x2005: "encoded_array_item",
    0x2006: "annotations_directory_item",
    0xF000: "hiddenapi_class_data_item",
}

# Dalvik opcode -> width in 16-bit code units. Everything not listed is 1.
# Pseudo-opcodes (nop variants) have variable width and are handled by the
# instruction walker, not this table.
_OPCODE_WIDTHS: dict[int, int] = {}

for _r in [(0x02, 0x03), (0x05, 0x06), (0x08, 0x09), (0x13, 0x13), (0x15, 0x16),
           (0x16, 0x16), (0x19, 0x19), (0x1a, 0x1a), (0x1c, 0x1c), (0x1f, 0x20),
           (0x22, 0x23), (0x29, 0x29)]:
    for _o in range(_r[0], _r[1] + 1):
        _OPCODE_WIDTHS[_o] = 2

for _o in (0x03, 0x06, 0x09, 0x14, 0x17, 0x1b, 0x24, 0x25, 0x26, 0x2a, 0x2b, 0x2c):
    _OPCODE_WIDTHS[_o] = 3

for _o in (0x18,):
    _OPCODE_WIDTHS[_o] = 5

_OPCODE_WIDTHS[0x21] = 2             # throw-verification-error (21c)

for _o in range(0x2d, 0x32):      # cmpl/cmpg-float, cmp-double, cmp-long (23x)
    _OPCODE_WIDTHS[_o] = 2

for _o in range(0x32, 0x38):      # if-eq..if-ne (22t)
    _OPCODE_WIDTHS[_o] = 2

for _o in range(0x38, 0x3e):      # if-*z (21t)
    _OPCODE_WIDTHS[_o] = 2

for _o in range(0x44, 0x52):      # aget/aput
    _OPCODE_WIDTHS[_o] = 2

for _o in range(0x52, 0x60):      # iget/iput
    _OPCODE_WIDTHS[_o] = 2

for _o in range(0x60, 0x6e):      # sget/sput
    _OPCODE_WIDTHS[_o] = 2

for _o in range(0x6e, 0x73):      # invoke-kind
    _OPCODE_WIDTHS[_o] = 3

for _o in range(0x74, 0x79):      # invoke-kind/range
    _OPCODE_WIDTHS[_o] = 3

for _o in range(0x90, 0xb0):      # binop
    _OPCODE_WIDTHS[_o] = 2

for _o in range(0xd0, 0xd8):      # binop/lit16
    _OPCODE_WIDTHS[_o] = 2

for _o in range(0xd8, 0xe3):      # binop/lit8
    _OPCODE_WIDTHS[_o] = 2

_OPCODE_WIDTHS[0xfa] = 4  # invoke-polymorphic (45cc)
_OPCODE_WIDTHS[0xfb] = 4  # invoke-polymorphic/range (4rcc)
_OPCODE_WIDTHS[0xfc] = 3  # invoke-custom (35c)
_OPCODE_WIDTHS[0xfd] = 3  # invoke-custom/range (3rc)
_OPCODE_WIDTHS[0xfe] = 2  # const-method-handle (21c)
_OPCODE_WIDTHS[0xff] = 2  # const-method-type (21c) - AOSP instruction-formats
                          # table says format 21c = 2 code units; the old value
                          # of 3 was wrong (caught cross-checking smali.py)

# Invoke opcodes whose BC operand is a method_id index.
INVOKE_OPCODES = set(range(0x6e, 0x73)) | set(range(0x74, 0x79)) | {0xfa, 0xfb, 0xfc, 0xfd}

# Opcodes that embed a string_id index in their operand.
STRING_OPCODES = {0x1a, 0x1b}


class DexError(ValueError):
    """Raised when a DEX file cannot be parsed at all."""


# ============================================================================
# PRIMITIVES: uleb128 / sleb128 / MUTF-8
# ============================================================================

def uleb128(data: bytes, offset: int, limit: int | None = None) -> tuple[int, int]:
    """Decode unsigned LEB128. Returns (value, next_offset)."""
    end = limit if limit is not None else len(data)
    result = 0
    shift = 0
    while True:
        if offset >= end:
            raise DexError(f"uleb128 runs past limit at {offset:#x}")
        byte = data[offset]
        offset += 1
        result |= (byte & 0x7F) << shift
        if not (byte & 0x80):
            return result, offset
        shift += 7
        if shift > 28:
            raise DexError("uleb128 exceeds 5 bytes")


def uleb128p1(data: bytes, offset: int, limit: int | None = None) -> tuple[int, int]:
    """Decode uleb128p1 (value+1; single-byte encoding of -1/NO_INDEX)."""
    value, offset = uleb128(data, offset, limit)
    return value - 1, offset


def sleb128(data: bytes, offset: int, limit: int | None = None) -> tuple[int, int]:
    """Decode signed LEB128. Returns (value, next_offset)."""
    end = limit if limit is not None else len(data)
    result = 0
    shift = 0
    while True:
        if offset >= end:
            raise DexError(f"sleb128 runs past limit at {offset:#x}")
        byte = data[offset]
        offset += 1
        result |= (byte & 0x7F) << shift
        shift += 7
        if not (byte & 0x80):
            if shift < 64 and (byte & 0x40):
                result |= -(1 << shift)
            return result, offset
        if shift > 63:
            raise DexError("sleb128 exceeds valid length")


def decode_mutf8(data: bytes, start: int, end: int) -> str:
    """Decode MUTF-8 (Modified UTF-8) bytes in data[start:end].

    Differences from UTF-8 per spec: U+0000 is encoded as 0xC0 0x80,
    supplementary planes are encoded as surrogate pairs (CESU-8 style).
    Decoding is tolerant: malformed sequences become U+FFFD instead of
    aborting the parse (hostile input still yields forensic data).
    """
    units: list[int] = []
    i = start
    while i < end:
        b0 = data[i]
        if b0 < 0x80:
            units.append(b0)
            i += 1
        elif b0 >> 5 == 0b110 and i + 1 < end:
            units.append(((b0 & 0x1F) << 6) | (data[i + 1] & 0x3F))
            i += 2
        elif b0 >> 4 == 0b1110 and i + 2 < end:
            units.append(((b0 & 0x0F) << 12) | ((data[i + 1] & 0x3F) << 6) | (data[i + 2] & 0x3F))
            i += 3
        else:
            units.append(0xFFFD)
            i += 1
    return "".join(
        chr(u) if not (0xD800 <= u <= 0xDFFF) else "\uFFFD" for u in units
    )


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class DexHeader:
    magic: bytes
    version: str
    checksum: int
    signature: bytes
    file_size: int
    header_size: int
    endian_tag: int
    link_size: int
    link_off: int
    map_off: int
    string_ids_size: int
    string_ids_off: int
    type_ids_size: int
    type_ids_off: int
    proto_ids_size: int
    proto_ids_off: int
    field_ids_size: int
    field_ids_off: int
    method_ids_size: int
    method_ids_off: int
    class_defs_size: int
    class_defs_off: int
    data_size: int
    data_off: int
    container_size: int | None = None
    header_offset: int | None = None


@dataclass
class ProtoId:
    shorty_idx: int
    return_type_idx: int
    parameters_off: int


@dataclass
class FieldId:
    class_idx: int
    type_idx: int
    name_idx: int


@dataclass
class MethodId:
    class_idx: int
    proto_idx: int
    name_idx: int


@dataclass
class EncodedField:
    field_idx: int
    access_flags: int


@dataclass
class EncodedMethod:
    method_idx: int
    access_flags: int
    code_off: int


@dataclass
class TryItem:
    start_addr: int
    insn_count: int
    handler_off: int


@dataclass
class CodeItem:
    offset: int
    registers_size: int
    ins_size: int
    outs_size: int
    tries_size: int
    debug_info_off: int
    insns_size: int
    insns_off: int
    tries: list[TryItem] = field(default_factory=list)

    def opcode_histogram(self, data: bytes) -> Counter:
        return walk_instructions_histogram(data, self)


@dataclass
class ClassDef:
    class_idx: int
    access_flags: int
    superclass_idx: int
    interfaces_off: int
    source_file_idx: int
    annotations_off: int
    class_data_off: int
    static_values_off: int
    static_fields: list[EncodedField] = field(default_factory=list)
    instance_fields: list[EncodedField] = field(default_factory=list)
    direct_methods: list[EncodedMethod] = field(default_factory=list)
    virtual_methods: list[EncodedMethod] = field(default_factory=list)


@dataclass
class MapItem:
    item_type: int
    size: int
    offset: int


@dataclass
class DexFile:
    data: bytes
    header: DexHeader
    string_offsets: list[int] = field(default_factory=list)
    type_ids: list[int] = field(default_factory=list)
    proto_ids: list[ProtoId] = field(default_factory=list)
    field_ids: list[FieldId] = field(default_factory=list)
    method_ids: list[MethodId] = field(default_factory=list)
    class_defs: list[ClassDef] = field(default_factory=list)
    map_items: list[MapItem] = field(default_factory=list)
    anomalies: list[str] = field(default_factory=list)
    checksum_ok: bool = False
    signature_ok: bool = False
    _string_cache: dict[int, str] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Lazy string access
    # ------------------------------------------------------------------

    def string(self, index: int) -> str | None:
        if index in self._string_cache:
            return self._string_cache[index]
        if not (0 <= index < len(self.string_offsets)):
            return None
        data = self.data
        offset = self.string_offsets[index]
        try:
            utf16_size, pos = uleb128(data, offset)
            end = data.index(b"\x00", pos)
            if end > len(data):
                raise DexError("string terminator past end of file")
            value = decode_mutf8(data, pos, end)
        except (DexError, ValueError):
            value = f"<malformed-string#{index}>"
        if len(self._string_cache) < 200_000:
            self._string_cache[index] = value
        return value

    # ------------------------------------------------------------------
    # Resolution helpers
    # ------------------------------------------------------------------

    def type_desc(self, index: int) -> str | None:
        if not (0 <= index < len(self.type_ids)):
            return None
        return self.string(self.type_ids[index])

    def proto_str(self, index: int) -> str:
        if not (0 <= index < len(self.proto_ids)):
            return "<bad-proto>"
        proto = self.proto_ids[index]
        ret = self.type_desc(proto.return_type_idx) or "?"
        params: list[str] = []
        if proto.parameters_off:
            try:
                count = struct.unpack_from("<I", self.data, proto.parameters_off)[0]
                for i in range(count):
                    type_idx = struct.unpack_from(
                        "<H", self.data, proto.parameters_off + 4 + i * 2
                    )[0]
                    params.append(self.type_desc(type_idx) or "?")
            except struct.error:
                return "<bad-proto-params>"
        return f"({', '.join(params)}) -> {ret}"

    def field_str(self, index: int) -> str:
        if not (0 <= index < len(self.field_ids)):
            return "<bad-field>"
        f = self.field_ids[index]
        cls = self.type_desc(f.class_idx) or "?"
        name = self.string(f.name_idx) or "?"
        typ = self.type_desc(f.type_idx) or "?"
        return f"{cls}->{name}:{typ}"

    def method_str(self, index: int) -> str:
        if not (0 <= index < len(self.method_ids)):
            return "<bad-method>"
        m = self.method_ids[index]
        cls = self.type_desc(m.class_idx) or "?"
        name = self.string(m.name_idx) or "?"
        return f"{cls}->{name}{self.proto_str(m.proto_idx)}"

    def code_for(self, method: EncodedMethod) -> CodeItem | None:
        if not method.code_off:
            return None
        return parse_code_item(self.data, method.code_off, self.anomalies)

    # ------------------------------------------------------------------
    # Forensic aggregation
    # ------------------------------------------------------------------

    def iter_code_items(self):
        for cdef in self.class_defs:
            for method in cdef.direct_methods + cdef.virtual_methods:
                code = self.code_for(method)
                if code is not None:
                    yield cdef, method, code

    def opcode_histogram(self) -> Counter:
        total: Counter = Counter()
        for _, _, code in self.iter_code_items():
            total.update(code.opcode_histogram(self.data))
        return total

    def api_references(self) -> Counter:
        """All method references found in invoke instructions (Phase 8 input)."""
        refs: Counter = Counter()
        data = self.data
        for _, _, code in self.iter_code_items():
            end = code.insns_off + code.insns_size * 2
            pos = code.insns_off
            while pos + 2 <= end:
                opcode = data[pos] if isinstance(data[pos], int) else data[pos]
                unit = struct.unpack_from("<H", data, pos)[0]
                op = unit & 0xFF
                if op in INVOKE_OPCODES and pos + 6 <= end:
                    # BC operand: 16-bit method index (bytecode formats 35c/3rc
                    # store it in code unit 1 for 35c, unit 1 for 3rc as well).
                    method_idx = struct.unpack_from("<H", data, pos + 2)[0]
                    if 0 <= method_idx < len(self.method_ids):
                        refs[self.method_str(method_idx)] += 1
                pos += instruction_width(data, pos, end) * 2
        return refs

    def string_constants(self) -> set[str]:
        """All string constants referenced by const-string instructions."""
        found: set[str] = set()
        data = self.data
        for _, _, code in self.iter_code_items():
            end = code.insns_off + code.insns_size * 2
            pos = code.insns_off
            while pos + 2 <= end:
                unit = struct.unpack_from("<H", data, pos)[0]
                op = unit & 0xFF
                if op in STRING_OPCODES:
                    size = 4 if op == 0x1b else 2
                    if pos + 2 + size <= end:
                        raw = data[pos + 2:pos + 2 + size]
                        idx = int.from_bytes(raw, "little")
                        value = self.string(idx)
                        if value is not None:
                            found.add(value)
                pos += instruction_width(data, pos, end) * 2
        return found

    def stats(self) -> dict:
        return {
            "version": self.header.version,
            "strings": len(self.string_offsets),
            "types": len(self.type_ids),
            "protos": len(self.proto_ids),
            "fields": len(self.field_ids),
            "methods": len(self.method_ids),
            "classes": len(self.class_defs),
            "checksum_ok": self.checksum_ok,
            "signature_ok": self.signature_ok,
            "anomalies": len(self.anomalies),
        }


# ============================================================================
# HEADER
# ============================================================================

def parse_dex_header(data: bytes) -> DexHeader:
    if len(data) < 112:
        raise DexError(f"buffer too small for DEX header: {len(data)} bytes")
    if data[:4] != DEX_MAGIC_PREFIX:
        raise DexError(f"bad DEX magic: {data[:4]!r}")

    version = data[4:7].decode("ascii", errors="replace")
    if data[7] != 0:
        raise DexError("DEX magic terminator is not NUL")

    (
        checksum,
    ) = struct.unpack_from("<I", data, 8)
    signature = data[12:32]
    file_size, header_size, endian_tag, link_size, link_off, map_off = struct.unpack_from(
        "<IIIIII", data, 32
    )
    (
        string_ids_size, string_ids_off,
        type_ids_size, type_ids_off,
        proto_ids_size, proto_ids_off,
        field_ids_size, field_ids_off,
        method_ids_size, method_ids_off,
        class_defs_size, class_defs_off,
        data_size, data_off,
    ) = struct.unpack_from("<IIIIIIIIIIIIII", data, 56)

    container_size = None
    header_offset = None
    if header_size >= HEADER_SIZE_V41:
        container_size, header_offset = struct.unpack_from("<II", data, 104)

    return DexHeader(
        magic=bytes(data[:8]),
        version=version,
        checksum=checksum,
        signature=bytes(signature),
        file_size=file_size,
        header_size=header_size,
        endian_tag=endian_tag,
        link_size=link_size,
        link_off=link_off,
        map_off=map_off,
        string_ids_size=string_ids_size,
        string_ids_off=string_ids_off,
        type_ids_size=type_ids_size,
        type_ids_off=type_ids_off,
        proto_ids_size=proto_ids_size,
        proto_ids_off=proto_ids_off,
        field_ids_size=field_ids_size,
        field_ids_off=field_ids_off,
        method_ids_size=method_ids_size,
        method_ids_off=method_ids_off,
        class_defs_size=class_defs_size,
        class_defs_off=class_defs_off,
        data_size=data_size,
        data_off=data_off,
        container_size=container_size,
        header_offset=header_offset,
    )


def verify_integrity(data: bytes, header: DexHeader, anomalies: list[str]) -> tuple[bool, bool]:
    """Recompute adler32 checksum and SHA-1 signature. Mismatches = tampering."""
    checksum_ok = False
    signature_ok = False

    computed_adler = zlib.adler32(data[12:]) & 0xFFFFFFFF
    checksum_ok = computed_adler == header.checksum
    if not checksum_ok:
        anomalies.append(
            f"adler32 checksum mismatch: stored 0x{header.checksum:08X} "
            f"computed 0x{computed_adler:08X} (modified/tampered file)"
        )

    computed_sha1 = hashlib.sha1(data[32:]).digest()
    signature_ok = computed_sha1 == header.signature
    if not signature_ok:
        anomalies.append(
            f"SHA-1 signature mismatch: stored {header.signature.hex()} "
            f"computed {computed_sha1.hex()}"
        )

    if header.endian_tag == REVERSE_ENDIAN_CONSTANT:
        anomalies.append("byte-swapped endian tag (REVERSE_ENDIAN_CONSTANT)")
    elif header.endian_tag != ENDIAN_CONSTANT:
        anomalies.append(f"unknown endian tag 0x{header.endian_tag:08X}")

    if header.file_size > len(data):
        anomalies.append(
            f"file_size {header.file_size:,} exceeds buffer {len(data):,} (truncated)"
        )
    elif header.file_size < len(data):
        anomalies.append(
            f"{len(data) - header.file_size:,} bytes appended after declared "
            "file_size (possible hidden data)"
        )

    if header.header_size not in (HEADER_SIZE_V35_TO_V40, HEADER_SIZE_V41):
        anomalies.append(f"unexpected header_size 0x{header.header_size:X}")
    if header.version == "041":
        anomalies.append("DEX v041 container format (experimental)")

    return checksum_ok, signature_ok


# ============================================================================
# INDEX TABLES
# ============================================================================

def parse_string_ids(data: bytes, header: DexHeader, anomalies: list[str]) -> list[int]:
    offsets: list[int] = []
    section = header.string_ids_off
    size = header.string_ids_size
    if size == 0:
        return offsets
    if section + size * 4 > len(data):
        raise DexError("string_ids section exceeds file")
    for i in range(size):
        off = struct.unpack_from("<I", data, section + i * 4)[0]
        if off >= len(data):
            anomalies.append(f"string_ids[{i}] offset {off:#x} out of range")
            off = 0
        offsets.append(off)
    return offsets


def parse_type_ids(data: bytes, header: DexHeader, anomalies: list[str]) -> list[int]:
    ids: list[int] = []
    section, size = header.type_ids_off, header.type_ids_size
    if size == 0:
        return ids
    if section + size * 4 > len(data):
        raise DexError("type_ids section exceeds file")
    for i in range(size):
        idx = struct.unpack_from("<I", data, section + i * 4)[0]
        if idx >= header.string_ids_size:
            anomalies.append(f"type_ids[{i}] descriptor_idx {idx} out of range")
        ids.append(idx)
    return ids


def parse_proto_ids(data: bytes, header: DexHeader, anomalies: list[str]) -> list[ProtoId]:
    protos: list[ProtoId] = []
    section, size = header.proto_ids_off, header.proto_ids_size
    if size == 0:
        return protos
    if section + size * 12 > len(data):
        raise DexError("proto_ids section exceeds file")
    for i in range(size):
        shorty_idx, return_type_idx, parameters_off = struct.unpack_from(
            "<III", data, section + i * 12
        )
        if return_type_idx >= header.type_ids_size:
            anomalies.append(f"proto_ids[{i}] return_type_idx out of range")
        if shorty_idx >= header.string_ids_size:
            anomalies.append(f"proto_ids[{i}] shorty_idx out of range")
        protos.append(ProtoId(shorty_idx, return_type_idx, parameters_off))
    return protos


def parse_field_ids(data: bytes, header: DexHeader, anomalies: list[str]) -> list[FieldId]:
    fields: list[FieldId] = []
    section, size = header.field_ids_off, header.field_ids_size
    if size == 0:
        return fields
    if section + size * 8 > len(data):
        raise DexError("field_ids section exceeds file")
    for i in range(size):
        class_idx, type_idx, name_idx = struct.unpack_from(
            "<HHI", data, section + i * 8
        )
        if class_idx >= header.type_ids_size or type_idx >= header.type_ids_size:
            anomalies.append(f"field_ids[{i}] type index out of range")
        if name_idx >= header.string_ids_size:
            anomalies.append(f"field_ids[{i}] name_idx out of range")
        fields.append(FieldId(class_idx, type_idx, name_idx))
    return fields


def parse_method_ids(data: bytes, header: DexHeader, anomalies: list[str]) -> list[MethodId]:
    methods: list[MethodId] = []
    section, size = header.method_ids_off, header.method_ids_size
    if size == 0:
        return methods
    if section + size * 8 > len(data):
        raise DexError("method_ids section exceeds file")
    for i in range(size):
        class_idx, proto_idx, name_idx = struct.unpack_from(
            "<HHI", data, section + i * 8
        )
        if class_idx >= header.type_ids_size:
            anomalies.append(f"method_ids[{i}] class_idx out of range")
        if proto_idx >= header.proto_ids_size:
            anomalies.append(f"method_ids[{i}] proto_idx out of range")
        if name_idx >= header.string_ids_size:
            anomalies.append(f"method_ids[{i}] name_idx out of range")
        methods.append(MethodId(class_idx, proto_idx, name_idx))
    return methods


# ============================================================================
# CLASS DATA
# ============================================================================

def parse_class_data(
    data: bytes,
    offset: int,
    header: DexHeader,
    cdef: ClassDef,
    anomalies: list[str],
) -> None:
    limit = len(data)
    static_n, pos = uleb128(data, offset, limit)
    instance_n, pos = uleb128(data, pos, limit)
    direct_n, pos = uleb128(data, pos, limit)
    virtual_n, pos = uleb128(data, pos, limit)

    def read_fields(count: int) -> list[EncodedField]:
        nonlocal pos
        fields: list[EncodedField] = []
        idx = 0
        for _ in range(count):
            diff, pos = uleb128(data, pos, limit)
            flags, pos = uleb128(data, pos, limit)
            idx += diff
            if idx >= header.field_ids_size:
                anomalies.append(
                    f"class {cdef.class_idx}: field_idx {idx} out of range"
                )
            fields.append(EncodedField(idx, flags))
        return fields

    def read_methods(count: int) -> list[EncodedMethod]:
        nonlocal pos
        methods: list[EncodedMethod] = []
        idx = 0
        for _ in range(count):
            diff, pos = uleb128(data, pos, limit)
            flags, pos = uleb128(data, pos, limit)
            code_off, pos = uleb128(data, pos, limit)
            idx += diff
            if idx >= header.method_ids_size:
                anomalies.append(
                    f"class {cdef.class_idx}: method_idx {idx} out of range"
                )
            if code_off and code_off >= limit:
                anomalies.append(
                    f"class {cdef.class_idx}: code_off {code_off:#x} out of range"
                )
            methods.append(EncodedMethod(idx, flags, code_off))
        return methods

    cdef.static_fields = read_fields(static_n)
    cdef.instance_fields = read_fields(instance_n)
    cdef.direct_methods = read_methods(direct_n)
    cdef.virtual_methods = read_methods(virtual_n)


def parse_class_defs(data: bytes, header: DexHeader, anomalies: list[str]) -> list[ClassDef]:
    defs: list[ClassDef] = []
    section, size = header.class_defs_off, header.class_defs_size
    if size == 0:
        return defs
    if section + size * 32 > len(data):
        raise DexError("class_defs section exceeds file")

    for i in range(size):
        (
            class_idx,
            access_flags,
            superclass_idx,
            interfaces_off,
            source_file_idx,
            annotations_off,
            class_data_off,
            static_values_off,
        ) = struct.unpack_from("<IIIIIIII", data, section + i * 32)

        if class_idx >= header.type_ids_size:
            anomalies.append(f"class_defs[{i}] class_idx out of range")
        if superclass_idx != NO_INDEX and superclass_idx >= header.type_ids_size:
            anomalies.append(f"class_defs[{i}] superclass_idx out of range")
        if source_file_idx != NO_INDEX and source_file_idx >= header.string_ids_size:
            anomalies.append(f"class_defs[{i}] source_file_idx out of range")

        cdef = ClassDef(
            class_idx=class_idx,
            access_flags=access_flags,
            superclass_idx=superclass_idx,
            interfaces_off=interfaces_off,
            source_file_idx=source_file_idx,
            annotations_off=annotations_off,
            class_data_off=class_data_off,
            static_values_off=static_values_off,
        )

        if class_data_off:
            if class_data_off >= len(data):
                anomalies.append(f"class_defs[{i}] class_data_off out of range")
            else:
                try:
                    parse_class_data(data, class_data_off, header, cdef, anomalies)
                except DexError as exc:
                    anomalies.append(f"class_defs[{i}] class_data: {exc}")

        defs.append(cdef)

    return defs


# ============================================================================
# CODE ITEMS
# ============================================================================

def parse_code_item(data: bytes, offset: int, anomalies: list[str]) -> CodeItem:
    if offset + 16 > len(data):
        raise DexError(f"code_item at {offset:#x} truncated")

    registers_size, ins_size, outs_size, tries_size = struct.unpack_from(
        "<HHHH", data, offset
    )
    debug_info_off, insns_size = struct.unpack_from("<II", data, offset + 8)
    insns_off = offset + 16

    if insns_off + insns_size * 2 > len(data):
        raise DexError(
            f"code_item insns at {insns_off:#x} exceed file "
            f"({insns_size} code units)"
        )

    code = CodeItem(
        offset=offset,
        registers_size=registers_size,
        ins_size=ins_size,
        outs_size=outs_size,
        tries_size=tries_size,
        debug_info_off=debug_info_off,
        insns_size=insns_size,
        insns_off=insns_off,
    )

    if tries_size:
        pos = insns_off + insns_size * 2
        if insns_size % 2:
            pos += 2  # 4-byte alignment padding
        if pos + tries_size * 8 > len(data):
            anomalies.append(f"code_item at {offset:#x}: tries exceed file")
            return code

        for i in range(tries_size):
            start_addr, insn_count, handler_off = struct.unpack_from(
                "<IHH", data, pos + i * 8
            )
            code.tries.append(TryItem(start_addr, insn_count, handler_off))
        # Handler list is present but not decoded yet (debug info scope).

    return code


# ============================================================================
# INSTRUCTION WALKING
# ============================================================================

def _payload_width(data: bytes, pos: int, end: int) -> int:
    """Width (in code units) of a nop pseudo-payload starting at pos.

    Returns 0 when the word is not a usable pseudo-payload: unknown ident, or
    a size field that claims more code units than remain in the code item.
    Such a word must be treated as a plain 1-unit nop - trusting its size
    jumps past insns_size and silently drops the rest of the method, and it is
    how a crafted fill-array-data used to turn into a multi-gigabyte
    allocation while rendering (see convert_payload consumers in tools.smali).
    """
    if pos + 4 > end:
        return 0
    ident = struct.unpack_from("<H", data, pos)[0]
    size = struct.unpack_from("<H", data, pos + 2)[0]
    if ident == 0x0100:  # packed-switch-payload
        width = 4 + 2 * size
    elif ident == 0x0200:  # sparse-switch-payload
        width = 2 + 4 * size
    elif ident == 0x0300:  # fill-array-data-payload
        if pos + 8 > end:
            return 0
        element_width = struct.unpack_from("<H", data, pos + 2)[0]
        count = struct.unpack_from("<I", data, pos + 4)[0]
        if element_width == 0:
            return 0
        width = 4 + (element_width * count + 1) // 2
    else:
        return 0
    # A payload may never claim more code units than the code item holds.
    return width if pos + width * 2 <= end else 0


def instruction_width(data: bytes, pos: int, end: int) -> int:
    """Width in 16-bit code units of the instruction at byte offset pos.

    A nop-tagged word whose pseudo-payload does not fit the code item decodes
    as a 1-unit nop, keeping the linear walk aligned instead of jumping past
    the end of the method.
    """
    if pos + 2 > end:
        return 0
    unit = struct.unpack_from("<H", data, pos)[0]
    opcode = unit & 0xFF
    if opcode == 0x00:
        # nop or pseudo-payload (packed-switch/sparse-switch/fill-array-data)
        if unit in (0x0100, 0x0200, 0x0300):
            width = _payload_width(data, pos, end)
            return width if width > 0 else 1
        return 1
    return _OPCODE_WIDTHS.get(opcode, 1)


def walk_instructions_histogram(data: bytes, code: CodeItem) -> Counter:
    """Walk insns linearly; count opcodes. Misalignment is an anomaly signal
    for callers that check the final position against insns_size."""
    histogram: Counter = Counter()
    end = code.insns_off + code.insns_size * 2
    pos = code.insns_off
    while pos + 2 <= end:
        unit = struct.unpack_from("<H", data, pos)[0]
        op = unit & 0xFF
        width = instruction_width(data, pos, end)
        if width <= 0 or pos + width * 2 > end:
            histogram["<overrun>"] += 1
            break
        histogram[op] += 1
        pos += width * 2
    return histogram


def validate_instruction_walk(data: bytes, code: CodeItem,
                              bad_payloads: list[int] | None = None) -> int | None:
    """Return the end position if the walk lands exactly at insns end,
    otherwise None (caller records an anomaly).

    bad_payloads, when given, collects the insns-relative addresses of
    nop-tagged payload words whose declared size does not fit the code item
    (truncated or anti-disassembly filler). Those decode as 1-unit nops, so
    the walk still lands - without this the signal would be lost.
    """
    end = code.insns_off + code.insns_size * 2
    pos = code.insns_off
    while pos + 2 <= end:
        unit = struct.unpack_from("<H", data, pos)[0]
        if (bad_payloads is not None and unit in (0x0100, 0x0200, 0x0300)
                and _payload_width(data, pos, end) == 0):
            bad_payloads.append(pos - code.insns_off)
        width = instruction_width(data, pos, end)
        if width <= 0 or pos + width * 2 > end:
            return None
        pos += width * 2
    return pos if pos == end else None


# ============================================================================
# MAP LIST
# ============================================================================

def parse_map_list(data: bytes, header: DexHeader, anomalies: list[str]) -> list[MapItem]:
    items: list[MapItem] = []
    map_off = header.map_off
    if not map_off or map_off + 4 > len(data):
        anomalies.append("map_off missing or out of range")
        return items

    size = struct.unpack_from("<I", data, map_off)[0]
    if map_off + 4 + size * 12 > len(data):
        anomalies.append("map_list exceeds file")
        return items

    for i in range(size):
        item_type, _unused, count, offset = struct.unpack_from(
            "<HHII", data, map_off + 4 + i * 12
        )
        items.append(MapItem(item_type, count, offset))

    # Cross-check header counts against the map (spec redundancy check).
    expected = {
        TYPE_STRING_ID_ITEM: header.string_ids_size,
        TYPE_TYPE_ID_ITEM: header.type_ids_size,
        TYPE_PROTO_ID_ITEM: header.proto_ids_size,
        TYPE_FIELD_ID_ITEM: header.field_ids_size,
        TYPE_METHOD_ID_ITEM: header.method_ids_size,
        TYPE_CLASS_DEF_ITEM: header.class_defs_size,
    }
    map_counts = {m.item_type: m.size for m in items}
    for item_type, expected_count in expected.items():
        if item_type in map_counts and map_counts[item_type] != expected_count:
            name = MAP_TYPE_NAMES.get(item_type, hex(item_type))
            anomalies.append(
                f"map_list {name} count {map_counts[item_type]} != header "
                f"count {expected_count}"
            )

    # Map entries must be ordered by offset.
    offsets = [m.offset for m in items]
    if offsets != sorted(offsets):
        anomalies.append("map_list entries are not ordered by offset")

    return items


# ============================================================================
# TOP-LEVEL PARSER
# ============================================================================

def parse_dex_bytes(data: bytes) -> DexFile:
    header = parse_dex_header(data)
    anomalies: list[str] = []

    checksum_ok, signature_ok = verify_integrity(data, header, anomalies)

    dex = DexFile(data=data, header=header, anomalies=anomalies,
                  checksum_ok=checksum_ok, signature_ok=signature_ok)

    dex.string_offsets = parse_string_ids(data, header, anomalies)
    dex.type_ids = parse_type_ids(data, header, anomalies)
    dex.proto_ids = parse_proto_ids(data, header, anomalies)
    dex.field_ids = parse_field_ids(data, header, anomalies)
    dex.method_ids = parse_method_ids(data, header, anomalies)
    dex.class_defs = parse_class_defs(data, header, anomalies)
    dex.map_items = parse_map_list(data, header, anomalies)

    # Self-validating instruction walk across every method.
    misaligned = 0
    oversized_payloads = 0
    for cdef, method, code in dex.iter_code_items():
        bad: list[int] = []
        if validate_instruction_walk(data, code, bad) is None:
            misaligned += 1
            if misaligned <= 5:
                anomalies.append(
                    f"method {dex.method_str(method.method_idx)}: instruction "
                    "walk does not land on insns_size (malformed/packed code)"
                )
        if bad:
            oversized_payloads += 1
            if oversized_payloads <= 5:
                anomalies.append(
                    f"method {dex.method_str(method.method_idx)}: {len(bad)} "
                    "nop-tagged payload word(s) declare more data than the "
                    "code item holds (truncated/anti-disassembly filler)"
                )
    if misaligned > 5:
        anomalies.append(f"... {misaligned - 5} more misaligned methods")
    if misaligned:
        anomalies.append(f"total misaligned methods: {misaligned}")
    if oversized_payloads > 5:
        anomalies.append(
            f"... {oversized_payloads - 5} more methods with oversized "
            "payload declarations"
        )
    if oversized_payloads:
        anomalies.append(
            f"total oversized payload methods: {oversized_payloads}"
        )

    return dex


# ============================================================================
# APK DISCOVERY (metadata-only, bounded reads)
# ============================================================================

_DEX_NAME_RE = re.compile(r"^(?:classes\d*\.dex)$", re.IGNORECASE)


def discover_dex_entries(apk_path: Path) -> list[str]:
    from tools.zip_audit import parse_zip

    entries = parse_zip(apk_path)
    names = [e.name for e in entries if _DEX_NAME_RE.match(e.name.split("/")[-1])]
    return sorted(names)


def read_dex_entry(apk_path: Path, entry_name: str, max_bytes: int = 256 * 1024 * 1024) -> bytes:
    try:
        return read_entry_bytes(apk_path, entry_name, max_uncompressed=max_bytes)
    except (ZipEntryError, ZipReadLimitError) as exc:
        raise DexError(f"{entry_name} rejected: {exc}") from exc


class FakeDexEntry(ValueError):
    """Raised when a classes*.dex entry does not contain DEX data at all
    (null padding, garbage, or a tarpit bomb)."""


def sniff_dex_magic(apk_path: Path, entry_name: str, sniff_bytes: int = 64 * 1024) -> bool:
    """Boundedly read the head of an entry and check for the DEX magic.

    prefix_only=True: read at most sniff_bytes of OUTPUT regardless of the
    declared size, and skip the ratio pre-check. A bomb can expand into at
    most sniff_bytes before the reader stops, so this is safe by construction.
    """
    head = read_entry_bytes(
        apk_path,
        entry_name,
        max_uncompressed=sniff_bytes,
        max_ratio=None,
        verify_crc=False,  # output may be truncated by the cap
        prefix_only=True,
    )
    return head[:4] == DEX_MAGIC_PREFIX


def parse_dex_from_apk(apk_path: Path, entry_name: str = "classes.dex") -> DexFile:
    return parse_dex_bytes(read_dex_entry(apk_path, entry_name))


def parse_all_dex(apk_path: Path) -> list[DexFile]:
    return [parse_dex_from_apk(apk_path, name) for name in discover_dex_entries(apk_path)]


# ============================================================================
# REPORTING
# ============================================================================

_ACCESS_FLAG_NAMES = [
    (ACC_PUBLIC, "public"), (ACC_PRIVATE, "private"), (ACC_PROTECTED, "protected"),
    (ACC_STATIC, "static"), (ACC_FINAL, "final"), (ACC_SYNCHRONIZED, "synchronized"),
    (ACC_VOLATILE, "volatile"), (ACC_BRIDGE, "bridge"), (ACC_TRANSIENT, "transient"),
    (ACC_VARARGS, "varargs"), (ACC_NATIVE, "native"), (ACC_INTERFACE, "interface"),
    (ACC_ABSTRACT, "abstract"), (ACC_STRICT, "strict"), (ACC_SYNTHETIC, "synthetic"),
    (ACC_ANNOTATION, "annotation"), (ACC_ENUM, "enum"),
    (ACC_CONSTRUCTOR, "constructor"), (ACC_DECLARED_SYNCHRONIZED, "declared-synchronized"),
]


def flags_to_str(flags: int) -> str:
    return " ".join(name for bit, name in _ACCESS_FLAG_NAMES if flags & bit)


def common_opcode_names() -> dict[int, str]:
    """Human-readable names for common opcodes (reporting only)."""
    names = {
        0x00: "nop", 0x01: "move", 0x0a: "move-result",
        0x0b: "move-result-wide", 0x0c: "move-result-object",
        0x0d: "move-exception", 0x0e: "return-void", 0x0f: "return",
        0x11: "return-object", 0x12: "const/4", 0x13: "const/16", 0x14: "const",
        0x1a: "const-string", 0x1b: "const-string/jumbo", 0x1c: "const-class",
        0x1f: "check-cast", 0x20: "instance-of", 0x22: "new-instance",
        0x23: "new-array", 0x24: "filled-new-array", 0x26: "fill-array-data",
        0x27: "throw", 0x28: "goto", 0x2b: "packed-switch", 0x2c: "sparse-switch",
    }
    for _i, _n in enumerate(
        "if-eqz if-nez if-ltz if-gez if-gtz if-lez".split()
    ):
        names[0x38 + _i] = _n
    for i, name in enumerate(
        "iget iget-wide iget-object iget-boolean iget-byte iget-char iget-short "
        "iput iput-wide iput-object iput-boolean iput-byte iput-char iput-short".split()
    ):
        names[0x52 + i] = name
    for i, name in enumerate(
        "sget sget-wide sget-object sget-boolean sget-byte sget-char sget-short "
        "sput sput-wide sput-object sput-boolean sput-byte sput-char sput-short".split()
    ):
        names[0x60 + i] = name
    for i, name in enumerate(
        "invoke-virtual invoke-super invoke-direct invoke-static invoke-interface".split()
    ):
        names[0x6e + i] = name
    for i, name in enumerate(
        "invoke-virtual/range invoke-super/range invoke-direct/range "
        "invoke-static/range invoke-interface/range".split()
    ):
        names[0x74 + i] = name
    return names


_OPCODE_NAMES = common_opcode_names()


def print_dex_report(dex: DexFile, max_classes: int = 15) -> None:
    header = dex.header
    print("=" * 70)
    print("DEX ANALYSIS")
    print("=" * 70)
    print(f"Magic        : {header.magic!r} (version {header.version})")
    print(f"File Size    : {header.file_size:,} (buffer {len(dex.data):,})")
    print(f"Checksum     : {'OK' if dex.checksum_ok else 'MISMATCH'}")
    print(f"SHA-1        : {'OK' if dex.signature_ok else 'MISMATCH'}")
    print(f"Strings      : {len(dex.string_offsets):,}")
    print(f"Types        : {len(dex.type_ids):,}")
    print(f"Protos       : {len(dex.proto_ids):,}")
    print(f"Fields       : {len(dex.field_ids):,}")
    print(f"Methods      : {len(dex.method_ids):,}")
    print(f"Classes      : {len(dex.class_defs):,}")
    print(f"Map Items    : {len(dex.map_items)}")

    print()
    print("-" * 70)
    print("TOP OPCODES (self-validating walk)")
    print("-" * 70)
    histogram = dex.opcode_histogram()
    for op, count in histogram.most_common(12):
        label = _OPCODE_NAMES.get(op, f"op_{op:#04x}") if isinstance(op, int) else op
        print(f"  {label:22} {count:8,}")

    print()
    print("-" * 70)
    print(f"FIRST {max_classes} CLASSES")
    print("-" * 70)
    for cdef in dex.class_defs[:max_classes]:
        name = dex.type_desc(cdef.class_idx) or "?"
        sup = dex.type_desc(cdef.superclass_idx) if cdef.superclass_idx != NO_INDEX else None
        flags = flags_to_str(cdef.access_flags)
        n_methods = len(cdef.direct_methods) + len(cdef.virtual_methods)
        src = dex.string(cdef.source_file_idx) if cdef.source_file_idx != NO_INDEX else None
        print(f"  {name}")
        print(f"    super: {sup or '<none>'}  flags: {flags}")
        print(f"    methods: {n_methods}  source: {src or '<unknown>'}")

    if dex.anomalies:
        print()
        print("=" * 70)
        print(f"DEX ANOMALIES ({len(dex.anomalies)})")
        print("=" * 70)
        for i, anomaly in enumerate(dex.anomalies, 1):
            print(f"[{i}] {anomaly}")


# ============================================================================
# STANDALONE
# ============================================================================

if __name__ == "__main__":
    APK_PATH = Path(__file__).parent.parent / "apks" / "sample.apk"

    names = discover_dex_entries(APK_PATH)
    print(f"DEX entries in APK: {names}")

    dex = parse_dex_from_apk(APK_PATH, names[0] if names else "classes.dex")
    print_dex_report(dex)

    print()
    print("STATS")
    for key, value in dex.stats().items():
        print(f"  {key}: {value}")

    print()
    refs = dex.api_references()
    print(f"Distinct API references: {len(refs)}")
    for ref, count in refs.most_common(10):
        print(f"  {count:5}x  {ref}")
