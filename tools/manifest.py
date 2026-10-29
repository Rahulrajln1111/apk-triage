"""
Binary Android XML (AXML) parser for AndroidManifest.xml.

This module intentionally parses the binary XML format directly instead of
depending on apktool/aapt. It is designed for APK forensics:

- validates chunk boundaries before reading
- supports UTF-8 and UTF-16 string pools
- supports the standard AXML chunk types
- resolves attributes through the string pool directly
- uses the optional resource map only for resource IDs
- tracks namespace scopes and prefixes
- decodes the standard Res_value data types
- tolerates unknown XML chunks instead of assuming they are semantic nodes

The on-disk structures and constants follow AOSP's androidfw/ResourceTypes.h.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable
import math
import re
import struct
import zipfile


# ============================================================================
# Top-level / chunk constants
# ============================================================================

RES_NULL_TYPE = 0x0000
RES_STRING_POOL_TYPE = 0x0001
RES_TABLE_TYPE = 0x0002
RES_XML_TYPE = 0x0003

# Binary XML node chunks.
RES_XML_START_NAMESPACE = 0x0100
RES_XML_END_NAMESPACE = 0x0101
RES_XML_START_ELEMENT = 0x0102
RES_XML_END_ELEMENT = 0x0103
RES_XML_CDATA = 0x0104

# Optional string -> resource-ID map.
RES_XML_RESOURCE_MAP = 0x0180

# Other resource-table chunks. They are not normal AXML chunks, but can occur
# in malformed/obfuscated input and should be recognized as "unknown" here.
RES_TABLE_PACKAGE_TYPE = 0x0200


# ============================================================================
# String-pool constants
# ============================================================================

SORTED_FLAG = 1 << 0
UTF8_FLAG = 1 << 8


# ============================================================================
# Res_value type constants
# ============================================================================

TYPE_NULL = 0x00
TYPE_REFERENCE = 0x01
TYPE_ATTRIBUTE = 0x02
TYPE_STRING = 0x03
TYPE_FLOAT = 0x04
TYPE_DIMENSION = 0x05
TYPE_FRACTION = 0x06
TYPE_DYNAMIC_REFERENCE = 0x07
TYPE_DYNAMIC_ATTRIBUTE = 0x08

TYPE_FIRST_INT = 0x10
TYPE_INT_DEC = 0x10
TYPE_INT_HEX = 0x11
TYPE_INT_BOOLEAN = 0x12

TYPE_FIRST_COLOR_INT = 0x1C
TYPE_INT_COLOR_ARGB8 = 0x1C
TYPE_INT_COLOR_RGB8 = 0x1D
TYPE_INT_COLOR_ARGB4 = 0x1E
TYPE_INT_COLOR_RGB4 = 0x1F
TYPE_LAST_INT = 0x1F


# Complex-value constants.
COMPLEX_UNIT_SHIFT = 0
COMPLEX_UNIT_MASK = 0xF

COMPLEX_UNIT_PX = 0
COMPLEX_UNIT_DIP = 1
COMPLEX_UNIT_SP = 2
COMPLEX_UNIT_PT = 3
COMPLEX_UNIT_IN = 4
COMPLEX_UNIT_MM = 5

COMPLEX_UNIT_FRACTION = 0
COMPLEX_UNIT_FRACTION_PARENT = 1

COMPLEX_RADIX_SHIFT = 4
COMPLEX_RADIX_MASK = 0x3

COMPLEX_RADIX_23P0 = 0
COMPLEX_RADIX_16P7 = 1
COMPLEX_RADIX_8P15 = 2
COMPLEX_RADIX_0P23 = 3

COMPLEX_MANTISSA_SHIFT = 8
COMPLEX_MANTISSA_MASK = 0xFFFFFF


# A null string-pool reference is encoded as UINT32_MAX.
NO_NAMESPACE = -1


# ============================================================================
# Exceptions
# ============================================================================

class BinaryXmlError(ValueError):
    """Raised when the binary XML structure is malformed or out of bounds."""


# ============================================================================
# Data classes
# ============================================================================

@dataclass
class XmlHeader:
    chunk_type: int
    header_size: int
    file_size: int


@dataclass
class StringPoolHeader:
    chunk_type: int
    header_size: int
    chunk_size: int
    string_count: int
    style_count: int
    flags: int
    strings_start: int
    styles_start: int
    chunk_start: int = 8


@dataclass
class StringPool:
    header: StringPoolHeader
    offsets: list[int]
    strings: list[str]


@dataclass
class XmlNodeHeader:
    chunk_type: int
    header_size: int
    chunk_size: int
    line_number: int
    comment_index: int


@dataclass
class StartElement:
    line_number: int
    namespace_idx: int
    name_idx: int
    attribute_start: int
    attribute_size: int
    attribute_count: int
    id_index: int
    class_index: int
    style_index: int


@dataclass
class EndElement:
    line_number: int
    namespace_idx: int
    name_idx: int


@dataclass
class ResourceMap:
    ids: list[int]


@dataclass
class XmlAttribute:
    namespace_idx: int
    name_idx: int
    raw_value_idx: int
    data_type: int
    data: int

    # Resolved values.
    name: str | None = None
    value: str | None = None

    # Optional metadata from the resource map / namespace scope.
    resource_id: int | None = None
    prefix: str | None = None
    namespace_uri: str | None = None


@dataclass
class XmlNamespace:
    line_number: int
    prefix_idx: int
    uri_idx: int
    prefix: str | None = None
    uri: str | None = None


@dataclass
class XmlElement:
    tag: str
    line: int
    attributes: list[XmlAttribute]
    children: list["XmlElement"]

    namespace_idx: int = NO_NAMESPACE
    namespace_uri: str | None = None
    prefix: str | None = None
    text: list[str] = field(default_factory=list)


# ============================================================================
# Small binary helpers
# ============================================================================

def _require_range(data: bytes, offset: int, size: int, what: str = "data") -> None:
    if offset < 0 or size < 0 or offset + size > len(data):
        raise BinaryXmlError(
            f"{what} out of bounds: offset={offset}, size={size}, file={len(data)}"
        )


def _u16(data: bytes, offset: int, what: str = "uint16") -> int:
    _require_range(data, offset, 2, what)
    return struct.unpack_from("<H", data, offset)[0]


def _u32(data: bytes, offset: int, what: str = "uint32") -> int:
    _require_range(data, offset, 4, what)
    return struct.unpack_from("<I", data, offset)[0]


def decode_ref(value: int) -> int:
    """Convert UINT32_MAX string-pool sentinel to -1."""
    return NO_NAMESPACE if value == 0xFFFFFFFF else value


def _safe_pool_string(pool: StringPool, index: int) -> str | None:
    if index < 0 or index >= len(pool.strings):
        return None
    return pool.strings[index]


# ============================================================================
# APK input
# ============================================================================

def read_manifest(apk_path: Path) -> bytes:
    """Read AndroidManifest.xml through the bounded, zip-bomb-safe reader."""
    from tools.zip_audit import ZipEntryError, ZipReadLimitError, read_entry_bytes

    try:
        return read_entry_bytes(
            apk_path,
            "AndroidManifest.xml",
            max_uncompressed=8 * 1024 * 1024,
            max_ratio=200.0,
        )
    except ZipEntryError as exc:
        raise BinaryXmlError(f"AndroidManifest.xml: {exc}") from exc
    except ZipReadLimitError as exc:
        # A manifest larger than 8 MB is not a manifest anymore; either the
        # APK is hostile or pathologically crafted. Either way it is evidence.
        raise BinaryXmlError(f"AndroidManifest.xml rejected: {exc}") from exc


# ============================================================================
# Top-level XML header
# ============================================================================

def parse_xml_header(data: bytes) -> XmlHeader:
    _require_range(data, 0, 8, "XML header")

    chunk_type, header_size, file_size = struct.unpack_from("<HHI", data, 0)

    if chunk_type != RES_XML_TYPE:
        raise BinaryXmlError(
            f"Not an AXML document: type=0x{chunk_type:04X}, "
            f"expected=0x{RES_XML_TYPE:04X}"
        )

    if header_size < 8:
        raise BinaryXmlError(f"Invalid XML header size: {header_size}")

    if file_size < header_size:
        raise BinaryXmlError(
            f"XML file size {file_size} is smaller than header size {header_size}"
        )

    if file_size > len(data):
        raise BinaryXmlError(
            f"XML file size {file_size} exceeds available bytes {len(data)}"
        )

    return XmlHeader(
        chunk_type=chunk_type,
        header_size=header_size,
        file_size=file_size,
    )


# ============================================================================
# Chunk walker
# ============================================================================

def walk_chunks(
    data: bytes,
    start: int,
    file_size: int,
) -> Iterable[tuple[int, int, int, int]]:
    """
    Yield (offset, type, header_size, chunk_size).

    Unknown chunk types are intentionally yielded rather than rejected.
    AOSP's XML parser can skip unknown XML blocks, which is important for
    forensic/tolerant parsing.
    """
    if start < 0 or file_size < start:
        raise BinaryXmlError("Invalid chunk walk range")

    file_size = min(file_size, len(data))
    offset = start

    while offset < file_size:
        if file_size - offset < 8:
            raise BinaryXmlError(
                f"Truncated chunk header at offset {offset}"
            )

        chunk_type, header_size, chunk_size = struct.unpack_from(
            "<HHI", data, offset
        )

        if header_size < 8:
            raise BinaryXmlError(
                f"Invalid chunk header size {header_size} "
                f"at offset {offset}"
            )

        if chunk_size < header_size:
            raise BinaryXmlError(
                f"Chunk size {chunk_size} < header size {header_size} "
                f"at offset {offset}"
            )

        end = offset + chunk_size
        if end > file_size:
            raise BinaryXmlError(
                f"Chunk at offset {offset} exceeds XML file: "
                f"end={end}, file_size={file_size}"
            )

        yield offset, chunk_type, header_size, chunk_size
        offset = end


# ============================================================================
# String pool
# ============================================================================

def parse_string_pool_header(
    data: bytes,
    offset: int = 8,
) -> StringPoolHeader:
    _require_range(data, offset, 28, "string-pool header")

    fields = struct.unpack_from("<HHIIIIII", data, offset)

    chunk_type, header_size, chunk_size = fields[:3]
    if chunk_type != RES_STRING_POOL_TYPE:
        raise BinaryXmlError(
            f"Expected string pool at 0x{offset:X}, "
            f"got type=0x{chunk_type:04X}"
        )

    if header_size < 28:
        raise BinaryXmlError(
            f"String-pool header too small: {header_size}"
        )

    if chunk_size < header_size:
        raise BinaryXmlError(
            f"String-pool chunk too small: {chunk_size}"
        )

    if offset + chunk_size > len(data):
        raise BinaryXmlError(
            "String-pool chunk extends beyond input"
        )

    return StringPoolHeader(
        chunk_type=chunk_type,
        header_size=header_size,
        chunk_size=chunk_size,
        string_count=fields[3],
        style_count=fields[4],
        flags=fields[5],
        strings_start=fields[6],
        styles_start=fields[7],
        chunk_start=offset,
    )


def parse_string_offsets(
    data: bytes,
    header: StringPoolHeader,
) -> list[int]:
    start = header.chunk_start + header.header_size
    bytes_needed = header.string_count * 4

    _require_range(
        data,
        start,
        bytes_needed,
        "string-pool index array",
    )

    return [
        _u32(
            data,
            start + i * 4,
            f"string offset #{i}",
        )
        for i in range(header.string_count)
    ]


def read_utf16_length(
    data: bytes,
    offset: int,
) -> tuple[int, int]:
    """
    Android UTF-16 string-pool length encoding.

    Returns:
        (UTF-16 code-unit count, bytes used by the encoded length)
    """
    first = _u16(data, offset, "UTF-16 string length")

    if (first & 0x8000) == 0:
        return first, 2

    second = _u16(
        data,
        offset + 2,
        "extended UTF-16 string length",
    )
    length = ((first & 0x7FFF) << 16) | second
    return length, 4


def read_utf8_length(
    data: bytes,
    offset: int,
) -> tuple[int, int]:
    """
    Android UTF-8 string-pool length encoding.

    The pool stores both:
      1. UTF-16 code-unit length
      2. UTF-8 byte length
    """
    _require_range(data, offset, 1, "UTF-8 string length")

    first = data[offset]
    if (first & 0x80) == 0:
        return first, 1

    _require_range(data, offset + 1, 1, "extended UTF-8 length")
    second = data[offset + 1]
    return ((first & 0x7F) << 8) | second, 2


def decode_utf16_string(
    data: bytes,
    absolute_offset: int,
) -> str:
    length, header_bytes = read_utf16_length(data, absolute_offset)

    payload_offset = absolute_offset + header_bytes
    payload_size = length * 2

    _require_range(
        data,
        payload_offset,
        payload_size,
        "UTF-16 string payload",
    )

    raw = data[payload_offset:payload_offset + payload_size]

    # The AXML string pool stores UTF-16LE. Decode errors are replaced instead
    # of allowing one malformed string to terminate an APK-wide analysis.
    return raw.decode("utf-16le", errors="replace")


def decode_utf8_string(
    data: bytes,
    absolute_offset: int,
) -> str:
    _utf16_length, first_bytes = read_utf8_length(
        data,
        absolute_offset,
    )

    byte_length, second_bytes = read_utf8_length(
        data,
        absolute_offset + first_bytes,
    )

    payload_offset = absolute_offset + first_bytes + second_bytes

    _require_range(
        data,
        payload_offset,
        byte_length,
        "UTF-8 string payload",
    )

    raw = data[payload_offset:payload_offset + byte_length]

    return raw.decode("utf-8", errors="replace")


def parse_string_pool(
    data: bytes,
    header: StringPoolHeader,
) -> StringPool:
    offsets = parse_string_offsets(data, header)
    strings: list[str] = []

    chunk_end = header.chunk_start + header.chunk_size
    string_data_start = header.chunk_start + header.strings_start

    if not (
        header.header_size <= header.strings_start < header.chunk_size
    ):
        raise BinaryXmlError(
            "String-pool strings_start is outside its chunk"
        )

    if header.style_count:
        if header.styles_start <= header.strings_start:
            raise BinaryXmlError(
                "styles_start must be after strings_start when styles exist"
            )
        string_data_end = header.chunk_start + header.styles_start
    else:
        string_data_end = chunk_end

    if string_data_end > chunk_end or string_data_start > string_data_end:
        raise BinaryXmlError("Invalid string-pool string-data range")

    for index, relative_offset in enumerate(offsets):
        absolute = string_data_start + relative_offset

        if absolute < string_data_start or absolute >= string_data_end:
            raise BinaryXmlError(
                f"String #{index} points outside the string-data area: "
                f"relative=0x{relative_offset:X}"
            )

        if header.flags & UTF8_FLAG:
            value = decode_utf8_string(data, absolute)
        else:
            value = decode_utf16_string(data, absolute)

        strings.append(value)

    return StringPool(
        header=header,
        offsets=offsets,
        strings=strings,
    )


# ============================================================================
# XML node header / namespace / element structures
# ============================================================================

def parse_node_header(
    data: bytes,
    offset: int,
) -> XmlNodeHeader:
    _require_range(data, offset, 16, "XML node header")

    chunk_type, header_size, chunk_size, line, comment = struct.unpack_from(
        "<HHIII",
        data,
        offset,
    )

    if header_size < 16:
        raise BinaryXmlError(
            f"XML node header is too small: {header_size}"
        )

    if chunk_size < header_size:
        raise BinaryXmlError(
            f"XML node size {chunk_size} < header size {header_size}"
        )

    return XmlNodeHeader(
        chunk_type=chunk_type,
        header_size=header_size,
        chunk_size=chunk_size,
        line_number=line,
        comment_index=decode_ref(comment),
    )


def parse_namespace(
    data: bytes,
    offset: int,
) -> XmlNamespace:
    node = parse_node_header(data, offset)

    if node.chunk_type not in (
        RES_XML_START_NAMESPACE,
        RES_XML_END_NAMESPACE,
    ):
        raise BinaryXmlError(
            f"Not a namespace chunk at 0x{offset:X}"
        )

    if node.chunk_size - node.header_size < 8:
        raise BinaryXmlError(
            f"Namespace chunk at 0x{offset:X} is too small"
        )

    prefix, uri = struct.unpack_from(
        "<II",
        data,
        offset + node.header_size,
    )

    return XmlNamespace(
        line_number=node.line_number,
        prefix_idx=decode_ref(prefix),
        uri_idx=decode_ref(uri),
    )


def resolve_namespace(
    ns: XmlNamespace,
    pool: StringPool,
) -> XmlNamespace:
    ns.prefix = _safe_pool_string(pool, ns.prefix_idx)
    ns.uri = _safe_pool_string(pool, ns.uri_idx)
    return ns


def parse_start_element(
    data: bytes,
    offset: int,
) -> StartElement:
    node = parse_node_header(data, offset)

    if node.chunk_type != RES_XML_START_ELEMENT:
        raise BinaryXmlError(
            f"Not a START_ELEMENT chunk at 0x{offset:X}"
        )

    ext_size = node.chunk_size - node.header_size
    if ext_size < 20:
        raise BinaryXmlError(
            f"START_ELEMENT extension is too small: {ext_size}"
        )

    values = struct.unpack_from(
        "<IIHHHHHH",
        data,
        offset + node.header_size,
    )

    start = StartElement(
        line_number=node.line_number,
        namespace_idx=decode_ref(values[0]),
        name_idx=values[1],
        attribute_start=values[2],
        attribute_size=values[3],
        attribute_count=values[4],
        id_index=values[5],
        class_index=values[6],
        style_index=values[7],
    )

    if start.attribute_count and start.attribute_size < 20:
        raise BinaryXmlError(
            f"Invalid attributeSize={start.attribute_size}; "
            "a standard attribute record is 20 bytes"
        )

    # AOSP defines attributeStart as an offset from the beginning of the
    # ResXMLTree_attrExt structure, which begins after the node header.
    attrs_rel_end = (
        start.attribute_start
        + start.attribute_count * start.attribute_size
    )

    if attrs_rel_end > ext_size:
        raise BinaryXmlError(
            "Attribute array exceeds START_ELEMENT chunk"
        )

    return start


def parse_end_element(
    data: bytes,
    offset: int,
) -> EndElement:
    node = parse_node_header(data, offset)

    if node.chunk_type != RES_XML_END_ELEMENT:
        raise BinaryXmlError(
            f"Not an END_ELEMENT chunk at 0x{offset:X}"
        )

    if node.chunk_size - node.header_size < 8:
        raise BinaryXmlError(
            f"END_ELEMENT chunk at 0x{offset:X} is too small"
        )

    ns, name = struct.unpack_from(
        "<II",
        data,
        offset + node.header_size,
    )

    return EndElement(
        line_number=node.line_number,
        namespace_idx=decode_ref(ns),
        name_idx=name,
    )


def parse_cdata(
    data: bytes,
    offset: int,
) -> tuple[int, int, int, int]:
    """
    Parse a CDATA node.

    Returns:
        (line_number, comment_index, data_string_index, data_type/data_raw)
    """
    node = parse_node_header(data, offset)

    if node.chunk_type != RES_XML_CDATA:
        raise BinaryXmlError(
            f"Not a CDATA chunk at 0x{offset:X}"
        )

    ext_size = node.chunk_size - node.header_size
    if ext_size < 12:
        raise BinaryXmlError(
            f"CDATA extension is too small: {ext_size}"
        )

    string_idx = decode_ref(
        _u32(data, offset + node.header_size, "CDATA string index")
    )

    typed_type = data[offset + node.header_size + 7]
    typed_data = _u32(
        data,
        offset + node.header_size + 8,
        "CDATA typed data",
    )

    return (
        node.line_number,
        string_idx,
        typed_type,
        typed_data,
    )


# ============================================================================
# Resource map
# ============================================================================

def parse_resource_map(
    data: bytes,
    offset: int,
    chunk_size: int,
    header_size: int = 8,
) -> ResourceMap:
    if chunk_size < header_size:
        raise BinaryXmlError("Invalid resource-map chunk size")

    payload = chunk_size - header_size
    if payload % 4 != 0:
        raise BinaryXmlError(
            f"Resource-map payload is not a multiple of 4: {payload}"
        )

    count = payload // 4
    _require_range(
        data,
        offset + header_size,
        payload,
        "resource-map data",
    )

    ids = [
        _u32(
            data,
            offset + header_size + i * 4,
            f"resource-map ID #{i}",
        )
        for i in range(count)
    ]

    return ResourceMap(ids=ids)


# ============================================================================
# Attributes
# ============================================================================

def attribute_array_offset(
    chunk_offset: int,
    element: StartElement,
    node_header_size: int = 16,
) -> int:
    # The attributeStart field is relative to ResXMLTree_attrExt, which
    # begins immediately after the XML node header.
    return chunk_offset + node_header_size + element.attribute_start


def parse_attribute(
    data: bytes,
    offset: int,
) -> XmlAttribute:
    _require_range(data, offset, 20, "ResXMLTree_attribute")

    namespace_idx, name_idx, raw_value_idx = struct.unpack_from(
        "<III",
        data,
        offset,
    )

    size, res0, data_type = struct.unpack_from(
        "<HBB",
        data,
        offset + 12,
    )

    value = _u32(
        data,
        offset + 16,
        "attribute typed value",
    )

    if size < 8:
        raise BinaryXmlError(
            f"Res_value size is invalid: {size}"
        )

    return XmlAttribute(
        namespace_idx=decode_ref(namespace_idx),
        name_idx=name_idx,
        raw_value_idx=decode_ref(raw_value_idx),
        data_type=data_type,
        data=value,
    )


def parse_attributes(
    data: bytes,
    chunk_offset: int,
    element: StartElement,
) -> list[XmlAttribute]:
    node = parse_node_header(data, chunk_offset)

    start = attribute_array_offset(
        chunk_offset,
        element,
        node.header_size,
    )

    attrs: list[XmlAttribute] = []

    for i in range(element.attribute_count):
        off = start + i * element.attribute_size

        # Read the complete standard 20-byte structure first.
        _require_range(
            data,
            off,
            20,
            f"attribute #{i}",
        )

        attrs.append(
            parse_attribute(
                data,
                off,
            )
        )

    return attrs


# ============================================================================
# Typed-value decoding
# ============================================================================

def _decode_complex_float(data: int) -> float:
    radix = (data >> COMPLEX_RADIX_SHIFT) & COMPLEX_RADIX_MASK

    mantissa = (
        data >> COMPLEX_MANTISSA_SHIFT
    ) & COMPLEX_MANTISSA_MASK

    # Sign-extend 23-bit mantissa.
    if mantissa & 0x800000:
        mantissa -= 0x1000000

    divisors = {
        COMPLEX_RADIX_23P0: 1.0,
        COMPLEX_RADIX_16P7: 1 << 7,
        COMPLEX_RADIX_8P15: 1 << 15,
        COMPLEX_RADIX_0P23: 1 << 23,
    }

    return mantissa / divisors.get(radix, 1.0)


def _format_float(value: float) -> str:
    if not math.isfinite(value):
        return repr(value)
    return f"{value:.7g}"


def _decode_typed_value(
    data_type: int,
    data: int,
    pool: StringPool,
) -> str | None:
    # No data / explicit empty.
    if data_type == TYPE_NULL:
        if data == 0:
            return None
        return ""

    # Resource references.
    if data_type == TYPE_REFERENCE:
        return f"@0x{data:08x}"

    if data_type == TYPE_DYNAMIC_REFERENCE:
        return f"@*0x{data:08x}"

    # Attribute references.
    if data_type == TYPE_ATTRIBUTE:
        return f"?0x{data:08x}"

    if data_type == TYPE_DYNAMIC_ATTRIBUTE:
        return f"?*0x{data:08x}"

    # String index into the containing XML string pool.
    if data_type == TYPE_STRING:
        value = _safe_pool_string(pool, data)
        return value if value is not None else f"<invalid-string#{data}>"

    # IEEE-754 float.
    if data_type == TYPE_FLOAT:
        value = struct.unpack("<f", struct.pack("<I", data))[0]
        return _format_float(value)

    # Complex dimension.
    if data_type == TYPE_DIMENSION:
        value = _decode_complex_float(data)
        unit = data & COMPLEX_UNIT_MASK

        units = {
            COMPLEX_UNIT_PX: "px",
            COMPLEX_UNIT_DIP: "dp",
            COMPLEX_UNIT_SP: "sp",
            COMPLEX_UNIT_PT: "pt",
            COMPLEX_UNIT_IN: "in",
            COMPLEX_UNIT_MM: "mm",
        }

        suffix = units.get(unit, f"unit{unit}")
        return f"{_format_float(value)}{suffix}"

    # Complex fraction.
    if data_type == TYPE_FRACTION:
        value = _decode_complex_float(data)
        unit = data & COMPLEX_UNIT_MASK

        if unit == COMPLEX_UNIT_FRACTION:
            return f"{_format_float(value * 100.0)}%"
        if unit == COMPLEX_UNIT_FRACTION_PARENT:
            return f"{_format_float(value * 100.0)}%p"

        return (
            f"{_format_float(value * 100.0)}%"
            f"(unit{unit})"
        )

    # Decimal integer.
    if data_type == TYPE_INT_DEC:
        # Typed Android integers are uint32 on disk; manifest values are
        # normally non-negative. Keep the exact unsigned data for fidelity.
        return str(data)

    # Hex integer.
    if data_type == TYPE_INT_HEX:
        return f"0x{data:08x}"

    # Boolean integer.
    if data_type == TYPE_INT_BOOLEAN:
        return "true" if data != 0 else "false"

    # Color integers.
    if TYPE_FIRST_COLOR_INT <= data_type <= TYPE_LAST_INT:
        if data_type == TYPE_INT_COLOR_ARGB8:
            return f"#{data:08x}"
        if data_type == TYPE_INT_COLOR_RGB8:
            return f"#{data & 0xFFFFFF:06x}"
        if data_type == TYPE_INT_COLOR_ARGB4:
            return f"#{data & 0xFFFF:04x}"
        if data_type == TYPE_INT_COLOR_RGB4:
            return f"#{data & 0x0FFF:03x}"

    # Unknown future/vendor type: do not guess its semantic meaning.
    return f"<type=0x{data_type:02x},data=0x{data:08x}>"


def resolve_attributes(
    attrs: list[XmlAttribute],
    pool: StringPool,
    resource_map: ResourceMap,
    namespace_scopes: list[XmlNamespace] | None = None,
) -> list[XmlAttribute]:
    """
    Resolve names and values.

    Important:
      - attribute.name_idx is a direct index into the string pool.
      - resource_map.ids[name_idx], when present, is the optional resource ID.
      - there is NO "-1" index shift.
    """
    namespace_scopes = namespace_scopes or []

    # The active namespace with the newest declaration wins.
    uri_to_prefix: dict[int, str] = {}
    for ns in namespace_scopes:
        if ns.uri_idx != NO_NAMESPACE and ns.prefix is not None:
            uri_to_prefix[ns.uri_idx] = ns.prefix

    for attr in attrs:
        attr.name = _safe_pool_string(pool, attr.name_idx)

        # Resource map indexes correspond directly to string-pool indexes.
        if 0 <= attr.name_idx < len(resource_map.ids):
            attr.resource_id = resource_map.ids[attr.name_idx]

        if attr.namespace_idx != NO_NAMESPACE:
            attr.namespace_uri = _safe_pool_string(
                pool,
                attr.namespace_idx,
            )
            attr.prefix = uri_to_prefix.get(attr.namespace_idx)

        # Preserve the original raw string when one exists.
        if attr.raw_value_idx != NO_NAMESPACE:
            attr.value = _safe_pool_string(
                pool,
                attr.raw_value_idx,
            )
            if attr.value is None:
                attr.value = (
                    f"<invalid-raw-string#{attr.raw_value_idx}>"
                )
        else:
            attr.value = _decode_typed_value(
                attr.data_type,
                attr.data,
                pool,
            )

    return attrs


# ============================================================================
# Tree construction
# ============================================================================

def _qualified_name(
    name: str,
    prefix: str | None,
) -> str:
    if prefix:
        return f"{prefix}:{name}"
    return name


def _namespace_match(
    expected: XmlNamespace,
    actual: XmlNamespace,
) -> bool:
    return (
        expected.prefix_idx == actual.prefix_idx
        and expected.uri_idx == actual.uri_idx
    )


def _remove_matching_namespace(
    stack: list[XmlNamespace],
    ending: XmlNamespace,
) -> None:
    # Normal AXML is properly nested. For malformed input, search backwards
    # so an unrelated namespace declaration does not poison the rest of the
    # document.
    for i in range(len(stack) - 1, -1, -1):
        if _namespace_match(stack[i], ending):
            del stack[i]
            return


def build_xml_tree(
    data: bytes,
    chunks: Iterable[tuple[int, int, int, int]],
    pool: StringPool,
    resource_map: ResourceMap,
) -> XmlElement | None:
    root: XmlElement | None = None
    stack: list[XmlElement] = []
    namespace_stack: list[XmlNamespace] = []

    for off, chunk_type, header_size, chunk_size in chunks:
        # XML nodes.
        if chunk_type == RES_XML_START_NAMESPACE:
            ns = resolve_namespace(
                parse_namespace(data, off),
                pool,
            )
            namespace_stack.append(ns)

        elif chunk_type == RES_XML_END_NAMESPACE:
            ns = resolve_namespace(
                parse_namespace(data, off),
                pool,
            )
            _remove_matching_namespace(
                namespace_stack,
                ns,
            )

        elif chunk_type == RES_XML_START_ELEMENT:
            start = parse_start_element(
                data,
                off,
            )

            tag = _safe_pool_string(
                pool,
                start.name_idx,
            )
            if tag is None:
                tag = f"<invalid-name#{start.name_idx}>"

            namespace_uri = None
            prefix = None

            if start.namespace_idx != NO_NAMESPACE:
                namespace_uri = _safe_pool_string(
                    pool,
                    start.namespace_idx,
                )

                for ns in reversed(namespace_stack):
                    if ns.uri_idx == start.namespace_idx:
                        prefix = ns.prefix
                        break

            attrs = parse_attributes(
                data,
                off,
                start,
            )

            resolve_attributes(
                attrs,
                pool,
                resource_map,
                namespace_stack,
            )

            node = XmlElement(
                tag=tag,
                line=start.line_number,
                attributes=attrs,
                children=[],
                namespace_idx=start.namespace_idx,
                namespace_uri=namespace_uri,
                prefix=prefix,
            )

            if stack:
                stack[-1].children.append(node)
            else:
                root = node

            stack.append(node)

        elif chunk_type == RES_XML_END_ELEMENT:
            end = parse_end_element(
                data,
                off,
            )

            if stack:
                current = stack[-1]

                # Normal AXML has matching names. Do not prevent forensic
                # analysis just because a crafted document mismatches them.
                if (
                    current.tag != _safe_pool_string(pool, end.name_idx)
                ):
                    pass

                stack.pop()

        elif chunk_type == RES_XML_CDATA:
            if stack:
                _line, string_idx, data_type, typed_data = parse_cdata(
                    data,
                    off,
                )

                if string_idx != NO_NAMESPACE:
                    text = _safe_pool_string(
                        pool,
                        string_idx,
                    )
                    if text is not None:
                        stack[-1].text.append(text)
                else:
                    typed_text = _decode_typed_value(
                        data_type,
                        typed_data,
                        pool,
                    )
                    if typed_text is not None:
                        stack[-1].text.append(typed_text)

        else:
            # Unknown XML chunk: intentionally ignore it.
            #
            # AOSP's XML parser also skips unknown XML block types while
            # continuing through the document.
            continue

    return root


# ============================================================================
# High-level parser
# ============================================================================

def parse_manifest_bytes(
    data: bytes,
) -> tuple[XmlHeader, StringPool, ResourceMap, XmlElement | None]:
    """
    Parse complete binary AndroidManifest.xml bytes.

    Returns:
        (xml_header, string_pool, resource_map, root_element)
    """
    xml_header = parse_xml_header(data)

    chunks = list(
        walk_chunks(
            data,
            xml_header.header_size,
            xml_header.file_size,
        )
    )

    pool_header: StringPoolHeader | None = None
    resource_map = ResourceMap(ids=[])

    for off, chunk_type, header_size, chunk_size in chunks:
        if chunk_type == RES_STRING_POOL_TYPE:
            if pool_header is None:
                pool_header = parse_string_pool_header(
                    data,
                    off,
                )

        elif chunk_type == RES_XML_RESOURCE_MAP:
            resource_map = parse_resource_map(
                data,
                off,
                chunk_size,
                header_size,
            )

    if pool_header is None:
        raise BinaryXmlError(
            "AXML has no string-pool chunk"
        )

    pool = parse_string_pool(
        data,
        pool_header,
    )

    root = build_xml_tree(
        data,
        chunks,
        pool,
        resource_map,
    )

    return (
        xml_header,
        pool,
        resource_map,
        root,
    )


def parse_manifest(
    apk_path: Path,
) -> tuple[XmlHeader, StringPool, ResourceMap, XmlElement | None]:
    """
    Convenience wrapper:
        APK -> AndroidManifest.xml -> AXML parser
    """
    return parse_manifest_bytes(
        read_manifest(apk_path)
    )


# ============================================================================
# Helpers preserved from the earlier version
# ============================================================================

def first_start_element(
    chunks: Iterable[tuple[int, int, int, int]],
) -> int | None:
    for off, chunk_type, _header_size, _chunk_size in chunks:
        if chunk_type == RES_XML_START_ELEMENT:
            return off
    return None


def first_xml_node_offset(
    pool_header: StringPoolHeader,
) -> int:
    return (
        pool_header.chunk_start
        + pool_header.chunk_size
    )

def _display_forensic_text(value: str) -> str:
    """
    Make control characters visible without changing the underlying value.

    This is ONLY for terminal output. The parsed value is never modified.
    """
    return (
        value
        .replace("\\", "\\\\")
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )


# Android manifest names are normally ASCII-style XML names.  For forensic
# presentation, the important property here is to reject whitespace/control
# characters rather than attempting to normalize the original bytes.
_XML_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.:-]*$")


def is_valid_xml_name(name: str | None) -> bool:
    """Return True when *name* is suitable for the semantic display tree."""
    return bool(name) and bool(_XML_NAME_RE.fullmatch(name))


def is_suspicious_attribute(attr: XmlAttribute) -> bool:
    """
    Classify an attribute name for presentation.

    IMPORTANT: this does NOT modify or delete the underlying parsed attribute.
    """
    return not is_valid_xml_name(attr.name)


def _display_name(name: str | None) -> str:
    """
    Return a safe terminal representation of an XML name.

    This helper is useful for forensic/debug output. The semantic tree filters
    suspicious attribute names separately so normal XML remains readable.
    """
    if not name:
        return "<invalid-name>"

    if is_valid_xml_name(name):
        return name

    return f"<SUSPICIOUS_NAME:{_display_forensic_text(name)}>"


def _escape_attribute_value(value: str) -> str:
    """Escape a value for XML-like terminal output without changing the model."""
    return (
        value
        .replace("\\", "\\\\")
        .replace("&", "&amp;")
        .replace('"', "&quot;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )


def print_tree(
    node: XmlElement | None,
    depth: int = 0,
) -> None:

    if node is None:
        return

    indent = "  " * depth

    # ------------------------------------------------------------
    # Element name
    # ------------------------------------------------------------

    tag = node.tag or f"<invalid-name#{node.namespace_idx}>"

    if node.prefix:
        tag = _qualified_name(
            tag,
            node.prefix,
        )

    # ------------------------------------------------------------
    # Normal attributes only
    # ------------------------------------------------------------

    attr_parts: list[str] = []

    for attr in node.attributes:

        # Keep suspicious attributes in forensic data, but don't put
        # them into the clean semantic XML representation.
        if attr.name is None:
            continue

        if not is_valid_xml_name(attr.name):
            continue

        name = attr.name

        if attr.prefix:
            name = _qualified_name(
                name,
                attr.prefix,
            )

        if attr.value is None:

            attr_parts.append(name)

        else:

            value = (
                str(attr.value)
                .replace("&", "&amp;")
                .replace('"', "&quot;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace("\n", "\\n")
                .replace("\r", "\\r")
                .replace("\t", "\\t")
            )

            attr_parts.append(
                f'{name}="{value}"'
            )

    # ------------------------------------------------------------
    # Opening tag
    # ------------------------------------------------------------

    attrs = ""

    if attr_parts:
        attrs = " " + " ".join(attr_parts)

    # ------------------------------------------------------------
    # Leaf node
    # ------------------------------------------------------------

    if not node.children and not node.text:

        print(
            f"{indent}<{tag}{attrs} />"
        )

        return

    # ------------------------------------------------------------
    # Opening tag
    # ------------------------------------------------------------

    print(
        f"{indent}<{tag}{attrs}>"
    )

    # ------------------------------------------------------------
    # Text / CDATA
    # ------------------------------------------------------------

    for text in node.text:

        safe_text = (
            text
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace("\n", "\\n")
            .replace("\r", "\\r")
            .replace("\t", "\\t")
        )

        print(
            f"{indent}  {safe_text}"
        )

    # ------------------------------------------------------------
    # Children
    # ------------------------------------------------------------

    for child in node.children:

        print_tree(
            child,
            depth + 1,
        )

    # ------------------------------------------------------------
    # Closing tag
    # ------------------------------------------------------------

    print(
        f"{indent}</{tag}>"
    )


def collect_xml_anomalies(
    node: XmlElement | None,
    path: str = "",
    results: list[dict] | None = None,
) -> list[dict]:
    """
    Collect suspicious attribute names while preserving the original values.

    Each result identifies where the attribute appeared and records the raw
    string-pool index, value type/data, and optional resource ID.
    """
    if results is None:
        results = []

    if node is None:
        return results

    current_path = f"{path}/{node.tag}" if path else f"/{node.tag}"

    for index, attr in enumerate(node.attributes):
        if not is_suspicious_attribute(attr):
            continue

        results.append(
            {
                "path": current_path,
                "attribute_index": index,
                "name_idx": attr.name_idx,
                "name": attr.name,
                "value": attr.value,
                "data_type": attr.data_type,
                "data": attr.data,
                "resource_id": attr.resource_id,
                "namespace_idx": attr.namespace_idx,
                "namespace_uri": attr.namespace_uri,
                "prefix": attr.prefix,
            }
        )

    for child in node.children:
        collect_xml_anomalies(child, current_path, results)

    return results


def print_anomalies(
    node: XmlElement | None,
) -> list[dict]:
    """Print and return the forensic AXML anomaly list."""
    anomalies = collect_xml_anomalies(node)

    print()
    print("=" * 60)
    print("AXML ANOMALIES")
    print("=" * 60)
    print(f"Suspicious attributes : {len(anomalies)}")

    for i, anomaly in enumerate(anomalies, 1):
        print()
        print(f"[{i}] Path       : {anomaly['path']}")
        print(f"    Index      : {anomaly['attribute_index']}")
        print(f"    name_idx   : {anomaly['name_idx']}")
        print(f"    Name       : {anomaly['name']!r}")
        print(f"    Value      : {anomaly['value']!r}")
        print(f"    Data Type  : 0x{anomaly['data_type']:02X}")
        print(f"    Data       : 0x{anomaly['data']:08X}")

        if anomaly["prefix"] is not None:
            print(f"    Prefix     : {anomaly['prefix']}")

        if anomaly["namespace_uri"] is not None:
            print(f"    Namespace  : {anomaly['namespace_uri']}")

        if anomaly["resource_id"] is not None:
            print(
                f"    Resource ID: 0x{anomaly['resource_id']:08X}"
            )

    return anomalies


def debug_attribute_bytes(
    data: bytes,
    chunk_offset: int,
    element: StartElement,
) -> None:
    node = parse_node_header(data, chunk_offset)

    start = attribute_array_offset(
        chunk_offset,
        element,
        node.header_size,
    )

    print("\nRAW ATTRIBUTE RECORDS")
    print("-" * 70)

    for i in range(element.attribute_count):
        off = start + i * element.attribute_size

        raw20 = data[off:off + 20]

        ns, name, raw = struct.unpack_from(
            "<III",
            data,
            off,
        )

        size, res0, dtype = struct.unpack_from(
            "<HBB",
            data,
            off + 12,
        )

        value = struct.unpack_from(
            "<I",
            data,
            off + 16,
        )[0]

        print(f"[{i}] @ {off}")
        print("bytes :", raw20.hex(" "))
        print(
            "ns="
            f"{ns} "
            f"name={name} "
            f"raw={raw} "
            f"type=0x{dtype:02X} "
            f"value={value}"
        )
        print()


def debug_single_string(
    data: bytes,
    pool: StringPool,
    index: int,
) -> None:
    if index < 0 or index >= len(pool.offsets):
        raise IndexError(index)

    offset = (
        pool.header.chunk_start
        + pool.header.strings_start
        + pool.offsets[index]
    )

    print(f"\nSTRING #{index}")
    print("-" * 40)
    print("Absolute Offset :", offset)

    if pool.header.flags & UTF8_FLAG:
        utf16_length, first_bytes = read_utf8_length(data, offset)
        byte_length, second_bytes = read_utf8_length(
            data,
            offset + first_bytes,
        )
        print("Declared UTF-16 Length :", utf16_length)
        print("Declared UTF-8 Length  :", byte_length)
        print("Header Bytes           :", first_bytes + second_bytes)
    else:
        length, header = read_utf16_length(
            data,
            offset,
        )
        print("Declared UTF-16 Length :", length)
        print("Header Bytes           :", header)

    raw = data[offset: offset + 64]
    print(
        "First 64 bytes        :",
        raw.hex(" "),
    )

    text = pool.strings[index]
    print("Decoded               :", repr(text))


# ============================================================================
# Optional standalone usage
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Parse AndroidManifest.xml from an APK"
    )
    parser.add_argument(
        "apk",
        type=Path,
        help="Path to APK",
    )

    args = parser.parse_args()

    try:
        header, pool, resource_map, root = parse_manifest(
            args.apk
        )

        print("AXML")
        print("-" * 50)
        print(
            f"Chunk Type : 0x{header.chunk_type:04X}"
        )
        print(
            f"Header Size: {header.header_size}"
        )
        print(
            f"File Size  : {header.file_size}"
        )
        print(
            f"Strings    : {len(pool.strings)}"
        )
        print(
            f"Resource IDs: {len(resource_map.ids)}"
        )
        print()

        print_tree(root)

    except (OSError, zipfile.BadZipFile, BinaryXmlError) as exc:
        raise SystemExit(
            f"manifest parse failed: {exc}"
        )
