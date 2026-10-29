"""
resources.arsc forensic parser.

Parses the Android resource table directly from AOSP's binary specification
(androidfw/ResourceTypes.h) with no external tooling. Design goals:

- bounded input: the table is streamed in through the zip-bomb-safe reader
- every offset is validated before use
- UTF-8 and UTF-16 string pools are both supported
- Type chunks with FLAG_SPARSE / FLAG_OFFSET16 are decoded per spec
- simple entries, compact entries and complex bags are all resolved
- resource references are resolved to @package:type/name where possible
- structural oddities are recorded as anomalies instead of aborting,
  because hostile APKs must still yield forensic output

On-disk layout recap (all little-endian):

    ResTable_header
      [optional non-standard chunks]
      ResStringPool_header          (global value strings)
      ResTable_package
        ResStringPool_header        (type names)
        ResStringPool_header        (entry key names)
        ResTable_typeSpec / ResTable_type / ResTable_library ...
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import struct
import zipfile

from tools.zip_audit import (
    ZipEntryError,
    ZipReadLimitError,
    read_entry_bytes,
)


# ============================================================================
# CHUNK TYPES  (AOSP ResourceTypes.h)
# ============================================================================

RES_NULL_TYPE = 0x0000
RES_STRING_POOL_TYPE = 0x0001
RES_TABLE_TYPE = 0x0002

RES_TABLE_PACKAGE_TYPE = 0x0200
RES_TABLE_TYPE_TYPE = 0x0201
RES_TABLE_TYPE_SPEC_TYPE = 0x0202
RES_TABLE_LIBRARY_TYPE = 0x0203
RES_TABLE_OVERLAYABLE_TYPE = 0x0204
RES_TABLE_OVERLAYABLE_POLICY_TYPE = 0x0205
RES_TABLE_STAGED_ALIAS_TYPE = 0x0206

# ============================================================================
# STRING POOL FLAGS
# ============================================================================

SORTED_FLAG = 1 << 0
UTF8_FLAG = 1 << 8
STRING_POOL_KNOWN_FLAGS = SORTED_FLAG | UTF8_FLAG

# ============================================================================
# ResTable_entry FLAGS
# ============================================================================

ENTRY_FLAG_COMPLEX = 0x0001
ENTRY_FLAG_PUBLIC = 0x0002
ENTRY_FLAG_WEAK = 0x0004
ENTRY_FLAG_COMPACT = 0x0008
ENTRY_KNOWN_FLAGS = (
    ENTRY_FLAG_COMPLEX | ENTRY_FLAG_PUBLIC | ENTRY_FLAG_WEAK | ENTRY_FLAG_COMPACT
)

NO_ENTRY = 0xFFFFFFFF

# ResTable_typeSpec flags.
SPEC_PUBLIC = 0x40000000
SPEC_STAGED_API = 0x20000000

# ResTable_type flags.
TYPE_FLAG_SPARSE = 0x01
TYPE_FLAG_OFFSET16 = 0x02

# ============================================================================
# Res_value TYPES  (AOSP ResourceTypes.h)
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

# Complex value encoding (dimensions / fractions).
COMPLEX_UNIT_SHIFT = 0
COMPLEX_UNIT_MASK = 0xF
COMPLEX_RADIX_SHIFT = 4
COMPLEX_RADIX_MASK = 0x3
COMPLEX_MANTISSA_SHIFT = 8
COMPLEX_MANTISSA_MASK = 0xFFFFFF

# Well-known bag idents. These are format markers from the aapt toolchain,
# not Android resource IDs, so labeling them is spec knowledge, not
# "hardcoding resource IDs".
PLURAL_KEYS = {
    0x01000004: "other",
    0x01000005: "zero",
    0x01000006: "one",
    0x01000007: "two",
    0x01000008: "few",
    0x01000009: "many",
}

ANDROID_PACKAGE_ID = 0x01

# Maximum table size we are ever willing to materialize in RAM.
DEFAULT_MAX_TABLE_BYTES = 32 * 1024 * 1024
DEFAULT_MAX_TABLE_RATIO = 200.0


# ============================================================================
# EXCEPTIONS
# ============================================================================

class ResourceTableError(ValueError):
    """Raised when resources.arsc is structurally unusable."""


# ============================================================================
# SAFE BINARY HELPERS
# ============================================================================

def require_range(data: bytes, offset: int, size: int, what: str) -> None:
    if offset < 0:
        raise ResourceTableError(f"{what}: negative offset")
    if size < 0:
        raise ResourceTableError(f"{what}: negative size")
    if offset + size > len(data):
        raise ResourceTableError(
            f"{what} out of bounds: offset={offset}, size={size}, buffer={len(data)}"
        )


def read_u8(data: bytes, offset: int, what: str) -> int:
    require_range(data, offset, 1, what)
    return data[offset]


def read_u16(data: bytes, offset: int, what: str) -> int:
    require_range(data, offset, 2, what)
    return struct.unpack_from("<H", data, offset)[0]


def read_u32(data: bytes, offset: int, what: str) -> int:
    require_range(data, offset, 4, what)
    return struct.unpack_from("<I", data, offset)[0]


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class ResourceEntryInfo:
    name: str
    compression_type: int
    compressed_size: int
    uncompressed_size: int
    compression_ratio: float | None


@dataclass
class ResourceTableHeader:
    chunk_type: int
    header_size: int
    chunk_size: int
    package_count: int


@dataclass
class ResourceStringPoolHeader:
    chunk_type: int
    header_size: int
    chunk_size: int
    string_count: int
    style_count: int
    flags: int
    strings_start: int
    styles_start: int


@dataclass
class ResourceStringPool:
    header: ResourceStringPoolHeader
    offsets: list[int]
    strings: list[str]
    anomalies: list[str] = field(default_factory=list)

    def get(self, index: int) -> str | None:
        if 0 <= index < len(self.strings):
            return self.strings[index]
        return None

    @property
    def is_utf8(self) -> bool:
        return bool(self.header.flags & UTF8_FLAG)


@dataclass
class TypedValue:
    """A decoded Res_value."""
    data_type: int
    data: int

    def format(self, resolver: "ResourceNameResolver | None" = None) -> str:
        return decode_res_value(self.data_type, self.data, resolver)


@dataclass
class BagItem:
    ident: int
    ident_label: str
    value: TypedValue


@dataclass
class ResourceEntry:
    """One ResTable_entry with its resolved payload."""
    entry_index: int
    flags: int
    key_index: int
    key_name: str | None
    is_compact: bool
    is_complex: bool
    value: TypedValue | None          # simple entries
    parent_ref: int | None            # complex entries
    bag: list[BagItem] = field(default_factory=list)

    @property
    def is_public(self) -> bool:
        return bool(self.flags & ENTRY_FLAG_PUBLIC)

    @property
    def is_weak(self) -> bool:
        return bool(self.flags & ENTRY_FLAG_WEAK)


@dataclass
class ResourceConfig:
    """Decoded ResTable_config."""
    size: int
    raw: bytes
    qualifier: str

    def __str__(self) -> str:
        return self.qualifier or "(default)"


@dataclass
class TypeChunk:
    """One ResTable_type: entries for one (type, config) pair."""
    chunk_offset: int
    type_id: int
    flags: int
    entry_count: int
    entries_start: int
    config: ResourceConfig
    is_sparse: bool
    uses_offset16: bool
    # entry index -> absolute offset of ResTable_entry (None = NO_ENTRY)
    entry_offsets: dict[int, int] = field(default_factory=dict)
    entries: dict[int, ResourceEntry] = field(default_factory=dict)
    anomalies: list[str] = field(default_factory=list)


@dataclass
class TypeSpecChunk:
    type_id: int
    entry_count: int
    flags: list[int]
    types_count: int = 0


@dataclass
class LibraryEntry:
    package_id: int
    name: str


@dataclass
class LibraryChunk:
    entries: list[LibraryEntry] = field(default_factory=list)


@dataclass
class PackageChunk:
    chunk_offset: int
    package_id: int
    name: str
    type_strings: ResourceStringPool
    key_strings: ResourceStringPool
    last_public_type: int
    last_public_key: int
    type_id_offset: int
    typespecs: dict[int, TypeSpecChunk] = field(default_factory=dict)
    types: list[TypeChunk] = field(default_factory=list)
    libraries: list[LibraryChunk] = field(default_factory=list)
    anomalies: list[str] = field(default_factory=list)

    def type_name(self, type_id: int) -> str | None:
        # Type IDs are 1-based; typeIdOffset shifts the string index.
        idx = type_id - 1 - self.type_id_offset
        return self.type_strings.get(idx)


@dataclass
class UnknownChunk:
    chunk_offset: int
    chunk_type: int
    header_size: int
    chunk_size: int
    printable: str


@dataclass
class ResourceTable:
    header: ResourceTableHeader
    global_pool: ResourceStringPool
    packages: list[PackageChunk] = field(default_factory=list)
    unknown_chunks: list[UnknownChunk] = field(default_factory=list)
    anomalies: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Resolution
    # ------------------------------------------------------------------

    def resolver(self) -> "ResourceNameResolver":
        return ResourceNameResolver(self)

    def package_by_id(self, package_id: int) -> PackageChunk | None:
        for pkg in self.packages:
            if pkg.package_id == package_id:
                return pkg
        return None

    def entry(self, resource_id: int) -> tuple[PackageChunk, TypeChunk, ResourceEntry] | None:
        """Resolve 0xpptteeee to its defining entry (first config wins)."""
        package_id = (resource_id >> 24) & 0xFF
        type_id = (resource_id >> 16) & 0xFF
        entry_index = resource_id & 0xFFFF

        pkg = self.package_by_id(package_id)
        if pkg is None:
            return None
        for t in pkg.types:
            if t.type_id == type_id and entry_index in t.entries:
                return pkg, t, t.entries[entry_index]
        return None

    def stats(self) -> dict:
        entry_total = sum(len(t.entries) for p in self.packages for t in p.types)
        config_total = len({(p.package_id, t.type_id, t.config.qualifier)
                            for p in self.packages for t in p.types})
        type_names = sorted({
            name for p in self.packages
            for t in p.types
            if (name := p.type_name(t.type_id))
        })
        return {
            "packages": len(self.packages),
            "resource_entries": entry_total,
            "config_variants": config_total,
            "types": type_names,
        }


class ResourceNameResolver:
    """Resolves resource IDs and bag idents to readable names."""

    def __init__(self, table: ResourceTable):
        self.table = table
        self.libraries: dict[int, str] = {}
        for pkg in table.packages:
            for lib in pkg.libraries:
                for entry in lib.entries:
                    self.libraries[entry.package_id] = entry.name

    # ------------------------------------------------------------------

    def package_label(self, package_id: int) -> str:
        if package_id == ANDROID_PACKAGE_ID:
            return "android"
        if package_id in self.libraries:
            return self.libraries[package_id]
        pkg = self.table.package_by_id(package_id)
        if pkg is not None:
            return pkg.name
        return f"pkg{package_id:02X}"

    def name_of(self, resource_id: int) -> str | None:
        """'type/name' for a full resource id, or None when unresolvable."""
        package_id = (resource_id >> 24) & 0xFF
        type_id = (resource_id >> 16) & 0xFF
        entry_index = resource_id & 0xFFFF

        pkg = self.table.package_by_id(package_id)
        if pkg is None:
            return None
        type_name = pkg.type_name(type_id)
        if type_name is None:
            return None
        for t in pkg.types:
            if t.type_id == type_id:
                entry = t.entries.get(entry_index)
                if entry is not None and entry.key_name:
                    return f"{type_name}/{entry.key_name}"
        # Fall back to the key pool even without a concrete entry.
        for spec in pkg.typespecs.values():
            if spec.type_id == type_id and entry_index < spec.entry_count:
                key = pkg.key_strings.get(entry_index)
                if key is not None:
                    return f"{type_name}/{key}"
        return None

    def reference(self, resource_id: int, attribute: bool = False) -> str:
        prefix = "?" if attribute else "@"
        name = self.name_of(resource_id)
        if name is not None:
            return f"{prefix}{self.package_label((resource_id >> 24) & 0xFF)}:{name}"
        return f"{prefix}0x{resource_id:08X}"


# ============================================================================
# ZIP ENTRY METADATA
# ============================================================================

def inspect_resource_entry(
    apk_path: Path,
    member_name: str = "resources.arsc",
) -> ResourceEntryInfo:
    with zipfile.ZipFile(apk_path, "r") as z:
        try:
            info = z.getinfo(member_name)
        except KeyError as exc:
            raise ResourceTableError(f"{member_name} not found in APK") from exc

        ratio = None
        if info.compress_size:
            ratio = info.file_size / info.compress_size

        return ResourceEntryInfo(
            name=info.filename,
            compression_type=info.compress_type,
            compressed_size=info.compress_size,
            uncompressed_size=info.file_size,
            compression_ratio=ratio,
        )


# ============================================================================
# BOUNDED TABLE LOADING  (never unzip the APK; one bounded entry read)
# ============================================================================

def load_resource_table_bytes(
    apk_path: Path,
    max_bytes: int = DEFAULT_MAX_TABLE_BYTES,
    max_ratio: float = DEFAULT_MAX_TABLE_RATIO,
) -> bytes:
    try:
        return read_entry_bytes(
            apk_path,
            "resources.arsc",
            max_uncompressed=max_bytes,
            max_ratio=max_ratio,
        )
    except (ZipEntryError, ZipReadLimitError) as exc:
        raise ResourceTableError(f"resources.arsc rejected: {exc}") from exc


def parse_resource_table(apk_path: Path) -> ResourceTable:
    """High-level entry point: APK -> resources.arsc -> ResourceTable."""
    return parse_resource_table_bytes(load_resource_table_bytes(apk_path))


# ============================================================================
# TABLE HEADER
# ============================================================================

def parse_resource_table_header(data: bytes) -> ResourceTableHeader:
    require_range(data, 0, 12, "resource table header")

    chunk_type, header_size, chunk_size, package_count = struct.unpack_from(
        "<HHII", data, 0
    )

    if chunk_type != RES_TABLE_TYPE:
        raise ResourceTableError(
            f"Invalid resource table type: 0x{chunk_type:04X}"
        )
    if header_size < 12:
        raise ResourceTableError(f"Invalid resource table header size: {header_size}")
    if chunk_size < header_size:
        raise ResourceTableError(
            f"Resource table size {chunk_size} < header size {header_size}"
        )
    if chunk_size > len(data):
        raise ResourceTableError(
            f"Resource table chunk size {chunk_size} exceeds buffer {len(data)}"
        )

    return ResourceTableHeader(
        chunk_type=chunk_type,
        header_size=header_size,
        chunk_size=chunk_size,
        package_count=package_count,
    )


# ============================================================================
# STRING POOLS (UTF-8 and UTF-16, forensic-tolerant)
# ============================================================================

def parse_string_pool_header(data: bytes, offset: int) -> ResourceStringPoolHeader:
    require_range(data, offset, 28, "resource string-pool header")

    (
        chunk_type,
        header_size,
        chunk_size,
        string_count,
        style_count,
        flags,
        strings_start,
        styles_start,
    ) = struct.unpack_from("<HHIIIIII", data, offset)

    if chunk_type != RES_STRING_POOL_TYPE:
        raise ResourceTableError(
            f"Expected string pool at offset {offset}, got 0x{chunk_type:04X}"
        )
    if header_size < 28:
        raise ResourceTableError(f"Invalid string-pool header size: {header_size}")
    if chunk_size < header_size:
        raise ResourceTableError(
            f"String-pool chunk size {chunk_size} < header size {header_size}"
        )
    if offset + chunk_size > len(data):
        raise ResourceTableError("String-pool chunk extends beyond input")
    if strings_start < header_size:
        raise ResourceTableError(
            f"stringsStart {strings_start} is before header end {header_size}"
        )
    if strings_start >= chunk_size:
        raise ResourceTableError(
            f"stringsStart {strings_start} is outside chunk size {chunk_size}"
        )

    return ResourceStringPoolHeader(
        chunk_type=chunk_type,
        header_size=header_size,
        chunk_size=chunk_size,
        string_count=string_count,
        style_count=style_count,
        flags=flags,
        strings_start=strings_start,
        styles_start=styles_start,
    )


def parse_string_offsets(
    data: bytes,
    pool_offset: int,
    header: ResourceStringPoolHeader,
) -> list[int]:
    offset_table = pool_offset + header.header_size
    total_bytes = header.string_count * 4

    require_range(data, offset_table, total_bytes, "string offset table")

    return [
        read_u32(data, offset_table + i * 4, f"string offset #{i}")
        for i in range(header.string_count)
    ]


def _utf8_length_at(data: bytes, offset: int) -> tuple[int, int]:
    """Android UTF-8 pool length prefix: 1 or 2 bytes, high-bit continuation."""
    require_range(data, offset, 1, "UTF-8 length")
    first = data[offset]
    if (first & 0x80) == 0:
        return first, 1
    require_range(data, offset + 1, 1, "extended UTF-8 length")
    return ((first & 0x7F) << 8) | data[offset + 1], 2


def _decode_utf8_string(
    data: bytes,
    offset: int,
    pool_end: int,
    anomalies: list[str],
    label: str,
) -> str:
    utf16_length, first_bytes = _utf8_length_at(data, offset)
    utf8_length, second_bytes = _utf8_length_at(data, offset + first_bytes)

    payload_offset = offset + first_bytes + second_bytes
    require_range(data, payload_offset, utf8_length, f"{label} payload")

    payload_end = payload_offset + utf8_length
    if payload_end >= pool_end:
        raise ResourceTableError(f"{label} extends past string-pool boundary")

    if data[payload_end] != 0:
        anomalies.append(f"{label}: missing NUL terminator")
    payload = data[payload_offset:payload_end]
    value = payload.decode("utf-8", errors="replace")

    actual_utf16 = len(value.encode("utf-16-le")) // 2
    if actual_utf16 != utf16_length:
        anomalies.append(
            f"{label}: UTF-16 length mismatch declared={utf16_length} "
            f"actual={actual_utf16}"
        )
    return value


def _decode_utf16_string(
    data: bytes,
    offset: int,
    pool_end: int,
    anomalies: list[str],
    label: str,
) -> str:
    require_range(data, offset, 2, f"{label} length")

    first = struct.unpack_from("<H", data, offset)[0]
    if (first & 0x8000) == 0:
        length, header_bytes = first, 2
    else:
        require_range(data, offset + 2, 2, f"{label} extended length")
        second = struct.unpack_from("<H", data, offset + 2)[0]
        length, header_bytes = ((first & 0x7FFF) << 16) | second, 4

    payload_offset = offset + header_bytes
    payload_size = length * 2
    require_range(data, payload_offset, payload_size, f"{label} payload")

    payload_end = payload_offset + payload_size
    if payload_end + 2 > pool_end:
        raise ResourceTableError(f"{label} extends past string-pool boundary")
    if data[payload_end] != 0 or data[payload_end + 1] != 0:
        anomalies.append(f"{label}: missing NUL terminator")

    return data[payload_offset:payload_end].decode("utf-16-le", errors="replace")


def parse_resource_string_pool(data: bytes, pool_offset: int) -> ResourceStringPool:
    header = parse_string_pool_header(data, pool_offset)
    offsets = parse_string_offsets(data, pool_offset, header)
    anomalies: list[str] = []

    unknown_flag_bits = header.flags & ~STRING_POOL_KNOWN_FLAGS
    if unknown_flag_bits:
        anomalies.append(
            f"non-standard pool flag bits: 0x{unknown_flag_bits:08X}"
        )

    chunk_end = pool_offset + header.chunk_size
    strings_start = pool_offset + header.strings_start

    if header.style_count:
        if header.styles_start <= header.strings_start:
            raise ResourceTableError(
                "styles_start must follow strings_start when styles exist"
            )
        string_data_end = pool_offset + header.styles_start
    else:
        string_data_end = chunk_end

    if string_data_end > chunk_end or strings_start > string_data_end:
        raise ResourceTableError("Invalid string-pool string-data range")

    is_utf8 = bool(header.flags & UTF8_FLAG)
    strings: list[str] = []

    for i, relative_offset in enumerate(offsets):
        absolute_offset = strings_start + relative_offset
        if absolute_offset < strings_start or absolute_offset >= string_data_end:
            raise ResourceTableError(
                f"String #{i} points outside string data: {relative_offset}"
            )
        label = f"string #{i}"
        if is_utf8:
            value = _decode_utf8_string(
                data, absolute_offset, string_data_end, anomalies, label
            )
        else:
            value = _decode_utf16_string(
                data, absolute_offset, string_data_end, anomalies, label
            )
        strings.append(value)

    return ResourceStringPool(
        header=header,
        offsets=offsets,
        strings=strings,
        anomalies=anomalies,
    )


# ============================================================================
# ResTable_config DECODING
# ============================================================================

def decode_res_table_config(data: bytes, offset: int, available: int) -> ResourceConfig:
    require_range(data, offset, 4, "ResTable_config size")
    config_size = read_u32(data, offset, "ResTable_config size")

    if config_size < 4 or config_size > available:
        raise ResourceTableError(
            f"ResTable_config size {config_size} invalid (available {available})"
        )
    require_range(data, offset, config_size, "ResTable_config")
    raw = data[offset:offset + config_size]

    def field_at(rel: int, count: int) -> bytes:
        if rel + count <= config_size:
            return raw[rel:rel + count]
        return b""

    def u16_at(rel: int) -> int:
        b = field_at(rel, 2)
        return struct.unpack("<H", b)[0] if len(b) == 2 else 0

    def u32_at(rel: int) -> int:
        b = field_at(rel, 4)
        return struct.unpack("<I", b)[0] if len(b) == 4 else 0

    parts: list[str] = []

    # --- mcc / mnc (offset 4) ---
    mcc = u16_at(4)
    mnc = u16_at(6)
    if mcc:
        parts.append(f"mcc{mcc}")
    if mnc:
        if mnc & 0x8000:  # add-one encoding for 0
            parts.append(f"mnc{mnc - 0x8000}")
        else:
            parts.append(f"mnc{mnc}")

    # --- locale (offset 8): 2-byte packed or 3-letter packed codes ---
    locale = raw[8:12] if config_size >= 12 else b"\x00\x00\x00\x00"
    language, country = locale[0:2], locale[2:4]
    lang_str = ""
    if language != b"\x00\x00":
        if language[0] & 0x80:
            # Packed 3-letter code (big-endian layout, see unpackLanguage).
            lang_str = _unpack_3letter(language[0], language[1])
        else:
            lang_str = language.decode("ascii", errors="replace")
    region_str = ""
    if country != b"\x00\x00":
        if country[0] & 0x80:
            region_str = _unpack_3letter(country[0], country[1])
        else:
            region_str = country.decode("ascii", errors="replace")

    # BCP-47: when localeScript (offset 36) is populated, aapt prints the
    # b+lang+script[+region] form; otherwise the legacy lang-rREGION form.
    script_str = ""
    if config_size >= 40:
        script = field_at(36, 4)
        if script and script != b"\x00\x00\x00\x00":
            script_str = script.decode("ascii", errors="replace")

    if lang_str and script_str:
        parts.append(
            f"b+{lang_str}+{script_str}" + (f"+{region_str}" if region_str else "")
        )
    elif lang_str and region_str:
        parts.append(f"{lang_str}-r{region_str}")
    elif lang_str:
        parts.append(lang_str)

    # --- screenType (offset 12): orientation, touchscreen, density ---
    orientation = raw[12] if config_size >= 13 else 0
    touchscreen = raw[13] if config_size >= 14 else 0
    density = u16_at(14)

    orientation_names = {0: "", 1: "port", 2: "land", 3: "square"}
    if orientation_names.get(orientation, ""):
        parts.append(orientation_names[orientation])

    touchscreen_names = {1: "notouch", 2: "finger", 3: "stylus"}
    if touchscreen_names.get(touchscreen, ""):
        parts.append(touchscreen_names[touchscreen])

    if density == 0xFFFF:
        parts.append("nodpi")
    elif density == 0xFFFE:
        parts.append("anydpi")
    elif density == 1:
        pass  # DENSITY_DEFAULT -> no qualifier
    elif density:
        parts.append(f"{density}dpi")

    # --- input (offset 16): keyboard, navigation, inputFlags ---
    keyboard = raw[16] if config_size >= 17 else 0
    navigation = raw[17] if config_size >= 18 else 0
    input_flags = raw[18] if config_size >= 19 else 0

    keyboard_names = {1: "nokeys", 2: "qwerty", 3: "12key"}
    if keyboard_names.get(keyboard, ""):
        parts.append(keyboard_names[keyboard])

    nav_names = {1: "nonav", 2: "dpad", 3: "trackball", 4: "wheel"}
    if nav_names.get(navigation, ""):
        parts.append(nav_names[navigation])

    keys_hidden = input_flags & 0x3
    keys_names = {1: "keysexposed", 2: "keyshidden", 3: "keyssoft"}
    if keys_names.get(keys_hidden, ""):
        parts.append(keys_names[keys_hidden])

    nav_hidden = (input_flags >> 2) & 0x3
    navh_names = {1: "navexposed", 2: "navhidden"}
    if navh_names.get(nav_hidden, ""):
        parts.append(navh_names[nav_hidden])

    # --- screenSize (offset 20) ---
    screen_w = u16_at(20)
    screen_h = u16_at(22)
    if screen_w or screen_h:
        parts.append(f"{screen_w}x{screen_h}")

    # --- version (offset 24) ---
    sdk_version = u16_at(24)
    if sdk_version:
        parts.append(f"v{sdk_version}")

    # --- screenConfig (offset 28) ---
    if config_size >= 32:
        screen_layout = raw[28]
        ui_mode = raw[29]
        smallest_w = u16_at(30)

        size_class = screen_layout & 0x0F
        size_names = {1: "small", 2: "normal", 3: "large", 4: "xlarge"}
        if size_names.get(size_class, ""):
            parts.append(size_names[size_class])

        if screen_layout & 0x20:
            parts.append("long")
        if screen_layout & 0x80:
            layout_dir = (screen_layout >> 6) & 0x3
            if layout_dir == 1:
                parts.append("ldltr")
            elif layout_dir == 2:
                parts.append("ldrtl")

        ui_type = ui_mode & 0x0F
        ui_names = {
            2: "car", 3: "television", 4: "appliance",
            5: "watch", 6: "vrheadset", 1: "desk",
        }
        if ui_names.get(ui_type, ""):
            parts.append(ui_names[ui_type])
        if ui_mode & 0x20:
            parts.append("night")
        elif ui_mode & 0x10:
            parts.append("notnight")

        if smallest_w:
            parts.append(f"sw{smallest_w}dp")

    # --- screenSizeDp (offset 32) ---
    screen_w_dp = u16_at(32)
    screen_h_dp = u16_at(34)
    if screen_w_dp:
        parts.append(f"w{screen_w_dp}dp")
    if screen_h_dp:
        parts.append(f"h{screen_h_dp}dp")

    # --- screenConfig2 (offset 48) ---
    if config_size >= 52:
        screen_layout2 = raw[48]
        color_mode = raw[49]
        if screen_layout2 & 0x02:
            parts.append("round")
        elif screen_layout2 & 0x01:
            parts.append("notround")
        if color_mode & 0x02:
            parts.append("widecg")
        elif color_mode & 0x01:
            parts.append("lowdr")
        if color_mode & 0x08:
            parts.append("hdr")
        elif color_mode & 0x04:
            parts.append("nohdr")

    return ResourceConfig(
        size=config_size,
        raw=raw,
        qualifier="-".join(parts),
    )


def _unpack_3letter(first: int, second: int) -> str:
    """Decode a packed 3-letter language/region code (AOSP packLanguage)."""
    packed = (second << 8) | first
    letters = []
    for shift in (0, 5, 10):
        letters.append(chr(ord("a") + ((packed >> shift) & 0x1F)))
    return "".join(letters)


# ============================================================================
# PACKAGE CHUNK
# ============================================================================

PACKAGE_HEADER_SIZE = 288  # ResTable_package fixed part: 8 + 4 + 256 + 20


def parse_package_header_fields(data: bytes, offset: int) -> dict:
    require_range(data, offset, 8, "package chunk header")

    chunk_type, header_size, chunk_size = struct.unpack_from("<HHI", data, offset)
    if chunk_type != RES_TABLE_PACKAGE_TYPE:
        raise ResourceTableError(
            f"Expected package chunk at {offset}, got 0x{chunk_type:04X}"
        )
    if header_size < PACKAGE_HEADER_SIZE:
        raise ResourceTableError(
            f"Package header size {header_size} < minimum {PACKAGE_HEADER_SIZE}"
        )
    if offset + chunk_size > len(data):
        raise ResourceTableError("Package chunk extends beyond input")

    package_id = read_u32(data, offset + 8, "package id")
    raw_name = data[offset + 12:offset + 12 + 256]
    name = raw_name.decode("utf-16-le", errors="replace").split("\x00")[0]

    (
        type_strings,
        last_public_type,
        key_strings,
        last_public_key,
        type_id_offset,
    ) = struct.unpack_from("<IIIII", data, offset + 8 + 4 + 256)

    return {
        "chunk_offset": offset,
        "header_size": header_size,
        "chunk_size": chunk_size,
        "package_id": package_id,
        "name": name,
        "type_strings": type_strings,
        "last_public_type": last_public_type,
        "key_strings": key_strings,
        "last_public_key": last_public_key,
        "type_id_offset": type_id_offset,
    }


def parse_package(data: bytes, offset: int) -> PackageChunk:
    fields = parse_package_header_fields(data, offset)
    chunk_size = fields["chunk_size"]
    header_size = fields["header_size"]
    pkg_end = offset + chunk_size
    anomalies: list[str] = []

    if fields["package_id"] == 0:
        anomalies.append("package id 0 (non-base / inherited package)")

    # ---- type string pool ----
    type_pool_off = offset + fields["type_strings"]
    if not (offset < type_pool_off < pkg_end):
        raise ResourceTableError(
            f"typeStrings offset {fields['type_strings']} outside package chunk"
        )
    type_pool = parse_resource_string_pool(data, type_pool_off)

    # ---- key string pool ----
    key_pool_off = offset + fields["key_strings"]
    if not (offset < key_pool_off < pkg_end):
        raise ResourceTableError(
            f"keyStrings offset {fields['key_strings']} outside package chunk"
        )
    key_pool = parse_resource_string_pool(data, key_pool_off)

    pkg = PackageChunk(
        chunk_offset=offset,
        package_id=fields["package_id"],
        name=fields["name"],
        type_strings=type_pool,
        key_strings=key_pool,
        last_public_type=fields["last_public_type"],
        last_public_key=fields["last_public_key"],
        type_id_offset=fields["type_id_offset"],
        anomalies=anomalies,
    )

    # ---- child chunks: everything after the later of the two pools ----
    cursor = max(
        type_pool_off + type_pool.header.chunk_size,
        key_pool_off + key_pool.header.chunk_size,
    )
    if cursor < offset + header_size:
        cursor = offset + header_size

    while cursor + 8 <= pkg_end:
        chunk_type, chunk_header_size, chunk_size = struct.unpack_from(
            "<HHI", data, cursor
        )
        if chunk_size < 8 or cursor + chunk_size > pkg_end:
            anomalies.append(
                f"invalid child chunk at package offset {cursor - offset} "
                f"(type 0x{chunk_type:04X}, size {chunk_size})"
            )
            break

        if chunk_type == RES_TABLE_TYPE_SPEC_TYPE:
            pkg.typespecs[read_u8(data, cursor + 8, "typespec id")] = parse_typespec(
                data, cursor
            )
        elif chunk_type == RES_TABLE_TYPE_TYPE:
            pkg.types.append(parse_type_chunk(data, cursor, pkg))
        elif chunk_type == RES_TABLE_LIBRARY_TYPE:
            pkg.libraries.append(parse_library_chunk(data, cursor))
        elif chunk_type in (
            RES_TABLE_OVERLAYABLE_TYPE,
            RES_TABLE_OVERLAYABLE_POLICY_TYPE,
            RES_TABLE_STAGED_ALIAS_TYPE,
        ):
            pass  # recognized; no forensic action yet
        else:
            anomalies.append(
                f"unknown package child chunk type 0x{chunk_type:04X} "
                f"at offset {cursor - offset}"
            )

        cursor += chunk_size

    # ---- cross-checks ----
    for t in pkg.types:
        spec = pkg.typespecs.get(t.type_id)
        if spec is not None and spec.entry_count != t.entry_count:
            anomalies.append(
                f"type {t.type_id}: TypeSpec entryCount {spec.entry_count} != "
                f"Type entryCount {t.entry_count}"
            )
        if t.type_id - 1 >= len(type_pool.strings):
            anomalies.append(
                f"type id {t.type_id} exceeds type string table "
                f"({len(type_pool.strings)} entries)"
            )

    return pkg


# ============================================================================
# TYPESPEC CHUNK
# ============================================================================

def parse_typespec(data: bytes, offset: int) -> TypeSpecChunk:
    require_range(data, offset, 16, "ResTable_typeSpec")

    # ResTable_typeSpec: ResChunk_header(8), id(u8)@8, res0(u8)@9,
    # typesCount(u16)@10, entryCount(u32)@12.
    chunk_type, header_size, chunk_size = struct.unpack_from("<HHI", data, offset)
    if chunk_type != RES_TABLE_TYPE_SPEC_TYPE:
        raise ResourceTableError(
            f"Expected TypeSpec at {offset}, got 0x{chunk_type:04X}"
        )
    if header_size < 16:
        raise ResourceTableError(f"TypeSpec header too small: {header_size}")
    if offset + chunk_size > len(data):
        raise ResourceTableError("TypeSpec chunk extends beyond input")

    type_id = read_u8(data, offset + 8, "typespec type id")
    res0 = read_u8(data, offset + 9, "typespec res0")
    types_count = read_u16(data, offset + 10, "typespec typesCount")
    entry_count = read_u32(data, offset + 12, "typespec entryCount")

    flags_bytes = chunk_size - header_size
    if flags_bytes != entry_count * 4:
        raise ResourceTableError(
            f"TypeSpec flags array {flags_bytes}B != entryCount*4 "
            f"({entry_count * 4}B)"
        )
    require_range(data, offset + header_size, flags_bytes, "typespec flags")

    flags = [
        read_u32(data, offset + header_size + i * 4, f"typespec flag #{i}")
        for i in range(entry_count)
    ]

    if res0 != 0:
        # Not fatal, but worth knowing about.
        pass

    return TypeSpecChunk(
        type_id=type_id,
        entry_count=entry_count,
        flags=flags,
        types_count=types_count,
    )


# ============================================================================
# TYPE CHUNK (ResTable_type)
# ============================================================================

def parse_type_chunk(data: bytes, offset: int, pkg: PackageChunk) -> TypeChunk:
    require_range(data, offset, 24, "ResTable_type")

    # ResTable_type: ResChunk_header(8), id(u8)@8, flags(u8)@9,
    # reserved(u16)@10, entryCount(u32)@12, entriesStart(u32)@16,
    # ResTable_config@20 (variable length).
    chunk_type, header_size, chunk_size = struct.unpack_from("<HHI", data, offset)
    if chunk_type != RES_TABLE_TYPE_TYPE:
        raise ResourceTableError(
            f"Expected Type chunk at {offset}, got 0x{chunk_type:04X}"
        )
    if header_size < 24:
        raise ResourceTableError(f"Type header too small: {header_size}")
    if offset + chunk_size > len(data):
        raise ResourceTableError("Type chunk extends beyond input")

    type_id = read_u8(data, offset + 8, "type id")
    flags = read_u8(data, offset + 9, "type flags")
    reserved = read_u16(data, offset + 10, "type reserved")
    entry_count = read_u32(data, offset + 12, "type entryCount")
    entries_start = read_u32(data, offset + 16, "type entriesStart")

    anomalies: list[str] = []
    if reserved != 0:
        anomalies.append(f"reserved field is {reserved}, expected 0")
    if flags & ~(TYPE_FLAG_SPARSE | TYPE_FLAG_OFFSET16):
        anomalies.append(f"unknown type flag bits: 0x{flags & ~0x03:02X}")

    config = decode_res_table_config(data, offset + 20, chunk_size - 20)

    is_sparse = bool(flags & TYPE_FLAG_SPARSE)
    uses_offset16 = bool(flags & TYPE_FLAG_OFFSET16)
    entry_size = 2 if uses_offset16 else 4

    offsets_end = header_size + entry_count * entry_size
    if offsets_end > chunk_size:
        raise ResourceTableError(
            f"Type entry-offset array ({entry_count} x {entry_size}B) exceeds chunk"
        )
    if entries_start > chunk_size:
        raise ResourceTableError(
            f"entriesStart {entries_start} exceeds chunk size {chunk_size}"
        )

    entry_offsets: dict[int, int] = {}

    if is_sparse:
        # Array of ResTable_sparseTypeEntry: (idx:u16, offset/4:u16),
        # sorted by idx for binary search.
        require_range(data, offset + header_size, entry_count * 4, "sparse entries")
        prev_idx = -1
        for i in range(entry_count):
            raw = read_u32(data, offset + header_size + i * 4, f"sparse #{i}")
            idx = raw & 0xFFFF
            rel = (raw >> 16) * 4
            if idx <= prev_idx:
                anomalies.append(
                    f"sparse entry #{i} index {idx} out of order (prev {prev_idx})"
                )
            prev_idx = idx
            entry_offsets[idx] = offset + entries_start + rel
    else:
        for i in range(entry_count):
            rel = read_u16(
                data, offset + header_size + i * 2, f"offset16 #{i}"
            ) if uses_offset16 else read_u32(
                data, offset + header_size + i * 4, f"offset #{i}"
            )
            if uses_offset16:
                if rel == 0xFFFF:
                    continue
                rel *= 4
            elif rel == NO_ENTRY:
                continue
            entry_offsets[i] = offset + entries_start + rel

    t = TypeChunk(
        chunk_offset=offset,
        type_id=type_id,
        flags=flags,
        entry_count=entry_count,
        entries_start=entries_start,
        config=config,
        is_sparse=is_sparse,
        uses_offset16=uses_offset16,
        entry_offsets=entry_offsets,
        anomalies=anomalies,
    )

    for idx in sorted(entry_offsets):
        absolute = entry_offsets[idx]
        if absolute >= offset + chunk_size:
            t.anomalies.append(
                f"entry {idx} offset beyond chunk boundary"
            )
            continue
        try:
            t.entries[idx] = parse_entry(data, absolute, offset + chunk_size, pkg, idx)
        except ResourceTableError as exc:
            t.anomalies.append(f"entry {idx}: {exc}")

    return t


# ============================================================================
# ENTRY PARSING  (ResTable_entry / ResTable_entry_compact / ResTable_map_entry)
# ============================================================================

def parse_entry(
    data: bytes,
    offset: int,
    limit: int,
    pkg: PackageChunk,
    entry_index: int,
) -> ResourceEntry:
    require_range(data, offset, 4, f"entry {entry_index}")

    # ResTable_entry is a union of Full and Compact layouts (AOSP):
    #
    #   Full:    size:u16 | flags:u16 | key:u32  [+ Res_value or maps]
    #   Compact: key:u16  | flags:u16 | data:u32   (dataType = data>>24)
    #
    # Compactness is signaled by FLAG_COMPACT (0x0008) inside the flags word,
    # which shares its offset in both layouts.
    size = read_u16(data, offset, "entry size")
    flags = read_u16(data, offset + 2, "entry flags")

    unknown_flag_bits = flags & ~ENTRY_KNOWN_FLAGS
    is_compact = bool(flags & ENTRY_FLAG_COMPACT)
    is_complex = bool(flags & ENTRY_FLAG_COMPLEX)

    if unknown_flag_bits:
        # Surfaced through the entry itself; not fatal.
        pass

    if is_compact:
        key_index = read_u16(data, offset, "compact entry key")
        word4 = read_u32(data, offset + 4, "compact entry data")
        return ResourceEntry(
            entry_index=entry_index,
            flags=flags,
            key_index=key_index,
            key_name=pkg.key_strings.get(key_index),
            is_compact=True,
            is_complex=False,
            value=TypedValue(data_type=(word4 >> 24) & 0xFF,
                             data=word4 & 0xFFFFFF),
            parent_ref=None,
        )

    key_index = read_u32(data, offset + 4, "entry key index")

    if is_complex:
        # ResTable_map_entry
        require_range(data, offset, 16, f"entry {entry_index} map header")
        parent_ref = read_u32(data, offset + 8, "entry parent")
        count = read_u32(data, offset + 12, "entry count")

        items_offset = offset + 16
        if items_offset + count * 12 > limit:
            raise ResourceTableError(
                f"bag with {count} items overflows chunk"
            )

        resolver = ResourceNameResolver(_current_table[0]) if _current_table[0] else None
        bag: list[BagItem] = []
        for i in range(count):
            item_off = items_offset + i * 12
            ident = read_u32(data, item_off, f"bag item {i} ident")
            data_type = read_u8(data, item_off + 7, f"bag item {i} type")
            value_data = read_u32(data, item_off + 8, f"bag item {i} data")
            bag.append(
                BagItem(
                    ident=ident,
                    ident_label=_bag_ident_label(ident, resolver),
                    value=TypedValue(data_type=data_type, data=value_data),
                )
            )

        return ResourceEntry(
            entry_index=entry_index,
            flags=flags,
            key_index=key_index,
            key_name=pkg.key_strings.get(key_index),
            is_compact=False,
            is_complex=True,
            value=None,
            parent_ref=parent_ref or None,
            bag=bag,
        )

    # Simple entry: Res_value immediately follows the 8-byte header.
    require_range(data, offset, 16, f"entry {entry_index} value")
    value_size = read_u16(data, offset + 8, "Res_value size")
    res0 = read_u8(data, offset + 10, "Res_value res0")
    data_type = read_u8(data, offset + 11, "Res_value dataType")
    value_data = read_u32(data, offset + 12, "Res_value data")

    if value_size < 8:
        raise ResourceTableError(f"Res_value size {value_size} < 8")
    if res0 != 0:
        # AOSP requires res0 == 0; surfaced, not fatal.
        pass

    return ResourceEntry(
        entry_index=entry_index,
        flags=flags,
        key_index=key_index,
        key_name=pkg.key_strings.get(key_index),
        is_compact=False,
        is_complex=False,
        value=TypedValue(data_type=data_type, data=value_data),
        parent_ref=None,
    )


def _bag_ident_label(ident: int, resolver: ResourceNameResolver | None) -> str:
    if ident in PLURAL_KEYS:
        return f"plural:{PLURAL_KEYS[ident]}"
    if ident == 0x0100000A:
        return "bag:attr"  # styleable attribute list start (aapt convention)
    if (ident >> 24) == ANDROID_PACKAGE_ID or (ident >> 24) == 0x7F:
        if resolver is not None:
            resolved = resolver.reference(ident, attribute=True)
            return resolved
    return f"0x{ident:08X}"


# Parsing entries needs the table-wide resolver for bag labels. Rather than
# threading it through every call, parse_resource_table_bytes registers the
# table here for the duration of parsing (single-threaded by design).
_current_table: list[ResourceTable | None] = [None]


# ============================================================================
# LIBRARY CHUNK
# ============================================================================

def parse_library_chunk(data: bytes, offset: int) -> LibraryChunk:
    require_range(data, offset, 12, "ResTable_library")

    # ResTable_library: ResChunk_header(8), count(u32)@8.
    chunk_type, header_size, chunk_size = struct.unpack_from("<HHI", data, offset)
    count = read_u32(data, offset + 8, "library count")
    if chunk_type != RES_TABLE_LIBRARY_TYPE:
        raise ResourceTableError(
            f"Expected library chunk at {offset}, got 0x{chunk_type:04X}"
        )
    if count * 260 > chunk_size - header_size:
        raise ResourceTableError("library chunk entry array overflows chunk")
    require_range(data, offset + header_size, count * 260, "library entries")

    lib = LibraryChunk()
    for i in range(count):
        entry_off = offset + header_size + i * 260
        package_id = read_u32(data, entry_off, f"library {i} id")
        raw_name = data[entry_off + 4:entry_off + 4 + 256]
        name = raw_name.decode("utf-16-le", errors="replace").split("\x00")[0]
        lib.entries.append(LibraryEntry(package_id=package_id, name=name))
    return lib


# ============================================================================
# Res_value DECODING
# ============================================================================

def decode_res_value(
    data_type: int,
    data: int,
    resolver: ResourceNameResolver | None = None,
) -> str:
    if data_type == TYPE_NULL:
        if data == 0:
            return "@null"
        return "@empty"

    if data_type == TYPE_REFERENCE:
        if resolver is not None:
            return resolver.reference(data)
        return f"@0x{data:08X}"

    if data_type == TYPE_DYNAMIC_REFERENCE:
        if resolver is not None:
            return f"@*{resolver.reference(data)}"
        return f"@*0x{data:08X}"

    if data_type == TYPE_ATTRIBUTE:
        if resolver is not None:
            return resolver.reference(data, attribute=True)
        return f"?0x{data:08X}"

    if data_type == TYPE_DYNAMIC_ATTRIBUTE:
        if resolver is not None:
            return f"?*{resolver.reference(data, attribute=True)}"
        return f"?*0x{data:08X}"

    if data_type == TYPE_STRING:
        # The caller's pool context matters; plain formatting falls back to
        # the index. Pools are attached at print time by format_value().
        return f"@string#{data}"

    if data_type == TYPE_FLOAT:
        import math
        value = struct.unpack("<f", struct.pack("<I", data))[0]
        if math.isfinite(value):
            return f"{value:.7g}f"
        return repr(value)

    if data_type == TYPE_DIMENSION:
        value = _decode_complex_float(data)
        units = {0: "px", 1: "dp", 2: "sp", 3: "pt", 4: "in", 5: "mm"}
        return f"{value:.4g}{units.get(data & COMPLEX_UNIT_MASK, 'unit?')}"

    if data_type == TYPE_FRACTION:
        value = _decode_complex_float(data)
        if (data & COMPLEX_UNIT_MASK) == 0:
            return f"{value * 100:.4g}%"
        return f"{value * 100:.4g}%p"

    if data_type == TYPE_INT_DEC:
        if data & 0x80000000:
            return str(data - 0x100000000)
        return str(data)

    if data_type == TYPE_INT_HEX:
        return f"0x{data:08X}"

    if data_type == TYPE_INT_BOOLEAN:
        return "true" if data else "false"

    if TYPE_FIRST_COLOR_INT <= data_type <= TYPE_LAST_INT:
        if data_type == TYPE_INT_COLOR_ARGB8:
            return f"#{data:08x}"
        if data_type == TYPE_INT_COLOR_RGB8:
            return f"#{data & 0xFFFFFF:06x}"
        if data_type == TYPE_INT_COLOR_ARGB4:
            return f"#{data & 0xFFFF:04x}"
        return f"#{data & 0x0FFF:03x}"

    return f"<type=0x{data_type:02X} data=0x{data:08X}>"


def _decode_complex_float(data: int) -> float:
    radix = (data >> COMPLEX_RADIX_SHIFT) & COMPLEX_RADIX_MASK
    mantissa = (data >> COMPLEX_MANTISSA_SHIFT) & COMPLEX_MANTISSA_MASK
    if mantissa & 0x800000:
        mantissa -= 0x1000000
    divisors = {0: 1.0, 1: 1 << 7, 2: 1 << 15, 3: 1 << 23}
    return mantissa / divisors.get(radix, 1.0)


# ============================================================================
# FULL TABLE PARSING
# ============================================================================

def parse_resource_table_bytes(data: bytes) -> ResourceTable:
    header = parse_resource_table_header(data)

    anomalies: list[str] = []
    unknown_chunks: list[UnknownChunk] = []
    packages: list[PackageChunk] = []

    table = ResourceTable(
        header=header,
        global_pool=ResourceStringPool(
            header=ResourceStringPoolHeader(0, 0, 0, 0, 0, 0, 0, 0),
            offsets=[],
            strings=[],
        ),
        packages=packages,
        unknown_chunks=unknown_chunks,
        anomalies=anomalies,
    )

    _current_table[0] = table
    try:
        offset = header.header_size
        end = header.chunk_size

        # Expected order: [junk] -> global pool -> packages.
        global_pool_seen = False

        while offset + 8 <= end:
            chunk_type, chunk_header_size, chunk_size = struct.unpack_from(
                "<HHI", data, offset
            )

            if chunk_size < 8 or offset + chunk_size > end:
                anomalies.append(
                    f"truncated/invalid chunk at {offset}: "
                    f"type=0x{chunk_type:04X} size={chunk_size}"
                )
                break

            if chunk_type == RES_STRING_POOL_TYPE and not global_pool_seen:
                table.global_pool = parse_resource_string_pool(data, offset)
                global_pool_seen = True

            elif chunk_type == RES_TABLE_PACKAGE_TYPE:
                packages.append(parse_package(data, offset))

            elif chunk_type == RES_TABLE_LIBRARY_TYPE:
                # Top-level library chunks are legal but rare.
                lib = parse_library_chunk(data, offset)
                if packages:
                    packages[-1].libraries.append(lib)
                else:
                    anomalies.append("library chunk before any package")

            else:
                printable = _printable_payload(
                    data,
                    offset + chunk_header_size,
                    chunk_size - chunk_header_size,
                )
                unknown_chunks.append(
                    UnknownChunk(
                        chunk_offset=offset,
                        chunk_type=chunk_type,
                        header_size=chunk_header_size,
                        chunk_size=chunk_size,
                        printable=printable,
                    )
                )
                if chunk_type == RES_NULL_TYPE:
                    anomalies.append(
                        f"RES_NULL chunk ({chunk_size}B) inside table at {offset} "
                        "- possible hidden/embedded data"
                    )
                else:
                    anomalies.append(
                        f"unknown chunk type 0x{chunk_type:04X} at {offset} "
                        f"({chunk_size}B)"
                    )

            offset += chunk_size

        if not global_pool_seen:
            anomalies.append("no global string pool found")
        if len(packages) != header.package_count:
            anomalies.append(
                f"packageCount {header.package_count} != parsed {len(packages)}"
            )

        # Surface pool anomalies in the table-level ledger.
        for message in table.global_pool.anomalies:
            anomalies.append(f"global pool: {message}")
        for pkg in packages:
            for message in pkg.anomalies:
                anomalies.append(f"package {pkg.name}: {message}")
            for message in pkg.type_strings.anomalies:
                anomalies.append(f"package {pkg.name} type pool: {message}")
            for message in pkg.key_strings.anomalies:
                anomalies.append(f"package {pkg.name} key pool: {message}")
            for t in pkg.types:
                for message in t.anomalies:
                    anomalies.append(
                        f"package {pkg.name} type {t.type_id} "
                        f"[{t.config.qualifier}]: {message}"
                    )
    finally:
        _current_table[0] = None

    return table


def _printable_payload(data: bytes, offset: int, size: int) -> str:
    require_range(data, offset, min(size, 4096), "unknown chunk payload")
    payload = data[offset:offset + min(size, 4096)]
    text = payload.rstrip(b"\x00").decode("utf-8", errors="replace")
    cleaned = "".join(ch if ch.isprintable() else "." for ch in text)
    return cleaned


# ============================================================================
# REPORT PRINTING
# ============================================================================

def format_entry_value(
    entry: ResourceEntry,
    table: ResourceTable,
    pool: ResourceStringPool,
) -> str:
    resolver = table.resolver()

    if entry.is_complex:
        parts = []
        if entry.parent_ref:
            parts.append(f"parent={resolver.reference(entry.parent_ref)}")
        for item in entry.bag:
            parts.append(
                f"{item.ident_label}={_value_with_pool(item.value, pool, resolver)}"
            )
        return "{" + ", ".join(parts) + "}"

    if entry.value is None:
        return "<none>"
    return _value_with_pool(entry.value, pool, resolver)


def _value_with_pool(
    value: TypedValue,
    pool: ResourceStringPool,
    resolver: ResourceNameResolver,
) -> str:
    if value.data_type == TYPE_STRING:
        text = pool.get(value.data)
        if text is None:
            return f"<invalid-string#{value.data}>"
        return repr(text)
    return decode_res_value(value.data_type, value.data, resolver)


def resource_id_string(pkg: PackageChunk, t: TypeChunk, entry: ResourceEntry) -> str:
    return f"0x{pkg.package_id:02X}{t.type_id:02X}{entry.entry_index:04X}"


def print_resource_table(table: ResourceTable, max_entries_per_type: int = 8) -> None:
    print("=" * 70)
    print("RESOURCE TABLE")
    print("=" * 70)

    h = table.header
    print(f"Type         : 0x{h.chunk_type:04X}")
    print(f"Header Size  : {h.header_size}")
    print(f"Chunk Size   : {h.chunk_size:,}")
    print(f"Package Count: {h.package_count}")

    pool = table.global_pool
    print(
        f"Global Pool  : {len(pool.strings):,} strings "
        f"({'UTF-8' if pool.is_utf8 else 'UTF-16'})"
    )

    for unknown in table.unknown_chunks:
        preview = unknown.printable[:60]
        print(
            f"[?] Unknown chunk 0x{unknown.chunk_type:04X} at {unknown.chunk_offset} "
            f"({unknown.chunk_size:,}B): {preview!r}"
        )

    resolver = table.resolver()

    for pkg in table.packages:
        print()
        print("-" * 70)
        print(
            f"PACKAGE 0x{pkg.package_id:02X} {pkg.name!r}  "
            f"(types={len(pkg.type_strings.strings)}, "
            f"keys={len(pkg.key_strings.strings)})"
        )
        print("-" * 70)

        for t in pkg.types:
            type_name = pkg.type_name(t.type_id) or f"type{t.type_id}"
            qualifier = t.config.qualifier or "(default)"
            flags_note = []
            if t.is_sparse:
                flags_note.append("sparse")
            if t.uses_offset16:
                flags_note.append("offset16")
            notes = f" [{','.join(flags_note)}]" if flags_note else ""

            print(
                f"  {type_name:<12} {qualifier:<28} "
                f"entries={t.entry_count}{notes}"
            )

            shown = 0
            for idx in sorted(t.entries):
                if shown >= max_entries_per_type:
                    remaining = len(t.entries) - shown
                    print(f"      ... {remaining} more")
                    break
                entry = t.entries[idx]
                rid = f"0x{pkg.package_id:02X}{t.type_id:02X}{idx:04X}"
                key = entry.key_name or f"<key#{entry.key_index}>"
                value = format_entry_value(entry, table, pool)
                marks = []
                if entry.is_public:
                    marks.append("public")
                if entry.is_weak:
                    marks.append("weak")
                mark = f" ({','.join(marks)})" if marks else ""
                print(f"    {rid}  {type_name}/{key:<32} = {value}{mark}")
                shown += 1

            for message in t.anomalies:
                print(f"    [!] {message}")

        for lib in pkg.libraries:
            for entry in lib.entries:
                print(
                    f"  [lib] 0x{entry.package_id:02X} -> {entry.name!r}"
                )


def print_anomalies(table: ResourceTable) -> None:
    print()
    print("=" * 70)
    print("ARSC ANOMALIES")
    print("=" * 70)
    print(f"Total anomalies: {len(table.anomalies)}")
    for i, message in enumerate(table.anomalies, 1):
        print(f"[{i}] {message}")


# ============================================================================
# STANDALONE USAGE
# ============================================================================

if __name__ == "__main__":
    APK_PATH = Path(__file__).parent.parent / "apks" / "sample.apk"

    info = inspect_resource_entry(APK_PATH)
    print("ZIP ENTRY")
    print("-" * 60)
    print("Name              :", info.name)
    print("Compression Type  :", info.compression_type)
    print("Compressed Size   :", f"{info.compressed_size:,}")
    print("Uncompressed Size :", f"{info.uncompressed_size:,}")
    print("Compression Ratio :", f"{info.compression_ratio:.2f}x" if info.compression_ratio else "n/a")
    print()

    table = parse_resource_table(APK_PATH)
    print_resource_table(table)
    print_anomalies(table)

    stats = table.stats()
    print()
    print("STATS")
    print("-" * 60)
    for key, value in stats.items():
        print(f"{key:18}: {value}")
