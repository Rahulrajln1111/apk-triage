"""
Smali reconstruction engine (Phase 5 core).

Reconstructs baksmali-style smali text directly from the Phase-4 DEX parser's
byte-level structures - no Apktool/JADX/Androguard. Everything derives from
the AOSP "Dalvik bytecode" + "instruction formats" specifications:

- full opcode mnemonic table (0x00-0xFF, incl. v038 invoke-polymorphic/custom)
- format-aware operand decoding for all 33 formats (10x..51l)
- two-pass emission: pass 1 collects branch targets / switch & array payloads,
  pass 2 renders instructions with resolved :labels
- payload pseudo-instructions (packed/sparse-switch, fill-array-data)
- try/catch ranges with .catch / .catchall annotations
- baksmali-style class emission (.class/.super/.field/.method)
- call-graph extraction from invoke instructions with manifest entry points
  and "never called" (dead-method candidate) detection
- Mermaid / Graphviz DOT export for visualization

UNIT DISCIPLINE (bug class caught during development):
  Branch offsets in the wire format are in 16-bit CODE UNITS. Every target
  this module stores is already converted to BYTE offsets relative to the
  start of the method's insns array. Switch payload targets are relative to
  the SWITCH OPCODE (not the payload) and are patched to absolute byte
  addresses inside decode_method_insns.

Design constraints honored from earlier phases:
- no unbounded reads; everything works on bytes already parsed by tools.dex
- hostile input degrades to <bad-...> placeholders, never crashes
- LLM-facing output keeps resource IDs as hex AND comments the resolved
  @type/name (the SmaliLLM hallucination fix)
"""

from __future__ import annotations

import re
import struct
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from tools.dex import (
    CodeItem,
    DexFile,
    NO_INDEX,
    TryItem,
    instruction_width,
)


class SmaliError(ValueError):
    """Raised when smali reconstruction is impossible (not merely degraded)."""


# ============================================================================
# OPCODE MNEMONICS (AOSP dalvik-bytecode summary table, verified 2026-09-28)
# ============================================================================

OPCODE_NAMES: dict[int, str] = {
    0x00: "nop",
    0x01: "move", 0x02: "move/from16", 0x03: "move/16",
    0x04: "move-wide", 0x05: "move-wide/from16", 0x06: "move-wide/16",
    0x07: "move-object", 0x08: "move-object/from16", 0x09: "move-object/16",
    0x0A: "move-result", 0x0B: "move-result-wide", 0x0C: "move-result-object",
    0x0D: "move-exception",
    0x0E: "return-void", 0x0F: "return", 0x10: "return-wide",
    0x11: "return-object",
    0x12: "const/4", 0x13: "const/16", 0x14: "const", 0x15: "const/high16",
    0x16: "const-wide/16", 0x17: "const-wide/32", 0x18: "const-wide",
    0x19: "const-wide/high16",
    0x1A: "const-string", 0x1B: "const-string/jumbo",
    0x1C: "const-class", 0x1D: "monitor-enter", 0x1E: "monitor-exit",
    0x1F: "check-cast", 0x20: "instance-of",
    0x21: "array-length", 0x22: "new-instance", 0x23: "new-array",
    0x24: "filled-new-array", 0x25: "filled-new-array/range",
    0x26: "fill-array-data", 0x27: "throw",
    0x28: "goto", 0x29: "goto/16", 0x2A: "goto/32",
    0x2B: "packed-switch", 0x2C: "sparse-switch",
    0x2D: "cmpl-float", 0x2E: "cmpg-float", 0x2F: "cmpl-double",
    0x30: "cmpg-double", 0x31: "cmp-long",
    0x32: "if-eq", 0x33: "if-ne", 0x34: "if-lt",
    0x35: "if-ge", 0x36: "if-gt", 0x37: "if-le",
    0x38: "if-eqz", 0x39: "if-nez", 0x3A: "if-ltz",
    0x3B: "if-gez", 0x3C: "if-gtz", 0x3D: "if-lez",
    # 0x3E-0x43 unused
    0x44: "aget", 0x45: "aget-wide", 0x46: "aget-object", 0x47: "aget-boolean",
    0x48: "aget-byte", 0x49: "aget-char", 0x4A: "aget-short",
    0x4B: "aput", 0x4C: "aput-wide", 0x4D: "aput-object", 0x4E: "aput-boolean",
    0x4F: "aput-byte", 0x50: "aput-char", 0x51: "aput-short",
    0x52: "iget", 0x53: "iget-wide", 0x54: "iget-object", 0x55: "iget-boolean",
    0x56: "iget-byte", 0x57: "iget-char", 0x58: "iget-short",
    0x59: "iput", 0x5A: "iput-wide", 0x5B: "iput-object", 0x5C: "iput-boolean",
    0x5D: "iput-byte", 0x5E: "iput-char", 0x5F: "iput-short",
    0x60: "sget", 0x61: "sget-wide", 0x62: "sget-object", 0x63: "sget-boolean",
    0x64: "sget-byte", 0x65: "sget-char", 0x66: "sget-short",
    0x67: "sput", 0x68: "sput-wide", 0x69: "sput-object", 0x6A: "sput-boolean",
    0x6B: "sput-byte", 0x6C: "sput-char", 0x6D: "sput-short",
    0x6E: "invoke-virtual", 0x6F: "invoke-super", 0x70: "invoke-direct",
    0x71: "invoke-static", 0x72: "invoke-interface",
    # 0x73 unused
    0x74: "invoke-virtual/range", 0x75: "invoke-super/range",
    0x76: "invoke-direct/range", 0x77: "invoke-static/range",
    0x78: "invoke-interface/range",
    # 0x79-0x7A unused
    # 0x7B-0x8F: unop (generated below)
    # 0x90-0xAF: binop (generated below)
    # 0xB0-0xCF: binop/2addr (generated below)
    0xD0: "add-int/lit16", 0xD1: "rsub-int", 0xD2: "mul-int/lit16",
    0xD3: "div-int/lit16", 0xD4: "rem-int/lit16", 0xD5: "and-int/lit16",
    0xD6: "or-int/lit16", 0xD7: "xor-int/lit16",
    0xD8: "add-int/lit8", 0xD9: "rsub-int/lit8", 0xDA: "mul-int/lit8",
    0xDB: "div-int/lit8", 0xDC: "rem-int/lit8", 0xDD: "and-int/lit8",
    0xDE: "or-int/lit8", 0xDF: "xor-int/lit8", 0xE0: "shl-int/lit8",
    0xE1: "shr-int/lit8", 0xE2: "ushr-int/lit8",
    # 0xE3-0xF9 unused
    0xFA: "invoke-polymorphic", 0xFB: "invoke-polymorphic/range",
    0xFC: "invoke-custom", 0xFD: "invoke-custom/range",
    0xFE: "const-method-handle", 0xFF: "const-method-type",
}

_int_names = [n + "-int" for n in "add sub mul div rem and or xor shl shr ushr".split()]
_long_names = [n + "-long" for n in "add sub mul div rem and or xor shl shr ushr".split()]
_float_names = [n + "-float" for n in "add sub mul div rem".split()]
_double_names = [n + "-double" for n in "add sub mul div rem".split()]
for _i, _n in enumerate(_int_names + _long_names + _float_names + _double_names):
    OPCODE_NAMES[0x90 + _i] = _n
    OPCODE_NAMES[0xB0 + _i] = _n + "/2addr"
for _i, _n in enumerate(
    ["neg-int", "not-int", "neg-long", "not-long", "neg-float", "neg-double",
     "int-to-long", "int-to-float", "int-to-double", "long-to-int",
     "long-to-float", "long-to-double", "float-to-int", "float-to-long",
     "float-to-double", "double-to-int", "double-to-long", "double-to-float",
     "int-to-byte", "int-to-char", "int-to-short"]
):
    OPCODE_NAMES[0x7B + _i] = _n


# op -> format id (10x, 12x, ..., 51l). The dispatch core of the decoder.
# Every op 0x00-0xFF is mapped; unused ops map to "10x" (zero operands).
OPCODE_FORMATS: dict[int, str] = {
    0x00: "10x",
    0x01: "12x", 0x02: "22x", 0x03: "32x",
    0x04: "12x", 0x05: "22x", 0x06: "32x",
    0x07: "12x", 0x08: "22x", 0x09: "32x",
    0x0A: "11x", 0x0B: "11x", 0x0C: "11x", 0x0D: "11x",
    0x0E: "10x", 0x0F: "11x", 0x10: "11x", 0x11: "11x",
    0x12: "11n", 0x13: "21s", 0x14: "31i", 0x15: "21h",
    0x16: "21s", 0x17: "31i", 0x18: "51l", 0x19: "21h",
    0x1A: "21c", 0x1B: "31c", 0x1C: "21c",
    0x1D: "11x", 0x1E: "11x", 0x1F: "21c", 0x20: "22c",
    0x21: "12x", 0x22: "21c", 0x23: "22c", 0x24: "35c", 0x25: "3rc",
    0x26: "31t", 0x27: "11x",
    0x28: "10t", 0x29: "20t", 0x2A: "30t", 0x2B: "31t", 0x2C: "31t",
    **{o: "23x" for o in range(0x2D, 0x32)},
    **{o: "22t" for o in range(0x32, 0x38)},
    **{o: "21t" for o in range(0x38, 0x3E)},
    **{o: "10x" for o in range(0x3E, 0x44)},
    **{o: "23x" for o in range(0x44, 0x52)},
    **{o: "22c" for o in range(0x52, 0x60)},
    **{o: "21c" for o in range(0x60, 0x6E)},
    **{o: "35c" for o in range(0x6E, 0x73)},
    0x73: "10x",
    **{o: "3rc" for o in range(0x74, 0x79)},
    **{o: "10x" for o in range(0x79, 0x7B)},
    **{o: "12x" for o in range(0x7B, 0x90)},
    **{o: "23x" for o in range(0x90, 0xB0)},
    **{o: "12x" for o in range(0xB0, 0xD0)},
    **{o: "22s" for o in range(0xD0, 0xD8)},
    **{o: "22b" for o in range(0xD8, 0xE3)},
    **{o: "10x" for o in range(0xE3, 0xFA)},
    0xFA: "45cc", 0xFB: "4rcc", 0xFC: "35c", 0xFD: "3rc",
    0xFE: "21c", 0xFF: "21c",
}

# op -> constant-pool kind for the *c formats ("kind@BBBB").
OPCODE_KINDS: dict[int, str] = {}
OPCODE_KINDS[0x1A] = OPCODE_KINDS[0x1B] = "string"
OPCODE_KINDS[0x1C] = OPCODE_KINDS[0x1F] = OPCODE_KINDS[0x20] = "type"
OPCODE_KINDS[0x22] = OPCODE_KINDS[0x23] = "type"
OPCODE_KINDS[0x24] = OPCODE_KINDS[0x25] = "type"
for _o in range(0x52, 0x60):
    OPCODE_KINDS[_o] = "field"
for _o in range(0x60, 0x6E):
    OPCODE_KINDS[_o] = "field"
for _o in list(range(0x6E, 0x73)) + list(range(0x74, 0x79)):
    OPCODE_KINDS[_o] = "meth"
OPCODE_KINDS[0xFA] = OPCODE_KINDS[0xFB] = "meth"
OPCODE_KINDS[0xFC] = OPCODE_KINDS[0xFD] = "site"
OPCODE_KINDS[0xFE] = "method_handle"
OPCODE_KINDS[0xFF] = "proto"

INVOKE_OPS = {0x6E, 0x6F, 0x70, 0x71, 0x72, 0x74, 0x75, 0x76, 0x77, 0x78,
              0xFA, 0xFB, 0xFC, 0xFD}


# ============================================================================
# SIGNED DECODE HELPERS
# ============================================================================

def _s8(v: int) -> int:
    return v - 0x100 if v >= 0x80 else v


def _s16(v: int) -> int:
    return v - 0x10000 if v >= 0x8000 else v


def _s32(lo: int, hi: int) -> int:
    v = lo | (hi << 16)
    return v - 0x1_0000_0000 if v >= 0x8000_0000 else v


def _s64(units: tuple[int, int, int, int]) -> int:
    v = units[0] | (units[1] << 16) | (units[2] << 32) | (units[3] << 48)
    return v - (1 << 64) if v >= (1 << 63) else v


def _nibble(unit: int, high: bool) -> int:
    return (unit >> 4) & 0xF if high else unit & 0xF


def _s4(v: int) -> int:
    """Signed 4-bit (sign bit is bit 3 of the nibble)."""
    return v - 0x10 if v >= 0x8 else v


def _ah(unit: int) -> int:
    """Field A of B|A|op formats: bits 8-11 (NOT the low nibble!)."""
    return (unit >> 8) & 0xF


def _bh(unit: int) -> int:
    """Field B of B|A|op formats: bits 12-15."""
    return (unit >> 12) & 0xF


# ============================================================================
# DECODED INSTRUCTION MODEL
# ============================================================================

@dataclass
class Insn:
    addr: int                      # BYTE offset within insns (even)
    opcode: int                    # low byte of first code unit
    format: str                    # Dalvik format id, e.g. "21c"
    width: int                     # width in code units
    regs: list[int] = field(default_factory=list)
    lit: int | None = None         # literal (already sign-adjusted)
    index: int | None = None       # constant pool index
    proto_index: int | None = None # 45cc/4rcc second index
    kind: str | None = None        # string/type/field/meth/site/...
    target: int | None = None      # branch offset in BYTES, RELATIVE to this
                                   # instruction (renderers add insn.addr);

    @property
    def name(self) -> str:
        return OPCODE_NAMES.get(self.opcode, f"op_{self.opcode:#04x}")


@dataclass
class PackedSwitchPayload:
    addr: int
    first_key: int
    targets: list[int] = field(default_factory=list)  # absolute BYTE addrs


@dataclass
class SparseSwitchPayload:
    addr: int
    keys: list[int] = field(default_factory=list)
    targets: list[int] = field(default_factory=list)  # absolute BYTE addrs


@dataclass
class ArrayPayload:
    addr: int
    element_width: int
    size: int
    data: bytes = b""


Payload = PackedSwitchPayload | SparseSwitchPayload | ArrayPayload


def decode_instruction(data: bytes, insns_off: int, end: int, pos: int) -> Insn:
    """Decode one instruction at byte offset pos. Format dispatch is total:
    every op 0x00-0xFF maps to a format per OPCODE_FORMATS.

    Branch offsets are stored as RELATIVE BYTE offsets (wire value * 2)."""
    unit0 = struct.unpack_from("<H", data, pos)[0]
    opcode = unit0 & 0xFF
    fmt = OPCODE_FORMATS[opcode]
    width = instruction_width(data, pos, end)
    insn = Insn(addr=pos - insns_off, opcode=opcode, format=fmt, width=width)

    def u1(n: int) -> int:
        return struct.unpack_from("<H", data, pos + n * 2)[0]

    if fmt == "10x":
        pass
    elif fmt == "12x":
        insn.regs = [_ah(unit0), _bh(unit0)]
    elif fmt == "11n":
        insn.regs = [_ah(unit0)]
        insn.lit = _s4(_bh(unit0))
    elif fmt == "11x":
        insn.regs = [(unit0 >> 8) & 0xFF]
    elif fmt == "10t":
        insn.target = _s8((unit0 >> 8) & 0xFF) * 2
    elif fmt == "20t":
        insn.target = _s16(u1(1)) * 2
    elif fmt == "22x":
        insn.regs = [(unit0 >> 8) & 0xFF, u1(1)]
    elif fmt == "21t":
        insn.regs = [(unit0 >> 8) & 0xFF]
        insn.target = _s16(u1(1)) * 2
    elif fmt == "21s":
        insn.regs = [(unit0 >> 8) & 0xFF]
        insn.lit = _s16(u1(1))
    elif fmt == "21h":
        insn.regs = [(unit0 >> 8) & 0xFF]
        insn.lit = _s16(u1(1))  # shift applied at render (op-dependent)
    elif fmt == "21c":
        insn.regs = [(unit0 >> 8) & 0xFF]
        insn.index = u1(1)
        insn.kind = OPCODE_KINDS.get(opcode, "?")
    elif fmt == "23x":
        w = u1(1)
        insn.regs = [(unit0 >> 8) & 0xFF, w & 0xFF, (w >> 8) & 0xFF]
    elif fmt == "22b":
        w = u1(1)
        insn.regs = [(unit0 >> 8) & 0xFF, w & 0xFF]
        insn.lit = _s8(w >> 8)
    elif fmt == "22t":
        insn.regs = [_ah(unit0), _bh(unit0)]
        insn.target = _s16(u1(1)) * 2
    elif fmt == "22s":
        insn.regs = [_ah(unit0), _bh(unit0)]
        insn.lit = _s16(u1(1))
    elif fmt == "22c":
        insn.regs = [_ah(unit0), _bh(unit0)]
        insn.index = u1(1)
        insn.kind = OPCODE_KINDS.get(opcode, "?")
    elif fmt == "30t":
        insn.target = _s32(u1(1), u1(2)) * 2
    elif fmt == "31i":
        insn.regs = [(unit0 >> 8) & 0xFF]
        insn.lit = _s32(u1(1), u1(2))
    elif fmt == "31t":
        insn.regs = [(unit0 >> 8) & 0xFF]
        insn.target = _s32(u1(1), u1(2)) * 2
    elif fmt == "31c":
        insn.regs = [(unit0 >> 8) & 0xFF]
        insn.index = u1(1) | (u1(2) << 16)
        insn.kind = "string"
    elif fmt == "32x":
        insn.regs = [u1(1), u1(2)]
    elif fmt in ("35c", "45cc"):
        # unit0 = A|G|op (spec): A = bits 12-15 = register COUNT,
        # G = bits 8-11 = 5th register. (Verified against real DEX:
        # invoke-direct {v0}, Object;-><init>()V wires as 0x1070.)
        count = (unit0 >> 12) & 0xF
        g = (unit0 >> 8) & 0xF
        insn.index = u1(1)
        unit2 = u1(2)
        c = unit2 & 0xF
        d = (unit2 >> 4) & 0xF
        e = (unit2 >> 8) & 0xF
        f = (unit2 >> 12) & 0xF
        insn.regs = [c, d, e, f, g][:count]
        insn.kind = OPCODE_KINDS.get(opcode, "?")
        if fmt == "45cc":
            insn.proto_index = u1(3)
    elif fmt in ("3rc", "4rcc"):
        # unit0 = AA|op, unit1 = BBBB (index), unit2 = CCCC (first reg)
        insn.regs = [(unit0 >> 8) & 0xFF, u1(2)]  # [count, first_reg]
        insn.index = u1(1)
        insn.kind = OPCODE_KINDS.get(opcode, "?")
        if fmt == "4rcc":
            insn.proto_index = u1(3)
    elif fmt == "51l":
        insn.regs = [(unit0 >> 8) & 0xFF]
        insn.lit = _s64((u1(1), u1(2), u1(3), u1(4)))
    else:  # pragma: no cover - table is total
        raise SmaliError(f"unhandled format {fmt}")

    return insn


def decode_payload(data: bytes, pos: int, end: int, rel_addr: int) -> Payload:
    """Decode a nop-tagged data payload at file offset pos.

    rel_addr is the payload's address relative to the start of insns - the
    single key space used by Insn.addr, try ranges and labels.
    Switch targets are kept RAW (code units, relative to the switch opcode);
    decode_method_insns patches them to absolute insns-relative byte
    addresses once it knows which switch instruction references this payload."""
    ident = struct.unpack_from("<H", data, pos)[0]
    size = struct.unpack_from("<H", data, pos + 2)[0]
    available = end - pos
    if ident == 0x0100:
        if 8 + size * 4 > available:
            raise SmaliError(
                f"packed-switch payload at {pos:#x} declares {size} "
                f"targets, {available} bytes remain")
        first_key = _s32(*struct.unpack_from("<HH", data, pos + 4))
        raw = [_s32(*struct.unpack_from("<HH", data, pos + 8 + i * 4))
               for i in range(size)]
        return PackedSwitchPayload(rel_addr, first_key, raw)
    if ident == 0x0200:
        if 4 + size * 8 > available:
            raise SmaliError(
                f"sparse-switch payload at {pos:#x} declares {size} "
                f"entries, {available} bytes remain")
        keys = [_s32(*struct.unpack_from("<HH", data, pos + 4 + i * 4))
                for i in range(size)]
        raw = [_s32(*struct.unpack_from("<HH", data, pos + 4 + size * 4 + i * 4))
               for i in range(size)]
        return SparseSwitchPayload(rel_addr, keys, raw)
    if ident == 0x0300:
        element_width = struct.unpack_from("<H", data, pos + 2)[0]
        count = struct.unpack_from("<I", data, pos + 4)[0]
        if element_width == 0:
            raise SmaliError(f"fill-array-data at {pos:#x}: zero element width")
        # Never trust the declared count: only bytes actually present in the
        # code item may be rendered (a crafted count used to allocate the
        # declared size here - gigabytes - and raise MemoryError).
        payload_bytes = element_width * count
        if payload_bytes > available - 8:
            raise SmaliError(
                f"fill-array-data at {pos:#x} declares {payload_bytes:,} "
                f"bytes ({count} x {element_width}), "
                f"{available - 8:,} remain")
        blob = data[pos + 8: pos + 8 + payload_bytes]
        return ArrayPayload(rel_addr, element_width, count, blob)
    raise SmaliError(f"not a payload at {pos:#x}")


def decode_method_insns(data: bytes, code: CodeItem) -> tuple[list[Insn], dict[int, Payload]]:
    """Walk a method's insns linearly. Returns (instructions, payloads).

    Linear walking is exactly what the spec allows for: payload data must
    never appear in normal flow, so every non-payload position is a real
    instruction boundary. After the walk, switch payloads get their targets
    patched to absolute byte addresses (relative to the switch opcode)."""
    insns: list[Insn] = []
    payloads: dict[int, Payload] = {}
    end = code.insns_off + code.insns_size * 2
    pos = code.insns_off
    while pos + 2 <= end:
        unit = struct.unpack_from("<H", data, pos)[0]
        if unit in (0x0100, 0x0200, 0x0300):
            try:
                payloads[pos - code.insns_off] = decode_payload(
                    data, pos, end, pos - code.insns_off)
            except (SmaliError, struct.error):
                pass  # hostile data: keep walking by the width table
            pos += instruction_width(data, pos, end) * 2
            continue
        insns.append(decode_instruction(data, code.insns_off, end, pos))
        pos += insns[-1].width * 2

    # Patch switch payload targets: wire targets are code-unit offsets
    # relative to the SWITCH opcode address.
    for insn in insns:
        if insn.name in ("packed-switch", "sparse-switch") and insn.target is not None:
            payload = payloads.get(insn.addr + insn.target)
            if isinstance(payload, (PackedSwitchPayload, SparseSwitchPayload)):
                payload.targets = [insn.addr + t * 2 for t in payload.targets]
    return insns, payloads


def decode_tries(data: bytes, code: CodeItem) -> list[tuple[TryItem, list, int | None]]:
    """Parse try items + encoded_catch_handler_list.

    Returns [(try_item, [(type_idx, handler_addr_bytes)...], catch_all_bytes)].
    Handler addresses are converted from code units to BYTE offsets from the
    start of insns. type_idx is None for catch-all entries.

    LAYOUT FACT the spec makes easy to miss: the handler list lives AFTER
    the try table and is NOT bounded by it - bounding uleb reads by the
    try-table end crashes on real code (found on sample.apk)."""
    if not code.tries_size:
        return []
    pos = code.insns_off + code.insns_size * 2
    if code.insns_size % 2:
        pos += 2
    table_end = pos + code.tries_size * 8
    if table_end > len(data):
        return []

    # handler_off is relative to the start of encoded_catch_handler_list,
    # which follows the try items (no alignment padding). The list starts
    # with a ULEB128 size - NOT a u32 (u32 read skewed every handler offset
    # and crashed on real code).
    list_size, hpos = uleb128(data, table_end, len(data))
    # Each encoded handler is >= 1 byte; cap hostile counts to the space left.
    max_handlers = max(0, len(data) - hpos)
    handlers: dict[int, tuple[list, int | None]] = {}
    try:
        for _ in range(min(list_size, max_handlers)):
            if hpos >= len(data):
                break
            off = hpos - table_end
            hsize, hpos = sleb128(data, hpos, len(data))
            pairs: list = []
            catch_all = None
            n = hsize if hsize > 0 else -hsize
            for _p in range(n):
                tidx, hpos = uleb128(data, hpos, len(data))
                addr, hpos = uleb128(data, hpos, len(data))
                pairs.append((tidx, addr * 2))
            if hsize <= 0:
                catch_all, hpos = uleb128(data, hpos, len(data))
                catch_all *= 2
            handlers[off] = (pairs, catch_all)
    except DexError:
        # hostile/corrupt handler list: return what we decoded so far
        pass

    result = []
    for t in code.tries:
        pairs, catch_all = handlers.get(t.handler_off, ([], None))
        result.append((t, pairs, catch_all))
    return result


from tools.dex import DexError, sleb128, uleb128  # noqa: E402  (used by decode_tries)


# ============================================================================
# RESOLUTION HELPERS
# ============================================================================

def type_str(dex: DexFile, index: int | None) -> str:
    if index is None:
        return "<no-type>"
    desc = dex.type_desc(index)
    return desc if desc else f"<type@{index}>"


def field_ref(dex: DexFile, index: int | None) -> str:
    if index is None:
        return "<no-field>"
    try:
        return dex.field_str(index)
    except Exception:
        return f"<field@{index}>"


def proto_signature(dex: DexFile, proto_idx: int) -> str:
    """Baksmali-style '(II)V' signature (descriptors concatenated)."""
    if not (0 <= proto_idx < len(dex.proto_ids)):
        return "(?)?"
    proto = dex.proto_ids[proto_idx]
    ret = dex.type_desc(proto.return_type_idx) or "?"
    params: list[str] = []
    if proto.parameters_off:
        try:
            count = struct.unpack_from("<I", dex.data, proto.parameters_off)[0]
            for i in range(count):
                tidx = struct.unpack_from(
                    "<H", dex.data, proto.parameters_off + 4 + i * 2)[0]
                params.append(dex.type_desc(tidx) or "?")
        except struct.error:
            return "(?)?"
    return f"({''.join(params)}){ret}"


def method_signature(dex: DexFile, method_idx: int) -> str:
    """'Lcls;->name(sig)ret' with concatenated descriptors."""
    if not (0 <= method_idx < len(dex.method_ids)):
        return f"<method@{method_idx}>"
    m = dex.method_ids[method_idx]
    cls = dex.type_desc(m.class_idx) or "?"
    name = dex.string(m.name_idx) or "?"
    return f"{cls}->{name}{proto_signature(dex, m.proto_idx)}"


def escape_string(value: str) -> str:
    out = value.replace("\\", "\\\\").replace('"', '\\"')
    out = out.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
    return '"' + "".join(
        c if ord(c) >= 0x20 else f"\\u{ord(c):04x}" for c in out
    ) + '"'


# ============================================================================
# RESOURCE-ID CONTEXT (LLM hallucination fix)
# ============================================================================

_RESOURCE_ID_RE = re.compile(r"0x7f[0-9a-fA-F]{6}")


def build_resource_context(apk_path: Path) -> dict[int, str]:
    """Map 0x7fXXXXXX -> '@pkg:type/name' for every entry in resources.arsc."""
    from tools.resource_table import parse_resource_table

    table = parse_resource_table(apk_path)
    out: dict[int, str] = {}
    for pkg in table.packages:
        for t in pkg.types:
            type_name = pkg.type_name(t.type_id)
            for idx, entry in t.entries.items():
                if entry.key_name and type_name:
                    rid = (pkg.package_id << 24) | (t.type_id << 16) | idx
                    out.setdefault(rid, f"@{pkg.name}:{type_name}/{entry.key_name}")
    return out


def annotate_resource_ids(text: str, context: dict[int, str]) -> str:
    """Append '# @pkg:type/name' comments after raw 0x7f... literals."""
    if not context:
        return text

    def _sub(match: re.Match) -> str:
        rid = int(match.group(0), 16)
        name = context.get(rid)
        return f"{match.group(0)}  # {name}" if name else match.group(0)

    return _RESOURCE_ID_RE.sub(_sub, text)


# ============================================================================
# TWO-PASS SMALI EMISSION
# ============================================================================

class MethodSmali:
    """Reconstructed smali for one method body."""

    def __init__(self, dex: DexFile, cls_name: str, method_name: str,
                 access_flags: int, proto_idx: int, code: CodeItem):
        self.dex = dex
        self.cls = cls_name
        self.method_name = method_name
        self.flags = access_flags
        self.proto_idx = proto_idx
        self.code = code

    # ------------------------------------------------------------------
    def _resolve_kind(self, insn: Insn) -> str:
        dex, kind, idx = self.dex, insn.kind, insn.index
        if kind == "string":
            value = dex.string(idx) if idx is not None else None
            return escape_string(value) if value is not None else f"<string@{idx}>"
        if kind == "type":
            return type_str(dex, idx)
        if kind == "field":
            return field_ref(dex, idx)
        if kind == "meth":
            return method_signature(dex, idx) if idx is not None else "<no-method>"
        if kind == "site":
            return f"call_site@{idx}"
        if kind == "method_handle":
            return f"method_handle@{idx}"
        if kind == "proto":
            return proto_signature(dex, idx) if idx is not None else "?"
        return f"@{idx}"

    def _lit_str(self, insn: Insn) -> str:
        if insn.format == "21h":
            if insn.opcode == 0x19:  # const-wide/high16: value = B << 48
                return f"0x{(insn.lit << 48) & 0xFFFFFFFFFFFFFFFF:x}L"
            return f"0x{(insn.lit << 16) & 0xFFFFFFFF:x}"
        if insn.lit is None:
            return "?"
        return hex(insn.lit)

    # ------------------------------------------------------------------
    def _collect_labels(self, insns: list[Insn],
                        payloads: dict[int, Payload]) -> set[int]:
        labels: set[int] = set()
        for insn in insns:
            if insn.target is not None:
                labels.add(insn.addr + insn.target)
        for payload in payloads.values():
            if isinstance(payload, (PackedSwitchPayload, SparseSwitchPayload)):
                labels.update(payload.targets)
        return labels

    def _render_array_data(self, payload: ArrayPayload, lines: list[str]) -> None:
        lines.append(f".array-data {payload.element_width}")
        vals: list[str] = []
        for i in range(payload.size):
            chunk = payload.data[i * payload.element_width:
                                 (i + 1) * payload.element_width]
            vals.append("0x" + chunk[::-1].hex())
            if len(vals) == 8:
                lines.append("    " + " ".join(vals))
                vals = []
        if vals:
            lines.append("    " + " ".join(vals))
        lines.append(".end array-data")

    def _render_switch_data(self, payload, lines: list[str],
                            labels) -> None:
        label = lambda a: f":label_{a:x}"
        if isinstance(payload, PackedSwitchPayload):
            lines.append(f".packed-switch 0x{payload.first_key & 0xFFFFFFFF:x}")
            for tgt in payload.targets:
                lines.append(f"    {label(tgt)}")
            lines.append(".end packed-switch")
        else:
            lines.append(".sparse-switch")
            for k, tgt in zip(payload.keys, payload.targets):
                lines.append(f"    0x{k & 0xFFFFFFFF:x} -> {label(tgt)}")
            lines.append(".end sparse-switch")

    # ------------------------------------------------------------------
    def render(self) -> str:
        dex, code = self.dex, self.code
        insns, payloads = decode_method_insns(dex.data, code)
        labels = self._collect_labels(insns, payloads)
        tries = decode_tries(dex.data, code)
        label = lambda addr: f":label_{addr:x}"

        lines: list[str] = []
        emit = lines.append

        try_starts = {t.start_addr for t, _p, _a in tries}
        try_ends = {t.start_addr + t.insn_count * 2 for t, _p, _a in tries}

        last_insn_end = insns[-1].addr + insns[-1].width * 2 if insns else 0

        for insn in insns:
            a = insn.addr
            if a in try_starts:
                emit(f":try_start_{a:x}")
            if a in labels:
                emit(label(a))
            if a in payloads:
                # in-flow payload reference target (rare, but legal to render)
                emit(f":pswitch_data_{a:x}" if isinstance(payloads[a], PackedSwitchPayload)
                     else f":sswitch_data_{a:x}" if isinstance(payloads[a], SparseSwitchPayload)
                     else f":array_data_{a:x}")

            # .catch annotations anchor at the try start
            for t, pairs, catch_all in tries:
                if t.start_addr == a:
                    end_addr = t.start_addr + t.insn_count * 2
                    for tidx, haddr in pairs:
                        tname = dex.type_desc(tidx) if tidx is not None else None
                        tname = tname or f"<type@{tidx}>"
                        emit(f".catch {tname} {{:try_start_{t.start_addr:x} .. "
                             f":try_end_{end_addr:x}}} {label(haddr)}")
                    if catch_all is not None:
                        emit(f".catchall {{:try_start_{t.start_addr:x} .. "
                             f":try_end_{end_addr:x}}} {label(catch_all)}")

            # payload pseudo-instructions (in-flow position)
            payload = payloads.get(a)
            if payload is not None:
                self._render_switch_data(payload, lines, labels) \
                    if isinstance(payload, (PackedSwitchPayload, SparseSwitchPayload)) \
                    else self._render_array_data(payload, lines)
                continue

            # regular instruction
            name = insn.name
            ops: list[str] = [f"v{r}" for r in insn.regs]

            if name.startswith("invoke"):
                if insn.format in ("3rc", "4rcc"):
                    count, first = insn.regs
                    regtxt = "{v%x .. v%x}" % (first, first + count - 1) if count else "{}"
                    ops = [regtxt, self._resolve_kind(insn)]
                else:
                    ops = ["{" + ", ".join(ops) + "}", self._resolve_kind(insn)]
                if insn.proto_index is not None:
                    ops.append(proto_signature(dex, insn.proto_index))
            elif insn.kind is not None:
                ops.append(self._resolve_kind(insn))
            elif insn.target is not None:
                note = ""
                if name in ("packed-switch", "sparse-switch", "fill-array-data"):
                    tgt = insn.addr + insn.target
                    p = payloads.get(tgt)
                    tag = ("pswitch_data" if isinstance(p, PackedSwitchPayload)
                           else "sswitch_data" if isinstance(p, SparseSwitchPayload)
                           else "array_data" if isinstance(p, ArrayPayload) else None)
                    if tag:
                        note = f"  # {tag}_{tgt:x}"
                ops.append(label(insn.addr + insn.target) + note)
            elif insn.lit is not None:
                ops.append(self._lit_str(insn))

            emit(f"{name}{' ' + ', '.join(ops) if ops else ''}")

            end_addr = a + insn.width * 2
            if end_addr in try_ends:
                emit(f":try_end_{end_addr:x}")

        # trailing payloads live after the last real instruction
        for addr in sorted(payloads):
            if addr < last_insn_end:
                continue
            payload = payloads[addr]
            if addr in labels:
                emit(label(addr))
            if isinstance(payload, (PackedSwitchPayload, SparseSwitchPayload)):
                self._render_switch_data(payload, lines, labels)
            else:
                self._render_array_data(payload, lines)

        if code.tries_size and not tries:
            emit(f"# note: {code.tries_size} try items present but undecodable")

        return "\n".join(lines)


# ============================================================================
# CLASS EMISSION
# ============================================================================

def _class_flags_str(flags: int) -> str:
    from tools.dex import (
        ACC_PUBLIC, ACC_PRIVATE, ACC_PROTECTED, ACC_STATIC, ACC_FINAL,
        ACC_INTERFACE, ACC_ABSTRACT, ACC_SYNTHETIC, ACC_ANNOTATION, ACC_ENUM,
    )
    names = []
    if flags & ACC_PUBLIC: names.append("public")
    if flags & ACC_PRIVATE: names.append("private")
    if flags & ACC_PROTECTED: names.append("protected")
    if flags & ACC_STATIC: names.append("static")
    if flags & ACC_FINAL: names.append("final")
    if flags & ACC_INTERFACE: names.append("interface")
    if flags & ACC_ABSTRACT: names.append("abstract")
    if flags & ACC_SYNTHETIC: names.append("synthetic")
    if flags & ACC_ANNOTATION: names.append("annotation")
    if flags & ACC_ENUM: names.append("enum")
    return " ".join(names)


def _method_flags_str(flags: int) -> str:
    from tools.dex import (
        ACC_PUBLIC, ACC_PRIVATE, ACC_PROTECTED, ACC_STATIC, ACC_FINAL,
        ACC_SYNCHRONIZED, ACC_NATIVE, ACC_ABSTRACT, ACC_STRICT,
        ACC_SYNTHETIC, ACC_CONSTRUCTOR, ACC_DECLARED_SYNCHRONIZED,
    )
    names = []
    if flags & ACC_PUBLIC: names.append("public")
    if flags & ACC_PRIVATE: names.append("private")
    if flags & ACC_PROTECTED: names.append("protected")
    if flags & ACC_STATIC: names.append("static")
    if flags & ACC_FINAL: names.append("final")
    if flags & ACC_SYNCHRONIZED: names.append("synchronized")
    if flags & ACC_NATIVE: names.append("native")
    if flags & ACC_ABSTRACT: names.append("abstract")
    if flags & ACC_STRICT: names.append("strictfp")
    if flags & ACC_SYNTHETIC: names.append("synthetic")
    if flags & ACC_CONSTRUCTOR: names.append("constructor")
    if flags & ACC_DECLARED_SYNCHRONIZED: names.append("declared-synchronized")
    return " ".join(names)


def _field_flags_str(flags: int) -> str:
    from tools.dex import (
        ACC_PUBLIC, ACC_PRIVATE, ACC_PROTECTED, ACC_STATIC, ACC_FINAL,
        ACC_VOLATILE, ACC_TRANSIENT, ACC_SYNTHETIC, ACC_ENUM,
    )
    names = []
    if flags & ACC_PUBLIC: names.append("public")
    if flags & ACC_PRIVATE: names.append("private")
    if flags & ACC_PROTECTED: names.append("protected")
    if flags & ACC_STATIC: names.append("static")
    if flags & ACC_FINAL: names.append("final")
    if flags & ACC_VOLATILE: names.append("volatile")
    if flags & ACC_TRANSIENT: names.append("transient")
    if flags & ACC_SYNTHETIC: names.append("synthetic")
    if flags & ACC_ENUM: names.append("enum")
    return " ".join(names)


def _render_field(dex: DexFile, f) -> str:
    fid = dex.field_ids[f.field_idx] if 0 <= f.field_idx < len(dex.field_ids) else None
    if fid is None:
        return f".field <bad-field@{f.field_idx}>"
    fname = dex.string(fid.name_idx) or "?"
    ftype = dex.type_desc(fid.type_idx) or "?"
    flags = _field_flags_str(f.access_flags)
    head = " ".join(x for x in (".field", flags, f"{fname}:{ftype}") if x)
    return head


def _render_method(dex: DexFile, cls_name: str, m, lines: list[str],
                   resource_context: dict[int, str] | None) -> None:
    mid = dex.method_ids[m.method_idx] if 0 <= m.method_idx < len(dex.method_ids) else None
    if mid is None:
        lines.append(f".method <bad-method@{m.method_idx}>")
        lines.append(".end method")
        return
    mname = dex.string(mid.name_idx) or "?"
    sig = proto_signature(dex, mid.proto_idx)
    flags = _method_flags_str(m.access_flags)
    lines.append("")
    lines.append(" ".join(
        x for x in (".method", flags, f"{mname}{sig}") if x))

    code = dex.code_for(m)
    if code is None:
        lines.append(".end method")
        return

    lines.append(f"    .registers {code.registers_size}")
    body = MethodSmali(dex, cls_name, mname, m.access_flags, mid.proto_idx, code)
    text = body.render()
    if resource_context:
        text = annotate_resource_ids(text, resource_context)
    for line in text.splitlines():
        lines.append(f"    {line}" if line else "")
    lines.append(".end method")


def render_class(dex: DexFile, cdef,
                 resource_context: dict[int, str] | None = None) -> str:
    """Baksmali-style .smali text for one class_def."""
    name = dex.type_desc(cdef.class_idx) or "<unknown>"
    lines: list[str] = []
    flags = _class_flags_str(cdef.access_flags)
    lines.append(" ".join(x for x in (".class", flags, name) if x))
    if cdef.superclass_idx != NO_INDEX:
        lines.append(f".super {type_str(dex, cdef.superclass_idx)}")
    else:
        lines.append(".super <none>")

    if cdef.source_file_idx != NO_INDEX:
        src = dex.string(cdef.source_file_idx)
        if src:
            lines.append(f'.source "{src}"')

    if cdef.interfaces_off:
        try:
            count = struct.unpack_from("<I", dex.data, cdef.interfaces_off)[0]
            for i in range(count):
                tidx = struct.unpack_from(
                    "<H", dex.data, cdef.interfaces_off + 4 + i * 2)[0]
                lines.append(f".implements {type_str(dex, tidx)}")
        except struct.error:
            lines.append(".implements <bad-interface-list>")

    if cdef.static_fields:
        lines.append("")
        lines.append("# static fields")
        for f in cdef.static_fields:
            lines.append(_render_field(dex, f))
    if cdef.instance_fields:
        lines.append("")
        lines.append("# instance fields")
        for f in cdef.instance_fields:
            lines.append(_render_field(dex, f))

    if cdef.direct_methods:
        lines.append("")
        lines.append("# direct methods")
        for m in cdef.direct_methods:
            _render_method(dex, name, m, lines, resource_context)
    if cdef.virtual_methods:
        lines.append("")
        lines.append("# virtual methods")
        for m in cdef.virtual_methods:
            _render_method(dex, name, m, lines, resource_context)

    return "\n".join(lines)


# ============================================================================
# CALL GRAPH
# ============================================================================

ANDROID_CALLBACKS = {
    "onCreate", "onStart", "onResume", "onPause", "onStop", "onRestart",
    "onDestroy", "onSaveInstanceState", "onRestoreInstanceState",
    "onActivityResult", "onNewIntent", "onConfigurationChanged",
    "onBackPressed", "onRequestPermissionsResult",
    "onCreateOptionsMenu", "onOptionsItemSelected",
    "onCreateView", "onViewCreated", "onAttach", "onDetach",
    "onClick", "onLongClick", "onTouch", "onKey",
    "onReceive", "onHandleIntent", "onStartCommand", "onBind", "onUnbind",
    "onServiceConnected", "onServiceDisconnected",
    "run", "call", "accept", "apply", "get", "compare", "onChanged",
    "doInBackground", "onPreExecute", "onProgressUpdate", "onPostExecute",
    "handleMessage", "main", "toString", "equals", "hashCode",
    "onDraw", "onMeasure", "onLayout", "onTouchEvent", "onScroll",
    "read", "write", "close", "flush", "visit", "describeContents",
}

ACC_ABSTRACT_F = 0x400
ACC_NATIVE_F = 0x100


@dataclass
class MethodInfo:
    cls: str
    name: str
    sig: str
    flags: int
    calls: list[str] = field(default_factory=list)   # method_signature strings
    called_by: list[str] = field(default_factory=list)
    entry: bool = False
    dead: bool = False
    has_code: bool = True

    @property
    def key(self) -> str:
        return f"{self.cls}->{self.name}{self.sig}"


@dataclass
class ClassInfo:
    name: str
    superclass: str | None
    interfaces: list[str] = field(default_factory=list)
    methods: dict[str, MethodInfo] = field(default_factory=dict)  # key -> MethodInfo
    component: str | None = None   # activity/service/receiver/provider/...


class CallGraph:
    def __init__(self):
        self.classes: dict[str, ClassInfo] = {}
        self.class_edges: Counter = Counter()          # (src, dst) class level
        self.external_refs: Counter = Counter()        # calls leaving the APK
        self.anomalies: list[str] = []

    # ------------------------------------------------------------------
    def stats(self) -> dict:
        methods = sum(len(c.methods) for c in self.classes.values())
        dead = sum(1 for c in self.classes.values()
                   for m in c.methods.values() if m.dead)
        return {
            "classes": len(self.classes),
            "methods": methods,
            "internal_edges": sum(self.class_edges.values()),
            "external_targets": len(self.external_refs),
            "dead_candidates": dead,
        }

    def dead_methods(self) -> list[MethodInfo]:
        return [m for c in self.classes.values()
                for m in c.methods.values() if m.dead]

    def entry_points(self) -> list[MethodInfo]:
        return [m for c in self.classes.values()
                for m in c.methods.values() if m.entry]

    # ------------------------------------------------------------------
    def to_mermaid(self, max_nodes: int = 40) -> str:
        """Class-level graph for the classes involved in the hottest edges."""
        pairs = [(s, d) for (s, d), _n in self.class_edges.most_common(max_nodes)]
        nodes = sorted({n for p in pairs for n in p})
        short = {n: f"C{i}" for i, n in enumerate(nodes)}
        lines = ["graph TD"]
        for n in nodes:
            cls = self.classes.get(n)
            label = n.split("/")[-1].rstrip(";").replace('"', "'")
            if cls is not None and cls.component:
                lines.append(f'    {short[n]}["{label} ({cls.component})"]')
            else:
                lines.append(f'    {short[n]}["{label}"]')
        for (s, d), n in self.class_edges.most_common(max_nodes):
            if (s, d) in pairs:
                lines.append(f"    {short[s]} -->|{n}| {short[d]}")
        return "\n".join(lines)

    def to_dot(self, max_nodes: int = 40) -> str:
        pairs = [(s, d) for (s, d), _n in self.class_edges.most_common(max_nodes)]
        nodes = sorted({n for p in pairs for n in p})
        lines = ["digraph callgraph {", "  rankdir=LR;"]
        for n in nodes:
            label = n.split("/")[-1].rstrip(";").replace('"', "'")
            shape = "box" if n in self.classes else "ellipse"
            lines.append(f'  "{label}" [shape={shape}];')
        for (s, d), n in self.class_edges.most_common(max_nodes):
            if (s, d) in pairs:
                lines.append(
                    f'  "{s.split("/")[-1].rstrip(";")}" -> '
                    f'"{d.split("/")[-1].rstrip(";")}" [label="{n}"];')
        lines.append("}")
        return "\n".join(lines)


def descriptor_from_java(name: str, package: str | None) -> str:
    """'com.foo.Bar' / '.Bar' / 'Bar' / 'Lcom/foo/Bar;' -> 'Lcom/foo/Bar;'."""
    if name.startswith("L") and name.endswith(";"):
        return name
    if name.startswith("."):
        name = (package or "") + name
    elif "." not in name and package:
        name = f"{package}.{name}"
    return "L" + name.replace(".", "/") + ";"


def manifest_components(root) -> list[tuple[str, str]]:
    """[(kind, Ldescriptor;)] from a parsed manifest XmlElement tree."""
    components: list[tuple[str, str]] = []
    package = None
    for a in root.attributes:
        if a.name == "package":
            package = a.value
            break

    stack = [root]
    kinds = {"activity", "activity-alias", "service", "receiver", "provider"}
    while stack:
        node = stack.pop()
        if node.tag in kinds:
            for a in node.attributes:
                if a.name == "name" and a.value:
                    kind = "activity" if node.tag == "activity-alias" else node.tag
                    components.append((kind, descriptor_from_java(a.value, package)))
                elif a.name == "targetActivity" and node.tag == "activity-alias" and a.value:
                    components.append(("activity", descriptor_from_java(a.value, package)))
        stack.extend(node.children)
    return components


def build_call_graph(dex: DexFile,
                     components: list[tuple[str, str]] | None = None) -> CallGraph:
    """Extract class->class edges, entry points and dead-method candidates.

    Entry points: methods of manifest-declared components matching Android
    callback names, plus any override of a known framework callback (the
    framework dispatches into these, so they are never 'dead').
    A method is a dead CANDIDATE when it has code, is not a constructor,
    not a callback override, not abstract/native, and nothing in the DEX
    calls it. Reflection / JNI / framework dispatch can still reach such
    methods - the report says candidate, never 'guaranteed dead'.
    """
    graph = CallGraph()
    comp_map: dict[str, str] = {desc: kind for kind, desc in (components or [])}

    for cdef in dex.class_defs:
        cls_name = dex.type_desc(cdef.class_idx) or f"<class@{cdef.class_idx}>"
        info = ClassInfo(
            name=cls_name,
            superclass=(dex.type_desc(cdef.superclass_idx)
                        if cdef.superclass_idx != NO_INDEX else None),
            component=comp_map.get(cls_name),
        )
        if cdef.interfaces_off:
            try:
                count = struct.unpack_from("<I", dex.data, cdef.interfaces_off)[0]
                for i in range(count):
                    tidx = struct.unpack_from(
                        "<H", dex.data, cdef.interfaces_off + 4 + i * 2)[0]
                    desc = dex.type_desc(tidx)
                    if desc:
                        info.interfaces.append(desc)
            except struct.error:
                graph.anomalies.append(f"{cls_name}: bad interface list")

        for m in cdef.direct_methods + cdef.virtual_methods:
            if not (0 <= m.method_idx < len(dex.method_ids)):
                continue
            mid = dex.method_ids[m.method_idx]
            mname = dex.string(mid.name_idx) or "?"
            sig = proto_signature(dex, mid.proto_idx)
            mi = MethodInfo(cls_name, mname, sig, m.access_flags,
                            has_code=m.code_off != 0)
            info.methods[mi.key] = mi

            code = dex.code_for(m)
            if code is None:
                continue

            data = dex.data
            end = code.insns_off + code.insns_size * 2
            pos = code.insns_off
            while pos + 2 <= end:
                op = struct.unpack_from("<H", data, pos)[0] & 0xFF
                if op in INVOKE_OPS and pos + 6 <= end:
                    method_idx = struct.unpack_from("<H", data, pos + 2)[0]
                    mi.calls.append(method_signature(dex, method_idx))
                pos += instruction_width(data, pos, end) * 2

        graph.classes[cls_name] = info

    # wire edges
    for cls in graph.classes.values():
        for mi in cls.methods.values():
            for target in mi.calls:
                tcls = target.split("->", 1)[0] if "->" in target else target
                callee_cls = graph.classes.get(tcls)
                if callee_cls is None:
                    graph.external_refs[target] += 1
                    continue
                graph.class_edges[(mi.cls, tcls)] += 1
                callee = callee_cls.methods.get(target)
                if callee is not None:
                    callee.called_by.append(mi.key)

    # entry points + dead candidates
    for cls in graph.classes.values():
        for mi in cls.methods.values():
            is_ctor = mi.name in ("<init>", "<clinit>")
            is_callback = mi.name in ANDROID_CALLBACKS
            is_abstract_native = bool(mi.flags & (ACC_ABSTRACT_F | ACC_NATIVE_F))
            if (cls.component is not None or cls.name in comp_map) and is_callback:
                mi.entry = True
            if is_callback and mi.has_code:
                mi.entry = True  # framework dispatch reaches overrides
            if (mi.has_code and not is_ctor and not is_callback
                    and not is_abstract_native and not mi.called_by):
                mi.dead = True

    return graph


def print_call_graph_report(graph: CallGraph, top: int = 12) -> None:
    stats = graph.stats()
    print(f"Classes           : {stats['classes']:,}")
    print(f"Methods           : {stats['methods']:,}")
    print(f"Internal edges    : {stats['internal_edges']:,}")
    print(f"External targets  : {stats['external_targets']:,}")
    print(f"Dead candidates   : {stats['dead_candidates']:,}")

    print()
    print("-" * 70)
    print("TOP CLASS-TO-CLASS EDGES (internal)")
    print("-" * 70)
    for (src, dst), n in graph.class_edges.most_common(top):
        print(f"  {n:4}x  {src} -> {dst}")

    entries = graph.entry_points()
    print()
    print("-" * 70)
    print(f"ENTRY POINTS ({len(entries)})")
    print("-" * 70)
    for mi in entries[:top]:
        print(f"  {mi.key}")

    dead = graph.dead_methods()
    print()
    print("-" * 70)
    print(f"NEVER-CALLED CANDIDATES ({len(dead)}) - reflection/JNI may still reach them")
    print("-" * 70)
    for mi in dead[:top]:
        print(f"  {mi.key}")


# ============================================================================
# FULL EXPORT (all DEX files in an APK)
# ============================================================================

# Characters that are legal inside a type descriptor but hostile in a
# filesystem path (Linux forbids only '/' and NUL, but '<', '>', ':',
# control chars etc. make files unmanageable; Windows interop forbids more).
_BAD_PATH_CHARS = set(chr(c) for c in range(32)) | set('<>:"|?*\\')


def _sanitize_segment(seg: str) -> str:
    seg = "".join("_" if ch in _BAD_PATH_CHARS else ch for ch in seg)
    if seg in ("", ".", ".."):
        seg = "_" + seg          # kills '..' traversal and empty segments
    return seg[:200]             # ext4 NAME_MAX is 255


def class_rel_path(desc: str) -> str:
    """Type descriptor -> relative smali path: Lcom/foo/Bar$1; ->
    com/foo/Bar$1.smali. Never escapes the output root and never
    returns an empty path, whatever the (possibly hostile) descriptor."""
    body = desc[1:-1] if desc.startswith("L") and desc.endswith(";") else desc
    parts = [_sanitize_segment(s) for s in body.split("/")]
    if not parts or not any(parts):
        parts = ["_class"]
    return "/".join(parts) + ".smali"


def export_smali_tree(apk_path: Path, out_dir: Path,
                      resource_context: dict | None = None,
                      progress_every: int = 250) -> dict:
    """Render EVERY class of EVERY real DEX file in the APK.

    Layout: <out>/<dex-stem>/<package path>/<Class>.smali (+ per-dex
    .mmd call graphs). Returns a stats dict; per-class failures are
    logged and counted, never fatal. Memory note: DEX files are parsed
    and released one at a time - the whole APK is never held in RAM."""
    from tools.dex import (DexError, discover_dex_entries,
                           parse_dex_from_apk, sniff_dex_magic)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = {"dex_files": 0, "written": 0, "failed": 0,
             "fake_dex": [], "failures": [], "per_dex": {}}

    try:
        names = discover_dex_entries(apk_path)
    except ValueError as exc:
        stats["failures"].append(f"{exc}")
        return stats

    for name in names:
        try:
            real = sniff_dex_magic(apk_path, name)
        except (DexError, ValueError):
            real = False
        if not real:
            stats["fake_dex"].append(name)
            continue

        try:
            dex = parse_dex_from_apk(apk_path, name)
        except (DexError, ValueError) as exc:
            stats["failures"].append(f"{name}: {exc}")
            stats["failed"] += 1
            continue

        dex_label = _sanitize_segment(
            name.rsplit("/", 1)[-1].rsplit(".", 1)[0]) or "_dex"
        dex_dir = out_dir / dex_label
        dex_dir.mkdir(parents=True, exist_ok=True)

        # call-graph artifact for this DEX (cheap: O(methods))
        graph = build_call_graph(dex)
        (dex_dir / f"_{dex_label}_callgraph.mmd").write_text(
            graph.to_mermaid())

        written = failed = 0
        used: set[str] = set()
        for cdef in dex.class_defs:
            desc = dex.type_desc(cdef.class_idx) or f"<class@{cdef.class_idx}>"
            rel = class_rel_path(desc)
            # distinct descriptors can sanitize to the same path -> dedupe
            if rel in used:
                stem, ext = rel[:-len(".smali")], ".smali"
                n = 2
                while f"{stem}~{n}{ext}" in used:
                    n += 1
                rel = f"{stem}~{n}{ext}"
            used.add(rel)

            try:
                text = render_class(dex, cdef,
                                    resource_context=resource_context)
                dst = dex_dir / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_text(text)
                written += 1
            except Exception as exc:
                failed += 1
                if len(stats["failures"]) < 20:
                    stats["failures"].append(f"{name} {desc}: {exc}")

            if progress_every and (written + failed) % progress_every == 0:
                print(f"  [{name}] {written + failed} classes ...")

        stats["dex_files"] += 1
        stats["written"] += written
        stats["failed"] += failed
        stats["per_dex"][name] = (written, failed)
        del dex          # release member bytes before parsing the next DEX
        print(f"[export] {name}: {written} written, {failed} failed")

    return stats


# ============================================================================
# STANDALONE
# ============================================================================

def _smali_cli(apk_path: Path, max_classes: int = 3) -> None:
    from tools.dex import parse_dex_from_apk

    dex = parse_dex_from_apk(apk_path, "classes.dex")

    print("=" * 70)
    print("SMALI RECONSTRUCTION (first classes)")
    print("=" * 70)
    for cdef in dex.class_defs[:max_classes]:
        name = dex.type_desc(cdef.class_idx) or "?"
        print(f"\n### {name}\n")
        print(render_class(dex, cdef))

    graph = build_call_graph(dex)
    print()
    print("=" * 70)
    print("CALL GRAPH")
    print("=" * 70)
    print_call_graph_report(graph)
    print()
    print("MERMAID (top edges)")
    print(graph.to_mermaid())


if __name__ == "__main__":
    import sys
    apk = Path(sys.argv[1]) if len(sys.argv) > 1 else \
        Path(__file__).parent.parent / "apks" / "sample.apk"
    _smali_cli(apk)
