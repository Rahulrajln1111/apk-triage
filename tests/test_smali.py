"""
Tests for the smali reconstruction engine (Phase 5).

The synthetic DEXes are built byte-by-byte from the AOSP specification - the
technique that caught real bugs in the arsc and dex parsers. The branch-unit
tests in particular exist because Dalvik branch offsets are in 16-bit code
units, and confusing them with byte offsets is the classic smali-engine bug.
"""

import struct
import zlib
import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from tools.dex import parse_dex_bytes, validate_instruction_walk
from tools.smali import (
    MethodSmali,
    annotate_resource_ids,
    build_call_graph,
    decode_instruction,
    decode_method_insns,
    descriptor_from_java,
    manifest_components,
    method_signature,
    print_call_graph_report,
    proto_signature,
    render_class,
)


# ============================================================================
# PRIMITIVE ENCODERS (test-side, spec vectors)
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


def sleb(value: int) -> bytes:
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if (value == 0 and not (byte & 0x40)) or (value == -1 and (byte & 0x40)):
            out.append(byte)
            return bytes(out)
        out.append(byte | 0x80)


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
    return bytes(out)


# ============================================================================
# SYNTHETIC DEX BUILDER (byte-by-byte, parameterized)
# ============================================================================

@dataclass
class TMethod:
    method_idx: int
    flags: int = 0x9            # public static
    insns: bytes | None = None  # None = abstract/native (no code)
    regs: int = 4


@dataclass
class TClass:
    type_idx: int
    methods: list = field(default_factory=list)
    super_idx: int | None = 0   # index into types; None = no superclass
    flags: int = 0x1            # public


def build_dex(*, strings, types, protos, methods, classes,
              tries: dict | None = None) -> bytes:
    """Assemble a spec-conformant DEX.

    strings : list[str]
    types   : list[int]          (string indices)
    protos  : list[(shorty_idx, ret_type_idx, params:list[int]|None)]
    methods : list[(class_type_idx, proto_idx, name_str_idx)]
    classes : list[TClass]
    tries   : {(ci, mi): ([(start_addr, insn_count, handler_off)], blob_bytes)}
              addresses in code units, handler_off relative to blob start
    """
    tries = tries or {}

    # ---- section blobs ---------------------------------------------------
    string_data = b"".join(uleb(len(s)) + mutf8(s) + b"\x00" for s in strings)
    string_offsets = []
    off = 0
    for s in strings:
        string_offsets.append(off)
        off += len(uleb(len(s))) + len(mutf8(s)) + 1
    string_ids = b"".join(struct.pack("<I", 0) for _ in strings)  # patched

    type_ids = b"".join(struct.pack("<I", t) for t in types)

    # type_list blobs for proto params (unique lists shared)
    param_lists: dict[tuple, bytes] = {}
    for _shorty, _ret, params in protos:
        if params and tuple(params) not in param_lists:
            param_lists[tuple(params)] = struct.pack("<I", len(params)) + b"".join(
                struct.pack("<H", p) for p in params)

    proto_ids = b"".join(
        struct.pack("<III", shorty, ret, 0)  # params_off patched below
        for shorty, ret, params in protos
    )
    proto_param_offsets = {}
    for i, (_shorty, _ret, params) in enumerate(protos):
        if params:
            proto_param_offsets[i] = tuple(params)

    method_ids = b"".join(struct.pack("<HHI", c, p, n) for c, p, n in methods)

    class_defs = b"".join(
        struct.pack("<IIIIIIII",
                    c.type_idx, c.flags,
                    NO_INDEX_SUPER(c.super_idx),
                    0, NO_INDEX_SRC, 0, 0, 0)
        for c in classes
    )

    # ---- layout ----------------------------------------------------------
    cursor = 112
    string_ids_off = cursor;  cursor += len(string_ids)
    type_ids_off = cursor;    cursor += len(type_ids)
    proto_ids_off = cursor;   cursor += len(proto_ids)
    method_ids_off = cursor;  cursor += len(method_ids)
    class_defs_off = cursor;  cursor += len(class_defs)

    type_list_offsets: dict[tuple, int] = {}
    for key, blob in param_lists.items():
        cursor = (cursor + 3) & ~3
        type_list_offsets[key] = cursor
        cursor += len(blob)

    string_data_off = cursor; cursor += len(string_data)

    # code items: one per method with insns, 4-byte aligned.
    # KEYED BY (class index, METHOD_IDX): class_data is encoded sorted by
    # method_idx, so lookup must not depend on declaration order.
    code_layout: dict[tuple, int] = {}      # (ci, method_idx) -> code_off
    code_meta: dict[tuple, TMethod] = {}
    for ci, c in enumerate(classes):
        for m in c.methods:
            if m.insns is not None:
                cursor = (cursor + 3) & ~3
                code_layout[(ci, m.method_idx)] = cursor
                code_meta[(ci, m.method_idx)] = m
                cursor += 16 + len(m.insns)
                if len(m.insns) % 4:
                    cursor += 2              # insns padding to 4 bytes

    # try table + encoded_catch_handler_list sit IMMEDIATELY after the
    # method's insns padding (spec: handler_off is an offset from the start
    # of the handler list, so the layout must place it predictably). The
    # list itself is uleb(size) + handlers; the size prefix is added here,
    # so callers pass only the raw handler bytes.
    try_meta: dict[tuple, tuple] = {}
    for key, (try_items, blob) in tries.items():
        # encoded_catch_handler_list.size counts HANDLERS, not bytes -
        # walk the blob: sleb hsize, |hsize| pairs of ulebs, +1 uleb if
        # hsize <= 0 (catch-all addr).
        n_handlers = 0
        p = 0
        while p < len(blob):
            hsize = blob[p] - 256 if blob[p] > 127 else blob[p]  # sleb 1st byte
            p += 1
            n = hsize if hsize > 0 else -hsize
            p += 2 * n
            if hsize <= 0:
                p += 1
            n_handlers += 1
        hlist = uleb(n_handlers) + blob
        try_meta[key] = (cursor, try_items, hlist)
        cursor += len(try_items) * 8 + len(hlist)

    # class_data blobs: encode with FINAL code offsets. In-place uleb
    # patching corrupts the file when an offset grows a byte (the extra
    # byte overruns into the next field).
    class_data_blobs: list[bytes] = []
    for ci, c in enumerate(classes):
        ms = sorted(c.methods, key=lambda m: m.method_idx)
        blob = uleb(0) + uleb(0) + uleb(len(ms)) + uleb(0)
        prev = 0
        for m in ms:
            blob += uleb(m.method_idx - prev) + uleb(m.flags)
            blob += uleb(code_layout.get((ci, m.method_idx), 0))
            prev = m.method_idx
        class_data_blobs.append(blob)

    class_data_off = cursor
    class_data_all = b"".join(class_data_blobs)
    cursor += len(class_data_all)

    map_off = (cursor + 3) & ~3

    # ---- map -------------------------------------------------------------
    map_items = [
        (0x0000, 1, 0),
        (0x0001, len(strings), string_ids_off),
        (0x0002, len(types), type_ids_off),
        (0x0003, len(protos), proto_ids_off),
        (0x0005, len(methods), method_ids_off),
        (0x0006, len(classes), class_defs_off),
    ]
    if param_lists:
        map_items.append((0x1001, len(param_lists),
                          min(type_list_offsets.values())))
    map_items.append((0x2002, len(strings), string_data_off))
    if classes:
        map_items.append((0x2000, len(classes), class_data_off))
    if code_layout:
        map_items.append((0x2001, len(code_layout),
                          min(code_layout.values())))
    map_items.sort(key=lambda t: t[2])
    map_list = struct.pack("<I", len(map_items)) + b"".join(
        struct.pack("<HHII", t, 0, n, o) for t, n, o in map_items)
    file_size = map_off + 4 + len(map_list)

    # ---- assemble --------------------------------------------------------
    buf = bytearray(file_size)
    buf[0:8] = b"dex\n035\x00"

    def put(at: int, blob: bytes) -> None:
        buf[at:at + len(blob)] = blob

    string_ids = b"".join(struct.pack("<I", string_data_off + o)
                          for o in string_offsets)
    put(string_ids_off, string_ids)
    put(type_ids_off, type_ids)
    put(proto_ids_off, proto_ids)
    # patch param type_list offsets AFTER the blob write, or they get wiped
    for i, key in proto_param_offsets.items():
        struct.pack_into("<I", buf, proto_ids_off + i * 12 + 8,
                         type_list_offsets[key])
    put(method_ids_off, method_ids)
    put(class_defs_off, class_defs)
    for key, at in type_list_offsets.items():
        put(at, param_lists[key])
    put(string_data_off, string_data)
    put(class_data_off, class_data_all)
    for (ci, midx), at in code_layout.items():
        m = code_meta[(ci, midx)]
        insns = m.insns
        tm = try_meta.get((ci, midx))
        n_tries = len(tm[1]) if tm else 0
        put(at, struct.pack("<HHHHII", m.regs, 0, 0, n_tries, 0, len(insns) // 2))
        put(at + 16, insns)
        if tm:
            t_at, t_items, hlist = tm
            for j, (start, count, hoff) in enumerate(t_items):
                struct.pack_into("<IHH", buf, t_at + j * 8, start, count, hoff)
            # handler list follows the try items IMMEDIATELY (no padding);
            # handler_off is relative to the start of the size prefix.
            put(t_at + len(t_items) * 8, hlist)

    # class_def patches: class_data_off (byte 24 of each 32-byte def);
    # per-class blobs are contiguous, so offsets just accumulate.
    cdata_at = class_data_off
    for i, blob in enumerate(class_data_blobs):
        struct.pack_into("<I", buf, class_defs_off + i * 32 + 24, cdata_at)
        cdata_at += len(blob)

    # ---- header ----------------------------------------------------------
    struct.pack_into("<I", buf, 32, file_size)
    struct.pack_into("<I", buf, 36, 112)
    struct.pack_into("<I", buf, 40, 0x12345678)
    struct.pack_into("<II", buf, 44, 0, 0)
    struct.pack_into("<I", buf, 52, map_off)
    struct.pack_into("<II", buf, 56, len(strings), string_ids_off)
    struct.pack_into("<II", buf, 64, len(types), type_ids_off)
    struct.pack_into("<II", buf, 72, len(protos), proto_ids_off)
    struct.pack_into("<II", buf, 80, 0, 0)
    struct.pack_into("<II", buf, 88, len(methods), method_ids_off)
    struct.pack_into("<II", buf, 96, len(classes), class_defs_off)
    struct.pack_into("<II", buf, 104, file_size - string_data_off, string_data_off)

    buf[12:32] = hashlib.sha1(bytes(buf[32:])).digest()
    struct.pack_into("<I", buf, 8, zlib.adler32(bytes(buf[12:])) & 0xFFFFFFFF)
    return bytes(buf)


NO_INDEX_SRC = 0xFFFFFFFF


def NO_INDEX_SUPER(idx):
    return 0xFFFFFFFF if idx is None else idx


def uleb_step(buf, pos):
    value = 0
    shift = 0
    while True:
        byte = buf[pos]
        pos += 1
        value |= (byte & 0x7F) << shift
        if not (byte & 0x80):
            return value, pos
        shift += 7


def patch_uleb(buf, pos, value):
    blob = uleb(value)
    buf[pos:pos + len(blob)] = blob


# ============================================================================
# SHARED FIXTURES
# ============================================================================

S = ["", "V", "LFoo;", "LBar;", "main", "()V", "go", "Hello",
     "unusedThing", "()I", "I", "a", "c", "b", "onCreate",
     "(I)V", "(I)I"]


def base_sections():
    # types: [0]=V, [1]=LFoo;, [2]=LBar;, [3]=I  (values are string indices)
    types = [1, 2, 3, 10]
    # protos: [0]=()V ; [1]=(I)I   (shorty_idx, ret_type_idx, param_type_idxs)
    protos = [(1, 0, None), (10, 3, [3])]
    # methods: LFoo;->main()V ; LBar;->go()V ; LFoo;->unusedThing()V
    methods = [(1, 0, 4), (2, 0, 6), (1, 0, 8)]
    return types, protos, methods


def simple_foo_class(insns: bytes, regs: int = 4) -> list:
    return [TClass(type_idx=1, methods=[TMethod(0, 0x9, insns, regs)])]


# ============================================================================
# DECODER UNIT TESTS
# ============================================================================

def test_branch_offsets_are_bytes_not_units():
    # insns: nop; goto +2 (code units); return-void; nop(target)
    # goto offset 2 units = 4 bytes -> target addr 2+4 = 6.
    insns = struct.pack("<4H", 0x0000, 0x0228, 0x000E, 0x0000)
    dex = parse_dex_bytes(build_dex(
        strings=S, types=base_sections()[0], protos=base_sections()[1],
        methods=base_sections()[2], classes=simple_foo_class(insns)))
    code = dex.code_for(dex.class_defs[0].direct_methods[0])
    insns_dec, payloads = decode_method_insns(dex.data, code)
    goto = insns_dec[1]
    assert goto.name == "goto"
    # Insn.target is the RELATIVE byte offset (wire 2 units * 2 = 4 bytes);
    # the absolute label address is insn.addr + target = 2 + 4 = 6.
    assert goto.target == 4, "branch offset must be converted to BYTES"
    text = MethodSmali(dex, "LFoo;", "main", 0x9, 0, code).render()
    assert ":label_6" in text, "label must resolve to byte addr 6"
    assert ":label_3" not in text, "unit-based label (2+1) would be 3"


def test_35c_register_extraction():
    # invoke-static {v2, v3}, LBar;->go()V
    # Spec 35c = A|G|op BBBB F|E|D|C: A(bits 12-15)=count=2, G(bits
    # 8-11)=0 -> unit0 = 0x2071; unit2 = F|E|D|C = 0x0032 (C=2, D=3).
    insns = struct.pack("<3H", 0x2071, 0x0001, 0x0032) + struct.pack("<H", 0x000E)
    dex = parse_dex_bytes(build_dex(
        strings=S, types=base_sections()[0], protos=base_sections()[1],
        methods=base_sections()[2], classes=simple_foo_class(insns)))
    code = dex.code_for(dex.class_defs[0].direct_methods[0])
    insns_dec, _ = decode_method_insns(dex.data, code)
    invoke = insns_dec[0]
    assert invoke.regs == [2, 3]
    text = MethodSmali(dex, "LFoo;", "main", 0x9, 0, code).render()
    assert "invoke-static {v2, v3}, LBar;->go()V" in text


def test_const_string_escaping_and_high16():
    # const-string v0, "Hello" ; const/high16 v1, 0x003C0000 ; return-void
    insns = struct.pack("<2H", 0x0A1A, 0x0007)      # vAA=10? no: A=10 too big for 8 bits? A is 8 bits (AA)
    # fix: A=0 -> 0x001A
    insns = struct.pack("<6H",
                        0x001A, 0x0007,   # const-string v0, string@7 "Hello"
                        0x0115, 0x003C,   # const/high16 v1, 0x003C0000
                        0x000E, 0x0000)   # return-void + pad
    dex = parse_dex_bytes(build_dex(
        strings=S, types=base_sections()[0], protos=base_sections()[1],
        methods=base_sections()[2], classes=simple_foo_class(insns)))
    code = dex.code_for(dex.class_defs[0].direct_methods[0])
    text = MethodSmali(dex, "LFoo;", "main", 0x9, 0, code).render()
    assert 'const-string v0, "Hello"' in text
    assert "const/high16 v1, 0x3c0000" in text


def test_const_wide_high16_shifts_48():
    insns = struct.pack("<4H",
                        0x0119, 0x0001,   # const-wide/high16 v1, 1<<48
                        0x000E, 0x0000)
    dex = parse_dex_bytes(build_dex(
        strings=S, types=base_sections()[0], protos=base_sections()[1],
        methods=base_sections()[2], classes=simple_foo_class(insns)))
    code = dex.code_for(dex.class_defs[0].direct_methods[0])
    text = MethodSmali(dex, "LFoo;", "main", 0x9, 0, code).render()
    assert "const-wide/high16 v1, 0x1000000000000L" in text


def test_packed_switch_payload_targets_patched():
    # packed-switch v0, +4(units) @0 ; return-void @6 ; payload @8
    # payload: ident, size=2, first_key=5, targets [+1, +3] units from opcode
    payload = struct.pack("<HH", 0x0100, 2) + struct.pack("<i", 5) + \
        struct.pack("<ii", 1, 3)
    assert len(payload) == 16
    insns = struct.pack("<3H", 0x012B, 0x0004, 0x0000)  # switch, +4 units
    insns += struct.pack("<H", 0x000E)                  # return-void @6
    insns += payload
    dex = parse_dex_bytes(build_dex(
        strings=S, types=base_sections()[0], protos=base_sections()[1],
        methods=base_sections()[2], classes=simple_foo_class(insns)))
    assert not any("misaligned" in a for a in dex.anomalies), dex.anomalies
    code = dex.code_for(dex.class_defs[0].direct_methods[0])
    text = MethodSmali(dex, "LFoo;", "main", 0x9, 0, code).render()
    # targets: opcode at byte 0 + 1 unit*2 = 2 ; +3 units = 6
    assert ".packed-switch 0x5" in text
    assert ":label_2" in text
    assert ":label_6" in text
    assert ":label_8  # pswitch_data_8" in text, "switch must reference its payload"


def test_sparse_switch_payload():
    payload = struct.pack("<HH", 0x0200, 2)
    payload += struct.pack("<ii", 1, 100)          # keys
    payload += struct.pack("<ii", 2, 6)            # targets (units from op)
    # sparse-switch (31t) at byte 0; return-void at byte 6; payload at byte 8
    insns = struct.pack("<3H", 0x012C, 0x0004, 0x0000)  # switch, +4 units
    insns += struct.pack("<H", 0x000E)
    insns += payload
    dex = parse_dex_bytes(build_dex(
        strings=S, types=base_sections()[0], protos=base_sections()[1],
        methods=base_sections()[2], classes=simple_foo_class(insns)))
    code = dex.code_for(dex.class_defs[0].direct_methods[0])
    text = MethodSmali(dex, "LFoo;", "main", 0x9, 0, code).render()
    assert ".sparse-switch" in text
    assert "0x1 -> :label_4" in text     # 0 + 2 units * 2 = byte 4
    assert "0x64 -> :label_c" in text    # 0 + 6 units * 2 = byte 12 = 0xc


def test_array_payload_renders_big_endian_elements():
    payload = struct.pack("<HH", 0x0300, 2) + struct.pack("<I", 2) + \
        struct.pack("<HH", 0x1234, 0xBEEF)
    insns = struct.pack("<3H", 0x0126, 0x0003, 0x0000)  # fill-array-data +3
    insns += struct.pack("<H", 0x000E)
    insns += payload
    dex = parse_dex_bytes(build_dex(
        strings=S, types=base_sections()[0], protos=base_sections()[1],
        methods=base_sections()[2], classes=simple_foo_class(insns)))
    code = dex.code_for(dex.class_defs[0].direct_methods[0])
    text = MethodSmali(dex, "LFoo;", "main", 0x9, 0, code).render()
    assert ".array-data 2" in text
    assert "0x1234" in text and "0xbeef" in text


def test_oversized_array_payload_is_bounded_and_walk_stays_aligned():
    # fill-array-data payload declaring 0x40000000 elements of width 4 (4 GiB)
    # with only the header words present. Regression: the declared count used
    # to be trusted, so the walk skipped past insns_size (silently dropping the
    # rest of the method) and the renderer allocated the declared size
    # (MemoryError on real APKs).
    payload = struct.pack("<HH", 0x0300, 4) + struct.pack("<I", 0x40000000)
    insns = struct.pack("<3H", 0x0126, 0x0004, 0x0000)   # fill-array-data v0, +4
    insns += struct.pack("<H", 0x000E)                   # return-void
    insns += payload                                    # words 4..7
    insns += struct.pack("<H", 0x000E)                   # trailing return-void
    dex = parse_dex_bytes(build_dex(
        strings=S, types=base_sections()[0], protos=base_sections()[1],
        methods=base_sections()[2], classes=simple_foo_class(insns)))
    code = dex.code_for(dex.class_defs[0].direct_methods[0])

    bad: list[int] = []
    end = code.insns_off + code.insns_size * 2
    assert validate_instruction_walk(dex.data, code, bad) == end
    assert bad == [4 * 2]                    # the payload ident, unit 4

    decoded, _payloads = decode_method_insns(dex.data, code)
    # both returns survive: the bad payload did not swallow the stream
    assert [i.name for i in decoded].count("return-void") == 2

    text = MethodSmali(dex, "LFoo;", "main", 0x9, 0, code).render()
    assert text.count("return-void") == 2
    assert any("oversized payload" in a for a in dex.anomalies), dex.anomalies
    assert not any("misaligned" in a for a in dex.anomalies), dex.anomalies


def test_try_catch_rendering():
    # insns: nop nop nop return-void (8 bytes)
    # try covers units 0..1 (bytes 0..4); handler for LBar; at unit 2 -> byte 4
    handler_blob = sleb(1) + uleb(2) + uleb(2)   # 1 pair: type@2(LBar;) addr 2
    # handler_off counts from the START of the list (its uleb size byte is
    # byte 0), so the first handler sits at offset 1.
    tries = {(0, 0): ([(0, 2, 1)], handler_blob)}
    insns = struct.pack("<4H", 0x0000, 0x0000, 0x0000, 0x000E)
    dex = parse_dex_bytes(build_dex(
        strings=S, types=base_sections()[0], protos=base_sections()[1],
        methods=base_sections()[2], classes=simple_foo_class(insns),
        tries=tries))
    code = dex.code_for(dex.class_defs[0].direct_methods[0])
    text = MethodSmali(dex, "LFoo;", "main", 0x9, 0, code).render()
    assert ":try_start_0" in text
    assert ":try_end_4" in text
    assert ".catch LBar; {:try_start_0 .. :try_end_4} :label_4" in text


def test_catchall_rendering():
    # size=0 (sleb) => ZERO typed pairs, then a catch-all addr follows.
    # (size<0 means |size| pairs PLUS a catch-all - sleb(-1) would be one
    # pair + catch-all, needing 3 fields.)
    handler_blob = uleb(0) + uleb(2)             # catch-all at unit 2
    tries = {(0, 0): ([(0, 2, 1)], handler_blob)}
    insns = struct.pack("<4H", 0x0000, 0x0000, 0x0000, 0x000E)
    dex = parse_dex_bytes(build_dex(
        strings=S, types=base_sections()[0], protos=base_sections()[1],
        methods=base_sections()[2], classes=simple_foo_class(insns),
        tries=tries))
    code = dex.code_for(dex.class_defs[0].direct_methods[0])
    text = MethodSmali(dex, "LFoo;", "main", 0x9, 0, code).render()
    assert ".catchall {:try_start_0 .. :try_end_4} :label_4" in text


# ============================================================================
# CLASS RENDERING
# ============================================================================

def test_class_rendering_structure():
    insns = struct.pack("<2H", 0x000E, 0x0000)
    dex = parse_dex_bytes(build_dex(
        strings=S, types=base_sections()[0], protos=base_sections()[1],
        methods=base_sections()[2],
        classes=[TClass(type_idx=1, methods=[TMethod(0, 0x9, insns)],
                        super_idx=2)]))
    text = render_class(dex, dex.class_defs[0])
    assert text.startswith(".class public LFoo;")
    assert ".super LBar;" in text
    assert ".method public static main()V" in text
    assert ".registers 4" in text
    assert ".end method" in text


def test_method_signature_format():
    dex = parse_dex_bytes(build_dex(
        strings=S, types=base_sections()[0], protos=base_sections()[1],
        methods=base_sections()[2],
        classes=simple_foo_class(struct.pack("<H", 0x000E))))
    assert proto_signature(dex, 0) == "()V"
    assert proto_signature(dex, 1) == "(I)I"
    assert method_signature(dex, 0) == "LFoo;->main()V"


def test_proto_params_render_in_invoke():
    # invoke-static {v1}, LFoo;->unusedThing()V is ()V; use method 1 proto (I)I
    # method_ids[2] = LFoo;->unusedThing()V -> change to use a (I)I method:
    # add method LBar;->go(I)I via a dedicated dex
    strings = S + ["go2"]
    types = base_sections()[0]
    protos = base_sections()[1]
    methods = base_sections()[2] + [(2, 1, len(strings) - 1)]  # LBar;->go2(I)I
    insns = struct.pack("<3H", 0x1071, len(methods) - 1, 0x0001)  # {v1}
    insns += struct.pack("<H", 0x000E)
    dex = parse_dex_bytes(build_dex(
        strings=strings, types=types, protos=protos, methods=methods,
        classes=simple_foo_class(insns)))
    code = dex.code_for(dex.class_defs[0].direct_methods[0])
    text = MethodSmali(dex, "LFoo;", "main", 0x9, 0, code).render()
    assert "LBar;->go2(I)I" in text


# ============================================================================
# CALL GRAPH
# ============================================================================

def callgraph_dex():
    """LFoo;: onCreate()V (entry) calls a()V ; a()V calls LBar;->b()V ;
    c()V never called. LBar;: b()V."""
    strings = ["", "V", "LFoo;", "LBar;", "onCreate", "()V", "a", "b",
               "c", "()I", "I"]
    types = [1, 2, 3, 10]                # V, LFoo;, LBar;, I (string idxs)
    protos = [(1, 0, None), (10, 3, None)]   # ()V ; ()I
    methods = [
        (1, 0, 4),   # LFoo;->onCreate()V
        (1, 0, 6),   # LFoo;->a()V
        (1, 0, 8),   # LFoo;->c()V
        (2, 0, 7),   # LBar;->b()V
    ]
    invoke_b = struct.pack("<3H", 0x1071, 3, 0x0000) + struct.pack("<H", 0x000E)
    invoke_a = struct.pack("<3H", 0x1071, 1, 0x0000) + struct.pack("<H", 0x000E)
    ret = struct.pack("<H", 0x000E) + struct.pack("<H", 0x0000)

    classes = [
        TClass(type_idx=1, methods=[
            TMethod(0, 0x9, invoke_a),   # onCreate -> a()
            TMethod(1, 0x9, invoke_b),   # a -> LBar;->b()
            TMethod(2, 0x9, ret),        # c -> nothing
        ]),
        TClass(type_idx=2, methods=[TMethod(3, 0x9, ret)]),
    ]
    return parse_dex_bytes(build_dex(
        strings=strings, types=types, protos=protos, methods=methods,
        classes=classes))


def test_call_graph_edges_and_dead_detection():
    dex = callgraph_dex()
    graph = build_call_graph(dex)

    assert graph.classes["LFoo;"].methods[
        "LFoo;->onCreate()V"].calls == ["LFoo;->a()V"]
    assert graph.class_edges[("LFoo;", "LBar;")] == 1
    assert "LBar;->b()V" in graph.external_refs or \
        graph.classes["LBar;"].methods["LBar;->b()V"].called_by

    # entry: onCreate is a framework callback with code
    entry_keys = {m.key for m in graph.entry_points()}
    assert "LFoo;->onCreate()V" in entry_keys
    # a() is called by onCreate -> not dead
    assert graph.classes["LFoo;"].methods["LFoo;->a()V"].dead is False
    # c() has code, nobody calls it, not a callback -> dead candidate
    assert graph.classes["LFoo;"].methods["LFoo;->c()V"].dead is True
    # b() is called by a() -> not dead
    assert graph.classes["LBar;"].methods["LBar;->b()V"].dead is False


def test_component_methods_are_entry_points():
    dex = callgraph_dex()
    graph = build_call_graph(dex, components=[("activity", "LFoo;")])
    assert graph.classes["LFoo;"].component == "activity"
    entry_keys = {m.key for m in graph.entry_points()}
    assert "LFoo;->onCreate()V" in entry_keys


def test_descriptor_from_java():
    assert descriptor_from_java("com.foo.Bar", None) == "Lcom/foo/Bar;"
    assert descriptor_from_java(".Bar", "com.foo") == "Lcom/foo/Bar;"
    assert descriptor_from_java("Bar", "com.foo") == "Lcom/foo/Bar;"
    assert descriptor_from_java("Lcom/x/Y;", None) == "Lcom/x/Y;"


def test_manifest_component_extraction():
    from tools.manifest import XmlAttribute, XmlElement
    package = XmlAttribute(0, 0, 0, 0, 0, "package", "com.foo", None, "", "")
    app = XmlElement("application", 2, [], [], 0, "", "", "")
    act = XmlElement(
        "activity", 3,
        [XmlAttribute(0, 0, 0, 0, 0, "name", "com.foo.Main", None, "", "")],
        [], 0, "", "", "")
    alias = XmlElement(
        "activity-alias", 4,
        [XmlAttribute(0, 0, 0, 0, 0, "name", ".Alias", None, "", ""),
         XmlAttribute(0, 0, 0, 0, 0, "targetActivity", "com.foo.Main",
                      None, "", "")],
        [], 0, "", "", "")
    app.children = [act, alias]
    root = XmlElement("manifest", 1, [package], [app], 0, "", "", "")

    comps = dict((desc, kind) for kind, desc in manifest_components(root))
    assert comps.get("Lcom/foo/Main;") == "activity"
    assert comps.get("Lcom/foo/Alias;") == "activity"


# ============================================================================
# RESOURCE-ID CONTEXT
# ============================================================================

def test_resource_annotation_appends_comment():
    ctx = {0x7F0B001D: "@zenith:layout/activity_main"}
    out = annotate_resource_ids("const v0, 0x7f0b001d", ctx)
    assert "0x7f0b001d  # @zenith:layout/activity_main" in out


def test_resource_annotation_leaves_unknown_ids():
    ctx = {0x7F0B001D: "@zenith:layout/activity_main"}
    out = annotate_resource_ids("const v0, 0x7f0a0001", ctx)
    assert "#" not in out


# ============================================================================
# REAL APK VALIDATION
# ============================================================================

APK_PATH = Path(__file__).parent.parent / "apks" / "sample.apk"


@pytest.mark.skipif(not APK_PATH.exists(), reason="sample.apk not present")
def test_real_apk_class_rendering_roundtrip():
    from tools.dex import parse_dex_from_apk
    dex = parse_dex_from_apk(APK_PATH, "classes.dex")
    # Render 25 real classes end to end: exercises every decoder path that
    # occurs in real code; a format bug or unit bug would corrupt the output.
    # (Obfuscated APKs start with empty marker interfaces, so only require
    # at least one rendered method across the batch.)
    with_method = 0
    for cdef in dex.class_defs[:25]:
        text = render_class(dex, cdef)
        assert text.startswith(".class ")
        assert "op_0x" not in text, "unknown mnemonic leaked into output"
        if ".end method" in text:
            with_method += 1
    assert with_method > 0, "no methods rendered in 25 classes?"


@pytest.mark.skipif(not APK_PATH.exists(), reason="sample.apk not present")
def test_real_apk_call_graph():
    from tools.dex import parse_dex_from_apk
    dex = parse_dex_from_apk(APK_PATH, "classes.dex")
    graph = build_call_graph(dex)
    stats = graph.stats()
    assert stats["classes"] == 142
    assert stats["internal_edges"] > 0
    assert stats["methods"] > 500
    # dead candidates must be a minority on real code (heuristics sanity)
    assert stats["dead_candidates"] < stats["methods"]


# ============================================================================
# FULL EXPORT (all-DEX tree export)
# ============================================================================

def test_class_rel_path_sanitizes_hostile_descriptors():
    from tools.smali import class_rel_path
    assert class_rel_path("Lcom/foo/Bar$1;") == "com/foo/Bar$1.smali"
    assert class_rel_path("Lcom/foo/Bar;") == "com/foo/Bar.smali"
    # traversal attempt must not escape: '..' segment becomes '_..'
    p = class_rel_path("Lcom/../../etc/passwd;")
    assert not p.startswith("/")
    assert ".." not in p.split("/")
    # hostile chars and control bytes become '_'
    assert "<" not in class_rel_path("La<b:c>;")
    # degenerate descriptor still yields a writable relative path
    p = class_rel_path("not-a-descriptor")
    assert p.endswith(".smali") and not p.startswith("/")


def test_export_smali_tree_synthetic_apk(tmp_path):
    import zipfile
    from tests.test_smali import (S, base_sections, simple_foo_class,
                                  build_dex)
    from tools.smali import export_smali_tree

    dex_bytes = build_dex(
        strings=S, types=base_sections()[0], protos=base_sections()[1],
        methods=base_sections()[2], classes=simple_foo_class(
            struct.pack("<2H", 0x000E, 0x0000)))
    apk = tmp_path / "mini.apk"
    with zipfile.ZipFile(apk, "w") as z:
        z.writestr("AndroidManifest.xml", b"PK-fake-manifest")
        z.writestr("classes.dex", dex_bytes)

    out = tmp_path / "smali_out"
    stats = export_smali_tree(apk, out, progress_every=0)
    assert stats["dex_files"] == 1 and stats["written"] >= 1
    assert stats["failed"] == 0
    files = sorted(out.rglob("*.smali"))
    assert files, "no smali written"
    assert files[0].read_text().startswith(".class public LFoo;")
    assert (out / "classes" / "_classes_callgraph.mmd").exists()
