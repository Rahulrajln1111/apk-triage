"""Build a tiny SYNTHETIC corpus so the ML pipeline can be smoke-tested.

These APKs are byte-built from the AOSP spec (tests/test_smali.build_dex):
they carry no manifest/arsc (those blocks stay NaN on purpose, which also
exercises the NaN path) and different DEX sizes so hashes/dedupe work.
Do NOT report metrics from this corpus - it exists to prove the wiring.
"""
import csv
import hashlib
import struct
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from tests.test_smali import TClass, TMethod, build_dex


def build_apk(path: Path, n_classes: int, n_calls: int, n_assets: int) -> None:
    strings = ["", "V", "main", "()V"]
    strings += [f"Lc{i};" for i in range(n_classes)]
    strings += [f"s{j}" for j in range(n_calls)]
    types = [1] + [4 + i for i in range(n_classes)]
    protos = [(2, 0, None)]
    methods = [(1 + i, 0, 3) for i in range(n_classes)]
    insns = b""
    for j in range(n_calls):
        insns += struct.pack("<2H", 0x001A, 4 + n_classes + j)   # const-string
    insns += struct.pack("<H", 0x000E)                           # return-void
    if len(insns) % 4:
        insns += struct.pack("<H", 0x0000)
    classes = [TClass(type_idx=1 + i,
                      methods=[TMethod(i, 0x9, insns, regs=4)])
               for i in range(n_classes)]
    dex = build_dex(strings=strings, types=types, protos=protos,
                    methods=methods, classes=classes)

    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("classes.dex", dex)
        for k in range(n_assets):
            zf.writestr(f"assets/blob{k}.txt", (b"ab" * 4096) if k % 2 else
                        (b"unique" * 512 + bytes([k])))


def main() -> int:
    out = Path(__file__).parent / "corpus"
    out.mkdir(exist_ok=True)
    rows = []
    for i in range(12):
        n_classes = 1 + (i * 3) % 11
        n_calls = 1 + (i * 5) % 17
        n_assets = i % 4
        path = out / f"app{i:02d}.apk"
        build_apk(path, n_classes, n_calls, n_assets)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        label = 1 if (n_classes + n_assets) % 2 else 0
        rows.append({"sha256": digest, "label": label, "family": f"fam{i % 4}"})
    with open(out / "labels.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["sha256", "label", "family"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} synthetic APKs -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
