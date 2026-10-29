"""Corpus -> JSONL dataset (stdlib only: no numpy/sklearn needed here).

Label file is a CSV with a header:

    sha256,label,family
    8f1c...e2,1,ztorg
    3ab9...77,0,fdroid

label accepts 0/1 or benign/malware (see LABEL_*). family is the grouping
key for leak-free cross-validation; leave it empty and the APK hash becomes
the group (safe, just less informative).

Usage:
    python3 -m tools.ml.dataset --apk-dir apks --labels data/labels.csv \
        --out data/dataset.jsonl --deep --workers 4
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from tools.ml.features import FEATURE_VERSION, FIELDS, extract_apk_features

LABEL_TRUE = {"1", "malware", "malicious", "bad", "positive", "true"}
LABEL_FALSE = {"0", "benign", "goodware", "good", "negative", "false"}


# ---------------------------------------------------------------------------
# IO HELPERS
# ---------------------------------------------------------------------------

def sha256_file(path: Path, chunk: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def read_labels(csv_path: Path) -> dict[str, dict]:
    """sha256 -> {label: 0|1, family: str}. Raises on a malformed row."""
    labels: dict[str, dict] = {}
    with open(csv_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = {"sha256", "label"} - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{csv_path}: missing column(s) {sorted(missing)}")
        for line_no, row in enumerate(reader, start=2):
            digest = (row.get("sha256") or "").strip().lower()
            raw_label = (row.get("label") or "").strip().lower()
            if not digest:
                raise ValueError(f"{csv_path}:{line_no}: empty sha256")
            if raw_label in LABEL_TRUE:
                label = 1
            elif raw_label in LABEL_FALSE:
                label = 0
            else:
                raise ValueError(
                    f"{csv_path}:{line_no}: unknown label {raw_label!r} "
                    f"(want 0/1 or benign/malware)")
            family = (row.get("family") or "").strip()
            if digest in labels and labels[digest]["label"] != label:
                raise ValueError(
                    f"{csv_path}:{line_no}: conflicting labels for {digest}")
            labels[digest] = {"label": label, "family": family}
    return labels


def iter_apks(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.suffix.lower() == ".apk")


# ---------------------------------------------------------------------------
# EXTRACTION (runs in a worker process when workers > 1)
# ---------------------------------------------------------------------------

def _extract_task(task: tuple[str, bool]) -> dict:
    path, deep = task
    row, meta = extract_apk_features(Path(path), deep=deep)
    return {"path": path, "features": row, "meta": meta}


def build_dataset(apk_dir: Path,
                  labels_csv: Path | None,
                  out_path: Path,
                  deep: bool = True,
                  limit: int | None = None,
                  include_unlabeled: bool = False,
                  workers: int = 1,
                  min_mb: float = 0.0,
                  max_mb: float = 800.0,
                  quiet: bool = False) -> dict:
    """Scan a directory of APKs and write one JSON line per app."""
    labels = read_labels(labels_csv) if labels_csv else {}
    paths = iter_apks(apk_dir)
    if limit:
        paths = paths[:limit]

    plans: list[tuple[Path, int, str]] = []
    skipped: list[dict] = []
    for path in paths:
        size_mb = path.stat().st_size / (1024 * 1024)
        if size_mb < min_mb or size_mb > max_mb:
            skipped.append({"path": str(path), "reason": "size",
                            "size_mb": round(size_mb, 2)})
            continue
        digest = sha256_file(path)
        info = labels.get(digest)
        if info is None:
            if not include_unlabeled:
                skipped.append({"path": str(path), "reason": "unlabeled",
                                "sha256": digest})
                continue
            info = {"label": -1, "family": ""}
        plans.append((path, info, digest))

    results: list[dict] = []
    tasks = [(str(path), deep) for path, _info, _digest in plans]
    if workers > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_extract_task, task): task for task in tasks}
            for done, future in enumerate(as_completed(futures), start=1):
                extracted = future.result()
                results.append(extracted)
                if not quiet:
                    print(f"[{done}/{len(tasks)}] {Path(extracted['path']).name}",
                          file=sys.stderr)
    else:
        for done, task in enumerate(tasks, start=1):
            results.append(_extract_task(task))
            if not quiet:
                print(f"[{done}/{len(tasks)}] {Path(task[0]).name}",
                      file=sys.stderr)

    by_path = {item["path"]: item for item in results}
    rows: list[dict] = []
    for path, info, digest in plans:
        extracted = by_path.get(str(path))
        if extracted is None:
            continue
        rows.append({
            "sha256": digest,
            "label": info["label"],
            "family": info["family"] or f"sha:{digest[:12]}",
            "path": str(path),
            "features": extracted["features"],
            "meta": extracted["meta"],
        })

    kept, dropped = dedupe_rows(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        for row in kept:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    summary = {
        "feature_version": FEATURE_VERSION,
        "apks_seen": len(paths),
        "rows_written": len(kept),
        "rows_dropped_duplicate": len(dropped),
        "skipped": skipped[:50],
        "duplicates": [{"sha256": r["sha256"], "reason": reason}
                       for r, reason in dropped[:50]],
        "label_counts": _count(kept, "label"),
        "family_counts": len({r["family"] for r in kept}),
        "out": str(out_path),
    }
    with open(out_path.with_suffix(".summary.json"), "w",
              encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def _count(rows, key: str) -> dict:
    out: dict = {}
    for row in rows:
        out[str(row.get(key))] = out.get(str(row.get(key)), 0) + 1
    return out


# ---------------------------------------------------------------------------
# DEDUPE
# ---------------------------------------------------------------------------

def dedupe_rows(rows: list[dict]) -> tuple[list[dict], list[tuple[dict, str]]]:
    """Drop duplicate APKs, then APKs sharing *all* DEX files.

    The second rule is the cheap near-duplicate proxy: repackaged variants
    reuse the payload DEX, and leaving them in both train and test is the
    classic accuracy inflation in APK-malware papers. Order matters: the
    first occurrence wins, so pass family-grouped/ordered input if you care
    which copy survives.
    """
    seen_sha: set[str] = set()
    seen_dex: set[str] = set()
    kept: list[dict] = []
    dropped: list[tuple[dict, str]] = []
    for row in rows:
        digest = row.get("sha256") or ""
        dex_set = (row.get("meta") or {}).get("dex_set_hash") or ""
        if digest and digest in seen_sha:
            dropped.append((row, "sha256"))
            continue
        if dex_set and dex_set in seen_dex:
            dropped.append((row, "dex_set"))
            continue
        if digest:
            seen_sha.add(digest)
        if dex_set:
            seen_dex.add(dex_set)
        kept.append(row)
    return kept, dropped


# ---------------------------------------------------------------------------
# LOAD
# ---------------------------------------------------------------------------

def load_dataset(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            missing = set(FIELDS) - set(row.get("features") or {})
            if missing:
                raise ValueError(
                    f"{path}:{line_no}: {len(missing)} feature(s) missing "
                    f"(stale dataset? rebuild it) e.g. {sorted(missing)[:3]}")
            rows.append(row)
    return rows


def _main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Build the APKAI ML dataset")
    parser.add_argument("--apk-dir", type=Path, required=True)
    parser.add_argument("--labels", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("data/dataset.jsonl"))
    parser.add_argument("--deep", action="store_true",
                        help="add opcode histogram + call graph features")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--include-unlabeled", action="store_true",
                        help="keep label=-1 rows (inference only)")
    parser.add_argument("--max-mb", type=float, default=800.0)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    summary = build_dataset(
        args.apk_dir, args.labels, args.out, deep=args.deep,
        limit=args.limit, include_unlabeled=args.include_unlabeled,
        workers=args.workers, max_mb=args.max_mb, quiet=args.quiet)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
