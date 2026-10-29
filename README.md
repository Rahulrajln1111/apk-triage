# APKAI — Android APK forensic analyzer with an ML triage layer

Statically parses APKs **from scratch** against the AOSP specifications — ZIP
central directory, Android binary XML, `resources.arsc`, DEX and Dalvik
bytecode through to smali — using **no Apktool, JADX or Androguard**. The
parser output then feeds a versioned feature vector that trains a calibrated
triage model with per-feature attribution.

Python 3.11+ · stdlib-only core · optional `numpy` / `scikit-learn` for the
ML layer.

## Why this exists

Most APK tooling treats the archive as trusted input. This one does not: the
whole design assumes the file is hostile, and the interesting signals are
*structural* — inconsistencies inside the container rather than the
permissions everyone already looks at.

- **Never unzip an APK wholesale.** Entries are read individually through a
  bounded reader; zip bombs and multi-GB decoys are a real malware trick.
- **Nothing is trusted before validation.** EOCD records are read from a
  bounded tail window, offsets are rebased against the real file size, and
  DEX checksums/signatures are verified rather than assumed.
- **Rule #9 — never trust memory.** Every structure is checked against the
  AOSP spec and against real bytes. Tests build DEX files byte-by-byte from
  the spec; that technique has caught real parser bugs three times (arsc,
  DEX try/handler layout, invoke register nibbles).
- **Synthetic fixtures can share a bug with the parser.** They did: a `35c`
  A/G nibble swap passed every synthetic test and broke on the first real
  APK. Real-APK validation is not optional.

## Pipeline

`python3 main.py [--export-smali] [--translate] [--ml]`

| Stage | What it does |
|---|---|
| 1. Binary verification | ZIP magic, EOCD, central-directory offsets, ZIP64 sentinel |
| 2. ZIP audit | per-entry ratios, traversal/absolute paths, duplicate names, null-byte names, symlinks, encryption, nested archives, data descriptors |
| 3. Binary XML | manifest chunk tree, permissions, components, flags, ghost/malformed attributes |
| 4. resources.arsc | packages/types/entries, config variants, compact/sparse tables, null chunks, rebuild fingerprints |
| 5. DEX | header + section integrity, per-DEX reports, fake-DEX sniffing, instruction-walk validation |
| 6. Smali + call graph | full bytecode decode, baksmali-style rendering, dead-method candidates, entry-point → sensitive-API reachability |
| 7. ML triage (`--ml`) | calibrated malware score plus the features that moved it |

Stages 1–6 need nothing but the standard library. Stage 7 runs only when a
trained artifact exists at `models/apkai_triage.joblib`; otherwise it prints
a skip notice.

### Outputs (all disposable, git-ignored)

`out/smali/`, `out/smali_full/` (mirrors DEX names: `classes/`, `classes2/`, …),
`out/callgraph.{mmd,dot}`, `out/api_report.txt`, `out/java/`.

## Quickstart

```bash
# core analyzer: standard library only
python3 main.py

# every class of every real DEX (large output, delete out/ afterwards)
python3 main.py --export-smali

# optional LLM smali -> Java pass for entry-point classes
python3 main.py --translate

# ML layer dependencies (only these need numpy/sklearn)
pip install numpy scikit-learn joblib      # optional: xgboost
```

### LLM configuration

Order: CLI flags > environment > defaults. Defaults target a local SmaliLLM
setup (`smalillm-8b` on Ollama).

```bash
APKAI_LLM_PROVIDER=ollama|llamacpp|openai
APKAI_LLM_URL=http://127.0.0.1:11434
APKAI_LLM_MODEL=smalillm-8b
APKAI_LLM_API_KEY=...     # openai-compatible endpoints only
```

Note on local hardware: the GPU driver modules must be loaded on demand
(`nvidia-current` + `nvidia-current-uvm`). Without the UVM device nodes
llama.cpp **silently falls back to CPU** (~3× slower) — worth checking before
a long batch run.

## ML triage layer

The hand-tuned additive score in `tools/report.py` is still computed and
printed as a baseline. Stage 7 replaces it with a model trained on the same
structural signals, and — more importantly — explains the score instead of
just producing one.

### Feature vector — 149 columns, 5 blocks

| Block | Columns | Source |
|---|---|---|
| `zip` | 31 | ratios, tarpit counts, path anomalies, DEX naming, nesting |
| `axml` | 26 | permissions, components, exported flags, SDK levels, risky services |
| `arsc` | 21 | table shape, entry kinds, null chunks, scramble score, rebuild fingerprints |
| `dex` | 40 | header counts, fake/parse failures, anomaly totals, opcode histogram + entropy |
| `graph` | 31 | call-graph density, dead-code ratio, entry points, sensitive-API reachability, constant hits |

Two rules keep the matrix honest:

- **Absence is NaN, never 0.** "No `resources.arsc`" and "`resources.arsc`
  with zero entries" are different forensic facts. Tree models handle NaN
  natively; the linear baseline imputes train medians.
- **Blocks fail independently.** A hostile APK degrades its own block and
  records the reason in `meta["errors"]`; it never aborts a corpus scan.

The schema is versioned (`FEATURE_VERSION`). A training artifact stores the
version it was built against and refuses to score a changed matrix instead of
silently returning nonsense.

### Commands

```bash
# 1. corpus -> one JSON line per APK (features + label + family group)
python3 -m tools.ml.dataset --apk-dir apks --labels data/labels.csv \
    --out data/dataset.jsonl --deep --workers 4

# 2. grouped cross-validation, calibration, ablation, permutation importance
python3 -m tools.ml.train --dataset data/dataset.jsonl \
    --out models/apkai_triage.joblib --folds 5 --seed 42 --ablation --max-fpr 0.01

# 3. score one APK and show what drove the score
python3 -m tools.ml.model --model models/apkai_triage.joblib --apk app.apk

# 4. or run the whole analyzer with the triage stage as stage 7
python3 main.py --ml
```

`data/labels.csv` is a plain table keyed by APK hash — labels accept
`0/1` or `benign/malware`:

```csv
sha256,label,family
8f1c…e2,1,ztorg
3ab9…77,0,fdroid
```

### Evaluation protocol

The numbers worth quoting are the ones produced by this protocol, not by a
random split:

- **Grouped cross-validation** (`StratifiedGroupKFold` on the `family`
  column). Repackaged variants of one family must not appear in both train
  and test — that leak is the classic source of inflated APK-detection
  accuracy.
- **Deduplication before splitting**, on APK hash *and* on the set of
  contained DEX hashes (a repack usually reuses the payload DEX).
- **The rule score is a baseline**, evaluated on the identical folds and seed.
  If the model cannot beat it, the model is not worth shipping.
- **Metrics:** PR-AUC, recall at a fixed FPR (default 1%), ROC-AUC, and Brier
  score after calibration — no bare accuracy on an imbalanced corpus.
- **Calibration on out-of-fold scores only** (isotonic), never on fit data, so
  the output is a usable probability and the operating threshold is derived
  from a stated FPR target.
- **Per-block ablation** so the write-up can say which signals carry the
  signal instead of "the model works".
- Every run writes `results/eval-<stamp>.json` plus `results/eval_latest.md`.

**No accuracy numbers are committed with this repository.** The pipeline
ships without a trained artifact: honest metrics need a labeled corpus (a few
hundred APKs is enough for grouped CV, thousands for a strong claim). Anything
quoted from the 12-APK synthetic corpus in `make_corpus.py` would be
meaningless — that corpus exists only to prove the wiring end to end.

### Why this model family

Histogram gradient boosting is the primary learner: the features are
medium-sized, heavy-tailed and full of NaN, which is exactly where tree
ensembles beat neural nets on tabular data. A Drebin-style L1 logistic
regression and a random forest are trained on the same folds as reference
points — the ladder (rule → linear → trees) is the evidence, not a single
number. Optional extras: `--models xgb` and `--ablation`.

### Explainability

`TriageModel.explain()` perturbs each feature back to its training median and
reports the change in score, so a verdict arrives with "these signals moved
it" rather than a black-box number. Global permutation importance is stored in
the artifact and printed by the CLI.

## Verified behaviour

Measured on this machine (no numbers are estimated):

| Check | Result |
|---|---|
| Test suite | `90 passed` (`python3 -m pytest tests/ -q`) |
| Stage 1–6 on a 13.9 MB sample | 0.64 s end to end |
| Deep feature extraction, same file | 0.89 s, 149 columns, 0 block errors |
| Deep feature extraction, 304 MB / 5-DEX app | 7.2 s, 237 MB peak RSS, 0 block errors (39,275 classes, 9,182-class call graph) |

## Tests

```bash
python3 -m pytest tests/ -q
```

Synthetic builders encode wire formats straight from the AOSP spec pages
(instruction-formats + dalvik-bytecode): 33 instruction formats, uleb128
`class_data`, try/handler layout (the handler list **follows** the try table,
its size field counts *handlers*, handler offsets are relative to the list
start), switch payloads relative to the **switch** opcode, and branch offsets
in code units (×2 for bytes).

The ML tests pin the contracts that are easy to break: exact feature schema,
NaN for absent artifacts, label parsing, dedupe order, grouped folds that
never share a family, and the artifact/feature-version gate.

## Project layout

```
main.py                  orchestrates stages 1-7
make_corpus.py           synthetic APK corpus for wiring smoke tests

tools/
  binary.py              bounded head/tail reads, EOCD lookup
  zip_audit.py           central-directory parsing + forensic audit
  manifest.py            binary XML (AXML) parser
  resource_table.py      resources.arsc parser
  dex.py                 DEX parser, integrity checks, instruction walk
  smali.py               bytecode decode + smali rendering + call graph
  api_report.py          sensitive-API reachability from entry points
  translate.py           optional LLM smali -> Java client
  features.py            numeric features for the rule score
  report.py              hand-tuned additive risk score (baseline)
  ml/
    features.py          149-column versioned feature extractor
    dataset.py           corpus scan, labels, dedupe, JSONL dataset
    train.py             grouped CV, calibration, ablation, importance
    model.py             artifact I/O, scoring, attribution

tests/                   pytest suite (parsers, audit, smali, ML contracts)
```

## Limitations

- **Static analysis only.** Nothing is executed; behaviour that only appears
  at runtime (droppers, dynamically fetched payloads) is out of scope.
- **No trained model is shipped.** The ML stage is inert until you train one
  on your own labeled corpus.
- **Adversarial input is handled defensively, not perfectly.** The parsers
  are hardened against malformed and oversized structures, but a crafted APK
  can still force a slow or partially degraded analysis; block-level errors
  are reported rather than hidden.
- **A high score is a review signal, not a verdict.** The explainability path
  exists precisely so a human can disagree with the model.

## Roadmap

- Signing-certificate and manifest-fingerprint features (current vector stops
  at structure).
- Cross-APK family clustering from the DEX-set hash plus code fingerprints.
- Optional `xgboost` and `TabPFN` reference models for small-label regimes.
