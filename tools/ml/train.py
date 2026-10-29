"""Train + evaluate the APKAI triage model.

Evaluation protocol (the part that makes the numbers mean something):
  * grouped CV by malware family / APK hash - a sample never shares its group
    with a test fold, so repackaged variants cannot leak across the split;
  * identical folds and seeds for every model, including the hand-tuned
    rule score from tools/report.py (that is the baseline to beat);
  * metrics: PR-AUC, recall at 1% FPR, ROC-AUC and Brier after isotonic
    calibration - never plain accuracy on an imbalanced corpus;
  * every run writes a JSON + Markdown report so results are comparable
    across feature-version bumps.

    python3 -m tools.ml.train --dataset data/dataset.jsonl \
        --out models/apkai_triage.joblib --folds 5 --seed 42 --ablation
"""
from __future__ import annotations

import argparse
import json
import math
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from tools.ml.dataset import load_dataset
from tools.ml.features import BLOCKS, FEATURE_VERSION, FIELDS, block_of
from tools.ml.model import CalibratedClassifier, positive_proba, save_artifact

DEFAULT_MODELS = ("hgb", "logreg_l1", "rf")


class DropConstantColumns(BaseEstimator, TransformerMixin):
    """NaN-safe feature mask fitted per fold.

    A column that never varies inside a training fold (an all-zero counter, a
    whole NaN block for an APK without resources.arsc) makes sklearn's
    histogram binner raise "window shape cannot be larger than input array
    shape". Fitting the mask on train and replaying it at predict keeps the
    test fold untouched, unlike dropping columns on the full dataset.
    """

    def fit(self, X, y=None):
        array = np.asarray(X, dtype=float)
        self.n_features_in_ = array.shape[1]
        with warnings.catch_warnings():      # all-NaN column: variance is NaN
            warnings.simplefilter("ignore", RuntimeWarning)
            variance = np.nanvar(array, axis=0, ddof=0)
        self.support_ = np.isfinite(variance) & (variance > 0)
        return self

    def transform(self, X):
        return np.asarray(X, dtype=float)[:, self.support_]

    def get_support(self, indices: bool = False):
        return np.flatnonzero(self.support_) if indices else self.support_.copy()


# ---------------------------------------------------------------------------
# MODEL FACTORIES (fresh instance per fold: sklearn estimators are stateful)
# ---------------------------------------------------------------------------

def make_logreg_l1(seed: int):
    """Drebin-style linear model: scale, then L1-select features.

    sklearn 1.8 deprecated `penalty` in favour of `l1_ratio`; pick the knob
    that actually means "L1" on the installed version instead of silently
    training an L2 model on an older release.
    """
    import re

    import sklearn
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    major, minor = (int(part) for part in
                    re.match(r"(\d+)\.(\d+)", sklearn.__version__).groups())
    kwargs = dict(solver="liblinear", C=0.1, class_weight="balanced",
                  max_iter=5000, random_state=seed)
    if (major, minor) >= (1, 8):
        kwargs["l1_ratio"] = 1.0
    else:
        kwargs["penalty"] = "l1"
    return Pipeline([
        ("impute", SimpleImputer(strategy="median",
                                 keep_empty_features=True)),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(**kwargs)),
    ])


def make_rf(seed: int):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline

    return Pipeline([
        ("impute", SimpleImputer(strategy="median",
                                keep_empty_features=True)),
        ("clf", RandomForestClassifier(
            n_estimators=400, min_samples_leaf=2, max_features="sqrt",
            class_weight="balanced_subsample", n_jobs=1, random_state=seed)),
    ])


def make_hgb(seed: int):
    """Workhorse: histogram gradient boosting, NaN-native, no extra deps.

    DropConstantColumns first: constant/all-NaN columns break the histogram
    binner and carry no information anyway.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.pipeline import Pipeline

    return Pipeline([
        ("drop_constant", DropConstantColumns()),
        ("clf", HistGradientBoostingClassifier(
            max_iter=400, learning_rate=0.06, max_leaf_nodes=31,
            min_samples_leaf=20, l2_regularization=1.0,
            class_weight="balanced",
            early_stopping=False,             # keep folds honest
            random_state=seed)),
    ])


def make_xgb(seed: int, positives_ratio: float = 1.0):
    """Optional: pip install xgboost. Same role as hgb, slightly stronger."""
    from xgboost import XGBClassifier

    return XGBClassifier(
        n_estimators=500, max_depth=5, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.7, reg_lambda=2.0,
        min_child_weight=1.0, tree_method="hist", eval_metric="aucpr",
        scale_pos_weight=positives_ratio, n_jobs=-1, random_state=seed,
    )


def build_models(seed: int, y: np.ndarray, names=DEFAULT_MODELS) -> dict:
    out: dict = {}
    for name in names:
        if name == "hgb":
            out[name] = lambda s=seed: make_hgb(s)
        elif name == "logreg_l1":
            out[name] = lambda s=seed: make_logreg_l1(s)
        elif name == "rf":
            out[name] = lambda s=seed: make_rf(s)
        elif name == "xgb":
            negatives = int((y == 0).sum())
            positives = max(1, int((y == 1).sum()))
            out[name] = lambda s=seed, r=negatives / positives: make_xgb(s, r)
        else:
            raise SystemExit(f"unknown model {name!r}")
    return out


# ---------------------------------------------------------------------------
# METRICS
# ---------------------------------------------------------------------------

def recall_at_fpr(y, scores, max_fpr: float = 0.01) -> float:
    """TPR at the lowest threshold whose FPR is still <= max_fpr."""
    from sklearn.metrics import roc_curve

    fpr, tpr, _ = roc_curve(np.asarray(y), np.asarray(scores))
    mask = fpr <= max_fpr
    return float(np.max(tpr[mask])) if mask.any() else math.nan


def summarize(y, scores, max_fpr: float = 0.01) -> dict:
    y = np.asarray(y).astype(int)
    scores = np.asarray(scores, dtype=float)
    out = {"n": int(y.size), "positives": int(y.sum())}
    if y.size == 0 or len(np.unique(y)) < 2:
        out.update({k: None for k in
                    ("pr_auc", "roc_auc", "recall_at_fpr", "brier")})
        return out
    from sklearn.metrics import (average_precision_score, brier_score_loss,
                                 roc_auc_score)
    finite = np.nan_to_num(scores, nan=0.5)
    out["pr_auc"] = float(average_precision_score(y, finite))
    out["roc_auc"] = float(roc_auc_score(y, finite))
    out["recall_at_fpr"] = recall_at_fpr(y, finite, max_fpr)
    out["brier"] = float(brier_score_loss(y, np.clip(finite, 0.0, 1.0)))
    return out


def aggregate(folds: list[dict]) -> tuple[dict, dict]:
    keys = ("pr_auc", "roc_auc", "recall_at_fpr", "brier")
    mean, std = {}, {}
    for key in keys:
        values = [f[key] for f in folds if f.get(key) is not None]
        mean[key] = float(np.mean(values)) if values else None
        std[key] = float(np.std(values)) if values else None
    return mean, std


def threshold_at_fpr(y, scores, max_fpr: float = 0.01) -> float:
    """Operating point: highest TPR with FPR <= max_fpr, else 0.5."""
    from sklearn.metrics import roc_curve

    fpr, tpr, thresholds = roc_curve(np.asarray(y), np.asarray(scores))
    mask = fpr <= max_fpr
    if not mask.any():
        return 0.5
    index = int(np.argmax(np.where(mask, tpr, -1.0)))
    return float(np.clip(thresholds[index], 0.0, 1.0))


# ---------------------------------------------------------------------------
# CROSS-VALIDATION
# ---------------------------------------------------------------------------

def make_folds(y: np.ndarray, groups, n_splits: int, seed: int):
    """StratifiedGroupKFold: class balance per fold, no group leakage."""
    from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

    y = np.asarray(y)
    groups = np.asarray(groups)
    k = max(2, min(int(n_splits), len(set(groups.tolist()))))
    try:
        splitter = StratifiedGroupKFold(n_splits=k, shuffle=True,
                                        random_state=seed)
        return list(splitter.split(np.zeros(len(y)), y, groups))
    except ValueError:
        return list(GroupKFold(n_splits=k).split(np.zeros(len(y)), y, groups))


def oof_scores(factory, X, y, folds) -> np.ndarray:
    scores = np.full(len(y), math.nan)
    for train_idx, test_idx in folds:
        model = factory()
        model.fit(X[train_idx], y[train_idx])
        scores[test_idx] = positive_proba(model, X[test_idx])
    return scores


def varying_columns(X, columns) -> list[int]:
    """Global (label-free) pre-filter so an empty block cannot be fitted."""
    keep = []
    for index in columns:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            variance = np.nanvar(X[:, index], ddof=0)
        if np.isfinite(variance) and variance > 0:
            keep.append(index)
    return keep


def cross_validate(factory, X, y, folds, max_fpr: float) -> dict:
    """Grouped CV for one estimator. A failure is reported, never raised:
    on a real corpus a single degenerate fold must not abort the run."""
    fold_metrics, scores = [], np.full(len(y), math.nan)
    empty = {"pr_auc": None, "roc_auc": None, "recall_at_fpr": None,
             "brier": None}
    try:
        for train_idx, test_idx in folds:
            model = factory()
            model.fit(X[train_idx], y[train_idx])
            proba = positive_proba(model, X[test_idx])
            scores[test_idx] = proba
            fold_metrics.append(summarize(y[test_idx], proba, max_fpr))
    except Exception as exc:                  # degenerate block / tiny fold
        return {"folds": fold_metrics, "mean": dict(empty), "std": dict(empty),
                "overall": dict(empty), "oof": scores,
                "error": f"{type(exc).__name__}: {exc}"}
    mean, std = aggregate(fold_metrics)
    return {
        "folds": fold_metrics,
        "mean": mean,
        "std": std,
        "overall": summarize(y, np.nan_to_num(scores, nan=0.5), max_fpr),
        "oof": scores,
        "error": None,
    }


# ---------------------------------------------------------------------------
# DATASET -> MATRIX
# ---------------------------------------------------------------------------

def to_matrix(rows: list[dict], columns=FIELDS) -> np.ndarray:
    return np.array([[row["features"].get(name, math.nan) for name in columns]
                     for row in rows], dtype=np.float64)


def load_xy(dataset: Path):
    rows = load_dataset(dataset)
    for row in rows:
        version = (row.get("meta") or {}).get("feature_version")
        if version != FEATURE_VERSION:
            raise SystemExit(
                f"{dataset}: row {row.get('path')} uses feature version "
                f"{version!r}, code is {FEATURE_VERSION!r}; rebuild the dataset")
    rows = [r for r in rows if r.get("label") in (0, 1)]
    if len(rows) < 6:
        raise SystemExit(
            f"need at least 6 labeled APKs to cross-validate, got {len(rows)}")
    X = to_matrix(rows)
    y = np.array([r["label"] for r in rows], dtype=int)
    groups = np.array([r.get("family") or r["sha256"] for r in rows])
    rule = np.array([(r["meta"].get("rule_score") or 0) / 100.0 for r in rows])
    return rows, X, y, groups, rule


# ---------------------------------------------------------------------------
# ABLATION + IMPORTANCE
# ---------------------------------------------------------------------------

def run_ablation(X, y, folds, max_fpr: float, seed: int) -> dict:
    """One model, one feature block at a time: what actually carries signal?"""
    out: dict = {}
    for block in BLOCKS + ("all",):
        columns = [i for i, name in enumerate(FIELDS)
                   if block == "all" or block_of(name) == block]
        columns = varying_columns(X, columns)
        if not columns:
            out[block] = {"n_features": 0,
                          "note": "no varying features in this corpus"}
            continue
        try:
            scores = oof_scores(lambda s=seed: make_hgb(s), X[:, columns], y,
                                folds)
        except Exception as exc:
            out[block] = {"n_features": len(columns),
                          "note": f"{type(exc).__name__}: {exc}"}
            continue
        out[block] = summarize(y, scores, max_fpr)
        out[block]["n_features"] = len(columns)
    return out


def run_importance(factory, X, y, seed: int, top: int = 20) -> list:
    from sklearn.inspection import permutation_importance

    model = factory()
    model.fit(X, y)
    result = permutation_importance(
        model, X, y, n_repeats=5, random_state=seed,
        scoring="average_precision", n_jobs=1)
    ranked = sorted(zip(FIELDS, result.importances_mean.tolist()),
                    key=lambda item: item[1], reverse=True)
    return [{"feature": name, "importance": float(value)}
            for name, value in ranked[:top]]


# ---------------------------------------------------------------------------
# REPORTING
# ---------------------------------------------------------------------------

def _fmt(value, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "  -  "
    return f"{value:.{digits}f}"


def print_table(results: dict, baseline: dict | None) -> None:
    header = (f"{'model':<12}{'PR-AUC':>9}{'±std':>8}"
              f"{'rec@1%FPR':>11}{'ROC-AUC':>9}{'Brier':>8}")
    print(header)
    print("-" * len(header))
    if baseline:
        print(f"{'rule_score':<12}{_fmt(baseline.get('pr_auc')):>9}"
              f"{'':>8}{_fmt(baseline.get('recall_at_fpr')):>11}"
              f"{_fmt(baseline.get('roc_auc')):>9}"
              f"{_fmt(baseline.get('brier')):>8}")
    for name, result in results.items():
        mean = result["mean"]
        if result.get("error"):
            print(f"{name:<12}  error: {result['error']}")
            continue
        print(f"{name:<12}{_fmt(mean.get('pr_auc')):>9}"
              f"{_fmt(result['std'].get('pr_auc')):>8}"
              f"{_fmt(mean.get('recall_at_fpr')):>11}"
              f"{_fmt(mean.get('roc_auc')):>9}"
              f"{_fmt(mean.get('brier')):>8}")


def write_report(results_dir: Path, payload: dict) -> tuple[Path, Path]:
    results_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    json_path = results_dir / f"eval-{stamp}.json"
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)

    lines = [f"# APKAI triage evaluation ({stamp})", "",
             f"- feature version: `{payload['dataset']['feature_version']}`",
             f"- rows: {payload['dataset']['n_rows']} "
             f"({payload['dataset']['positives']} malware, "
             f"{payload['dataset']['n_rows'] - payload['dataset']['positives']} benign), "
             f"groups: {payload['dataset']['n_groups']}",
             f"- grouped CV: {payload['dataset']['folds']} folds, "
             f"seed {payload['dataset']['seed']}, "
             f"FPR target {payload['dataset']['max_fpr']:.0%}", "",
             "| model | PR-AUC | ±std | recall@1%FPR | ROC-AUC | Brier |",
             "|---|---|---|---|---|---|"]
    for name, result in payload["models"].items():
        mean, std = result["mean"], result["std"]
        lines.append(f"| {name} | {_fmt(mean.get('pr_auc'))} | "
                     f"{_fmt(std.get('pr_auc'))} | "
                     f"{_fmt(mean.get('recall_at_fpr'))} | "
                     f"{_fmt(mean.get('roc_auc'))} | "
                     f"{_fmt(mean.get('brier'))} |")
    if payload.get("baseline"):
        base = payload["baseline"]
        lines.append(f"| rule_score (current heuristic) | {_fmt(base.get('pr_auc'))} "
                     f"| - | {_fmt(base.get('recall_at_fpr'))} | "
                     f"{_fmt(base.get('roc_auc'))} | {_fmt(base.get('brier'))} |")

    if payload.get("ablation"):
        lines += ["", "## Feature-block ablation (hgb)", "",
                  "| block | features | PR-AUC | recall@1%FPR |",
                  "|---|---|---|---|"]
        for block, metrics in payload["ablation"].items():
            note = f" {metrics['note']}" if metrics.get("note") else ""
            lines.append(f"| {block} | {metrics.get('n_features')} | "
                         f"{_fmt(metrics.get('pr_auc'))}{note} | "
                         f"{_fmt(metrics.get('recall_at_fpr'))} |")
    if payload.get("importance"):
        lines += ["", "## Top permutation importance", "", "| feature | ΔPR-AUC |",
                  "|---|---|"]
        for item in payload["importance"]:
            lines.append(f"| {item['feature']} | {item['importance']:+.5f} |")

    md_path = results_dir / "eval_latest.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Train the APKAI triage model")
    parser.add_argument("--dataset", type=Path, default=Path("data/dataset.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("models/apkai_triage.joblib"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-fpr", type=float, default=0.01)
    parser.add_argument("--ablation", action="store_true")
    parser.add_argument("--no-importance", action="store_true")
    args = parser.parse_args(argv)

    rows, X, y, groups, rule = load_xy(args.dataset)
    folds = make_folds(y, groups, args.folds, args.seed)
    names = tuple(n.strip() for n in args.models.split(",") if n.strip())
    factories = build_models(args.seed, y, names)

    print(f"dataset : {args.dataset} ({len(rows)} rows, "
          f"{int(y.sum())} malware, {len(set(groups.tolist()))} groups)")
    print(f"features: {X.shape[1]} columns, version {FEATURE_VERSION}")
    print(f"cv      : {len(folds)} grouped folds, seed {args.seed}\n")

    baseline = summarize(y, rule, args.max_fpr)
    results = {name: cross_validate(factory, X, y, folds, args.max_fpr)
               for name, factory in factories.items()}
    print_table(results, baseline)

    best_name = max(results, key=lambda k: results[k]["mean"]["pr_auc"] or 0.0)
    best_factory = factories[best_name]
    oof = results[best_name]["oof"]

    # ---- calibration on grouped out-of-fold scores ----------------------
    from sklearn.isotonic import IsotonicRegression

    oof_finite = np.nan_to_num(oof, nan=0.5)
    iso = IsotonicRegression(out_of_bounds="clip").fit(oof_finite, y)
    calibrated = summarize(y, iso.predict(oof_finite), args.max_fpr)
    threshold = threshold_at_fpr(y, iso.predict(oof_finite), args.max_fpr)
    base_model = best_factory().fit(X, y)
    model = CalibratedClassifier(base_model, iso)

    print(f"\nbest    : {best_name}")
    print(f"calibrated OOF: PR-AUC {_fmt(calibrated['pr_auc'])}, "
          f"recall@1%FPR {_fmt(calibrated['recall_at_fpr'])}, "
          f"Brier {_fmt(calibrated['brier'])} (was "
          f"{_fmt(results[best_name]['mean']['brier'])} raw)")
    print(f"threshold @ {args.max_fpr:.0%} FPR: {threshold:.4f}")

    importance = []
    if not args.no_importance:
        importance = run_importance(best_factory, X, y, args.seed)
        print("\ntop features (permutation importance, ΔPR-AUC):")
        for item in importance[:10]:
            print(f"  {item['importance']:+.5f}  {item['feature']}")

    ablation = {}
    if args.ablation:
        ablation = run_ablation(X, y, folds, args.max_fpr, args.seed)
        print("\nfeature-block ablation (hgb):")
        for block, metrics in ablation.items():
            note = metrics.get("note")
            print(f"  {block:<6}{metrics.get('n_features', 0):>4} cols  "
                  f"PR-AUC {_fmt(metrics.get('pr_auc'))}  "
                  f"recall@1%FPR {_fmt(metrics.get('recall_at_fpr'))}"
                  + (f"  ({note})" if note else ""))

    with warnings.catch_warnings():          # all-NaN columns are expected
        warnings.simplefilter("ignore", RuntimeWarning)
        baseline_median = np.nanmedian(X, axis=0)
    baseline_median = np.where(np.isnan(baseline_median), 0.0, baseline_median)
    save_artifact(
        args.out, model=model, metrics={"calibrated": calibrated,
                                        "raw": results[best_name]["mean"],
                                        "baseline_rule": baseline},
        threshold=threshold, seed=args.seed, n_train=len(y),
        n_groups=len(set(groups.tolist())), feature_names=FIELDS,
        baseline=baseline_median, importance=importance,
        extra={"best_model": best_name, "folds": len(folds)})
    print(f"\nartifact: {args.out}")

    payload = {
        "dataset": {
            "path": str(args.dataset),
            "feature_version": FEATURE_VERSION,
            "n_rows": int(len(y)),
            "positives": int(y.sum()),
            "n_groups": int(len(set(groups.tolist()))),
            "folds": len(folds),
            "seed": args.seed,
            "max_fpr": args.max_fpr,
        },
        "baseline": baseline,
        "models": {name: {"mean": r["mean"], "std": r["std"],
                          "folds": r["folds"]}
                   for name, r in results.items()},
        "calibrated": calibrated,
        "threshold": threshold,
        "best_model": best_name,
        "ablation": ablation,
        "importance": importance,
    }
    json_path, md_path = write_report(args.results_dir, payload)
    print(f"reports : {json_path}\n          {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
