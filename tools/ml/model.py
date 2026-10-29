"""Artifact I/O, inference and local explanations for the triage model.

The trained artifact is a plain joblib dict so it stays inspectable:

    feature_names / feature_version / block_of / baseline (train medians)
    model (CalibratedClassifier) / metrics / threshold / seed / metadata

`TriageModel.explain()` answers the only question a forensic operator
actually asks: "which signals moved this app's score?". It perturbs one
feature at a time back to the training median and reports the delta. That is
cheap, dependency-free and model-agnostic; swap in SHAP TreeExplainer when
`pip install shap` is available and you get the same table with better
credit assignment for correlated features.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

from tools.ml.features import FEATURE_VERSION


class ArtifactVersionError(ValueError):
    """Artifact was trained on a different feature schema."""


# ---------------------------------------------------------------------------
# CALIBRATION WRAPPER
# ---------------------------------------------------------------------------

class CalibratedClassifier:
    """base estimator -> isotonic calibration fitted on grouped-OOF scores."""

    def __init__(self, base, iso):
        self.base = base
        self.iso = iso

    def fit(self, X, y):                      # pragma: no cover - already fitted
        raise NotImplementedError("CalibratedClassifier is fitted by train.py")

    def predict_proba(self, X) -> np.ndarray:
        raw = positive_proba(self.base, X)
        return np.asarray(self.iso.predict(raw), dtype=float)


def positive_proba(estimator, X) -> np.ndarray:
    """P(malware) for any binary sklearn estimator, NaN/degenerate safe."""
    proba = estimator.predict_proba(X)
    proba = np.asarray(proba, dtype=float)
    if proba.ndim == 1:
        return proba
    classes = list(getattr(estimator, "classes_", [0, 1]))
    if len(classes) == 1:                     # single-class fold
        return np.full(len(X), float(classes[0]))
    return proba[:, classes.index(1)]


# ---------------------------------------------------------------------------
# ARTIFACT
# ---------------------------------------------------------------------------

def save_artifact(path, *, model, metrics, threshold, seed, n_train, n_groups,
                  feature_names, baseline, importance=None, extra=None) -> None:
    import joblib

    artifact = {
        "feature_version": FEATURE_VERSION,
        "feature_names": list(feature_names),
        "baseline": [float(v) for v in baseline],
        "model": model,
        "metrics": metrics,
        "threshold": float(threshold),
        "seed": int(seed),
        "n_train": int(n_train),
        "n_groups": int(n_groups),
        "importance": list(importance or []),
        "extra": dict(extra or {}),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path)


def load_artifact(path) -> dict:
    import joblib

    artifact = joblib.load(path)
    if not isinstance(artifact, dict) or "model" not in artifact:
        raise ArtifactVersionError(f"{path}: not an APKAI triage artifact")
    if artifact.get("feature_version") != FEATURE_VERSION:
        raise ArtifactVersionError(
            f"{path}: trained on {artifact.get('feature_version')!r}, "
            f"code is {FEATURE_VERSION!r} - retrain (features changed)")
    return artifact


# ---------------------------------------------------------------------------
# INFERENCE
# ---------------------------------------------------------------------------

@dataclass
class TriageResult:
    score: float
    threshold: float
    label: str
    contributions: list[tuple[str, float]]

    def as_lines(self, top: int | None = None) -> list[str]:
        shown = self.contributions[:top] if top else self.contributions
        lines = [f"score={self.score:.4f} threshold={self.threshold:.4f} "
                 f"verdict={self.label}"]
        for name, delta in shown:
            arrow = "+" if delta >= 0 else "-"
            lines.append(f"  {arrow}{abs(delta):.4f}  {name}")
        return lines


class TriageModel:
    def __init__(self, artifact: dict):
        self.artifact = artifact
        self.model = artifact["model"]
        self.feature_names: list[str] = list(artifact["feature_names"])
        self.baseline = np.asarray(artifact["baseline"], dtype=float)
        self.threshold = float(artifact["threshold"])
        self.metrics = artifact.get("metrics") or {}
        self.importance: list = artifact.get("importance") or []

    @classmethod
    def load(cls, path) -> "TriageModel":
        return cls(load_artifact(path))

    # -- vectors ---------------------------------------------------------
    def vector(self, row: dict) -> np.ndarray:
        """Row dict -> (1, n_features), missing/NaN filled from train medians."""
        values = np.empty(len(self.feature_names), dtype=float)
        for index, name in enumerate(self.feature_names):
            value = row.get(name, np.nan)
            try:
                value = float(value)
            except (TypeError, ValueError):
                value = np.nan
            values[index] = self.baseline[index] if value != value else value
        return values.reshape(1, -1)

    def score_vector(self, vector: np.ndarray) -> float:
        return float(np.clip(self.model.predict_proba(vector)[0], 0.0, 1.0))

    def score(self, row: dict) -> float:
        return self.score_vector(self.vector(row))

    # -- explanation -----------------------------------------------------
    def explain(self, row: dict, top: int = 8) -> TriageResult:
        vector = self.vector(row)
        score = self.score_vector(vector)
        deltas: list[tuple[str, float]] = []
        for index, name in enumerate(self.feature_names):
            if vector[0, index] == self.baseline[index]:
                continue                       # already at the reference
            probe = vector.copy()
            probe[0, index] = self.baseline[index]
            deltas.append((name, score - self.score_vector(probe)))
        deltas.sort(key=lambda item: abs(item[1]), reverse=True)
        verdict = "malware" if score >= self.threshold else "benign"
        return TriageResult(score, self.threshold, verdict, deltas[:top])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _main(argv=None) -> int:
    from tools.ml.features import extract_apk_features

    parser = argparse.ArgumentParser(description="Score one APK")
    parser.add_argument("--model", default="models/apkai_triage.joblib")
    parser.add_argument("--apk", required=True)
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--fast", action="store_true",
                        help="skip opcode + call-graph features")
    args = parser.parse_args(argv)

    model = TriageModel.load(args.model)
    features, meta = extract_apk_features(args.apk, deep=not args.fast)
    result = model.explain(features, top=args.top)
    print(f"APK        : {args.apk}")
    print(f"Rule score : {meta.get('rule_score')}/100   (existing heuristic)")
    for line in result.as_lines():
        print(line)
    if model.importance:
        print("\ntop global features (permutation importance):")
        for item in model.importance[:args.top]:
            if isinstance(item, dict):
                name, value = item.get("feature"), item.get("importance", 0.0)
            else:                              # tuple form, older artifacts
                name, value = item
            print(f"  {value:+.5f}  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
