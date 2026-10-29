"""Tests for the ML pipeline (tools/ml).

The feature tests build a synthetic APK byte-by-byte (same builder the DEX
tests use) and deliberately ship NO manifest/arsc: the contract is that an
absent artifact becomes NaN, never 0.
"""
import math
import struct
import zipfile

import pytest

from tests.test_smali import S, base_sections, build_dex, simple_foo_class
from tools.ml.dataset import dedupe_rows, read_labels, sha256_file
from tools.ml.features import (BLOCKS, FEATURE_VERSION, FIELDS, block_of,
                               extract_apk_features)


def _mini_apk(path, payload: bytes = b"0" * 5000):
    insns = struct.pack("<2H", 0x000E, 0x0000)
    dex = build_dex(strings=S, types=base_sections()[0],
                    protos=base_sections()[1], methods=base_sections()[2],
                    classes=simple_foo_class(insns))
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("classes.dex", dex)
        zf.writestr("assets/blob.bin", payload)
    return dex


# ---------------------------------------------------------------------------
# FEATURES
# ---------------------------------------------------------------------------

def test_feature_row_schema_and_missing_artifacts_are_nan(tmp_path):
    apk = tmp_path / "mini.apk"
    _mini_apk(apk)

    row, meta = extract_apk_features(apk, deep=True)

    assert set(row) == set(FIELDS)                    # exactly the schema
    assert meta["feature_version"] == FEATURE_VERSION
    assert meta["archive_ok"] is True
    assert row["dex.real_count"] == 1
    assert row["dex.classes"] == 1
    # no AndroidManifest.xml / resources.arsc in this fixture
    assert math.isnan(row["axml.permission_count"])
    assert math.isnan(row["arsc.package_count"])


def test_feature_blocks_are_complete_and_unique():
    assert set(block_of(name) for name in FIELDS) == set(BLOCKS)
    assert len(set(FIELDS)) == len(FIELDS)
    assert all(block_of(name) in BLOCKS for name in FIELDS)


def test_deep_and_fast_agree_on_header_features(tmp_path):
    apk = tmp_path / "mini.apk"
    _mini_apk(apk)

    fast, _ = extract_apk_features(apk, deep=False)
    deep, _ = extract_apk_features(apk, deep=True)

    assert fast["dex.classes"] == deep["dex.classes"]
    assert fast["zip.total_entries"] == deep["zip.total_entries"]
    # graph/opcode columns only exist in deep mode
    assert all(math.isnan(fast[name]) for name in FIELDS
               if block_of(name) == "graph")
    assert not math.isnan(deep["dex.op_entropy"])


def test_non_apk_input_degrades_to_nan_row(tmp_path):
    apk = tmp_path / "junk.apk"
    apk.write_bytes(b"not a zip at all")

    row, meta = extract_apk_features(apk, deep=True)

    assert set(row) == set(FIELDS)
    assert meta["archive_ok"] is False
    assert meta["errors"]
    assert all(math.isnan(value) for value in row.values())


# ---------------------------------------------------------------------------
# DATASET
# ---------------------------------------------------------------------------

def test_labels_parsing_accepts_words_and_numbers(tmp_path):
    path = tmp_path / "labels.csv"
    path.write_text("sha256,label,family\nAA,malware,fam1\nbb,0,\n")

    labels = read_labels(path)

    assert labels["aa"] == {"label": 1, "family": "fam1"}
    assert labels["bb"] == {"label": 0, "family": ""}


def test_labels_reject_unknown_value(tmp_path):
    path = tmp_path / "labels.csv"
    path.write_text("sha256,label\ncc,maybe\n")
    with pytest.raises(ValueError):
        read_labels(path)


def test_dedupe_drops_same_apk_then_same_dex_set():
    rows = [
        {"sha256": "a", "label": 1, "meta": {"dex_set_hash": "x"}},
        {"sha256": "a", "label": 1, "meta": {"dex_set_hash": "x"}},   # same file
        {"sha256": "b", "label": 0, "meta": {"dex_set_hash": "x"}},   # payload twin
        {"sha256": "c", "label": 1, "meta": {"dex_set_hash": "y"}},
    ]

    kept, dropped = dedupe_rows(rows)

    assert [row["sha256"] for row in kept] == ["a", "c"]
    assert [reason for _row, reason in dropped] == ["sha256", "dex_set"]


def test_sha256_file_matches_hashlib(tmp_path):
    import hashlib

    apk = tmp_path / "x.apk"
    apk.write_bytes(b"payload" * 1000)
    assert sha256_file(apk) == hashlib.sha256(apk.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# METRICS / SPLITS / ARTIFACT (need sklearn)
# ---------------------------------------------------------------------------

def test_metrics_rank_perfect_separation():
    np = pytest.importorskip("numpy")
    pytest.importorskip("sklearn")
    from tools.ml.train import recall_at_fpr, summarize, threshold_at_fpr

    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])

    metrics = summarize(y, scores, max_fpr=0.25)
    assert metrics["pr_auc"] == 1.0
    assert metrics["roc_auc"] == 1.0
    assert recall_at_fpr(y, scores, 0.25) == 1.0
    assert 0.4 < threshold_at_fpr(y, scores, 0.25) <= 0.6


def test_grouped_folds_never_share_a_family():
    np = pytest.importorskip("numpy")
    pytest.importorskip("sklearn")
    from tools.ml.train import make_folds

    y = np.array([0, 1] * 6)
    groups = np.array([f"fam{i % 4}" for i in range(12)])

    folds = make_folds(y, groups, n_splits=4, seed=0)

    assert len(folds) == 4
    for train_idx, test_idx in folds:
        assert not (set(groups[train_idx]) & set(groups[test_idx]))


def test_artifact_roundtrip_and_single_feature_explanation(tmp_path):
    np = pytest.importorskip("numpy")
    pytest.importorskip("sklearn")
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression

    from tools.ml.model import (ArtifactVersionError, CalibratedClassifier,
                                TriageModel, load_artifact, save_artifact)

    names = ["zip.total_entries", "dex.classes", "graph.dead_ratio"]
    X = np.array([[1.0, 1.0, 0.0], [2.0, 2.0, 0.0], [8.0, 9.0, 0.5],
                  [9.0, 8.0, 0.6]])
    y = np.array([0, 0, 1, 1])
    base = LogisticRegression().fit(X, y)
    iso = IsotonicRegression(out_of_bounds="clip").fit([0.05, 0.4, 0.8, 0.95], y)

    path = tmp_path / "model.joblib"
    save_artifact(path, model=CalibratedClassifier(base, iso), metrics={},
                  threshold=0.5, seed=0, n_train=4, n_groups=2,
                  feature_names=names, baseline=[0.0, 0.0, 0.0])

    model = TriageModel.load(path)
    row = {"zip.total_entries": 9.0, "dex.classes": 9.0,
           "graph.dead_ratio": 0.6}
    result = model.explain(row, top=3)

    assert 0.0 <= result.score <= 1.0
    assert result.label in ("benign", "malware")
    assert result.contributions                       # non-empty attribution
    assert all(name in names for name, _delta in result.contributions)

    # a stale artifact must fail loudly instead of silently mis-scoring
    artifact = load_artifact(path)
    artifact["feature_version"] = "apkai-ml-0"
    import joblib
    joblib.dump(artifact, path)
    with pytest.raises(ArtifactVersionError):
        load_artifact(path)


def test_drop_constant_columns_is_nan_safe():
    np = pytest.importorskip("numpy")
    pytest.importorskip("sklearn")
    from tools.ml.train import DropConstantColumns

    X = np.array([[1.0, np.nan, 5.0],
                  [2.0, np.nan, 5.0],
                  [3.0, np.nan, 6.0]])

    mask = DropConstantColumns().fit(X)
    kept = mask.transform(X)

    assert kept.shape == (3, 2)          # all-NaN and constant column dropped
    assert list(mask.get_support(indices=True)) == [0, 2]
