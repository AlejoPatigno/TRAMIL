import json

import numpy as np
import pandas as pd
import pytest

from TRAMIL import e1_run, exp_e1, features

ITEMS = [f"PHQ8_{i}" for i in range(1, 9)]
SUBJECTS = list(range(301, 313))  # 12 train subjects, 6 positive


def _make_input(root):
    """Instances dataset (CSVs only, no wavs) beside a flat features dataset, as mounted on Kaggle."""
    inst, feat = root / "ds" / "instances", root / "ds" / "features"
    inst.mkdir(parents=True)
    feat.mkdir(parents=True)
    rows = [[s, *([2] * 8 if i % 2 else [0] * 8), 1, "train"] for i, s in enumerate(SUBJECTS)]
    pd.DataFrame(rows, columns=["Participant", *ITEMS, "Depression_label", "split"]).to_csv(
        inst / "detailed_lables.csv", index=False)
    pd.DataFrame([[s, i, f"a {s} {i}", 1.0] for s in SUBJECTS for i in range(3)],
                 columns=["id", "idx", "a_text", "a_dur"]).to_csv(inst / "instances_clean.csv", index=False)
    g = np.random.default_rng(0)
    for i, s in enumerate(SUBJECTS):
        X = g.standard_normal((3, 88)).astype(np.float32)
        X[:, :8] += 2.0 * (i % 2)
        np.save(feat / f"{s}_X.npy", X)
        np.save(feat / f"{s}_E.npy", np.eye(768, dtype=np.float32)[:3])
    return root


@pytest.fixture(autouse=True)
def fake_embeddings(monkeypatch):
    monkeypatch.setattr(features, "embed_texts", lambda t, batch_size=32: np.ones((len(list(t)), 768), np.float32) / 28)


def _run(tmp_path, **kw):
    kw = {"seeds": (11,), "max_epochs": 2, "repetitions": 1, "n_folds": 2, **kw}
    return e1_run.run(tmp_path / "in", tmp_path / "out", **kw)


def _partitions(tmp_path):
    return sorted(p.name for p in (tmp_path / "out").glob("partition_r*_f*.json"))


def test_each_unit_writes_a_partition_file_with_every_arm(tmp_path):
    _make_input(tmp_path / "in")
    _run(tmp_path, repetitions=2)
    assert _partitions(tmp_path) == [f"partition_r{r}_f{k}.json" for r in (0, 1) for k in (0, 1)]
    p = json.loads((tmp_path / "out" / "partition_r1_f0.json").read_text())
    assert set(p["arms"]) == {"residual", "learned_only", "abmil", "gated_abmil", "svm"}
    for a in p["arms"].values():
        assert 0 <= a["fold_auroc"] <= 1 and a["oof"] and a["seconds"] >= 0
    assert p["device"]["kind"] in {"cpu", "cuda"} and p["seeds"] == [11] and p["max_epochs"] == 2
    prog = json.loads((tmp_path / "out" / "progress.json").read_text())
    assert prog["done"] == 4 and prog["total"] == 4


def test_existing_partitions_are_skipped_on_resume(tmp_path, monkeypatch):
    _make_input(tmp_path / "in")
    _run(tmp_path)
    calls = []
    real = exp_e1._neural
    monkeypatch.setattr(exp_e1, "_neural", lambda *a, **k: calls.append(1) or real(*a, **k))
    _run(tmp_path)
    assert calls == []
    (tmp_path / "out" / "partition_r0_f1.json").unlink()
    _run(tmp_path)
    assert len(calls) == 4  # one retrained unit: four neural arms, one seed
    # copies from a previous run's directory are picked up too
    (tmp_path / "prev").mkdir()
    (tmp_path / "out" / "partition_r0_f0.json").rename(tmp_path / "prev" / "partition_r0_f0.json")
    calls.clear()
    _run(tmp_path, resume_from=tmp_path / "prev")
    assert calls == [] and (tmp_path / "out" / "partition_r0_f0.json").exists()


def test_partitions_argument_restricts_units(tmp_path):
    _make_input(tmp_path / "in")
    _run(tmp_path, repetitions=2, partitions=["r1_f0", (0, 1)])
    assert _partitions(tmp_path) == ["partition_r0_f1.json", "partition_r1_f0.json"]


def test_merge_is_partial_until_all_units_exist(tmp_path):
    _make_input(tmp_path / "in")
    _run(tmp_path, partitions=["r0_f0"])
    out = tmp_path / "out"
    part = json.loads((out / "E1_full.json").read_text())
    assert part["stage"] == "PARTIAL" and part["missing"] == ["r0_f1"]
    assert "PARTIAL" in (out / "E1_full.md").read_text().splitlines()[0]
    _run(tmp_path)
    full = json.loads((out / "E1_full.json").read_text())
    assert full["stage"] == "FULL" and full["missing"] == []
    assert full["absent_arms"] == ["symptom_guided"]
    assert all(len(a["fold_auroc"]) == 2 for a in full["arms"].values())
    assert sorted(full["arms"]["svm"]["oof"]["r0"]) == [str(s) for s in SUBJECTS]
    assert "abmil" in full["comparisons"]


def test_units_never_leak_test_subjects_into_training(tmp_path):
    _make_input(tmp_path / "in")
    _run(tmp_path)
    tested = []
    for k in (0, 1):
        p = json.loads((tmp_path / "out" / f"partition_r0_f{k}.json").read_text())
        assert set(p["train"]).isdisjoint(p["test"]) and len(p["train"]) + len(p["test"]) == len(SUBJECTS)
        tested += p["test"]
    assert sorted(tested) == SUBJECTS


def test_bags_build_from_csv_and_features_without_wav_files(tmp_path):
    root = _make_input(tmp_path / "in")
    assert not list(root.rglob("*.wav"))
    found = e1_run._find([root])
    items, info = exp_e1.load_items(found["labels"], found["instances"].parent, found["features"])
    assert info["bags_used"] == len(SUBJECTS) and items[0]["X"].shape[1] == 88


def test_stamp_only_under_the_runner_with_commit_device_and_epoch_timing(tmp_path, monkeypatch):
    _make_input(tmp_path / "in")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("FORGE_RUN_MODE", raising=False)
    _run(tmp_path)
    assert not (tmp_path / "shard.json").exists()
    monkeypatch.setenv("FORGE_RUN_MODE", "smoke")
    monkeypatch.setenv("FORGE_CLONE_COMMIT", "a" * 40)
    (tmp_path / "out" / "partition_r0_f0.json").unlink()
    _run(tmp_path)
    s = json.loads((tmp_path / "shard.json").read_text())
    assert s["evidence"]["commit"] and s["environment"]["device"]["kind"] in {"cpu", "cuda"}
    spe = s["runs"]["secondsPerEpoch"]
    assert set(spe) == {"residual", "learned_only", "abmil", "gated_abmil"} and all(v > 0 for v in spe.values())


def test_mounted_datasets_are_found_through_the_forge_map(tmp_path, monkeypatch):
    root = _make_input(tmp_path / "in")
    monkeypatch.setenv("FORGE_INPUT_DATASETS", json.dumps({
        e1_run.FEATURES_REF: str(root / "ds" / "features"), e1_run.INSTANCES_REF: str(root / "ds" / "instances")}))
    found = e1_run._find(e1_run._default_roots())
    assert found["features"].name == "features" and found["labels"].name == "detailed_lables.csv"
