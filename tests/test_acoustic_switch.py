"""E1 acoustic source switch (egemaps | wavlm): loaders, model input dim, SVM bag stats, result separation."""
import json

import numpy as np
import pytest
import torch

from TRAMIL import comparators, diag_instance_svm, e1_run, exp_e1, features, model, standardize
from test_e1_run import SUBJECTS, _make_input

D_W = 16  # stand-in for the 768 WavLM dims


@pytest.fixture(autouse=True)
def fake_embeddings(monkeypatch):
    monkeypatch.setattr(features, "embed_texts", lambda t, batch_size=32: np.ones((len(list(t)), 768), np.float32) / 28)


def _add_w(root, text_dir_split=False):
    """_W.npy caches (rows aligned with _X.npy); optionally in a separate dataset folder without text embeddings."""
    feat = root / "ds" / "features"
    wdir = root / "ds" / "wavlm" if text_dir_split else feat
    wdir.mkdir(parents=True, exist_ok=True)
    g = np.random.default_rng(1)
    for i, s in enumerate(SUBJECTS):
        W = g.standard_normal((3, D_W)).astype(np.float32)
        W[:, 0] += 2.0 * (i % 2)
        np.save(wdir / f"{s}_W.npy", W)
    return wdir


def _items(root, **kw):
    return exp_e1.load_items(root / "ds" / "instances" / "detailed_lables.csv", root / "ds" / "instances",
                             root / "ds" / "features", **kw)


def test_model_accepts_an_arbitrary_input_dim_and_keeps_88_by_default():
    X, R = torch.randn(5, 768), torch.rand(5, 8)
    for m in (model.ResidualBipartite(d_in=768), model.ResidualBipartite(learned_only=True, d_in=768),
              comparators.ABMIL(d_in=768), comparators.ABMIL(gated=True, d_in=768)):
        assert m(X, R).shape == () and torch.isfinite(m(X, R))
    assert model.ResidualBipartite().encoder.fc.in_features == 88
    assert comparators.ABMIL().encoder.fc.in_features == 88
    assert model.n_params(model.ResidualBipartite(d_in=768)) - model.n_params(model.ResidualBipartite()) == (768 - 88) * 32


def test_standardize_is_uniform_over_all_dims_for_wavlm():
    g = np.random.default_rng(0)
    bags = [g.normal(3.0, 2.0, size=(4, 768)).astype(np.float32) for _ in range(6)]
    mu, s = standardize.fit(bags)
    Z = np.concatenate([standardize.apply(b, mu, s) for b in bags])
    assert Z.shape[1] == 768 and np.allclose(Z.mean(0), 0, atol=1e-4) and np.allclose(Z.std(0), 1, atol=1e-4)


def test_svm_bag_stats_are_mean_plus_sd_of_the_input_dim():
    assert comparators.bag_stats(np.ones((3, 768), np.float32)).shape == (1536,)
    assert comparators.bag_stats(np.ones((3, 88), np.float32)).shape == (176,)


def test_egemaps_loader_is_unchanged_by_default_and_wavlm_reads_w(tmp_path):
    _make_input(tmp_path)
    _add_w(tmp_path)
    X_saved = np.load(tmp_path / "ds" / "features" / f"{SUBJECTS[0]}_X.npy")
    W_saved = np.load(tmp_path / "ds" / "features" / f"{SUBJECTS[0]}_W.npy")
    default, _ = _items(tmp_path)
    explicit, _ = _items(tmp_path, acoustic="egemaps")
    wavlm, info = _items(tmp_path, acoustic="wavlm")
    assert default[0]["subject"] == SUBJECTS[0]
    assert default[0]["X"].shape == (3, 88) and np.array_equal(default[0]["X"], X_saved)
    assert np.array_equal(default[0]["X"], explicit[0]["X"]) and np.array_equal(default[0]["R"], explicit[0]["R"])
    assert wavlm[0]["X"].shape == (3, D_W) and np.array_equal(wavlm[0]["X"], W_saved)
    assert [it["subject"] for it in wavlm] == [it["subject"] for it in default]
    assert np.array_equal(wavlm[0]["R"], default[0]["R"]) and info["bags_used"] == len(SUBJECTS)


def test_run_e1_trains_every_arm_on_a_non_88_input_dim(tmp_path):
    _make_input(tmp_path)
    _add_w(tmp_path)
    items, _ = _items(tmp_path, acoustic="wavlm")
    res = exp_e1.run_e1(items, n_folds=2, seeds=(11,), max_epochs=1)
    assert set(res["arms"]) == {"residual", "learned_only", "abmil", "gated_abmil", "svm"}


def test_e1_run_wavlm_finds_w_features_text_elsewhere_and_writes_to_a_separate_dir(tmp_path, monkeypatch):
    _make_input(tmp_path / "in")
    root = tmp_path / "in"
    wdir = _add_w(root, text_dir_split=True)
    monkeypatch.setenv("TRAMIL_ACOUSTIC", "wavlm")
    monkeypatch.setenv("FORGE_CLONE_ROOT", str(tmp_path / "clone"))
    monkeypatch.delenv("FORGE_RUN_MODE", raising=False)
    found = e1_run._find([root], acoustic="wavlm")
    assert found["features"] == wdir and found["text"] == root / "ds" / "features"
    assert e1_run._find([root])["features"] == root / "ds" / "features"  # default still locates _X.npy
    e1_run.run(root, None, seeds=(11,), max_epochs=1, repetitions=1, n_folds=2)
    out = tmp_path / "clone" / "TRAMIL" / "Results" / "E1_wavlm"
    assert (out / "partition_r0_f0.json").exists() and (out / "E1_full.json").exists()
    assert not (tmp_path / "clone" / "TRAMIL" / "Results" / "E1").exists()


def test_diag_report_files_for_wavlm_do_not_overwrite_the_egemaps_ones(tmp_path):
    items = [{"subject": i, "y": i % 2, "X": np.random.default_rng(i).normal(size=(4, 5)).astype(np.float32)}
             for i in range(10)]
    res = diag_instance_svm.run(items, repetitions=1, n_folds=2, grid=[0.1])
    diag_instance_svm.report(res, tmp_path, acoustic="wavlm")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["diag_instance_svm_wavlm.json", "diag_instance_svm_wavlm.md"]
    assert "WavLM" in (tmp_path / "diag_instance_svm_wavlm.md").read_text().splitlines()[0]
    json.loads((tmp_path / "diag_instance_svm_wavlm.json").read_text())
    diag_instance_svm.report(res, tmp_path)
    assert (tmp_path / "diag_instance_svm.md").read_text().splitlines()[0] == "# Diagnostic: instance-level SVM (eGeMAPS only)"
