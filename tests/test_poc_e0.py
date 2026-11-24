import json
import os

import numpy as np
import pytest
import torch
from sklearn.linear_model import LogisticRegression

from TRAMIL import poc_e0 as e0


def test_generator_is_deterministic():
    a, b = e0.make_data_e0(3, 4, 0, 0.5), e0.make_data_e0(3, 4, 0, 0.5)
    for x, y in zip(a, b):
        for k in ("X", "R", "Rc", "G", "s", "y", "z"):
            assert torch.equal(x[k], y[k])
    c = e0.make_data_e0(4, 4, 0, 0.5)
    assert not torch.equal(a[0]["X"], c[0]["X"]) or len(a[0]["X"]) != len(c[0]["X"])


def test_bags_differ_across_rho_only_by_the_rho_p_term():
    p, _ = e0.e0_params(2)
    hi, lo = e0.make_data_e0(2, 5, 1, 0.5), e0.make_data_e0(2, 5, 1, 0.0)
    for h, l in zip(hi, lo):
        assert torch.equal(h["z"], l["z"]) and torch.equal(h["s"], l["s"]) and torch.equal(h["R"], l["R"])
        assert torch.equal(h["G"], l["G"]) and h["y"] == l["y"]
        expect = 0.5 * torch.tensor(p, dtype=torch.float64)[h["z"]]
        assert torch.allclose((h["X"] - l["X"]).double(), expect, atol=1e-5)


def test_projection_and_signal_direction_supports():
    p, w = e0.e0_params(7)
    assert p.shape == (8, 16) and np.allclose(p @ p.T, np.eye(8), atol=1e-10)
    assert np.all(p[:, :4] == 0) and np.isclose(np.linalg.norm(w), 1) and np.all(w[4:] == 0)


def _segments(seed, n, split, rho):
    bags = e0.make_data_e0(seed, n, split, rho)
    return torch.cat([b["X"] for b in bags]).numpy(), torch.cat([b["z"] for b in bags]).numpy()


def test_identity_is_decodable_from_audio_only_when_rho_is_positive():
    acc = {}
    for rho in (0.5, 0.0):
        Xtr, ztr = _segments(0, 200, 0, rho)
        Xte, zte = _segments(0, 80, 1, rho)
        acc[rho] = LogisticRegression(max_iter=500).fit(Xtr, ztr).score(Xte, zte)
    assert acc[0.0] < 0.2  # chance is 0.125
    assert acc[0.5] > 0.18 and acc[0.5] > acc[0.0] + 0.06


def test_prior_rows_and_label():
    for b in e0.make_data_e0(1, 6, 0, 0.5):
        n = len(b["X"])
        assert 20 <= n <= 60 and b["R"].shape == (n, 8) and b["R"].dtype == torch.float32
        assert torch.allclose(b["Rc"], b["R"] - b["R"].mean(0), atol=1e-6)
        assert int(b["y"]) == int(b["s"].sum() >= 10)


def test_surrogate_range_centering_and_graph():
    torch.manual_seed(0)
    m = e0.ModelE0("surrogate")
    b = e0.make_data_e0(0, 1, 0, 0.5)[0]
    R, Rc, G = m.surrogate(b["X"])
    R = R.detach()
    assert float(R.min()) > 0 and float(R.max()) < 1
    assert torch.allclose(Rc.sum(0), torch.zeros(8), atol=1e-4)
    assert torch.allclose(G, G.T, atol=1e-5) and float(G.abs().max()) <= 1.0


def test_surrogate_forward_never_reads_the_prompt_prior():
    torch.manual_seed(0)
    m = e0.ModelE0("surrogate")
    b = e0.make_data_e0(0, 1, 0, 0.5)[0]
    ref, _ = m({"X": b["X"]})
    nan = {"X": b["X"], "R": b["R"] * float("nan"), "Rc": b["Rc"] * float("nan"), "G": b["G"] * float("nan")}
    out, rc = m(nan)
    assert torch.equal(out, ref) and rc is not None
    t = e0.ModelE0("text")
    with pytest.raises(KeyError):
        t({"X": b["X"]})
    assert t(b)[1] is None and e0.ModelE0("none")({"X": b["X"]})[1] is None


def test_penalty_gradient_reaches_only_the_surrogate_and_encoder():
    torch.manual_seed(0)
    m = e0.ModelE0("surrogate", wo_init="normal")
    b = e0.make_data_e0(0, 1, 0, 0.5)[0]
    _, rc = m({"X": b["X"]})
    e0.penalty(rc, b["Rc"]).backward()
    g = lambda mod: sum(float(p.grad.abs().sum()) if p.grad is not None else 0.0 for p in mod.parameters())  # noqa: E731
    assert g(m.W_r) > 0 and g(m.enc) > 0
    for mod in (m.W_o, m.head, m.W_g, m.W_a):
        assert g(mod) == 0.0


def test_ssw_zero_for_constant_one_for_perfect_and_omega_from_train_only():
    tr = e0.make_data_e0(0, 40, 0, 0.5)
    te = e0.make_data_e0(0, 20, 2, 0.5)
    ow = e0.omega_plus(tr)
    m1 = sum(int(b["y"]) for b in tr)
    assert np.isclose(ow, (len(tr) - m1) / m1)
    w_te = e0.bag_weights(te, ow)
    assert all(w == (ow if int(b["y"]) else 1.0) for w, b in zip(w_te, te))
    sbar = e0.weighted_train_mean(tr, ow)
    assert np.isclose(e0.ss_w([sbar.expand(8)] * len(te), te, sbar, w_te), 0.0, atol=1e-12)
    assert np.isclose(e0.ss_w([b["s"].double() for b in te], te, sbar, w_te), 1.0)


def test_holm_and_rank_biserial_known_cases():
    assert np.allclose(e0.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    assert np.allclose(e0.holm([0.5, 0.9]), [1.0, 1.0])
    assert np.isclose(e0.rank_biserial(np.array([1.0, 2.0, 3.0, -4.0])), 0.2)
    assert e0.rank_biserial(np.array([1.0, 2.0, 3.0])) == 1.0


def test_bootstrap_ci_is_deterministic_and_covers_the_mean():
    d = np.array([0.1, 0.3, 0.2, 0.4, 0.25, 0.15])
    a, b = e0.bootstrap_ci(d, B=500, seed=1), e0.bootstrap_ci(d, B=500, seed=1)
    assert a == b and a[0] <= d.mean() <= a[1]
    assert e0.bootstrap_ci(d, B=500, seed=2) != a


def test_training_reduces_validation_error_and_restores_best_epoch():
    tr, va = e0.make_data_e0(0, 24, 0, 0.5), e0.make_data_e0(0, 12, 1, 0.5)
    m = e0.train_e0(tr, "none", seed=0, epochs=6, lr=1e-2, val=va, omega_plus=e0.omega_plus(tr))
    h = m.history["val"]
    assert len(h) == 6 and m.history["best_epoch"] == int(np.argmin(h)) + 1
    assert e0.val_mse(m, va) == pytest.approx(min(h), rel=1e-5)


def test_smoke_run_end_to_end_writes_report_and_resume_skips_cached(tmp_path, monkeypatch):
    monkeypatch.delenv("FORGE_SESSION_DEADLINE", raising=False)
    out = e0.run_e0(tmp_path, smoke=True, workers=1)
    js = json.loads((tmp_path / "poc_e0.json").read_text(encoding="utf-8"))
    assert js["smoke"] is True and (tmp_path / "poc_e0.md").exists()
    assert {"gate0", "arms", "comparisons", "curves", "verdicts"} <= set(js)
    assert set(js["gate0"]["configs"]) and js["gate0"]["verdict"] in ("PASS", "FAIL")
    if True:  # smoke runs the arms whatever the gate says
        assert {"P1", "P2a", "P2b", "P4", "P3"} <= set(js["comparisons"])
        assert {"full|rho=0.5", "r12|rho=0", "constant|rho=0.5"} <= set(js["arms"])
        assert (tmp_path / "e0_runs").is_dir() and out["verdicts"] == js["verdicts"]
        runs = sorted((tmp_path / "e0_runs").iterdir())
        stamp = {p.name: p.stat().st_mtime_ns for p in runs}
        e0.run_e0(tmp_path, smoke=True, workers=1)
        assert stamp == {p.name: p.stat().st_mtime_ns for p in sorted((tmp_path / "e0_runs").iterdir())}


def test_partial_json_written_when_the_session_deadline_has_passed(tmp_path, monkeypatch):
    work = tmp_path / "cwd"
    work.mkdir()
    monkeypatch.chdir(work)
    monkeypatch.setenv("FORGE_SESSION_DEADLINE", "1")
    res = e0.run_e0(tmp_path / "out", smoke=True, workers=1)
    part = json.loads((work / "partial.json").read_text(encoding="utf-8"))
    assert part["remaining"] and res.get("partial") is True
    assert not os.path.exists(tmp_path / "out" / "poc_e0.json")
