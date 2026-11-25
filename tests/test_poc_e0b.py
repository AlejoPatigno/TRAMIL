import numpy as np
import torch

from TRAMIL import poc_e0 as e0
from TRAMIL import poc_e0b as eb


def test_vbar_makes_the_zero_predictor_score_one():
    tr = e0.make_data_e0(0, 64, 0, 1.0)
    vbar = eb.rc_scale(tr)
    err = np.mean([eb.npenalty(torch.zeros_like(b["Rc"]), b["Rc"], vbar).item() for b in tr])
    assert abs(err - 1.0) < 0.01  # per-bag mean vs pooled vbar: about 1, not exactly


def test_vbar_does_not_depend_on_rho():
    assert eb.rc_scale(e0.make_data_e0(1, 16, 0, 0.0)) == eb.rc_scale(e0.make_data_e0(1, 16, 0, 2.0))


def test_patience_stops_early_and_restores_best():
    tr, va = e0.make_data_e0(0, 16, 0, 1.0), e0.make_data_e0(0, 8, 1, 1.0)
    m = eb.train_e0b(tr, "none", epochs=50, lr=0.05, val=va, patience=3, vbar=1.0)
    h = m.history
    assert len(h["val"]) <= h["best_epoch"] + 3 and len(h["val"]) < 50
    assert h["val"][h["best_epoch"] - 1] == min(h["val"])


def test_penalty_is_active_in_training():
    tr = e0.make_data_e0(0, 16, 0, 1.0)
    vbar = eb.rc_scale(tr)
    a = eb.train_e0b(tr, "surrogate", mu_R=0.0, epochs=3, lr=1e-2, vbar=vbar)
    b = eb.train_e0b(tr, "surrogate", mu_R=10.0, epochs=3, lr=1e-2, vbar=vbar)
    err = lambda m: np.mean([eb.npenalty(m(x)[1], x["Rc"], vbar).item() for x in tr])  # noqa: E731
    with torch.no_grad():
        assert err(b) < err(a)


def test_select_breaks_ties_toward_smaller_lr_then_mu():
    c = [dict(lr=1e-2, mu=1.0, val=1.0), dict(lr=1e-3, mu=10.0, val=1.0), dict(lr=1e-3, mu=0.1, val=1.0),
         dict(lr=3e-3, mu=0.1, val=1.1)]
    assert eb.select(c) == c[2]


def _arms(f, u, fs, us, n=10):
    rng = np.random.default_rng(0)
    mk = lambda m, s=0.0: list(m + 0.001 * rng.standard_normal(n) + s)  # noqa: E731
    arms = {}
    for rho in eb.RHOS:
        arms[f"full|rho={rho:g}"] = dict(ss_w=mk(f), surr_err=mk(0.5), spearman_mean=mk(fs))
        arms[f"untargeted|rho={rho:g}"] = dict(ss_w=mk(u), surr_err=mk(0.9), spearman_mean=mk(us))
        arms[f"r12|rho={rho:g}"] = dict(ss_w=mk(0.2), surr_err=mk(np.nan), spearman_mean=mk(np.nan))
    return arms


def test_compare_holm_over_four_and_p2_needs_all_three():
    rows, ver = eb.compare(_arms(0.1, 0.05, 0.3, 0.0))
    assert set(rows) >= {"P1", "P2a", "P2b", "P2c", "P3", "P4"}
    assert all(rows[k]["p_adj"] >= rows[k]["p_raw"] for k in ("P1", "P2a", "P2b", "P2c"))
    assert ver["P2"] == "supported"
    rows, ver = eb.compare(_arms(0.1, 0.05, 0.0, 0.3))  # no recovery: P2c fails
    assert ver["P2"] == "partially supported" and not rows["P2c"]["pass"]
    assert ver["H8_carried_forward"] is False


def test_smoke_runs_end_to_end(tmp_path):
    js = eb.run_e0b(out_dir=tmp_path, smoke=True, workers=1)
    assert js["smoke"] and "verdicts" in js and (tmp_path / "poc_e0b.md").exists()
    assert set(js["arms"]) >= {f"full|rho={r:g}" for r in eb.RHOS}
