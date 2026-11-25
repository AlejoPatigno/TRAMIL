"""Shared hyperparameter search (r22 Ec. 102; v07 E1 'Hyperparameter search'): units, reducer, frozen JSON."""
import json
import random

import numpy as np
import pytest

from TRAMIL import hpo, validation
from test_frontend import _items
from test_item_comparators import _Q


def _res(arm, k, p, curves):
    """Three inner-fold results of one (arm, k, p) trial with the given per-fold curves."""
    return [{"rep": 0, "fold": 0, "arm": arm, "k": k, "p": p, "inner": r, "curve": list(c)} for r, c in enumerate(curves)]


def _flat(arm, k, p, mse):
    return _res(arm, k, p, [[mse + 1.0, mse]] * 3)


def test_trial_objective_is_minus_the_ec39_inner_mse_at_the_earliest_best_epoch():
    out = hpo.reduce(_res("full", 16, 0.0, [[0.9, 0.5, 0.5], [0.9, 0.3, 0.3], [0.9, 0.4, 0.4]]), kstar_arms=["full"])
    (row,) = out["table"]
    assert row["epoch"] == 2 and row["objective"] == pytest.approx(-0.4)


def test_shared_k_is_the_argmax_over_k_of_the_arm_mean_of_the_best_p_objective():
    rs = []
    # arm a prefers k=16 slightly, arm b prefers k=32 strongly: mean of per-arm best p picks k=32 (Ec. 102)
    rs += _flat("a", 16, 0.0, 0.50) + _flat("a", 16, 0.3, 0.60) + _flat("a", 32, 0.0, 0.52) + _flat("a", 32, 0.3, 0.70)
    rs += _flat("b", 16, 0.0, 0.90) + _flat("b", 16, 0.3, 0.95) + _flat("b", 32, 0.0, 0.80) + _flat("b", 32, 0.3, 0.60)
    out = hpo.reduce(rs, kstar_arms=["a", "b"])
    assert out["k_star"] == 32
    assert out["p_star"] == {"a": 0.0, "b": 0.3}


def test_ties_go_to_the_smaller_k_then_the_smaller_p():
    rs = []
    for arm in ("a", "b"):
        for k in (16, 32):
            for p in (0.0, 0.15, 0.3):
                rs += _flat(arm, k, p, 0.5)
    out = hpo.reduce(rs, kstar_arms=["a", "b"])
    assert out["k_star"] == 16 and out["p_star"] == {"a": 0.0, "b": 0.0}


def test_arms_outside_the_shared_k_get_their_own_p_and_the_768_reference_its_own():
    rs = _flat("full", 16, 0.0, 0.5) + _flat("full", 16, 0.3, 0.4) + _flat("permuted", 16, 0.0, 0.3)
    rs += _flat("permuted", 16, 0.3, 0.6) + _flat("permuted", 32, 0.0, 0.1) + _flat("full", 32, 0.0, 0.9)
    rs += _flat("prompt_only", 8, 0.0, 0.7) + _flat("prompt_only", 8, 0.3, 0.2)
    rs += _flat("full", 768, 0.0, 0.8) + _flat("full", 768, 0.3, 0.6)
    out = hpo.reduce(rs, kstar_arms=["full"])
    assert out["k_star"] == 16  # permuted does not vote, its k = 32 optimum is ignored
    assert out["p_star"]["permuted"] == 0.0 and out["p_star"]["full"] == 0.3
    assert out["p_star"]["prompt_only"] == 0.3 and out["p768"] == 0.3


def test_reducer_is_deterministic_under_any_result_order():
    rs = []
    for arm in ("a", "b"):
        for k in (16, 32):
            for p in (0.0, 0.3):
                rs += _res(arm, k, p, np.random.default_rng(hash((arm, k, p)) % 2**32).random((3, 4)))
    ref = hpo.reduce(rs, kstar_arms=["a", "b"])
    for s in range(3):
        sh = list(rs)
        random.Random(s).shuffle(sh)
        assert hpo.reduce(sh, kstar_arms=["a", "b"]) == ref


def test_every_arm_gets_the_same_trial_values_and_unit_specs_are_independent():
    specs = hpo.units(0, 1, arms=["full", "abmil8", "permuted", "prompt_only"], K=(16, 32), P=(0.0, 0.3))
    grid = lambda arm: {(s["k"], s["p"]) for s in specs if s["arm"] == arm}  # noqa: E731
    audio = {(k, p) for k in (16, 32) for p in (0.0, 0.3)}
    assert grid("abmil8") == grid("permuted") == audio
    assert grid("full") == audio | {(768, 0.0), (768, 0.3)}
    assert grid("prompt_only") == {(8, 0.0), (8, 0.3)}
    assert all(s["seed"] == 11 and s["inner"] in (0, 1, 2) for s in specs)
    assert len({json.dumps(s, sort_keys=True) for s in specs}) == len(specs)
    assert hpo.P_GRID == (0.0, 0.15, 0.3, 0.45) and hpo.K_GRID == (16, 32, 64, 128) and hpo.SEED == 11


def test_a_unit_never_touches_an_outer_test_subject():
    items = _items(n=15)
    spec = {"rep": 0, "fold": 1, "arm": "full", "k": 3, "p": 0.0, "inner": 0, "seed": 11, "max_epochs": 2}
    r = hpo.run_unit(spec, items)
    y = [it["y"] for it in items]
    test = {items[i]["subject"] for i in validation.outer_splits(y, 0)[1]}
    assert test.isdisjoint(r["train_subjects"]) and test.isdisjoint(r["val_subjects"])
    assert set(r["train_subjects"]).isdisjoint(r["val_subjects"])
    changed = [dict(it, X=it["X"] * 50 + 3) if it["subject"] in test else it for it in items]
    assert hpo.run_unit(spec, changed)["curve"] == r["curve"] and len(r["curve"]) == 2


def test_fold_search_writes_frozen_json_with_provenance_and_the_loader_reads_it_back(tmp_path):
    items = _items(n=15)
    out = hpo.run_fold(items, rep=0, fold=0, out_dir=tmp_path, Q=_Q(), arms=["full", "abmil8"], K=(2, 3),
                       P=(0.0, 0.3), max_epochs=2, kstar_arms=["full", "abmil8"])
    js = json.loads((tmp_path / "hpo_r0_f0.json").read_text())
    assert js["k_star"] in (2, 3) and set(js["p_star"]) == {"full", "abmil8"} and "p768" in js
    assert js["provenance"]["seed"] == 11 and "commit" in js["provenance"] and js["grid"]["P"] == [0.0, 0.3]
    assert len(js["table"]) == 2 * 2 * 2 + 2  # two arms on the grid plus the k = 768 reference of the full model
    frozen = hpo.load_frozen(tmp_path, 0, 0)
    assert frozen == {"k_star": out["k_star"], "p_star": out["p_star"], "p768": out["p768"]}
