import json
import tempfile
from pathlib import Path

import numpy as np

from TRAMIL import exp_e1


def _items(n=24, seed=0):
    g = np.random.default_rng(seed)
    out = []
    for i in range(n):
        y = int(i % 3 == 0)
        X = g.standard_normal((int(g.integers(3, 6)), 88)).astype(np.float32)
        X[:, :8] += 1.5 * y
        out.append({"subject": 100 + i, "X": X, "R": g.random((len(X), 8)).astype(np.float32), "y": y})
    return out


def test_e1_rehearsal_runs_every_arm_on_every_fold_without_leakage():
    res = exp_e1.run_e1(_items(), n_folds=2, seeds=(11,), max_epochs=2, repetition=0)
    assert set(res["arms"]) == {"residual", "learned_only", "abmil", "gated_abmil", "svm"}
    subjects = {it["subject"] for it in _items()}
    assert len(res["folds"]) == 2
    for f in res["folds"]:
        assert set(f["train"]).isdisjoint(f["test"])
    assert sorted(s for f in res["folds"] for s in f["test"]) == sorted(subjects)  # every subject tested once
    for name, a in res["arms"].items():
        assert len(a["fold_auroc"]) == 2 and all(0.0 <= v <= 1.0 for v in a["fold_auroc"])
        assert sorted(a["oof"]) == sorted(subjects)
    assert all(1 <= e <= 2 for a in res["arms"].values() for e in a.get("epochs", [1]))


def test_results_table_is_stamped_pilot_in_json_and_markdown():
    res = exp_e1.run_e1(_items(), n_folds=2, seeds=(11,), max_epochs=2, repetition=0)
    with tempfile.TemporaryDirectory() as d:
        exp_e1.write_results(res, Path(d), extra={"note": "unit"})
        js = json.loads((Path(d) / "E1_pilot.json").read_text())
        md = (Path(d) / "E1_pilot.md").read_text()
    assert js["stage"] == "PILOT" and "PILOT" in md.splitlines()[0]
    assert "plumbing" in md.lower() and "residual" in md
