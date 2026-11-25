"""Unit dispatcher over worker processes (T6) and the timing probe (T7)."""
import json

import numpy as np

from TRAMIL import dispatch, e1v07_run as run, timing_v07
from test_frontend import _items
from test_item_comparators import _Q


def _specs():
    base = {"phase": "search", "rep": 0, "fold": 0, "k": 3, "seed": 11, "max_epochs": 2}
    return [dict(base, arm=a, model=a, p=p, inner=i) for a in ("full", "abmil8") for p in (0.0, 0.3) for i in (0, 1)]


def test_results_do_not_depend_on_the_number_of_workers():
    data = {"items": _items(n=20), "Q": _Q()}
    one = dispatch.run(run.run_unit, _specs(), data, workers=1)
    two = dispatch.run(run.run_unit, _specs(), data, workers=2)
    assert [r["arm"] for r in one] == [r["arm"] for r in two]
    for a, b in zip(one, two):
        assert np.allclose(a["curve"], b["curve"], rtol=1e-5, atol=1e-7)


def _count(spec, data):
    data["cache"]["calls"] = data["cache"].get("calls", 0) + 1
    return data["cache"]["calls"]


def test_a_warm_up_unit_runs_in_each_worker_before_the_timed_units_and_one_worker_is_the_default():
    assert dispatch.run(_count, ["a", "b"], {}, warmup="w") == [2, 3]  # call 1 was the warm-up
    assert dispatch.run(_count, ["a", "b"], {}) == [1, 2]
    import inspect
    assert inspect.signature(dispatch.run).parameters["workers"].default == 1
    assert inspect.signature(run.run).parameters["workers"].default == 1


def test_timing_probe_writes_per_fit_and_concurrent_measurements_in_the_existing_schema(tmp_path):
    p = tmp_path / "timing.json"
    p.write_text(json.dumps({"schema": 2, "hosts": {}, "measurements": [{"arm": "old"}]}))
    data = {"items": _items(n=20), "Q": _Q()}
    timing_v07.probe(data, p, arms=("full", "svm"), epochs=1, budget=80, workers_grid=(1, 2), k=3)
    js = json.loads(p.read_text())
    new = [m for m in js["measurements"] if m.get("unit") == "fit"]
    assert js["measurements"][0] == {"arm": "old"} and js["schema"] == 2
    assert {(m["arm"], m["processes"]) for m in new} == {("full", 1), ("full", 2), ("svm", 1), ("svm", 2)}
    for m in new:
        assert {"host", "device", "arm", "unit", "secondsPerUnit", "processes", "measuredAt"} <= set(m)
        assert m["secondsPerUnit"] > 0 and m["fitsPerHour"] > 0
    full = next(m for m in new if m["arm"] == "full" and m["processes"] == 1)
    assert full["extrapolated"] and full["epochsMeasured"] == 1 and full["epochBudget"] == 80
    assert js["hosts"][full["host"]]["cpuCount"] >= 1
