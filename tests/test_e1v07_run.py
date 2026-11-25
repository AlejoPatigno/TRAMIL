"""E1 v07 runner: search -> real -> perm -> report on synthetic bags, resumable and shardable units."""
import json

import numpy as np
import pytest

from TRAMIL import e1v07_run as run, item_training, validation
from test_frontend import _items
from test_item_comparators import _Q

CFG = dict(reps=1, seeds=(11,), K=(2,), P=(0.0, 0.3), max_epochs=2, r_pi=2,
           arms=("full", "abmil8", "prompt_only"), kstar_arms=("full", "abmil8"))


@pytest.fixture(scope="module")
def data():
    return {"items": _items(n=30), "Q": _Q()}


@pytest.fixture(scope="module")
def done(tmp_path_factory, data):
    out = tmp_path_factory.mktemp("e1v07")
    ran = {ph: run.run_phase(ph, data, out, **CFG) for ph in ("search", "real", "perm", "report")}
    return out, ran


def test_phases_run_end_to_end_and_the_report_fills_the_expected_e1_tables(done):
    out, ran = done
    assert (out / "hpo" / "hpo_r0_f0.json").exists() and len(list((out / "hpo").glob("hpo_r0_f*.json"))) == 5
    js = json.loads((out / "E1_v07.json").read_text())
    models = {r["model"] for r in js["assay"]}
    assert models == {"full", "abmil8", "svm", "prompt_only"}
    for r in js["assay"]:
        assert r["R_pi"] == 2 and 0 < r["p"] <= 1 and isinstance(r["passes"], bool)
    assert {r["comparator"] for r in js["noninferiority"]} == {"abmil8", "svm"}
    assert js["h1"]["verdict"] in {"supported", "partially supported", "not supported", "inconclusive", "estimate"}
    assert "testable" in js["h1"] and "ss_omega" in js and js["ss_omega"]["full"]["point"] < 1
    assert {r["model"] for r in js["descriptive"]} >= {"full", "abmil8", "svm", "full_k768"}
    md = (out / "E1_v07.md").read_text()
    assert "Passes Ec. 91" in md and "Margin attainable (Ec. 87)" in md and "Corrected t (Ec. 63)" in md


def test_real_units_cover_every_arm_seed_and_fold_with_frozen_hyperparameters(done):
    out, _ = done
    specs = run.units("real", out, **CFG)
    arms = {s["arm"] for s in specs}
    assert arms == {"full", "abmil8", "prompt_only", "full_k768", "svm"}
    frozen = json.loads((out / "hpo" / "hpo_r0_f2.json").read_text())
    s = next(s for s in specs if s["arm"] == "abmil8" and s["fold"] == 2)
    assert s["k"] == frozen["k_star"] and s["p"] == frozen["p_star"]["abmil8"]
    s = next(s for s in specs if s["arm"] == "full_k768" and s["fold"] == 2)
    assert s["model"] == "full" and s["k"] == 768 and s["p"] == frozen["p768"]
    assert next(s for s in specs if s["arm"] == "prompt_only")["k"] == 8
    r = json.loads((out / "real" / f"{run.uid(s)}.json").read_text())
    assert len(r["oof"]) == 6 and all(len(v) == 8 for v in r["oof"].values())


def test_phases_are_resumable_and_skip_existing_result_files(done, data):
    out, ran = done
    assert ran["real"]["ran"] and not run.run_phase("real", data, out, **CFG)["ran"]


def test_shards_partition_the_units_and_an_explicit_list_restricts_them(done):
    out, _ = done
    ids = [run.uid(s) for s in run.units("perm", out, **CFG)]
    parts = [[run.uid(s) for s in run.select(run.units("perm", out, **CFG), shard=f"{i}/3")] for i in (1, 2, 3)]
    assert sorted(sum(parts, [])) == sorted(ids) and len(set(sum(parts, []))) == len(ids)
    pick = run.select(run.units("perm", out, **CFG), only=ids[:2])
    assert [run.uid(s) for s in pick] == ids[:2]


def test_permutation_units_use_repetition_one_seed_11_and_move_item_scores_as_a_unit(done, data):
    out, _ = done
    specs = run.units("perm", out, **CFG)
    assert {s["arm"] for s in specs} == {"full", "abmil8", "prompt_only", "svm"}
    assert {s["perm"] for s in specs} == {0, 1} and all(s["rep"] == 0 for s in specs)
    assert all(s["seed"] == 11 for s in specs if s["arm"] != "svm")
    svm = next(s for s in specs if s["arm"] == "svm")
    assert svm["grid"]["svc__kernel"] and len(svm["grid"]["svc__C"]) == 1  # frozen from the real run
    items = run.permuted_items(data["items"], 1)
    S = np.stack([it["s"] for it in data["items"]])
    assert np.array_equal(np.stack([it["s"] for it in items]), item_training.permute_items(S, run.PERM_SEED + 1))
    assert all(it["y"] == int(it["s"].sum() >= 10) for it in items)
    assert [it["X"] is jt["X"] for it, jt in zip(items, data["items"])] == [True] * 30


def test_a_real_unit_never_reads_its_outer_test_subjects_for_fitting(done, data):
    out, _ = done
    spec = next(s for s in run.units("real", out, **CFG) if s["arm"] == "full" and s["fold"] == 1)
    a = run.run_unit(spec, {**data, "cache": {}})
    y = [it["y"] for it in data["items"]]
    test = set(validation.outer_splits(y, 0)[1].tolist())
    items = [dict(it, X=it["X"] * 50 + 3, R=it["R"][::-1].copy()) if i in test else it
             for i, it in enumerate(data["items"])]
    b = run.run_unit(spec, {"items": items, "Q": data["Q"], "cache": {}})
    assert a["epoch"] == b["epoch"] and a["curves"] == b["curves"]  # fitting and epoch choice never read the test fold
    assert set(a["oof"]) == {str(data["items"][i]["subject"]) for i in test} and a["oof"] != b["oof"]
