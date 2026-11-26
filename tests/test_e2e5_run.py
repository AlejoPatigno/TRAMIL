"""E2E5 runner: search at the E1 k*, real units with extras, resumable shardable phases, zips, history control."""
import json
import zipfile

import numpy as np
import pytest

from TRAMIL import e1v07_run as e1, e2e5_run as run, frontend, validation
from test_frontend import _items
from test_item_comparators import _Q

E1CFG = dict(reps=1, seeds=(11,), K=(2,), P=(0.0, 0.3), max_epochs=2, arms=("full", "permuted", "untargeted"),
             kstar_arms=("full",))
CFG = dict(reps=1, seeds=(11,), P=(0.0, 0.3), max_epochs=2,
           arms=("graph_corr", "binary_sup", "free_desc", "permuted_quint", "full_x"))


@pytest.fixture(scope="module")
def data():
    return {"items": _items(n=30), "Q": _Q()}


@pytest.fixture(scope="module")
def done(tmp_path_factory, data):
    out = tmp_path_factory.mktemp("e2e5")
    e1.run_phase("search", data, out, **E1CFG)  # the frozen E1 hpo (k* = 2) the new arms import
    ran = {ph: run.run_phase(ph, data, out, **CFG) for ph in ("search", "real")}
    return out, ran


def test_search_units_run_at_the_e1_kstar_over_the_p_grid_for_the_arms_that_need_their_own_p(done):
    out, _ = done
    specs = run.units("search", out, **CFG)
    assert {s["arm"] for s in specs} == {"graph_corr", "binary_sup", "free_desc", "permuted_quint"}  # full_x: alias
    assert {s["k"] for s in specs} == {2} and {s["p"] for s in specs} == {0.0, 0.3}
    assert len(specs) == 5 * 4 * 2 * 3 and all(s["seed"] == 11 for s in specs)


def test_the_search_freezes_one_p_star_per_arm_beside_the_e1_hpo_without_touching_it(done):
    out, _ = done
    js = json.loads((out / "hpo" / "e2e5_r0_f2.json").read_text())
    assert js["k_star"] == 2 and set(js["p_star"]) == {"graph_corr", "binary_sup", "free_desc", "permuted_quint"}
    assert all(p in (0.0, 0.3) for p in js["p_star"].values())
    e1_js = json.loads((out / "hpo" / "hpo_r0_f2.json").read_text())
    assert e1_js["k_star"] == 2 and "graph_corr" not in e1_js["p_star"]


def test_real_units_use_k_star_and_their_own_p_star_and_the_aliases_use_the_e1_p_star(done):
    out, _ = done
    specs = run.units("real", out, **CFG)
    assert len(specs) == 5 * 5 * 1 and {s["arm"] for s in specs} == set(CFG["arms"])
    own = json.loads((out / "hpo" / "e2e5_r0_f1.json").read_text())["p_star"]
    e1p = json.loads((out / "hpo" / "hpo_r0_f1.json").read_text())["p_star"]
    for s in specs:
        if s["fold"] == 1:
            assert s["k"] == 2 and s["p"] == (e1p["full"] if s["arm"] == "full_x" else own[s["arm"]])
    assert next(s for s in specs if s["arm"] == "full_x")["model"] == "full_x"


def test_real_units_store_item_predictions_and_extras_by_subject(done):
    out, _ = done
    full = json.loads(next(out.joinpath("real").glob("r0_f0_full_x_s11.json")).read_text())
    assert full["oof"] and all(len(v) == 8 for v in full["oof"].values())
    ex = full["extra"]
    assert set(ex) == set(full["oof"])
    e = next(iter(ex.values()))
    n = len(e["att"])
    assert 3 <= n <= 8 and len(e["att"][0]) == 8 and np.allclose(np.sum(e["att"], axis=0), 1.0, atol=1e-4)
    assert len(e["rank"]) == 8 and len(e["rank"][0]) == n and len(e["const"]) == 8 and 0 < e["feff"] <= 1
    assert len(e["fid"]) == 8 and e["al"] >= 0 and e["rms"] >= 0 and full["drift"] is None
    desc = json.loads(next(out.joinpath("real").glob("r0_f0_free_desc_s11.json")).read_text())
    assert len(desc["drift"]) == 8 and all(-1 <= d <= 1 for d in desc["drift"])
    b = json.loads(next(out.joinpath("real").glob("r0_f0_binary_sup_s11.json")).read_text())
    assert all(len(v) == 1 for v in b["oof"].values()) and b["extra"]


def test_the_quintile_derangement_seeds_are_recorded_per_fit_and_differ_between_fits(done):
    out, _ = done
    q = json.loads(next(out.joinpath("real").glob("r0_f0_permuted_quint_s11.json")).read_text())
    seeds = q["derangement_seeds"]
    assert set(seeds) == {"inner0", "inner1", "inner2", "outer"} and len(set(seeds.values())) == 4
    q2 = json.loads(next(out.joinpath("real").glob("r0_f1_permuted_quint_s11.json")).read_text())
    assert q2["derangement_seeds"]["outer"] != seeds["outer"]
    assert "derangement_seeds" not in json.loads(next(out.joinpath("real").glob("r0_f0_graph_corr_s11.json")).read_text())


def test_phases_are_resumable_and_shards_partition_the_units(done, data):
    out, ran = done
    assert ran["real"]["ran"] and not run.run_phase("real", data, out, **CFG)["ran"]
    ids = [e1.uid(s) for s in run.units("real", out, **CFG)]
    parts = [[e1.uid(s) for s in e1.select(run.units("real", out, **CFG), shard=f"{i}/2")] for i in (1, 2)]
    assert sorted(sum(parts, [])) == sorted(ids)


def test_a_real_unit_never_reads_its_outer_test_subjects_for_fitting(done, data):
    out, _ = done
    spec = next(s for s in run.units("real", out, **CFG) if s["arm"] == "graph_corr" and s["fold"] == 1)
    a = run.run_unit(spec, {**data, "cache": {}})
    y = [it["y"] for it in data["items"]]
    test = set(validation.outer_splits(y, 0)[1].tolist())
    items = [dict(it, X=it["X"] * 50 + 3, R=it["R"][::-1].copy()) if i in test else it
             for i, it in enumerate(data["items"])]
    b = run.run_unit(spec, {"items": items, "Q": data["Q"], "cache": {}})
    assert a["epoch"] == b["epoch"] and a["curves"] == b["curves"]


def test_the_history_arm_needs_the_frozen_prompt_list_and_trains_on_reduced_bags(done, data):
    spec = {"phase": "real", "rep": 0, "fold": 0, "arm": "full_nohist", "model": "full_nohist", "k": 2, "p": 0.0,
            "seed": 11, "max_epochs": 1}
    with pytest.raises(ValueError, match="history"):
        run.run_unit(spec, {**data, "cache": {}})
    items = _items(n=30)
    for it in items:
        it["prompts"] = ["Were you ever diagnosed with depression?"] + [f"Topic {i}." for i in range(1, len(it["R"]))]
    r = run.run_unit(spec, {"items": items, "Q": data["Q"], "cache": {}, "history": ["were you ever diagnosed with depression"]})
    assert r["oof"] and r["history_removed_segments"] == 30


def test_a_real_run_zips_under_the_e2e5_name_and_a_smoke_keeps_loose_units(tmp_path, monkeypatch, data):
    monkeypatch.setattr(e1, "load_data", lambda input_root=None, limit_bags=None: (data, {}))
    e1.run_phase("search", data, tmp_path / "e1", **E1CFG)
    root = tmp_path / "in" / "notebooks" / "o" / "e1job" / "E1_v07"
    (root / "hpo").mkdir(parents=True)
    for f in (tmp_path / "e1" / "hpo").glob("hpo_*.json"):
        (root / "hpo" / f.name).write_text(f.read_text())
    small = {**CFG, "arms": ("graph_corr",)}
    smoke = tmp_path / "smoke"
    run.run(["search", "real"], smoke, input_root=str(tmp_path / "in"), limit_bags=5, **small)
    assert list((smoke / "real").glob("*.json")) and not list(smoke.glob("*.zip"))
    real = tmp_path / "real"
    r1 = run.run("search", real, input_root=str(tmp_path / "in"), **small)
    r2 = run.run("real", real, input_root=str(tmp_path / "in"), shard="1/2", **small)
    assert sorted(z.name for z in real.glob("*.zip")) == ["E2E5_v07_real_s1of2.zip", "E2E5_v07_search.zip"]
    assert r1["imported"] == 5 and r2["imported"] == 0 and r2["real"]["ran"] == 3  # 5 folds, shard 1/2
    assert not list((real / "real").glob("*.json")) and not list((real / "search").glob("*.json"))
    assert len(list((real / "hpo").glob("hpo_r0_f*.json"))) == 5 and len(list((real / "hpo").glob("e2e5_r0_f*.json"))) == 5
    with zipfile.ZipFile(real / "E2E5_v07_real_s1of2.zip") as z:
        assert all(n.startswith("E2E5_v07/real/") for n in z.namelist())
    assert run.run("real", real, input_root=str(tmp_path / "in"), shard="1/2", **small)["real"]["ran"] == 0  # resumed from the zip
