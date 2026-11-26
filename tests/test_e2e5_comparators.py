"""Option A: the seven E1 arms without a persisted model (and the E1 SVM) re-run in e2e5-real as *_x so E6 and E7 find them."""
import json

import numpy as np
import pytest

from TRAMIL import e1_arms, e1v07_run as e1, e2e5_run as run, frozen, hpo
from test_frontend import _items
from test_item_comparators import _Q

NEW = ("abmil8_x", "gated_abmil8_x", "symptom_guided_x", "r06_8_x", "abmil_r8_x", "prompt_only_x", "r12_x")
E1CFG = dict(reps=1, seeds=(11,), K=(2,), P=(0.0, 0.3), max_epochs=2, kstar_arms=("full",),
             arms=("full", "abmil8", "gated_abmil8", "symptom_guided", "r06_8", "abmil_r8", "prompt_only", "r12"))
CFG = dict(reps=1, seeds=(11,), P=(0.0, 0.3), max_epochs=2, arms=(*NEW, "svm_x"))


@pytest.fixture(scope="module")
def data():
    return {"items": _items(n=30), "Q": _Q()}


@pytest.fixture(scope="module")
def done(tmp_path_factory, data):
    out = tmp_path_factory.mktemp("cmp")
    for ph in ("search", "real"):
        e1.run_phase(ph, data, out, **E1CFG)
    run.run_phase("search", data, out, **CFG)
    run.run_phase("real", data, out, **CFG)
    return out


def test_the_aliases_are_registered_and_default_in_the_e2e5_job():
    assert all(a in e1_arms.ARMS and e1_arms.ALIAS[a] == a[:-2] for a in NEW)
    assert e1_arms.SOURCE["prompt_only_x"] == "R" and "svm_x" not in e1_arms.ARMS
    assert set(NEW) | {"svm_x"} <= set(run.DEFAULTS["arms"])
    assert set(run.search_arms(run._cfg(CFG))) == set()  # aliases search nothing: E1 k* and p*


def test_alias_units_use_the_e1_k_star_p_star_and_prompt_only_the_prompt_k(done):
    specs = {s["arm"]: s for s in run.units("real", done, **CFG) if s["fold"] == 1}
    e1p = json.loads((done / "hpo" / "hpo_r0_f1.json").read_text())
    for a in NEW:
        assert specs[a]["p"] == e1p["p_star"][a[:-2]], a
        assert specs[a]["k"] == (hpo.K_PROMPT if a == "prompt_only_x" else e1p["k_star"]), a
    assert specs["svm_x"]["model"] == "svm" and specs["svm_x"]["seed"] == 0
    assert len([s for s in run.units("real", done, **CFG)]) == 5 * (len(NEW) + 1)  # one seed here


def test_every_alias_unit_persists_a_model_that_reproduces_its_predictions(done, data):
    store = frozen.Store([done])
    for a in NEW:
        uid = f"r0_f2_{a}_s11"
        js = json.loads((done / "real" / f"{uid}.json").read_text())
        assert all(len(v) == 8 and np.isfinite(v).all() for v in js["oof"].values()), a
        assert ("extra" in js) == (a in ("prompt_only_x", "r12_x")), a  # only surrogate models carry attention extras
        m, eb, items = frozen.load(store, uid, data["items"], data["Q"])
        got = frozen.predict(m, eb)
        assert all(np.allclose(r, js["oof"][str(it["subject"])], atol=1e-5) for it, r in zip(items, got)), a


def test_svm_x_persists_the_pipeline_and_selects_the_same_kernel_and_c_as_e1(done, data):
    js = json.loads((done / "real" / "r0_f2_svm_x_s0.json").read_text())
    ref = e1.run_unit({"phase": "real", "rep": 0, "fold": 2, "arm": "svm", "model": "svm", "seed": 0}, {**data, "cache": {}})
    assert js["grid"] == ref["grid"] and js["oof"] == ref["oof"]
    pipe = frozen.Store([done]).svm("r0_f2_svm_x_s0")
    assert pipe.named_steps["svc"].kernel == js["grid"]["svc__kernel"][0]
    assert not (done / "real" / "r0_f2_svm_x_s0.pt").exists()


def test_the_report_still_reads_the_new_units(done, data):

    scores, profiles = e1._score_matrix(e1._load_dir(done / "real"), [it["subject"] for it in data["items"]], 1)
    assert set(NEW) | {"svm_x"} <= set(scores) and "graph_corr" not in scores
