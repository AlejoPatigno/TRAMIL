"""U4: every real unit persists its trained model and frozen front end, so E6 and E7 run inference only."""
import json
import zipfile

import numpy as np
import pytest
import torch

from TRAMIL import comparators, e1v07_run as e1, e2e5_run as run, frozen, frontend, validation
from test_frontend import _items
from test_item_comparators import _Q

E1CFG = dict(reps=1, seeds=(11,), K=(2,), P=(0.0, 0.3), max_epochs=2, arms=("full", "abmil8"), kstar_arms=("full",))
CFG = dict(reps=1, seeds=(11,), P=(0.0, 0.3), max_epochs=2, arms=("graph_corr", "full_x"))


@pytest.fixture(scope="module")
def data():
    return {"items": _items(n=30), "Q": _Q()}


@pytest.fixture(scope="module")
def done(tmp_path_factory, data):
    out = tmp_path_factory.mktemp("frozen")
    for ph in ("search", "real"):
        e1.run_phase(ph, data, out, **E1CFG)
    for ph in ("search", "real"):
        run.run_phase(ph, data, out, **CFG)
    return out


def test_real_units_write_a_state_file_beside_the_json_and_one_front_end_per_fold_and_k(done):
    for arm in ("full", "abmil8", "graph_corr", "full_x"):
        stem = f"r0_f0_{arm}_s11"
        assert (done / "real" / f"{stem}.json").exists() and (done / "real" / f"{stem}.pt").exists(), arm
    js = json.loads((done / "real" / "r0_f0_full_s11.json").read_text())
    assert "_state" not in js and "state_dict" not in js  # the JSON stays small
    fes = sorted(p.name for p in (done / "frontend").glob("*.pt"))
    # one per (fold, k): full, graph_corr, full_x share k* = 2; abmil8 reads its own reference k
    assert [n for n in fes if n.endswith("_k2_X.pt")] == [f"fe_r0_f{f}_k2_X.pt" for f in range(5)]
    assert len(fes) == len(set(fes)) == 10
    assert not list((done / "search").glob("*.pt"))  # only real units persist models


def test_a_stored_model_reproduces_its_out_of_fold_predictions_and_extras_by_inference_only(done, data):
    store = frozen.Store([done])
    y = [it["y"] for it in data["items"]]
    for arm in ("full", "abmil8", "graph_corr", "full_x"):
        uid = f"r0_f1_{arm}_s11"
        st = store.state(uid)
        assert st["spec"]["arm"] == arm and st["epoch"] >= 1 and st["k"] == 2
        test = validation.outer_splits(y, 0)[1]
        assert st["test_subjects"] == [data["items"][i]["subject"] for i in test]
        assert all(v.dtype == torch.float32 for v in st["state_dict"].values())
        m, eb, items = frozen.load(store, uid, data["items"], data["Q"])
        got = frozen.predict(m, eb)
        want = json.loads((done / "real" / f"{uid}.json").read_text())["oof"]
        for it, row in zip(items, got):
            assert np.allclose(row, want[str(it["subject"])], atol=1e-5), (arm, it["subject"])
    ex = json.loads((done / "real" / "r0_f1_full_x_s11.json").read_text())["extra"]
    m, eb, items = frozen.load(store, "r0_f1_full_x_s11", data["items"], data["Q"])
    again = run._extras(m, eb, items)
    for b, e in ex.items():
        assert np.allclose(again[b]["att"], e["att"], atol=1e-4)


def test_the_front_end_is_the_one_fitted_on_the_training_subjects_only(done, data):
    store = frozen.Store([done])
    fe = store.frontend("fe_r0_f2_k2_X")
    y = [it["y"] for it in data["items"]]
    train = validation.outer_splits(y, 0)
    tr = np.setdiff1d(np.arange(len(y)), train[2])
    ref = frontend.FrontEnd(2).fit([data["items"][i]["X"] for i in tr])
    assert np.allclose(fe.V, ref.V) and np.allclose(fe.mu, ref.mu) and np.allclose(fe.sd, ref.sd)


def test_the_zip_carries_states_and_front_ends_and_the_loaders_read_them_back_from_it(tmp_path, data, monkeypatch):
    monkeypatch.setattr(e1, "load_data", lambda input_root=None, limit_bags=None: (data, {}))
    out = tmp_path / "e1"
    r = e1.run(["search", "real"], out, **{**E1CFG, "arms": ("full",)})
    names = {p.name for p in out.glob("*.zip")}
    assert names == {"E1_v07_search.zip", "E1_v07_real.zip"}
    with zipfile.ZipFile(out / "E1_v07_real.zip") as z:
        pts = [n for n in z.namelist() if n.endswith(".pt")]
        assert sum("/real/" in n for n in pts) == 10 and sum("/frontend/" in n for n in pts) == 10  # k* and the k = d rotation ablation
    assert not list((out / "real").glob("*")) and not list((out / "frontend").glob("*"))  # loose copies gone
    store = frozen.Store([out])
    assert len(store.uids()) == 15  # + the bag-level SVM pickles and store.state(store.uids()[0])["spec"]["arm"] == "full"
    m, eb, items = frozen.load(store, store.uids()[0], data["items"], data["Q"])
    assert frozen.predict(m, eb).shape == (len(items), 8)
    svm = store.svm("r0_f0_svm_s0")
    assert svm.decision_function(np.stack([comparators.bag_stats(it["X"]) for it in data["items"]])).shape == (30,)
    # a resumed run re-zips without losing the states of the earlier zip
    e1.run(["real"], out, **{**E1CFG, "arms": ("full",)})
    assert len(frozen.Store([out]).uids()) == 15
    assert r["real"]["ran"] == 15


def test_report_imports_never_copy_state_files(tmp_path, done):
    dst = tmp_path / "rep"
    with zipfile.ZipFile(tmp_path / "E1_v07_real.zip", "w") as z:
        for p in (done / "real").glob("r0_f0_full_s11.*"):
            z.write(p, f"E1_v07/real/{p.name}")
    got = e1.import_upstream_files([tmp_path], dst, "E1_v07", ("real",))
    assert [g.suffix for g in got] == [".json"]
