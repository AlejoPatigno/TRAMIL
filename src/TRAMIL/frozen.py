"""Frozen models of the real units (U4): what a real unit persists so that a later experiment (E6 deletion passes, E7 sealed
read) runs inference only and never re-fits.

Per real unit (rep, fold, arm, seed) the runner writes, beside its result JSON, `<uid>.pt`: the trained state_dict (float32,
cpu), the arm, k, p, vbar, the selected epoch, the held-out subject order and the name of its front end; and once per
(rep, fold, k, source) `frontend/<name>.pt`: the scaler and the V_k projection fitted on the outer training subjects only.
The bag-level SVM of E1 is a pickled sklearn pipeline `<uid>.pkl`. Everything travels inside the job's verified zip
(<name>/real/<uid>.pt, <name>/frontend/<name>.pt). Store indexes loose files and zips under any roots; load rebuilds the
model and the held-out bags for inference.

Mounted upstream zips are trusted artifacts of this project: the .pkl members are unpickled."""
import io
import os
import pickle
import zipfile
from pathlib import Path

import numpy as np
import torch

from . import e1_arms, frontend, item_training

__provenance__ = {
    "revision": "experiments-experimental-plan-for-the-residual-semantic-prior-bipartite-v07.md",
    "sections": ["E6", "E7"],
    "experiments": ["E6", "E7"],
    "invariants": [
        "a_real_unit_persists_its_state_dict_and_the_front_end_fitted_on_its_training_subjects",
        "inference_from_a_stored_state_reproduces_the_stored_out_of_fold_predictions",
        "front_ends_are_stored_once_per_fold_k_and_source",
    ],
}

KINDS = ("real", "frontend")
EXTS = (".pt", ".pkl")


def fe_name(spec, k, source):
    return f"fe_r{spec['rep']}_f{spec['fold']}_k{k}_{source}" + ("_nohist" if spec.get("arm") == "full_nohist" else "")


def _bytes(obj):
    b = io.BytesIO()
    torch.save(obj, b)
    return b.getvalue()


def pack_state(spec, m, vbar, k, epoch, test_items, fe):
    """Bytes of <uid>.pt."""
    keep = ("phase", "rep", "fold", "arm", "model", "seed", "p")
    return _bytes({"spec": {a: spec[a] for a in keep if a in spec}, "k": int(k), "p": float(spec["p"]),
                   "vbar": float(vbar), "epoch": int(epoch), "frontend": fe,
                   "test_subjects": [int(it["subject"]) for it in test_items],
                   "state_dict": {n: v.detach().to("cpu", torch.float32).clone() for n, v in m.state_dict().items()}})


def pack_frontend(fe):
    """Bytes of frontend/<name>.pt (tensors keep the fitted dtype)."""
    return _bytes({"k": int(fe.k), "mu": torch.from_numpy(np.asarray(fe.mu)), "sd": torch.from_numpy(np.asarray(fe.sd)),
                   "V": torch.from_numpy(np.asarray(fe.V))})


def _unpack_frontend(d):
    fe = frontend.FrontEnd(d["k"])
    fe.mu, fe.sd, fe.V = d["mu"].numpy(), d["sd"].numpy(), d["V"].numpy()
    return fe


class Store:
    """Index of the persisted models below roots: loose <kind>/*.pt|.pkl files and members <anything>/<kind>/<file> of any
    zip, whatever the mount layout. An earlier root wins a duplicate name."""

    def __init__(self, roots):
        self.loc = {k: {} for k in KINDS}
        for root in map(Path, roots):
            for here, dirs, files in os.walk(root):
                dirs[:] = [d for d in dirs if not d.startswith((".", "instances_audio"))]
                here = Path(here)
                for f in files:
                    if f.endswith(".zip"):
                        self._zip(here / f)
                    elif f.endswith(EXTS) and here.name in KINDS:
                        self.loc[here.name].setdefault(Path(f).stem, ("file", here / f))

    def _zip(self, z):
        try:
            with zipfile.ZipFile(z) as zf:
                for m in zf.namelist():
                    p = m.split("/")
                    if len(p) >= 2 and p[-2] in KINDS and p[-1].endswith(EXTS) and ".." not in p:
                        self.loc[p[-2]].setdefault(Path(p[-1]).stem, ("zip", z, m))
        except zipfile.BadZipFile:
            pass

    def _read(self, kind, name):
        loc = self.loc[kind][name]
        if loc[0] == "file":
            return loc[1].read_bytes()
        with zipfile.ZipFile(loc[1]) as zf:
            return zf.read(loc[2])

    def uids(self):
        return sorted(self.loc["real"])

    def state(self, uid):
        return torch.load(io.BytesIO(self._read("real", uid)), map_location="cpu")

    def frontend(self, name):
        return _unpack_frontend(torch.load(io.BytesIO(self._read("frontend", name)), map_location="cpu"))

    def svm(self, uid):
        return pickle.loads(self._read("real", uid))  # the SVM unit stores <uid>.pkl


def load(store, uid, items, Q=None):
    """(model in eval mode on the training device, collated held-out bags, held-out items in the stored order) of one
    stored unit. `items` is the full item list the unit was run on (the history-removed list for full_nohist)."""
    st = store.state(uid)
    fe = store.frontend(st["frontend"])
    spec = st["spec"]
    by = {int(it["subject"]): it for it in items}
    test = [by[s] for s in st["test_subjects"]]
    m = e1_arms.build(spec["model"], k=st["k"], p_drop=st["p"], vbar=st["vbar"], Q=Q)()
    m.load_state_dict(st["state_dict"])
    m = m.to(item_training.device()).eval()
    source = e1_arms.SOURCE.get(spec["model"], "X")
    eb = item_training.collate([frontend.prepare(it, fe, source) for it in test], item_training.device())
    return m, eb, test


def predict(m, eb):
    """(M, 8) item predictions (a binary-head model: (M, 1) logits) of stored models, inference only."""
    return item_training.predict(m, eb)
