"""E2 to E5 of experiments v07 on top of the E1 runner: search -> real -> report, every phase resumable and shardable
(e1v07_run.select), one result file per unit under <out>/<phase>/<uid>.json, the job's results zipped at the end of a real run.

The E1 jobs already run permuted, untargeted, r12 and prompt_only in the E1 protocol (and the prompt-only Ec. 91 null in
the E1 perm phase), so their scores are read from there. This module adds what E1 does not store or run:

- search: p* of the new arms (e1_arms.E2E5_ARMS minus the aliases) over the same p grid at the E1 shared k* (read from the
  frozen hpo_r{rep}_f{fold}.json of E1, never recomputed), seed 11, three inner folds; a completed outer training set is
  frozen to hpo/e2e5_r{rep}_f{fold}.json (p_star per arm, k_star copied). The E1 file is not touched.
- real: (rep, fold, arm, seed) at k*, p* (own p*, or the E1 p* of the aliased arm for full_x, permuted_x, untargeted_x), the
  E1 recipe (inner epoch selection on three inner folds, refit on the outer training set) plus the extras the E2 to E5
  analyses need: per held-out subject the attention columns of Ec. 20 (rounded, with exact average ranks and the constant
  flags of Ec. 44), F_eff (Ec. 52), the surrogate fidelity per item, Al_b and RMS^Delta_b (Ec. 107); the query drift of the
  descriptor-initialized arm. permuted_quint draws one quintile derangement per fit (seeds recorded in the unit);
  full_nohist trains and reads the bags without the history prompts frozen in the run configuration.
- report: e2e5_report (E2 to E5 tables).

CLI: python -m TRAMIL.e2e5_run {search,real,report} [--out DIR] [--input ROOT] [--shard i/n] [--units id,id] [--workers N]"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from . import dispatch, e1_arms, e1v07_run as e1, e2e5_stats as st, frontend, frozen, hpo, item_training, pooled_stats, validation

__provenance__ = {
    "revision": "experiments-experimental-plan-for-the-residual-semantic-prior-bipartite-v07.md",
    "sections": ["E2", "E3", "E4", "E5"],
    "experiments": ["E2", "E3", "E4", "E5"],
    "invariants": [
        "search_runs_the_new_arms_at_the_e1_kstar_over_the_p_grid_and_freezes_p_star_beside_the_e1_hpo",
        "real_units_use_kstar_and_the_arms_own_p_star_and_aliases_use_the_e1_p_star",
        "real_units_store_attention_fidelity_and_graph_diagnostics_by_subject",
        "derangement_seeds_are_recorded_per_fit_for_the_quintile_control",
        "a_real_unit_never_reads_its_outer_test_subjects_for_fitting",
        "history_arm_needs_the_frozen_prompt_list",
        "phases_are_resumable_shardable_and_zipped_under_the_e2e5_name",
    ],
}

NAME = "E2E5_v07"
PHASES = ("search", "real", "report")
DER_SEED = 2026  # quintile derangement of the fit (rep, fold, inner or outer, seed) uses SeedSequence([DER_SEED, ...])
# full_nohist is opt-in: it needs the frozen history prompt list (history_prompts), which real data does not carry yet
DEFAULTS = dict(reps=e1.REPS, seeds=e1.SEEDS, P=hpo.P_GRID, max_epochs=item_training.MAX_EPOCHS,
                arms=(*(a for a in e1_arms.E2E5_ARMS if a != "full_nohist"), "svm_x"), B=pooled_stats.B, k_star=None, p_alias=None, history_prompts=None)
OWN_SUBS = ("search", "hpo", "real")  # frozen models travel in the same zips (frontend/ and real/*.pt)
IMPORT_E1 = {"search": ("hpo",), "real": ("hpo",), "report": ("real", "perm", "hpo")}
IMPORT_OWN = {"search": ("search", "hpo"), "real": ("search", "hpo"), "report": ("real",)}


def _cfg(cfg):
    return {**DEFAULTS, **{k: v for k, v in cfg.items() if k in DEFAULTS}}


def search_arms(c):
    """Arms that search their own p*: the new arms except the aliases of E1 arms."""
    return [a for a in c["arms"] if a in e1_arms.E2E5_ARMS and a not in e1_arms.ALIAS]


def own_path(out, rep, fold):
    return Path(out) / "hpo" / f"e2e5_r{rep}_f{fold}.json"


def _kstar(out, rep, fold, c):
    return c["k_star"] if c["k_star"] else hpo.load_frozen(Path(out) / "hpo", rep, fold)["k_star"]


def units(phase, out, **cfg):
    c, out = _cfg(cfg), Path(out)
    specs = []
    if phase == "search":
        for rep in range(c["reps"]):
            for fold in range(e1.FOLDS):
                k = _kstar(out, rep, fold, c)
                specs += [{"phase": "search", "rep": rep, "fold": fold, "arm": a, "model": a, "k": int(k), "p": float(p),
                           "inner": r, "seed": hpo.SEED, "max_epochs": int(c["max_epochs"])}
                          for a in search_arms(c) for p in c["P"] for r in range(hpo.INNER_FOLDS)]
    elif phase == "real":
        for rep in range(c["reps"]):
            for fold in range(e1.FOLDS):
                k = _kstar(out, rep, fold, c)
                own = json.loads(own_path(out, rep, fold).read_text())["p_star"] if search_arms(c) else {}
                for a in c["arms"]:
                    if a == "svm_x":  # the E1 bag-level SVM again, to persist its pipeline (same inner grid search)
                        specs.append({"phase": "real", "rep": rep, "fold": fold, "arm": a, "model": "svm", "seed": 0})
                        continue
                    if a in e1_arms.ALIAS:
                        p = c["p_alias"] if c["p_alias"] is not None else \
                            hpo.load_frozen(out / "hpo", rep, fold)["p_star"][e1_arms.ALIAS[a]]
                    else:
                        p = own[a]
                    ka = hpo.K_PROMPT if e1_arms.SOURCE.get(a) == "R" else k
                    specs += [{"phase": "real", "rep": rep, "fold": fold, "arm": a, "model": a, "k": int(ka), "p": p,
                               "seed": s, "max_epochs": int(c["max_epochs"])} for s in c["seeds"]]
    return specs


# ---------------- unit execution ----------------
def der_seed(spec, tag):
    """Seed of the quintile derangement of one fit: tag = inner fold index, or 9 for the outer refit."""
    return int(np.random.SeedSequence([DER_SEED, spec["rep"], spec["fold"], tag, spec["seed"]]).generate_state(1)[0])


def _arm_items(spec, data):
    if spec["arm"] != "full_nohist":
        return data["items"]
    if "items_nohist" not in data:
        if not data.get("history"):
            raise ValueError("the full_nohist arm needs history_prompts in the run configuration (history prompt list)")
        data["items_nohist"] = frontend.drop_history(data["items"], data["history"])
    return data["items_nohist"]


def _bags(spec, data, items, tr, ev, k, tag, rec):
    """Prepared collated bags of one fit; the quintile control retargets the penalty target of the training bags."""
    post = salt = None
    if spec["arm"] == "permuted_quint":
        salt = der_seed(spec, tag)
        rec["inner%d" % tag if tag != 9 else "outer"] = salt

        def post(tb, idx, s=salt):
            donor = frontend.quintile_derangement([len(items[i]["R"]) for i in idx], s)
            frontend.retarget_quint(tb, [items[i] for i in idx], donor)
    elif spec["arm"] == "full_nohist":
        salt = "nohist"
    src = e1_arms.SOURCE.get(spec["model"], "X")
    key = (spec.get("perm", -1), spec["rep"], spec["fold"], "outer" if tag == 9 else tag, k, src, salt)
    return (*e1._bags(data, key, items, tr, ev, k, src, post), data["cache"][("fe",) + key])


def _extras(m, eb, items_test):
    """Per held-out subject: attention columns, exact ranks, constant flags, F_eff, surrogate fidelity, Al_b, RMS^Delta_b."""
    m.eval()
    out, M = {}, len(eb["_n"])
    with torch.no_grad():
        for i0 in range(0, M, 64):
            b = item_training.take(eb, range(i0, min(i0 + 64, M)))
            P = m._parts(b)
            for r in range(b["Y"].shape[0]):
                n = eb["_n"][i0 + r]
                A = P["A"][r, :n].double().cpu().numpy()
                ranks, const = st.column_ranks(A)
                Rc_hat = P["Rc_hat"]
                fid = [None] * 8 if Rc_hat is None else [None if np.isnan(v) else float(v) for v in st.spearman_fidelity(
                    Rc_hat[r, :n].double().cpu().numpy(), b["Rc"][r, :n].double().cpu().numpy())]
                al, rms = st.edge_diagnostics(P["E"][r, :n, :n].double().cpu().numpy(),
                                              b["G"][r, :n, :n].double().cpu().numpy(),
                                              P["Delta"][r, :n, :n].double().cpu().numpy())
                out[str(items_test[i0 + r]["subject"])] = {
                    "att": np.round(A, 5).tolist(), "rank": (2 * ranks).astype(int).T.tolist(),
                    "const": [bool(v) for v in const], "feff": st.feff(A), "fid": fid, "al": al, "rms": rms}
    return out


def run_unit(spec, data):
    """One unit; data = {"items", "Q", "cache"[, "history"]}. Returns the spec plus its result, extras and seconds."""
    e1._sync()
    t0 = time.perf_counter()
    if spec["model"] == "svm":
        return e1.run_unit(spec, data)  # svm_x: the E1 unit itself, under its own name
    items = _arm_items(spec, data)
    src = e1_arms.SOURCE.get(spec["model"], "X")
    train, test = e1._split(spec, data)  # the outer partition comes from the labels, which every variant shares
    k = min(spec["k"], items[0][src].shape[1])
    y = [it["y"] for it in items]
    out, rec = dict(spec), {}

    def inner(r):
        tr, va = validation.inner_splits(y, train, spec["fold"])[r]
        tb, vb, vbar, _ = _bags(spec, data, items, tr, va, k, r, rec)
        make = e1_arms.build(spec["model"], k=k, p_drop=spec["p"], vbar=vbar, Q=data.get("Q"))
        return item_training.train(make, tb, spec["max_epochs"], spec["seed"], eval_bags=vb)[1]

    if spec["phase"] == "search":
        out["curve"] = [float(v) for v in inner(spec["inner"])]
    else:
        curves = [inner(r) for r in range(hpo.INNER_FOLDS)]
        e = item_training.select_epoch(curves)
        tb, eb, vbar, fe = _bags(spec, data, items, train, test, k, 9, rec)
        make = e1_arms.build(spec["model"], k=k, p_drop=spec["p"], vbar=vbar, Q=data.get("Q"))
        m, _ = item_training.train(make, tb, e, spec["seed"])
        P = item_training.predict(m, eb)
        out.update(epoch=e, curves=[[float(v) for v in c] for c in curves],
                   oof={str(items[i]["subject"]): [float(v) for v in row] for i, row in zip(test, P)},
                   **({"extra": _extras(m, eb, [items[i] for i in test])} if hasattr(m, "_parts") else {}),
                   drift=[float(v) for v in m.query_drift().detach()] if getattr(m, "free", None) == "desc" else None)
        name = frozen.fe_name(spec, k, src)
        out["_files"] = {".pt": frozen.pack_state(spec, m, vbar, k, e, [items[i] for i in test], name)}
        out["_frontend"] = (name, frozen.pack_frontend(fe))
        if rec:
            out["derangement_seeds"] = rec
        if spec["arm"] == "full_nohist":
            out["history_removed_segments"] = int(sum(len(a["R"]) - len(b["R"]) for a, b in zip(data["items"], items)))
    e1._sync()
    out["seconds"] = time.perf_counter() - t0
    return out


# ---------------- phases ----------------
def reduce_p(results):
    """p* per arm at k*: obj(arm, p) = - min_e mean over the inner folds of the Ec. 39 curve (earliest epoch), ties to
    the smaller p (Ec. 102 at a fixed k)."""
    curves = {}
    for r in results:
        curves.setdefault((r["arm"], r["p"]), {})[r["inner"]] = r["curve"]
    obj = {}
    for key, byfold in sorted(curves.items()):
        mean = np.mean([byfold[i] for i in sorted(byfold)], axis=0)
        obj[key] = -float(mean[item_training.select_epoch(mean[None, :]) - 1])
    return {a: hpo._best([(p, o) for (aa, p), o in obj.items() if aa == a]) for a in sorted({a for a, _ in obj})}, obj


def _freeze_complete(out, c):
    arms = search_arms(c)
    if not arms:
        return
    have = {}
    for r in e1._load_dir(Path(out) / "search"):
        have.setdefault((r["rep"], r["fold"]), []).append(r)
    need = len(arms) * len(c["P"]) * hpo.INNER_FOLDS
    for (rep, fold), res in sorted(have.items()):
        res = [r for r in res if r["arm"] in arms]
        if len(res) == need and not own_path(out, rep, fold).exists():
            p_star, obj = reduce_p(res)
            e1._write(own_path(out, rep, fold), {
                "rep": rep, "fold": fold, "k_star": res[0]["k"], "p_star": p_star,
                "table": [{"arm": a, "p": p, "objective": o} for (a, p), o in sorted(obj.items())],
                "grid": {"P": list(c["P"]), "arms": arms, "inner_folds": hpo.INNER_FOLDS, "max_epochs": c["max_epochs"]},
                "provenance": {"seed": hpo.SEED, "commit": hpo._commit(), "k_star_from": "E1 hpo_r{rep}_f{fold}.json"}})


def run_phase(phase, data, out, workers=1, shard=None, only=None, **cfg):
    c, out = _cfg(cfg), Path(out)
    if phase == "report":
        from . import e2e5_report

        return {"ran": ["report"], "report": e2e5_report.report(data, out, **c)}
    _freeze_complete(out, c)
    specs = e1.select(units(phase, out, **c), shard, only)
    pending = [s for s in specs if not (out / phase / f"{e1.uid(s)}.json").exists()]
    dispatch.run(run_unit, pending, data, workers, on_result=lambda s, r: e1.write_unit(out / phase, e1.uid(s), r))
    _freeze_complete(out, c)
    return {"ran": [e1.uid(s) for s in pending], "skipped": len(specs) - len(pending)}


# ---------------- remote entry ----------------
def run(phases, out_dir=None, input_root=None, shard=None, units=None, workers=1, limit_bags=None,
        import_from_inputs=True, **cfg):
    """Remote callable (run-config.json): like e1v07_run.run. Upstream results are imported per phase from the E1 outputs
    (E1_v07: the frozen hpo, and for the report the real and perm units) and from earlier E2E5 outputs (E2E5_v07), loose
    or zipped. A real run (no limit_bags) ends by zipping its unit results under the E2E5_v07 name."""
    from . import e1_run

    phases = [phases] if isinstance(phases, str) else list(phases)
    out = Path(out_dir) if out_dir else e1._default_out().with_name(NAME)
    out.mkdir(parents=True, exist_ok=True)
    e1.unpack_own_zips(out, NAME, OWN_SUBS, phases=phases)
    got = []
    if import_from_inputs:
        roots = e1._input_roots(input_root)
        for name, table in ((e1.NAME, IMPORT_E1), (NAME, IMPORT_OWN)):
            subs = sorted({s for ph in phases for s in table[ph]})
            got += e1.import_upstream_files(roots, out, name, subs)
    data, info = e1.load_data(input_root, limit_bags)
    data["history"] = cfg.get("history_prompts")
    summary = {"imported": len(got), "workers": workers, "shard": shard, **info}
    for ph in phases:
        res = run_phase(ph, data, out, workers=workers, shard=shard, only=units, **cfg)
        summary[ph] = {"ran": len(res["ran"]), "skipped": res.get("skipped", 0)}
    if limit_bags is None:
        summary["zips"] = [z.name for z in e1.zip_outputs(out, phases, shard, exclude=got, name=NAME)]
        for f in got:
            if f.parent.name != "hpo":
                f.unlink(missing_ok=True)
    e1_run._stamp(summary)
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m TRAMIL.e2e5_run")
    ap.add_argument("phase", choices=PHASES)
    ap.add_argument("--out", default=str(e1._default_out().with_name(NAME)))
    ap.add_argument("--input")
    ap.add_argument("--shard")
    ap.add_argument("--units")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--limit-bags", type=int)
    ap.add_argument("--max-epochs", type=int, default=item_training.MAX_EPOCHS)
    a = ap.parse_args(argv)
    res = run(a.phase, a.out, a.input, a.shard, a.units.split(",") if a.units else None, a.workers, a.limit_bags,
              max_epochs=a.max_epochs)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
