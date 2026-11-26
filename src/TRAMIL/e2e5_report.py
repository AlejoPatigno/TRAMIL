"""Report of E2 to E5 of experiments v07 ('Expected report E2' to 'E5') from the out-of-fold units of the E1 real and perm
phases (full, permuted, untargeted, r12, prompt_only, abmil8, symptom_guided, r06_8 and their Ec. 91 nulls) and of the e2e5
real phase (graph_corr, graph_prior, binary_sup, free_desc, free_zero, permuted_quint, full_nohist and the *_x re-runs that
carry attention, fidelity and graph diagnostics) -> E2E5_v07.json and .md.

Every bound reads the shared subject bootstrap of pooled_stats (B = 9999, alpha = 0.05). A section whose arms are not all
present is reported as missing, never filled. What is deliberately not computed here: the figures, LiftSem^S (E6 deletion
influence) and the qualitative panels of E3 (edge attention by interviewer topic) and E5 (influence by topic)."""
import json
from pathlib import Path

import numpy as np

from . import e1v07_run as e1, e2e5_stats as st, pooled_stats, statistics, validation

__provenance__ = {
    "revision": "experiments-experimental-plan-for-the-residual-semantic-prior-bipartite-v07.md",
    "sections": ["E2", "E3", "E4", "E5"],
    "experiments": ["E2", "E3", "E4", "E5"],
    "invariants": [
        "e2_reports_h5_from_the_skill_bounds_and_the_h8_family_with_bh",
        "e3_reports_the_graph_contrasts_with_bounds_bh_and_the_h2_matrix",
        "e4_reports_discrimination_variance_and_stability_readings_under_both_conventions",
        "e5_reports_the_shortcut_controls_the_prompt_only_null_and_the_h4_matrix",
        "missing_arms_are_reported_as_missing_and_never_filled",
    ],
}

E2_ARMS = ("full", "permuted", "untargeted", "r12", "abmil8", "symptom_guided")
LABELS = {"full": "Full model", "permuted": "Permuted target (Ec. 37)", "untargeted": "Untargeted, exploratory (H8)",
          "r12": "r12 reference, exploratory (H8)", "abmil8": "ABMIL-8, secondary estimate",
          "symptom_guided": "Symptom-Guided, secondary estimate", "wmean": "Class-weighted training item mean",
          "r06_8": "(a) no graph layer (r06 with item heads)", "graph_corr": "(b) correction only",
          "graph_prior": "(c) prior only", "binary_sup": "Binary supervision", "free_desc": "Descriptor-initialized",
          "free_zero": "Zero-initialized", "prompt_only": "Prompt-only reference",
          "permuted_quint": "Prompt rows permuted across subjects (quintile derangement)",
          "full_nohist": "Mental-health-history responses removed"}
ITEMS = ("no interest", "depressed", "sleep", "tired", "appetite", "failure", "concentration", "psychomotor")


# ---------------- unit readers ----------------
def run_matrix(results, arm, subjects, reps, seeds):
    """(R, M) scores of the R = reps x seeds runs of an arm (R ordered by repetition then seed; the sum of the eight item
    predictions, or the logit of a binary-head model); None when a run is incomplete."""
    acc = {}
    for r in results:
        if r["arm"] == arm:
            acc.setdefault((r["rep"], r["seed"]), {}).update(r["oof"])
    rows = []
    for q in range(reps):
        for s in seeds:
            d = acc.get((q, s), {})
            if len(d) != len(subjects):
                return None
            rows.append([float(np.sum(d[str(b)])) for b in subjects])
    return np.array(rows)


def fold_aucs(runs, y, reps, seeds):
    """The R_v = reps x seeds x 5 fold-level AUROCs of an arm (descriptive F and Brown-Forsythe, Ec. 64, 65)."""
    out = []
    for i, q in enumerate(range(reps)):
        for j in range(len(seeds)):
            out += [pooled_stats.auroc(runs[i * len(seeds) + j][te], y[te]) for te in validation.outer_splits(y, q)]
    return out


def extras_runs(results, arm, subjects, reps, seeds):
    """Per subject, the extras of the runs in (repetition, seed) order; None when an arm has no complete extras."""
    acc = {}
    for r in results:
        if r["arm"] == arm and r.get("extra"):
            acc.setdefault((r["rep"], r["seed"]), {}).update(r["extra"])
    order = [(q, s) for q in range(reps) for s in seeds]
    if any(len(acc.get(k, {})) != len(subjects) for k in order):
        return None
    return [[acc[k][str(b)] for k in order] for b in subjects]


def stability(ex):
    """Stab_bj (M, 8) with nan for undefined pairs (Ec. 95) and the mean pairwise TV of the nine attention columns."""
    S, tv = [], []
    for runs in ex:
        ranks = np.stack([np.array(e["rank"]).T / 2 for e in runs])
        S.append(st.stab_subject(ranks, np.array([e["const"] for e in runs])))
        tv.append(st.mean_pairwise_tv(np.stack([np.array(e["att"]) for e in runs])))
    return np.array(S), float(np.mean(tv))


def diag(ex):
    """Mean over subjects and runs of F_eff, Al_b, RMS^Delta_b and the surrogate fidelity (per-item mean first)."""
    pick = lambda key: float(np.mean([e[key] for runs in ex for e in runs]))  # noqa: E731
    fid = np.array([[np.nan if v is None else v for v in e["fid"]] for runs in ex for e in runs], float)
    with np.errstate(all="ignore"):
        item = np.nanmean(fid, axis=0) if np.isfinite(fid).any() else np.full(8, np.nan)
    return {"feff": pick("feff"), "al": pick("al"), "rms_delta": pick("rms"),
            "fidelity_items": item.tolist(), "fidelity": float(np.nanmean(item)) if np.isfinite(item).any() else None}


# ---------------- sections ----------------
def _skill(P, S, y, folds_q, I):
    Sh, w, refw = pooled_stats.ss_inputs(P, S, y, folds_q)
    boot = pooled_stats.boot_ss_omega(Sh, S, w, refw, I)
    Pp, refp = st.plain_inputs(P, S, folds_q)
    return {"point": pooled_stats.ss_omega(Sh, S, w, refw), "boot": boot, "lower": st.lower(boot), "Pp": Pp,
            "ref": refp, "refw": refw, "w": w}


def _metrics(Pp, S, ref):
    return {"ss": st.ss_plain(Pp, S, ref), "macro": float(np.mean(st.ss_items(Pp, S, ref))),
            "shape": st.ss_shape(Pp, S, ref), "kappa": st.weighted_kappa(Pp, S), "mae": st.mae(Pp, S)}


def e2(profiles, pooled, ex, S, y, folds_q, I, subjects):
    if "full" not in profiles:
        return {"missing": ["full"]}
    sk = {a: _skill(profiles[a], S, y, folds_q, I) for a in E2_ARMS if a in profiles}
    pi = sk.get("permuted")
    rows = []
    for a, k in sk.items():
        r = {"arm": a, "ss_omega": k["point"], "lower": k["lower"], **_metrics(k["Pp"], S, k["ref"])}
        if pi is not None and a != "permuted":
            d = k["boot"] - pi["boot"]
            r.update(diff_to_permuted=k["point"] - pi["point"], diff_lower=st.lower(d), diff_p=st.p_one_sided(d))
        rows.append(r)
    ref = sk["full"]
    rows.append({"arm": "wmean", "ss_omega": 0.0, **_metrics(ref["refw"], S, ref["ref"])})
    rho_f, p_f = st.item_spearman(sk["full"]["Pp"], S)
    rho_p = st.item_spearman(pi["Pp"], S)[0] if pi is not None else np.full(8, np.nan)
    holm = np.full(8, np.nan)
    ok = np.isfinite(p_f)
    if ok.any():
        holm[ok] = statistics.holm(p_f[ok])
    items = [{"item": j + 1, "name": ITEMS[j], "ss": float(st.ss_items(ref["Pp"], S, ref["ref"])[j]),
              "spearman_full": float(rho_f[j]), "spearman_permuted": float(rho_p[j]), "holm_p": float(holm[j])}
             for j in range(8)]
    h8, raw = [], []
    fx, ux = ex.get("full_x"), ex.get("untargeted_x")
    if "untargeted" in sk:
        d = sk["full"]["boot"] - sk["untargeted"]["boot"]
        h8.append({"name": "full minus untargeted, SS omega", "diff": sk["full"]["point"] - sk["untargeted"]["point"],
                   "lower": st.lower(d), "p": st.p_one_sided(d),
                   "fidelity": {"full": diag(fx)["fidelity"] if fx else None,
                                "untargeted": diag(ux)["fidelity"] if ux else None}})
    if fx and ux:
        c = st.stab_contrast(stability(fx)[0], stability(ux)[0], I)["complete_case"]
        h8.append({"name": "full minus untargeted, attention stability (Ec. 95, 101)", "diff": c["diff"],
                   "lower": c["lower"], "p": st.p_one_sided(c["boot"]), "fidelity": None})
    if "r12" in pooled and "full" in pooled:
        h8.append({"name": "full minus r12 reference, AUROC (cost of removing text)", "diff": pooled["r12"]["delta_bar"],
                   "lower": pooled["r12"]["ell"], "p": st.p_one_sided(pooled["r12"]["delta_boot"]), "fidelity": None})
    for r, a in zip(h8, st.bh([h["p"] for h in h8])):
        r["bh_p"] = float(a)
    h5 = {"a_lower": ref["lower"], "a_p": st.p_one_sided(ref["boot"]), "a_holds": bool(ref["lower"] > 0)}
    if pi is not None:
        d = ref["boot"] - pi["boot"]
        h5.update(b_diff=ref["point"] - pi["point"], b_lower=st.lower(d), b_p=st.p_one_sided(d),
                  b_holds=bool(st.lower(d) > 0))
        h5["verdict"] = ("supported" if h5["a_holds"] and h5["b_holds"] else
                         "partially supported" if h5["a_holds"] else "not supported")
    panel = st.declared_profile_subjects(S, subjects)
    idx = [subjects.index(b) for b in panel]
    return {"rows": rows, "items": items, "h8": h8, "h5": h5,
            "panel": [{"subject": b, "true": S[i].tolist(), "full": ref["Pp"][i].tolist(), "mean": ref["refw"][i].tolist()}
                      for b, i in zip(panel, idx)]}


def _contrast(pooled, m):
    o = pooled[m]
    return {"diff": o["delta_bar"], "lower": o["ell"], "p": st.p_one_sided(o["delta_boot"]),
            "distinguishable": bool(o["ell"] > 0)}


def e3(scores, profiles, pooled, ex, S, y, folds_q, I):
    arms = [a for a in ("r06_8", "graph_corr", "graph_prior") if a in pooled]
    if "full" not in pooled or not arms:
        return {"missing": [a for a in ("full", "r06_8", "graph_corr", "graph_prior") if a not in pooled]}
    rows = [{"arm": "full", "auroc": pooled["full"]["A_bar"]}] + [{"arm": a, "auroc": pooled[a]["A_bar"],
                                                                    **_contrast(pooled, a)} for a in arms]
    for r, adj in zip(rows[1:], st.bh([r["p"] for r in rows[1:]])):
        r["bh_p"] = float(adj)
    for r in rows:
        r["ss_omega"] = _skill(profiles[r["arm"]], S, y, folds_q, I)["point"] if r["arm"] in profiles else None
        key = "full_x" if r["arm"] == "full" else r["arm"]
        d = diag(ex[key]) if ex.get(key) else {}
        r.update(al=d.get("al"), rms_delta=d.get("rms_delta"))
    n = sum(r["distinguishable"] for r in rows[1:])
    verdict = "supported" if n == len(rows) - 1 == 3 else "partially supported" if n else "not supported"
    return {"rows": rows, "h2": verdict, "n_distinguishable": n,
            "novelty_supported": bool(all(r["distinguishable"] for r in rows[1:] if r["arm"] in ("r06_8", "graph_prior")))}


def e4(results, pooled, ex, y, reps, seeds, I, subjects):
    full = run_matrix(results, "full", subjects, reps, seeds)
    out = {"contrasts": [], "stability": [], "tv": {}, "drift": None}
    if full is None or "full" not in pooled:
        return {"missing": ["full"]}
    raw = []
    for tag, arm, ranks in (("H3 full minus binary supervision (percentile ranks)", "binary_sup", True),
                            ("H3b full minus descriptor-initialized", "free_desc", False),
                            ("full minus zero-initialized (estimate)", "free_zero", False)):
        runs = run_matrix(results, arm, subjects, reps, seeds)
        if runs is None or arm not in pooled:
            out["contrasts"].append({"name": tag, "missing": [arm]})
            continue
        a, b = (st.percentile_ranks(full), st.percentile_ranks(runs)) if ranks else (full, runs)
        lv, boot = st.log_variance_ratio(b, a, I)
        c = _contrast(pooled, arm)
        out["contrasts"].append({"name": tag, "arm": arm, "auroc_diff": c["diff"], "auroc_lower": c["lower"],
                                 "auroc_p": c["p"], "log_var_ratio": lv, "var_lower": st.lower(boot),
                                 "var_p": st.p_one_sided(boot),
                                 **st.fisher_brown_forsythe(fold_aucs(full, y, reps, seeds), fold_aucs(runs, y, reps, seeds))})
        raw += [c["p"], st.p_one_sided(boot)]
    adj = iter(st.bh(raw))
    for c in out["contrasts"]:
        if "arm" in c:
            c["bh_p_auroc"], c["bh_p_var"] = float(next(adj)), float(next(adj))
            ok = (c["auroc_lower"] > 0, c["var_lower"] > 0)
            c["verdict"] = "supported" if all(ok) else "partially supported" if any(ok) else "not supported"
    stabs = {a: stability(ex[a]) for a in ("full_x", "permuted_x", "free_desc", "free_zero") if ex.get(a)}
    for a, name in (("permuted_x", "full minus permuted target"), ("free_desc", "full minus descriptor-initialized"),
                    ("free_zero", "full minus zero-initialized")):
        if "full_x" in stabs and a in stabs:
            c = st.stab_contrast(stabs["full_x"][0], stabs[a][0], I)
            out["stability"].append({"name": name, "arm": a, "cc_diff": c["complete_case"]["diff"],
                                     "cc_lower": c["complete_case"]["lower"], "z_diff": c["zero_filled"]["diff"],
                                     "z_lower": c["zero_filled"]["lower"], "dropped_full": c["dropped_a"],
                                     "dropped_other": c["dropped_b"], "dropped_full_item": c["dropped_a_item"],
                                     "dropped_other_item": c["dropped_b_item"], "cc_p": st.p_one_sided(c["complete_case"]["boot"])})
    out["tv"] = {a: v[1] for a, v in stabs.items()}
    pi = next((s for s in out["stability"] if s["arm"] == "permuted_x"), None)
    if pi:
        both = pi["cc_lower"] > 0 and pi["z_lower"] > 0
        out["anchoring"] = {"supported": bool(both), "fraction_gap": abs(pi["dropped_full"] - pi["dropped_other"])}
    d = [r["drift"] for r in results if r["arm"] == "free_desc" and r.get("drift")]
    if d:
        out["drift"] = {"mean_per_item": np.mean(d, axis=0).tolist(), "n": len(d)}
    return out


def e5(results, scores, pooled, ex, perm_results, data, subjects, h1):
    if "full" not in pooled:
        return {"missing": ["full"]}
    nulls = e1._nulls(perm_results, data, subjects)
    rows = []
    for a in ("full", "permuted", "prompt_only", "permuted_quint"):
        if a not in pooled:
            continue
        o = pooled[a]
        r = {"arm": a, "auroc": o["A_bar"], "diff": o.get("delta_bar"), "diff_lower": o.get("ell"), "bound": o["lower"]}
        if a in nulls:
            r.update(pooled_stats.assay(o["lower"], o["A_bar"], nulls[a]), b=int((np.asarray(nulls[a]) >= o["A_bar"]).sum()))
        rows.append(r)
    pq = next((r for r in rows if r["arm"] == "prompt_only" and "passes" in r), None)
    ell = pooled["permuted"]["ell"] if "permuted" in pooled else None
    verdict = None
    if ell is not None and pq is not None:
        verdict = ("not supported" if ell <= 0 else "supported" if not pq["passes"] else "partially supported")
        if h1 is not None and h1 != "supported":
            verdict = f"inconclusive (read as an estimate: H1 was {h1}); base reading {verdict}"
    contrasts = [(a, _contrast(pooled, a)) for a in ("permuted_quint", "full_nohist") if a in pooled]
    for (a, c), adj in zip(contrasts, st.bh([c["p"] for _, c in contrasts])):
        c["bh_p"] = float(adj)
    region = {}
    for tag, arm in (("complete", "full_x"), ("reduced", "full_nohist")):
        if ex.get(arm):
            region[tag] = {"auroc": pooled[arm]["A_bar"] if arm in pooled else None, "feff": diag(ex[arm])["feff"],
                           "liftsem_s": "not computed (E6 deletion influence)"}
    return {"rows": rows, "h4": verdict, "h1_verdict": h1, "exploratory": dict(contrasts), "region": region,
            "complete_minus_reduced": dict(contrasts).get("full_nohist")}


# ---------------- report ----------------
def report(data, out, **cfg):
    out = Path(out)
    reps, seeds, B = cfg.get("reps", e1.REPS), tuple(cfg.get("seeds", e1.SEEDS)), cfg.get("B", pooled_stats.B)
    items = data["items"]
    subjects = [it["subject"] for it in items]
    y = np.array([it["y"] for it in items])
    S = np.stack([it["s"] for it in items]).astype(float)
    results = e1._load_dir(out / "real")
    scores, profiles = e1._score_matrix(results, subjects, reps)
    I = pooled_stats.bootstrap_indices(y, B)
    pooled = pooled_stats.pooled(scores, y, I, ref="full") if "full" in scores else {}
    folds_q = [validation.outer_splits(y, q) for q in range(reps)]
    ex = {a: extras_runs(results, a, subjects, reps, seeds) for a in sorted({r["arm"] for r in results if r.get("extra")})}
    h1 = None
    if (out / "E1_v07.json").exists():
        h1 = json.loads((out / "E1_v07.json").read_text())["h1"]["verdict"]
    js = {"stage": "E2 to E5 v07", "n_subjects": len(y), "n_positive": int(y.sum()), "B": B,
          "arms_present": sorted(scores), "arms_with_extras": [a for a, v in ex.items() if v],
          "E2": e2(profiles, pooled, ex, S, y, folds_q, I, subjects),
          "E3": e3(scores, profiles, pooled, ex, S, y, folds_q, I),
          "E4": e4(results, pooled, ex, y, reps, seeds, I, subjects),
          "E5": e5(results, scores, pooled, ex, e1._load_dir(out / "perm"), data, subjects, h1),
          "boot_seed": pooled_stats.BOOT_SEED}
    e1._write(out / "E2E5_v07.json", _clean(js))
    (out / "E2E5_v07.md").write_text(_markdown(js))
    return js


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items() if k not in ("boot", "Pp", "ref", "refw", "w")}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return o


def _t(head, rows):
    return ["| " + " | ".join(head) + " |", "|" + "---|" * len(head), *("| " + " | ".join(r) + " |" for r in rows), ""]


def _markdown(js):
    f = e1._f
    L = lambda a: LABELS.get(a, a)  # noqa: E731
    md = ["# E2 to E5 (experiments v07) report", "", f"Subjects {js['n_subjects']} ({js['n_positive']} positive); B = {js['B']}.",
          f"Arms present: {', '.join(js['arms_present'])}. Arms with extras: {', '.join(js['arms_with_extras']) or '-'}.", ""]
    E2 = js["E2"]
    md += ["## E2 The approximate per-subject PHQ-8 item profile", ""]
    if "missing" in E2:
        md += [f"Missing: {E2['missing']}", ""]
    else:
        md += _t(["Arm", "SS omega (Ec. 93)", "SS omega lower bound", "Paired difference to permuted target",
                  "Lower bound of the difference (Ec. 99)", "Unweighted SS (Ec. 88)", "Macro SS (Ec. 92)",
                  "Shape skill (Ec. 98)", "Weighted kappa", "Mean absolute item error"],
                 [[L(r["arm"]), f(r["ss_omega"]), f(r.get("lower")), f(r.get("diff_to_permuted")), f(r.get("diff_lower")),
                   f(r["ss"]), f(r["macro"]), f(r["shape"]), f(r["kappa"]), f(r["mae"])] for r in E2["rows"]])
        md += _t(["PHQ-8 item", "Per-item SS, full model (Ec. 92)", "Spearman, full model (Ec. 61)", "Spearman, permuted target",
                  "Holm p"], [[f"Item {r['item']}, {r['name']}", f(r["ss"]), f(r["spearman_full"]),
                              f(r["spearman_permuted"]), f(r["holm_p"], 4)] for r in E2["items"]])
        md += _t(["H8 exploratory estimate", "Mean paired difference", "One-sided lower bound", "BH-adjusted p",
                  "Surrogate fidelity, mean per-item Spearman (full / untargeted)"],
                 [[h["name"], f(h["diff"]), f(h["lower"]), f(h["bh_p"], 4),
                   "n/a" if not h["fidelity"] else f"{f(h['fidelity']['full'])} / {f(h['fidelity']['untargeted'])}"]
                  for h in E2["h8"]])
        h5 = E2["h5"]
        md += _t(["Comparison", "Test", "Statistic", "Raw p", "Effect size interval", "n", "alpha"],
                 [["H5 (a), SS omega of the full model above zero", "one-sided subject bootstrap (Ec. 99)",
                   f"lower = {f(h5['a_lower'])}", f(h5["a_p"], 4), f"[{f(h5['a_lower'])}, inf)", str(js["n_subjects"]), "0.05"],
                  ["H5 (b), full minus permuted target, SS omega", "one-sided subject bootstrap (Ec. 99)",
                   f"lower = {f(h5.get('b_lower'))}", f(h5.get("b_p"), 4), f"[{f(h5.get('b_lower'))}, inf)",
                   str(js["n_subjects"]), "0.05"]])
        md += [f"H5 verdict: {h5.get('verdict', 'n/a (permuted target missing)')}.", ""]
    E3 = js["E3"]
    md += ["## E3 Graph ablations of the second use of the prior", ""]
    if "missing" in E3:
        md += [f"Missing: {E3['missing']}", ""]
    else:
        md += _t(["Variant", "Pooled AUROC", "Paired difference to full model (Ec. 83)", "One-sided lower bound (Ec. 85)",
                  "BH-adjusted p", "SS omega", "Mean Al_b (Ec. 107)", "Mean RMS of Delta_b (Ec. 107)"],
                 [[("Full model" if r["arm"] == "full" else L(r["arm"])), f(r["auroc"]), f(r.get("diff")), f(r.get("lower")),
                   f(r.get("bh_p"), 4), f(r["ss_omega"]), f(r["al"]), f(r["rms_delta"])] for r in E3["rows"]])
        md += [f"H2: {E3['h2']} ({E3['n_distinguishable']} of 3 distinguishable). Novelty claim of the graph "
               f"(ablations (a) and (c) distinguishable): {f(E3['novelty_supported'])}.", ""]
    E4 = js["E4"]
    md += ["## E4 Item supervision and the text-distilled prior against free queries", ""]
    if "missing" in E4:
        md += [f"Missing: {E4['missing']}", ""]
    else:
        md += _t(["Contrast", "Paired AUROC difference", "AUROC lower bound", "Log variance ratio (Ec. 89)",
                  "Variance ratio lower bound", "BH-adjusted p (AUROC / variance)", "F (Ec. 64), descriptive",
                  "Brown-Forsythe W (Ec. 65), descriptive", "Verdict"],
                 [[c["name"], f(c.get("auroc_diff")), f(c.get("auroc_lower")), f(c.get("log_var_ratio")),
                   f(c.get("var_lower")), f"{f(c.get('bh_p_auroc'), 4)} / {f(c.get('bh_p_var'), 4)}" if "arm" in c else "missing",
                   f(c.get("F")), f(c.get("W")), c.get("verdict", "inconclusive")] for c in E4["contrasts"]])
        md += _t(["Stability contrast (Ec. 101)", "Complete-case difference", "Complete-case lower bound",
                  "Zero-filled difference", "Zero-filled lower bound", "Dropped pairs, full model", "Dropped pairs, other arm"],
                 [[s["name"], f(s["cc_diff"]), f(s["cc_lower"]), f(s["z_diff"]), f(s["z_lower"]), f(s["dropped_full"]),
                   f(s["dropped_other"])] for s in E4["stability"]])
        a = E4.get("anchoring")
        md += [f"Anchoring (l_stab_pi > 0 under both readings): {f(a['supported']) if a else 'n/a'}.",
               "Mean pairwise TV between the nine attention columns: " +
               (", ".join(f"{k} {f(v)}" for k, v in E4["tv"].items()) or "n/a") + ".",
               "Query drift (descriptor-initialized, mean cosine per item): " +
               (", ".join(f(v) for v in E4["drift"]["mean_per_item"]) if E4["drift"] else "n/a") + ".", ""]
    E5 = js["E5"]
    md += ["## E5 Shortcut controls", ""]
    if "missing" in E5:
        md += [f"Missing: {E5['missing']}", ""]
    else:
        md += _t(["Arm", "Pooled AUROC", "Paired difference to full model", "One-sided lower bound of the difference",
                  "Bootstrap lower bound of the AUROC (Ec. 91)", "Largest of the 23 null AUROCs",
                  "Exceedances b, point estimate", "Descriptive permutation p", "Passes Ec. 91"],
                 [[L(r["arm"]), f(r["auroc"]), f(r.get("diff")), f(r.get("diff_lower")), f(r["bound"]),
                   f(r.get("null_max")), f(r.get("b")), f(r.get("p")), f(r["passes"]) if "passes" in r else "no null run"]
                  for r in E5["rows"]])
        reg = E5["region"]
        md += _t(["Interview-region reading", "AUROC", "LiftSem S", "F_eff (Ec. 52)"],
                 [[("Complete bags" if k == "complete" else "Mental-health-history responses removed"), f(v["auroc"]),
                   v["liftsem_s"], f(v["feff"])] for k, v in reg.items()])
        c = E5["complete_minus_reduced"]
        md += [f"H4: {E5['h4'] or 'n/a'}. Complete minus history-removed AUROC: "
               f"{f(c['diff']) if c else 'n/a'} (lower {f(c['lower']) if c else 'n/a'}, BH p {f(c['bh_p'], 4) if c else 'n/a'}).", ""]
    md += ["Figures, LiftSem^S and the qualitative panels of E3 and E5 are deferred."]
    return "\n".join(md) + "\n"
