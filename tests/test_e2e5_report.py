"""E2 to E5 report on synthetic unit results: the tables fill, the verdict rules follow the v07 hypothesis matrices."""
import json

import numpy as np
import pytest

from TRAMIL import e1v07_run as e1, e2e5_report as rep, e2e5_stats as st, validation

M, REPS, SEEDS = 40, 2, (11, 23)
NOISE = {"full": 0.3, "permuted": 1.5, "untargeted": 0.5, "r12": 0.4, "abmil8": 0.8, "symptom_guided": 0.9,
         "r06_8": 0.9, "graph_corr": 0.6, "graph_prior": 0.7, "free_desc": 0.8, "free_zero": 0.7, "permuted_quint": 1.4,
         "prompt_only": 2.0, "binary_sup": 0.5, "full_nohist": 0.35, "full_x": 0.3, "permuted_x": 1.5, "untargeted_x": 0.5}
ATT_NOISE = {"full_x": 0.05, "permuted_x": 1.0, "untargeted_x": 0.5, "free_desc": 0.6, "free_zero": 0.8, "full_nohist": 0.1}


def _items():
    g = np.random.default_rng(0)
    out = []
    for i in range(M):
        s = (g.random(8) < (0.65 if i % 3 == 0 else 0.2)).astype(float) * g.integers(1, 4, 8)
        out.append({"subject": 500 + i, "s": s.astype(np.float32), "y": int(s.sum() >= 10),
                    "R": np.zeros((int(g.integers(4, 9)), 8), np.float32)})
    return out


def _extra(g, n, noise, base):
    A = np.abs(base + noise * g.random((n, 8))) + 1e-3
    A = A / A.sum(0)
    ranks, const = st.column_ranks(A)
    return {"att": np.round(A, 5).tolist(), "rank": (2 * ranks).astype(int).T.tolist(), "const": [bool(c) for c in const],
            "feff": st.feff(A), "fid": [float(v) for v in g.random(8)], "al": 0.1, "rms": 0.2}


def _build(tmp_path, items, arms, noise=NOISE):
    g = np.random.default_rng(1)
    y = np.array([it["y"] for it in items])
    S = np.stack([it["s"] for it in items])
    out = tmp_path / "out"
    (out / "real").mkdir(parents=True)
    bases = {it["subject"]: g.random((len(it["R"]), 8)) for it in items}
    for q in range(REPS):
        for fold, te in enumerate(validation.outer_splits(y, q)):
            for arm in arms:
                for seed in SEEDS:
                    oof = {}
                    for i in te:
                        if arm == "binary_sup":
                            oof[str(items[i]["subject"])] = [float((S[i].sum() - 10) / 4 + g.normal(0, noise[arm]))]
                        else:
                            oof[str(items[i]["subject"])] = (S[i] + g.normal(0, noise[arm], 8)).tolist()
                    r = {"phase": "real", "rep": q, "fold": fold, "arm": arm, "seed": seed, "oof": oof}
                    if arm in ATT_NOISE:
                        r["extra"] = {str(items[i]["subject"]): _extra(g, len(items[i]["R"]), ATT_NOISE[arm],
                                                                       bases[items[i]["subject"]]) for i in te}
                    if arm == "free_desc":
                        r["drift"] = [0.9] * 8
                    (out / "real" / f"r{q}_f{fold}_{arm}_s{seed}.json").write_text(json.dumps(r))
    (out / "perm").mkdir()
    for p in range(3):
        for fold, te in enumerate(validation.outer_splits(y, 0)):
            for arm in ("full", "prompt_only"):
                oof = {str(items[i]["subject"]): [float(g.normal())] * 8 for i in te}
                (out / "perm" / f"p{p}_f{fold}_{arm}.json").write_text(json.dumps(
                    {"phase": "perm", "perm": p, "fold": fold, "arm": arm, "oof": oof}))
    return out




@pytest.fixture(scope="module")
def built(tmp_path_factory):
    items = _items()
    out = _build(tmp_path_factory.mktemp("rep"), items, tuple(NOISE))
    js = rep.report({"items": items}, out, reps=REPS, seeds=SEEDS, B=300)
    return out, js, items


def test_e2_fills_the_profile_table_the_item_table_the_h8_family_and_the_h5_rule(built):
    _, js, _ = built
    E2 = js["E2"]
    assert [r["arm"] for r in E2["rows"]] == ["full", "permuted", "untargeted", "r12", "abmil8", "symptom_guided", "wmean"]
    full = E2["rows"][0]
    assert full["ss_omega"] > 0 and full["lower"] > 0 and full["diff_lower"] > 0 and E2["h5"]["verdict"] == "supported"
    assert E2["rows"][-1]["ss_omega"] == 0.0 and len(E2["items"]) == 8 and len(E2["panel"]) == 8
    assert len(E2["h8"]) == 3 and all(0 <= h["bh_p"] <= 1 for h in E2["h8"])
    raw = sorted(h["p"] for h in E2["h8"])
    assert min(h["bh_p"] for h in E2["h8"]) >= raw[0] - 1e-12
    assert E2["h8"][0]["fidelity"]["full"] is not None


def test_e2_h5_is_not_supported_when_the_model_is_worse_than_the_mean(tmp_path):
    items = _items()
    out = _build(tmp_path, items, ("full", "permuted"), noise=dict(NOISE, full=6.0, permuted=7.0))
    js = rep.report({"items": items}, out, reps=REPS, seeds=SEEDS, B=300)
    assert not js["E2"]["h5"]["a_holds"] and js["E2"]["h5"]["verdict"] == "not supported"


def test_e3_reports_the_graph_contrasts_with_bounds_bh_and_the_h2_matrix(built):
    _, js, _ = built
    E3 = js["E3"]
    assert [r["arm"] for r in E3["rows"]] == ["full", "r06_8", "graph_corr", "graph_prior"]
    for r in E3["rows"][1:]:
        assert r["lower"] < r["diff"] and 0 <= r["bh_p"] <= 1 and isinstance(r["distinguishable"], bool)
    assert E3["rows"][0]["al"] == pytest.approx(0.1) and E3["rows"][1]["al"] is None  # (a) has no graph diagnostics
    assert E3["h2"] in ("supported", "partially supported", "not supported")
    assert E3["n_distinguishable"] == sum(r["distinguishable"] for r in E3["rows"][1:])


def test_e4_reports_discrimination_variance_and_stability_under_both_readings(built):
    _, js, _ = built
    E4 = js["E4"]
    names = [c["name"] for c in E4["contrasts"]]
    assert names[0].startswith("H3 ") and names[1].startswith("H3b") and len(names) == 3
    h3 = E4["contrasts"][0]
    assert h3["log_var_ratio"] is not None and 0 <= h3["bh_p_var"] <= 1 and h3["F"] is not None and h3["W"] is not None
    assert h3["verdict"] in ("supported", "partially supported", "not supported")
    stab = {s["arm"]: s for s in E4["stability"]}
    assert set(stab) == {"permuted_x", "free_desc", "free_zero"}
    # the full model attends the same segments run after run, the permuted-target arm does not
    assert stab["permuted_x"]["cc_lower"] > 0 and stab["permuted_x"]["z_lower"] > 0
    assert E4["anchoring"]["supported"] is True and E4["drift"]["mean_per_item"] == [0.9] * 8
    assert set(E4["tv"]) == {"full_x", "permuted_x", "free_desc", "free_zero"} and E4["tv"]["full_x"] < E4["tv"]["permuted_x"]


def test_e5_reports_the_shortcut_controls_the_prompt_only_null_and_the_h4_matrix(built):
    _, js, _ = built
    E5 = js["E5"]
    rows = {r["arm"]: r for r in E5["rows"]}
    assert list(rows) == ["full", "permuted", "prompt_only", "permuted_quint"]
    assert rows["permuted"]["diff_lower"] > 0
    assert rows["prompt_only"]["null_max"] is not None and isinstance(rows["prompt_only"]["passes"], bool)
    assert "passes" in rows["full"] and "passes" not in rows["permuted"]
    assert E5["h4"] in ("supported", "partially supported") and set(E5["exploratory"]) == {"permuted_quint", "full_nohist"}
    assert set(E5["region"]) == {"complete", "reduced"} and E5["region"]["complete"]["feff"] > 0


def test_h4_is_an_estimate_when_h1_was_not_supported(built, tmp_path):
    out, _, items = built
    (out / "E1_v07.json").write_text(json.dumps({"h1": {"verdict": "not supported"}}))
    js = rep.report({"items": items}, out, reps=REPS, seeds=SEEDS, B=300)
    assert js["E5"]["h4"].startswith("inconclusive (read as an estimate")


def test_the_markdown_and_json_files_are_written_and_missing_arms_are_reported_as_missing(built, tmp_path):
    out, _, items = built
    md = (out / "E2E5_v07.md").read_text()
    for needle in ("## E2", "## E3", "## E4", "## E5", "Class-weighted training item mean", "(c) prior only",
                   "Anchoring", "Passes Ec. 91"):
        assert needle in md
    assert json.loads((out / "E2E5_v07.json").read_text())["E2"]["h5"]["verdict"]
    items = _items()
    out2 = _build(tmp_path, items, ("full", "abmil8"))
    js = rep.report({"items": items}, out2, reps=REPS, seeds=SEEDS, B=100)
    assert js["E3"]["missing"] and js["E4"]["contrasts"][0]["missing"] == ["binary_sup"] and js["E5"]["rows"][0]["arm"] == "full"
    assert not js["E2"].get("h5", {}).get("verdict")
