"""Remote entry point: the full E1 (4 neural arms + SVM), one resumable result file per (repetition, outer fold)."""
import json
import os
import platform
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from . import comparators, exp_e1, features, training, validation

__provenance__ = {
    "revision": "experiments-experimental-plan-for-the-residual-semantic-prior-bipartite-v04.md",
    "sections": ["E1"],
    "experiments": ["E1"],
    "invariants": [
        "each_unit_writes_a_partition_file_with_every_arm",
        "existing_partitions_are_skipped_on_resume",
        "partitions_argument_restricts_units",
        "merge_is_partial_until_all_units_exist",
        "units_never_leak_test_subjects_into_training",
        "bags_build_from_csv_and_features_without_wav_files",
        "stamp_only_under_the_runner_with_commit_device_and_epoch_timing",
        "mounted_datasets_are_found_through_the_forge_map",
        "wavlm_variant_reads_w_features_and_writes_to_e1_wavlm",
    ],
}

FEATURES_REF = "alejopatio/tramil-edaic-features"
WAVLM_REF = "alejopatio/tramil-edaic-wavlm"
INSTANCES_REF = "alejopatio/tramil-edaic-instances"
ABSENT_ARMS = ["symptom_guided"]
NEURAL = list(exp_e1.ARMS)
INNER_FOLDS = 3  # validation.inner_splits default; epochs trained per seed = INNER_FOLDS * max_epochs + e*
_MAX_DEPTH = 6


def _find(roots, depth=_MAX_DEPTH, acoustic="egemaps"):
    """Labels, instances CSV and the flat features folder, searched below each root (never inside wav folders).
    acoustic="wavlm": features = folder with _W.npy; text = folder with the mpnet _E.npy."""
    suffix = "_W.npy" if acoustic == "wavlm" else "_X.npy"
    found = {}
    for root in map(Path, roots):
        for here, dirs, files in os.walk(root):
            for f, key in (("detailed_lables.csv", "labels"), ("instances_clean.csv", "instances")):
                if f in files:
                    found.setdefault(key, Path(here) / f)
            if any(f.endswith(suffix) for f in files):
                found.setdefault("features", Path(here))
            if acoustic == "wavlm" and any(f.endswith("_E.npy") for f in files):
                found.setdefault("text", Path(here))
            dirs[:] = [d for d in dirs if not d.startswith("instances_audio")]
            if len(Path(here).relative_to(root).parts) >= depth:
                dirs[:] = []
    missing = {"labels", "instances", "features", *(["text"] if acoustic == "wavlm" else [])} - found.keys()
    if missing:
        raise FileNotFoundError(f"{sorted(missing)} not found below {[str(r) for r in roots]}")
    return found


def _default_roots(acoustic="egemaps"):
    mounted = json.loads(os.environ.get("FORGE_INPUT_DATASETS") or "{}")
    refs = (FEATURES_REF, INSTANCES_REF, *([WAVLM_REF] if acoustic == "wavlm" else []))
    roots = [mounted[r] for r in refs if r in mounted]
    return roots or ["TRAMIL/Data"]


def _default_out_dir(acoustic="egemaps"):
    name = "E1_wavlm" if acoustic == "wavlm" else "E1"
    return Path(os.environ.get("FORGE_CLONE_ROOT", ".")) / "TRAMIL" / "Results" / name


def _unit_id(r, k):
    return f"r{r}_f{k}"


def _units(repetitions, n_folds, partitions):
    wanted = None
    if partitions is not None:
        wanted = {p if isinstance(p, str) else _unit_id(*p) for p in partitions}
    return [(r, k) for r in range(repetitions) for k in range(n_folds) if wanted is None or _unit_id(r, k) in wanted]


def _write_json(path, obj):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=float))
    os.replace(tmp, path)


def _device():
    dev = training.device()
    if dev.type == "cuda":
        import torch

        return {"kind": "cuda", "name": torch.cuda.get_device_name(0)}
    return {"kind": "cpu", "name": platform.processor() or "cpu"}


def _unit(items, r, k, n_folds, seeds, max_epochs):
    """One (repetition, outer fold): every arm trained on the outer-training subjects, scored on the test fold."""
    y = np.array([it["y"] for it in items])
    subj = [it["subject"] for it in items]
    test_idx = validation.outer_splits(y, r, n_folds)[k]
    train_idx = np.setdiff1d(np.arange(len(items)), test_idx)
    arms = {}
    for name in [*NEURAL, "svm"]:
        t0 = time.time()
        if name == "svm":
            pipe = comparators.fit_svm([items[i]["X"] for i in train_idx], y[train_idx], validation.INNER_SEED + k)
            scores = pipe.decision_function(np.stack([comparators.bag_stats(items[i]["X"]) for i in test_idx]))
            epochs = []
        else:
            scores, epochs = exp_e1._neural(exp_e1.ARMS[name], items, train_idx, test_idx, k, seeds, max_epochs)
        arms[name] = {"fold_auroc": float(roc_auc_score(y[test_idx], scores)),
                      "oof": {str(subj[i]): float(s) for i, s in zip(test_idx, scores)},
                      "epochs": [int(e) for e in epochs], "seconds": time.time() - t0,
                      "epochs_trained": int(sum(INNER_FOLDS * max_epochs + e for e in epochs))}
    return {"unit": _unit_id(r, k), "repetition": r, "fold": k, "device": _device(), "seeds": list(seeds),
            "max_epochs": max_epochs, "train": [subj[i] for i in train_idx], "test": [subj[i] for i in test_idx],
            "n_positive": int(y.sum()), "arms": arms}


def merge(out_dir, repetitions=3, n_folds=5):
    """E1 result from the partition files: FULL only when every (repetition, fold) exists, else PARTIAL."""
    out = Path(out_dir)
    have = {}
    for r in range(repetitions):
        for k in range(n_folds):
            f = out / f"partition_{_unit_id(r, k)}.json"
            if f.exists():
                have[(r, k)] = json.loads(f.read_text())
    missing = [_unit_id(r, k) for r in range(repetitions) for k in range(n_folds) if (r, k) not in have]
    arms = {n: {"fold_auroc": [], "oof": {}, "epochs": [], "seconds": 0.0} for n in [*NEURAL, "svm"]}
    for (r, k), p in sorted(have.items()):
        for n, a in p["arms"].items():
            arms[n]["fold_auroc"].append(a["fold_auroc"])
            arms[n]["oof"].setdefault(f"r{r}", {}).update(a["oof"])
            arms[n]["epochs"] += a["epochs"]
            arms[n]["seconds"] += a["seconds"]
    first = next(iter(have.values()), None)
    res = {"stage": "PARTIAL" if missing else "FULL", "units_done": len(have), "units_total": repetitions * n_folds,
           "missing": missing, "absent_arms": ABSENT_ARMS, "arms": arms,
           "comparisons": exp_e1._comparisons(arms) if have else {},
           "config": {"repetitions": repetitions, "folds": n_folds,
                      "seeds": first["seeds"] if first else None, "max_epochs": first["max_epochs"] if first else None}}
    out.mkdir(parents=True, exist_ok=True)
    _write_json(out / "E1_full.json", res)
    rows = ["| Arm | Mean AUROC | Fold AUROC | Seconds |", "| --- | --- | --- | --- |"]
    for n, a in arms.items():
        if a["fold_auroc"]:
            rows.append("| {} | {:.3f} | {} | {:.0f} |".format(
                n, np.mean(a["fold_auroc"]), ", ".join(f"{v:.3f}" for v in a["fold_auroc"]), a["seconds"]))
    md = [f"# E1 -- {res['stage']} ({len(have)}/{repetitions * n_folds} units)", ""]
    if missing:
        md += [f"Missing units: {', '.join(missing)}", ""]
    md += [f"Absent arms (deferred): {', '.join(ABSENT_ARMS)}", ""] + rows + ["", "Paired comparisons against residual:", ""]
    md += [f"- {n}: mean diff {c['mean_diff']:.3f}, TOST 90% {c['tost_90']}, p {c['p']:.3f} (R={c['R']})"
           for n, c in res["comparisons"].items()]
    (out / "E1_full.md").write_text("\n".join(md) + "\n")
    return res


def _stamp(summary):
    """shard.json beside the results, only under the runner (FORGE_RUN_MODE): what the smoke verdict reads."""
    mode = os.environ.get("FORGE_RUN_MODE")
    if not mode:
        return None
    commit = None
    try:
        r = subprocess.run(["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "HEAD"],
                           capture_output=True, text=True)
        commit = r.stdout.strip() if r.returncode == 0 and r.stdout.strip() else None
    except OSError:
        pass
    commit = commit or os.environ.get("FORGE_CLONE_COMMIT")
    payload = {"mode": mode, "evidence": {"commit": commit},
               "environment": {"device": _device()}, "runs": summary}
    path = Path.cwd() / "shard.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def run(input_root=None, out_dir=None, partitions=None, seeds=(11, 23, 37), max_epochs=80, repetitions=3,
        n_folds=5, limit_bags=None, resume_from=None, acoustic=None):
    """Run the pending (repetition, fold) units of E1, one partition file each; resumable, merges at the end.
    acoustic: "egemaps" (default) or "wavlm"; falls back to $TRAMIL_ACOUSTIC. WavLM results go to Results/E1_wavlm."""
    acoustic = acoustic or features.acoustic_from_env()
    found = _find([input_root] if input_root else _default_roots(acoustic), acoustic=acoustic)
    out = Path(out_dir) if out_dir else _default_out_dir(acoustic)
    out.mkdir(parents=True, exist_ok=True)
    skip_refs = {FEATURES_REF, INSTANCES_REF, WAVLM_REF}
    mounted = json.loads(os.environ.get("FORGE_INPUT_DATASETS") or "{}")
    for src in [resume_from] if resume_from else [v for ref, v in mounted.items() if ref not in skip_refs]:
        for f in Path(src).rglob("partition_r*_f*.json"):
            if not (out / f.name).exists():
                shutil.copy(f, out / f.name)
    units = _units(repetitions, n_folds, partitions)
    pending = [(r, k) for r, k in units if not (out / f"partition_{_unit_id(r, k)}.json").exists()]
    items = None
    ran = []
    secs, epochs = dict.fromkeys(NEURAL, 0.0), dict.fromkeys(NEURAL, 0)
    for r, k in pending:
        if items is None:  # features are read only when a unit actually has to run
            items, info = exp_e1.load_items(found["labels"], found["instances"].parent, found["features"], limit_bags,
                                         acoustic=acoustic, text_dir=found.get("text"))
        p = _unit(items, r, k, n_folds, tuple(seeds), max_epochs)
        _write_json(out / f"partition_{_unit_id(r, k)}.json", p)
        ran.append(_unit_id(r, k))
        for n in NEURAL:
            secs[n] += p["arms"][n]["seconds"]
            epochs[n] += p["arms"][n]["epochs_trained"]
        done = sum((out / f"partition_{_unit_id(a, b)}.json").exists()
                   for a in range(repetitions) for b in range(n_folds))
        _write_json(out / "progress.json", {"done": done, "total": repetitions * n_folds, "last": _unit_id(r, k)})
    done = sum((out / f"partition_{_unit_id(a, b)}.json").exists() for a in range(repetitions) for b in range(n_folds))
    _write_json(out / "progress.json", {"done": done, "total": repetitions * n_folds})
    res = merge(out, repetitions, n_folds)
    summary = {"stage": res["stage"], "units_run": ran, "units_skipped": [_unit_id(*u) for u in units if _unit_id(*u) not in ran],
               "missing": res["missing"], "absent_arms": ABSENT_ARMS,
               "secondsPerEpoch": {n: secs[n] / epochs[n] for n in NEURAL if epochs[n]}}
    _stamp(summary)
    return summary
