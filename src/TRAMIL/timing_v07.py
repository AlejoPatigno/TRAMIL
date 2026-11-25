"""Timing probe for the v07 E1 runner (T7): per-fit wall time of every arm and concurrent throughput with N workers.

One fit = one inner-fold training run of max_epochs (search/real unit work). Neural arms are timed for `epochs` epochs
and extrapolated linearly to `budget` epochs (extrapolated=True, epochsMeasured, epochBudget recorded); pass
epochs == budget for an unextrapolated measurement. The SVM fit (inner grid search) is timed whole. Concurrent rows:
`processes` = N workers each running fits at once through dispatch.run; secondsPerUnit = wall / fits (so the
fits per hour of the whole device are 3600 / secondsPerUnit). Rows are appended to Results/timing.json (schema 2).

CLI: python -m TRAMIL.timing_v07 [--out TRAMIL/Results/timing.json] [--epochs 2] [--budget 80] [--workers 1,2,4,8]"""
import argparse
import json
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from . import dispatch, e1v07_run, item_training

__provenance__ = {
    "revision": "experiments-experimental-plan-for-the-residual-semantic-prior-bipartite-v07.md",
    "sections": ["E1"],
    "experiments": ["E1"],
    "invariants": ["timing_probe_writes_per_fit_and_concurrent_measurements_in_the_existing_schema"],
}

ARMS = (*e1v07_run.E1_NEURAL, "prompt_only", "svm")


def _spec(arm, epochs, k, i):
    if arm == "svm":
        return {"phase": "real", "rep": 0, "fold": i % 5, "arm": "svm", "model": "svm", "seed": 0}
    return {"phase": "search", "rep": 0, "fold": i % 5, "inner": 0, "arm": arm, "model": arm, "k": k, "p": 0.0,
            "seed": 11, "max_epochs": epochs}


def _wall(arm, data, n, epochs, k):
    """Seconds per fit with n workers each running one fit at once (the n fits start together)."""
    specs = [_spec(arm, epochs, k, i) for i in range(n)]
    dispatch.run(e1v07_run.run_unit, specs[:1], data, 1)  # warm-up: prepares fold tensors, compiles kernels
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    dispatch.run(e1v07_run.run_unit, specs, data, n)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n


def probe(data, path, arms=ARMS, epochs=2, budget=item_training.MAX_EPOCHS, workers_grid=(1, 2, 4, 8),
          k=64, host=None):
    dev = item_training.device()
    host = host or os.environ.get("FORGE_HOST") or platform.node() or "local"
    p = Path(path)
    js = json.loads(p.read_text()) if p.exists() else {"schema": 2, "hosts": {}, "measurements": []}
    js.setdefault("hosts", {})[host] = {"cpuCount": os.cpu_count() or 1, "ramGB": None,
                                        "mps": bool(getattr(torch.backends, "mps", None)
                                                    and torch.backends.mps.is_available()),
                                        "cuda": torch.cuda.is_available(),
                                        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    rows = []
    for arm in arms:
        scale = 1.0 if arm == "svm" else budget / epochs
        for n in workers_grid:
            s = _wall(arm, data, n, epochs, k) * scale
            rows.append({"measuredAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "host": host,
                         "candidate": f"{host}-{dev.type}-{n}", "device": dev.type, "arm": arm, "unit": "fit",
                         "secondsPerUnit": round(s, 4), "unitsMeasured": n, "processes": n,
                         "fitsPerHour": round(3600 / s, 2), "extrapolated": arm != "svm" and epochs != budget,
                         "epochsMeasured": None if arm == "svm" else epochs,
                         "epochBudget": None if arm == "svm" else budget, "k": None if arm == "svm" else k})
    js["measurements"] = js.get("measurements", []) + rows
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(p) + ".tmp")
    tmp.write_text(json.dumps(js, indent=2))
    os.replace(tmp, p)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m TRAMIL.timing_v07")
    ap.add_argument("--out", default=str(Path(os.environ.get("FORGE_CLONE_ROOT", ".")) / "TRAMIL" / "Results" /
                                         "timing.json"))
    ap.add_argument("--input")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--budget", type=int, default=item_training.MAX_EPOCHS)
    ap.add_argument("--workers", default="1,2,4,8")
    ap.add_argument("--arms", default=",".join(ARMS))
    a = ap.parse_args(argv)
    data, _ = e1v07_run.load_data(a.input)
    rows = probe(data, a.out, a.arms.split(","), a.epochs, a.budget, tuple(int(n) for n in a.workers.split(",")))
    print(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
