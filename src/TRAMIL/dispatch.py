"""Unit dispatcher (T6): run independent units over N worker processes that share one device.

Each worker receives the dataset once (initializer), keeps its own cache of prepared fold tensors already moved to the
device (training.device(), so CUDA_VISIBLE_DEVICES decides the GPU), and runs fn(spec, data) per unit. Every unit
seeds itself (item_training.train), so results do not depend on N or on the order units land on workers; floating
sums may differ in the last bits with the thread count. Results come back in spec order."""
import json
import os
from pathlib import Path

import torch

__provenance__ = {
    "revision": "experiments-experimental-plan-for-the-residual-semantic-prior-bipartite-v07.md",
    "sections": ["E1"],
    "experiments": ["E1"],
    "invariants": [
        "results_do_not_depend_on_the_number_of_workers",
        "default_workers_come_from_the_best_measured_throughput",
    ],
}

_DATA = None


def _init(data, threads):
    global _DATA
    torch.set_num_threads(threads)
    _DATA = {**data, "cache": {}}


def _call(args):
    fn, spec = args
    return fn(spec, _DATA)


def run(fn, specs, data, workers=1, on_result=None):
    """Map fn over specs; on_result(spec, result) is called in the parent as each result arrives, in spec order."""
    specs = list(specs)
    out = []
    if workers <= 1:
        _init(data, torch.get_num_threads())
        results = (fn(s, _DATA) for s in specs)
        pool = None
    else:
        threads = max(1, (os.cpu_count() or 1) // workers)
        pool = torch.multiprocessing.get_context("spawn").Pool(workers, initializer=_init, initargs=(data, threads))
        results = pool.imap(_call, [(fn, s) for s in specs])
    try:
        for s, r in zip(specs, results):
            if on_result:
                on_result(s, r)
            out.append(r)
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    return out


def default_workers(timing_path, device_kind):
    """Process count with the best measured fits per second (processes / secondsPerUnit) on this device kind."""
    p = Path(timing_path)
    if not p.exists():
        return 1
    rows = [m for m in json.loads(p.read_text()).get("measurements", [])
            if m.get("unit") == "fit" and m.get("device") == device_kind and m.get("secondsPerUnit")]
    if not rows:
        return 1
    best = max(rows, key=lambda m: (m["processes"] / m["secondsPerUnit"], -m["processes"]))
    return int(best["processes"])
