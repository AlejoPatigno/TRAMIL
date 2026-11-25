"""Unit dispatcher (T6): run independent units over N worker processes that share one device.

Each worker receives the dataset once (initializer), keeps its own cache of prepared fold tensors already moved to the
device (training.device(), so CUDA_VISIBLE_DEVICES decides the GPU), and runs fn(spec, data) per unit. Every unit
seeds itself (item_training.train), so results do not depend on N or on the order units land on workers; floating
sums may differ in the last bits with the thread count. Results come back in spec order. One worker per GPU is the
default: with batched training several processes on one T4 were 7-12x slower per fit (Kaggle timing, pin 15f438b).
An optional warm-up spec runs once in each worker before its units (timing: excludes start-up and data load)."""
import os

import torch

__provenance__ = {
    "revision": "experiments-experimental-plan-for-the-residual-semantic-prior-bipartite-v07.md",
    "sections": ["E1"],
    "experiments": ["E1"],
    "invariants": [
        "results_do_not_depend_on_the_number_of_workers",
        "a_warm_up_unit_runs_in_each_worker_before_the_timed_units_and_one_worker_is_the_default",
    ],
}

_DATA = None


def _init(data, threads, fn=None, warmup=None):
    global _DATA
    torch.set_num_threads(threads)
    _DATA = {**data, "cache": {}}
    if warmup is not None:
        fn(warmup, _DATA)


def _call(args):
    fn, spec = args
    return fn(spec, _DATA)


def run(fn, specs, data, workers=1, on_result=None, warmup=None):
    """Map fn over specs; on_result(spec, result) is called in the parent as each result arrives, in spec order."""
    specs = list(specs)
    out = []
    if workers <= 1:
        _init(data, torch.get_num_threads(), fn, warmup)
        results = (fn(s, _DATA) for s in specs)
        pool = None
    else:
        threads = max(1, (os.cpu_count() or 1) // workers)
        pool = torch.multiprocessing.get_context("spawn").Pool(workers, initializer=_init, initargs=(data, threads, fn, warmup))
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
