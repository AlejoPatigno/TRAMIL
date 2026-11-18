"""Timing-probe entry points: one training epoch per E1 arm on the full-scale training bags (Results/timing.json)."""
import os
from pathlib import Path

import numpy as np
import torch

from . import exp_e1, standardize, training

__provenance__ = {
    "revision": "experiments-experimental-plan-for-the-residual-semantic-prior-bipartite-v04.md",
    "sections": ["E1"],
    "experiments": ["E1"],
    "invariants": ["timing_epoch_runs_one_epoch_through_wrap"],
}

ARMS = ("residual", "learned_only", "abmil", "gated_abmil")
SEED = 11
_cache = None


def _items():
    """Prepared (X, R, y) for one outer-fold training set: 4/5 of the 141 training bags, read from the feature cache."""
    global _cache
    if _cache is None:
        items, _ = exp_e1.load_items(os.environ["TRAMIL_LABELS"], os.environ["TRAMIL_EDAIC"],
                                     Path(os.environ["TRAMIL_FEATURES"]))
        keep = np.random.default_rng(0).permutation(len(items))[: len(items) * 4 // 5]
        fit = [items[i] for i in keep]
        mu, s = standardize.fit([it["X"] for it in fit])
        _cache = training.prepare([(it["X"], it["R"], it["y"]) for it in fit], mu, s)
    return _cache


def _epoch(arm, settings, wrap):
    items = _items()
    torch.manual_seed(SEED)
    m = exp_e1.ARMS[arm]()
    opt = training.make_optimizer(m)
    w = training.class_weight([y for *_, y in items])
    training.run_epoch(m, opt, items, w, torch.Generator().manual_seed(SEED), wrap)


def epoch_residual(settings, wrap):
    _epoch("residual", settings, wrap)


def epoch_learned_only(settings, wrap):
    _epoch("learned_only", settings, wrap)


def epoch_abmil(settings, wrap):
    _epoch("abmil", settings, wrap)


def epoch_gated_abmil(settings, wrap):
    _epoch("gated_abmil", settings, wrap)


if os.environ.get("TRAMIL_FEATURES"):  # load before the probe's clock starts: data loading is setup, not an epoch
    _items()
