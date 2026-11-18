import numpy as np
import torch

from TRAMIL import timing


def _toy(n=12, seed=0):
    rng = np.random.default_rng(seed)
    return [(torch.from_numpy(rng.normal(size=(5, 88)).astype(np.float32)),
             torch.from_numpy(rng.random((5, 8)).astype(np.float32)), float(i % 2)) for i in range(n)]


def test_timing_epoch_runs_one_epoch_through_wrap(monkeypatch):
    items = _toy()
    monkeypatch.setattr(timing, "_items", lambda: items)
    seen = []

    def wrap(it):
        for b in it:
            seen.append(len(b))
            yield b

    for arm in timing.ARMS:
        seen.clear()
        getattr(timing, f"epoch_{arm}")({"device": "cpu"}, wrap)
        assert sum(seen) == len(items), arm  # every training bag visited exactly once
