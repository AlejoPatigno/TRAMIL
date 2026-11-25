"""Item supervision (r22 Ec. 39, 75-77): class weights, class-weighted item MSE, decision, epoch selection, loop."""
import numpy as np
import pytest
import torch
from torch import nn

from TRAMIL import comparators, item_training as it


def _bag(s, N=5, d=4, seed=0):
    g = np.random.default_rng(seed)
    s = torch.tensor(s, dtype=torch.float32)
    return {"Y": torch.from_numpy(g.standard_normal((N, d)).astype(np.float32)), "s": s,
            "y": int(s.sum() >= 10), "subject": seed}


def test_class_is_item_sum_at_least_10_and_weight_is_negatives_over_positives():
    S = np.array([[1] * 8, [2] * 8, [0] * 8, [1, 1, 1, 1, 1, 1, 1, 3]])  # totals 8, 16, 0, 10
    assert it.binary(S).tolist() == [0, 1, 0, 1]
    assert it.omega_plus(it.binary(S)) == 1.0
    assert it.omega_plus([0, 0, 0, 1]) == 3.0


def test_weighted_item_mse_matches_the_item_term_of_ec_77():
    s_hat = torch.tensor([[1.0] * 8, [0.0] * 8])
    s = torch.tensor([[3.0] * 8, [1.0] * 8])  # bag 0 positive (24), bag 1 negative (8)
    y = [1, 0]
    # (1/M) sum_b w_b (1/8) sum_j (s_hat - s)^2 = (2.5 * 4 + 1 * 1) / 2
    assert torch.isclose(it.weighted_item_mse(s_hat, s, y, 2.5), torch.tensor(5.5))


def test_decision_and_score_read_the_predicted_total():
    s_hat = np.array([[1.25] * 8, [1.0] * 8, [2.0, -1, 0, 0, 0, 0, 0, 0]])
    assert it.score(s_hat).tolist() == [10.0, 8.0, 1.0]
    assert it.decide(s_hat).tolist() == [1, 0, 0]


def test_epoch_selection_takes_the_lowest_mean_inner_weighted_mse_earliest_on_ties():
    curves = np.array([[0.9, 0.5, 0.5, 0.6], [0.9, 0.3, 0.3, 0.2], [0.9, 0.4, 0.4, 0.7]])
    assert it.select_epoch(curves) == 2  # means 0.9, 0.4, 0.4, 0.5; epochs 2 and 3 tie -> 2
    assert it.select_epoch(np.array([[0.3, 0.2, 0.1]])) == 3


class _Mean(nn.Module):
    """Toy item model: mean of the instance rows through eight linear heads; optional constant penalty."""

    def __init__(self, pen=0.0):
        super().__init__()
        self.fc = nn.Linear(4, 8)
        self.pen = pen

    def forward(self, b):
        return self.fc(b["Y"].mean(0))

    def fit_terms(self, b):
        s_hat = self(b)
        return s_hat, self.pen * (self.fc.weight ** 2).sum()


def _toy(n, seed):
    g = np.random.default_rng(seed)
    out = []
    for i in range(n):
        lvl = 3 * (i % 2)
        b = _bag([lvl] * 8, seed=seed * 1000 + i)
        b["Y"] = b["Y"] + float(lvl)
        out.append(b)
    return out


def test_training_fits_items_records_weighted_val_mse_and_is_deterministic_per_seed():
    tr, va = _toy(16, 0), _toy(8, 1)
    m1, c1 = it.train(_Mean, tr, epochs=40, seed=11, eval_bags=va)
    assert len(c1) == 40 and c1[-1] < c1[0]
    _, c2 = it.train(_Mean, tr, epochs=40, seed=11, eval_bags=va)
    _, c3 = it.train(_Mean, tr, epochs=40, seed=23, eval_bags=va)
    assert np.allclose(c1, c2) and not np.allclose(c1, c3)
    P = it.predict(m1, va)
    assert P.shape == (8, 8) and np.isfinite(P).all()


def test_eval_curve_is_the_class_weighted_item_mse_without_the_penalty():
    tr, va = _toy(8, 0), _toy(4, 1)
    m, c = it.train(lambda: _Mean(pen=100.0), tr, epochs=1, seed=11, eval_bags=va)
    ow = it.omega_plus([b["y"] for b in tr])
    P = torch.from_numpy(it.predict(m, va))
    want = it.weighted_item_mse(P, torch.stack([b["s"] for b in va]), [b["y"] for b in va], ow).item()
    assert c[0] == pytest.approx(want, rel=1e-6)


def test_model_penalty_enters_the_training_loss():
    tr = _toy(8, 0)
    m0, _ = it.train(lambda: _Mean(pen=0.0), tr, epochs=5, seed=11)
    m1, _ = it.train(lambda: _Mean(pen=10.0), tr, epochs=5, seed=11)
    assert (m1.fc.weight ** 2).sum() < (m0.fc.weight ** 2).sum()


def test_optimizer_is_the_e1_protocol_adamw():
    assert (it.LR, it.WEIGHT_DECAY, it.BATCH, it.MAX_EPOCHS) == (1e-3, 1e-4, 8, 80)


def test_svm_accepts_the_class_derived_from_item_sums():
    g = np.random.default_rng(2)
    S = np.array([[2] * 8 if i % 2 else [0] * 8 for i in range(24)])
    y = it.binary(S)
    X = [(g.standard_normal((5, 6)) + 1.5 * yy).astype(np.float32) for yy in y]
    pipe = comparators.fit_svm(X, y, inner_seed=7301)
    assert set(pipe.classes_.tolist()) == {0, 1}
