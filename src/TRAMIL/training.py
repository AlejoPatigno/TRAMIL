"""Objective and optimisation: weighted BCE on subjects (Ec. 38), AdamW lr 1e-3, wd 1e-4, batches of 8 bags."""
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from . import standardize

__provenance__ = {
    "revision": "research-concept-residual-semantic-prior-bipartite-mil-r06.md",
    "sections": ["12"],
    "equations": [38],
    "invariants": [
        "class_weight_is_negatives_over_positives",
        "weighted_bce_matches_the_formula_of_ec_38",
        "training_uses_adamw_lr_1e3_wd_1e4_batch_8",
        "training_learns_a_separable_toy_problem_and_is_deterministic_per_seed",
        "training_device_is_cuda_when_available_else_cpu",
    ],
}

LR, WEIGHT_DECAY, BATCH = 1e-3, 1e-4, 8


def device():
    """cuda when available, else cpu; probabilities returned by predict stay numpy on the cpu."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def class_weight(y):
    """omega_+ = M_0 / M_1, from the training set only."""
    y = np.asarray(y)
    return float((y == 0).sum() / (y == 1).sum())


def weighted_bce(eta, y, w):
    """-(1/M) sum [w y log p + (1 - y) log(1 - p)] through the stable logits form."""
    return torch.nn.functional.binary_cross_entropy_with_logits(eta, y, pos_weight=torch.tensor(w, device=eta.device))


def make_optimizer(m):
    return torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)


def prepare(items, mu, s):
    """items: (X raw, R, y) per bag -> tensors with the scaler of the run applied to X (R is not standardised)."""
    return [(torch.from_numpy(standardize.apply(X, mu, s)), torch.from_numpy(R), float(y)) for X, R, y in items]


def predict(m, items):
    m.eval()
    dev = next(m.parameters()).device
    with torch.no_grad():
        return np.array([torch.sigmoid(m(X.to(dev), R.to(dev))).item() for X, R, _ in items])


def run_epoch(m, opt, items, w, gen, wrap=iter):
    """One pass over the shuffled bags in mini-batches of BATCH; `wrap` sees the batch iterator (timing probe)."""
    m.train()
    dev = next(m.parameters()).device
    order = torch.randperm(len(items), generator=gen).tolist()
    batches = ([items[j] for j in order[i : i + BATCH]] for i in range(0, len(order), BATCH))
    for batch in wrap(batches):
        eta = torch.stack([m(X.to(dev), R.to(dev)) for X, R, _ in batch])
        loss = weighted_bce(eta, torch.tensor([y for *_, y in batch], device=dev), w)
        opt.zero_grad()
        loss.backward()
        opt.step()


def train(make_model, items, epochs, seed, eval_items=None):
    """Re-initialise with seed and train; returns (model, per-epoch AUROC on eval_items or [])."""
    torch.manual_seed(seed)
    m = make_model().to(device())
    opt = make_optimizer(m)
    w = class_weight([y for *_, y in items])
    gen = torch.Generator().manual_seed(seed)
    curve = []
    for _ in range(epochs):
        run_epoch(m, opt, items, w, gen)
        if eval_items is not None:
            curve.append(roc_auc_score([y for *_, y in eval_items], predict(m, eval_items)))
    return m, curve
