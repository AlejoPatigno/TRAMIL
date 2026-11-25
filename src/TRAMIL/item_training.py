"""Item supervision (r22 Section 9, 12, 13): class weights, class-weighted item MSE (Ec. 77, item term), decision on the
predicted total (Ec. 76), epoch selection on the inner-fold class-weighted item MSE (Ec. 39) and the training loop.

A bag is a dict of tensors with at least "s" (8,) item scores and "y" = 1[sum s >= 10]; the model maps a bag to the
eight item scores. A model with a training-only penalty exposes fit_terms(bag) -> (s_hat, penalty), the penalty already
multiplied by its weight mu_R; models without one are called as model(bag)."""
import numpy as np
import torch

from .training import device

__provenance__ = {
    "revision": "research-concept-residual-semantic-prior-bipartite-mil-r22.md",
    "sections": ["9", "12", "13"],
    "equations": [39, 75, 76, 77],
    "experiments": ["E1"],
    "invariants": [
        "class_is_item_sum_at_least_10_and_weight_is_negatives_over_positives",
        "weighted_item_mse_matches_the_item_term_of_ec_77",
        "decision_and_score_read_the_predicted_total",
        "epoch_selection_takes_the_lowest_mean_inner_weighted_mse_earliest_on_ties",
        "training_fits_items_records_weighted_val_mse_and_is_deterministic_per_seed",
        "eval_curve_is_the_class_weighted_item_mse_without_the_penalty",
        "model_penalty_enters_the_training_loss",
        "optimizer_is_the_e1_protocol_adamw",
        "svm_accepts_the_class_derived_from_item_sums",
        "label_permutation_moves_the_eight_item_scores_as_a_unit",
    ],
}

LR, WEIGHT_DECAY, BATCH, MAX_EPOCHS = 1e-3, 1e-4, 8, 80  # r22 Section 12; v07 validation-scheme table
THRESHOLD = 10


def binary(S):
    """y_b = 1[sum_j s_bj >= 10] for an (M, 8) array of item scores."""
    return (np.asarray(S).sum(-1) >= THRESHOLD).astype(int)


def omega_plus(y):
    """omega_+ = M_0 / M_1 of the given (training) set."""
    y = np.asarray(y)
    return float((y == 0).sum() / (y == 1).sum())


def permute_items(S, seed):
    """Label permutation across subjects (v07 Ec. 91 nulls): the eight item scores of a subject move as one row, so the
    total, the class and omega_+ are the donor's (recompute y with binary)."""
    S = np.asarray(S)
    return S[np.random.default_rng(seed).permutation(len(S))]


def weighted_item_mse(s_hat, s, y, ow):
    """(1/M) sum_b omega_b (1/8) sum_j (s_hat_bj - s_bj)^2, omega_b = ow for positives, 1 otherwise (Ec. 77, 39)."""
    w = torch.tensor([ow if int(v) else 1.0 for v in y], dtype=s_hat.dtype, device=s_hat.device)
    return (w * (s_hat - s).pow(2).mean(-1)).mean()


def score(S_hat):
    """score_b = sum_j s_hat_bj (Ec. 76), the AUROC input."""
    return np.asarray(S_hat).sum(-1)


def decide(S_hat):
    """y_hat_b = 1[score_b >= 10] (Ec. 76)."""
    return (score(S_hat) >= THRESHOLD).astype(int)


def select_epoch(curves):
    """e* = argmin_e mean over inner folds of the class-weighted item MSE (Ec. 39), earliest on ties; 1-based."""
    return int(np.argmin(np.asarray(curves).mean(0))) + 1  # argmin returns the first minimum


def make_optimizer(m):
    return torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)


def _to(b, dev):
    return {k: v.to(dev) if torch.is_tensor(v) else v for k, v in b.items()}


def predict(m, bags):
    """(M, 8) numpy item predictions in eval mode (dropout off)."""
    m.eval()
    dev = next(m.parameters()).device
    with torch.no_grad():
        return np.stack([m(_to(b, dev)).double().cpu().numpy() for b in bags])


def val_mse(m, bags, ow):
    """Ec. 39 inner-fold objective: class-weighted item MSE without the penalty."""
    P = torch.from_numpy(predict(m, bags))
    S = torch.stack([b["s"].cpu() for b in bags]).double()  # bags may be preloaded on the GPU; P is on the CPU
    return float(weighted_item_mse(P, S, [b["y"] for b in bags], ow))


def run_epoch(m, opt, bags, ow, gen, wrap=iter):
    """One pass over the shuffled bags in mini-batches of BATCH: mean over the batch of the weighted item error plus
    the model's own penalty (Ec. 77); `wrap` sees the batch iterator (timing probe)."""
    m.train()
    dev = next(m.parameters()).device
    order = torch.randperm(len(bags), generator=gen).tolist()
    terms = getattr(m, "fit_terms", None)
    for idx in wrap(order[i : i + BATCH] for i in range(0, len(order), BATCH)):
        loss = 0.0
        for j in idx:
            b = _to(bags[j], dev)
            s_hat, pen = terms(b) if terms else (m(b), 0.0)
            loss = loss + (ow if int(b["y"]) else 1.0) * (s_hat - b["s"]).pow(2).mean() + pen
        opt.zero_grad()
        (loss / len(idx)).backward()
        opt.step()


def train(make_model, bags, epochs, seed, eval_bags=None):
    """Re-initialise with seed and train; returns (model, per-epoch Ec. 39 objective on eval_bags or []).
    omega_+ comes from the training bags and weights the evaluation bags too (Ec. 39)."""
    torch.manual_seed(seed)
    m = make_model().to(device())
    opt = make_optimizer(m)
    ow = omega_plus([b["y"] for b in bags])
    gen = torch.Generator().manual_seed(seed)
    curve = []
    for _ in range(epochs):
        run_epoch(m, opt, bags, ow, gen)
        if eval_bags is not None:
            curve.append(val_mse(m, eval_bags, ow))
    return m, curve
