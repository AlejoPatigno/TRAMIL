"""Item supervision (r22 Section 9, 12, 13): class weights, class-weighted item MSE (Ec. 77, item term), decision on the
predicted total (Ec. 76), epoch selection on the inner-fold class-weighted item MSE (Ec. 39) and the training loop.

A bag is a dict of tensors with at least "Y" (N, k), "s" (8,) item scores and "y" = 1[sum s >= 10]. Training and
prediction run on batches: collate() pads a set of bags once into tensors [M, Nmax, ...] with an instance "mask" (on
the device), take() slices a batch out of it by index (trimmed to its longest bag), and a model maps a batch to the
(B, 8) item scores in one forward. A model with a training-only penalty exposes fit_terms(batch) -> (s_hat, penalty),
the penalty per bag (B,) or a scalar, already multiplied by its weight mu_R; models without one are called as
model(batch). Padded rows never reach a valid output or gradient (masked softmaxes, centerings and sums)."""
import numpy as np
import torch
from torch.nn import functional as F

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
        "batched_forward_equals_the_per_bag_models",
        "batched_loss_and_gradients_equal_the_per_bag_sum",
        "extra_padding_never_changes_outputs_or_gradients",
        "a_batch_is_an_index_slice_trimmed_to_its_longest_bag",
        "training_on_batches_matches_the_per_bag_loop_without_dropout",
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


INSTANCE_KEYS = ("Y", "R", "Rc", "Rc_perm")  # (N, c) per bag
PAIR_KEYS = ("G",)  # (N, N) per bag
PREDICT_CHUNK = 256


def is_batch(b):
    return "mask" in b


def collate(bags, dev=None, n_max=None):
    """Pad bags into one batch: instance keys to [M, Nmax, c], G to [M, Nmax, Nmax], mask [M, Nmax] (True on valid
    rows), N [M]; other tensors are stacked, numbers become tensors. Built once per fold, on dev."""
    dev = dev if dev is not None else bags[0]["Y"].device
    N = [int(b["Y"].shape[0]) for b in bags]
    n = max(N) if n_max is None else n_max
    out = {"_n": N, "N": torch.tensor(N, device=dev),
           "mask": torch.arange(n, device=dev)[None, :] < torch.tensor(N, device=dev)[:, None]}
    for key, v0 in bags[0].items():
        if key in INSTANCE_KEYS:
            out[key] = torch.stack([F.pad(b[key].to(dev), (0, 0, 0, n - len(b[key]))) for b in bags])
        elif key in PAIR_KEYS:
            out[key] = torch.stack([F.pad(b[key].to(dev), (0, n - len(b[key]), 0, n - len(b[key]))) for b in bags])
        elif torch.is_tensor(v0):
            out[key] = torch.stack([b[key].to(dev) for b in bags])
        else:
            out[key] = torch.tensor([b[key] for b in bags], device=dev)
    return out


def take(T, idx):
    """The batch of bags idx: an index slice of the collated tensors, trimmed to the longest bag in it."""
    idx = list(idx)
    nb = max(T["_n"][i] for i in idx)
    ii = torch.as_tensor(idx, device=T["mask"].device)
    out = {"_n": [T["_n"][i] for i in idx]}
    for k, v in T.items():
        if k == "_n":
            continue
        if k in INSTANCE_KEYS or k == "mask":
            out[k] = v[ii, :nb]
        elif k in PAIR_KEYS:
            out[k] = v[ii, :nb, :nb]
        else:
            out[k] = v[ii]
    return out


def batch_of(b):
    """(batch, single): a lone bag (2-D Y) becomes a batch of one, so models keep a per-bag call for inspection."""
    return (b, False) if is_batch(b) else (collate([b]), True)


def _batched(bags, dev):
    return bags if is_batch(bags) else collate(list(bags), dev)


def predict(m, bags):
    """(M, 8) numpy item predictions in eval mode (dropout off), in a few large batches."""
    m.eval()
    T = _batched(bags, next(m.parameters()).device)
    M = len(T["_n"])
    with torch.no_grad():
        return np.concatenate([m(take(T, range(i, min(i + PREDICT_CHUNK, M)))).double().cpu().numpy()
                               for i in range(0, M, PREDICT_CHUNK)])


def val_mse(m, bags, ow):
    """Ec. 39 inner-fold objective: class-weighted item MSE without the penalty."""
    T = _batched(bags, next(m.parameters()).device)
    P = torch.from_numpy(predict(m, T))
    return float(weighted_item_mse(P, T["s"].cpu().double(), T["y"].cpu().tolist(), ow))


def batch_loss(m, b, ow):
    """Mean over the batch of omega_b (1/8) ||s_hat_b - s_b||^2 plus the model's own penalty (Ec. 77)."""
    terms = getattr(m, "fit_terms", None)
    s_hat, pen = terms(b) if terms else (m(b), 0.0)
    w = torch.where(b["y"] == 1, ow, 1.0).to(s_hat.dtype)
    return (w * (s_hat - b["s"]).pow(2).mean(-1)).mean() + torch.as_tensor(pen, dtype=s_hat.dtype).mean()


def run_epoch(m, opt, bags, ow, gen, wrap=iter):
    """One pass over the shuffled bags in mini-batches of BATCH, each one padded forward (batch_loss); `wrap` sees
    the batch iterator (timing probe)."""
    m.train()
    T = _batched(bags, next(m.parameters()).device)
    order = torch.randperm(len(T["_n"]), generator=gen).tolist()
    for idx in wrap(order[i : i + BATCH] for i in range(0, len(order), BATCH)):
        loss = batch_loss(m, take(T, idx), ow)
        opt.zero_grad()
        loss.backward()
        opt.step()


def train(make_model, bags, epochs, seed, eval_bags=None):
    """Re-initialise with seed and train; returns (model, per-epoch Ec. 39 objective on eval_bags or []).
    omega_+ comes from the training bags and weights the evaluation bags too (Ec. 39). Bags are collated once on the
    device (or passed already collated)."""
    torch.manual_seed(seed)
    m = make_model().to(device())
    opt = make_optimizer(m)
    T = _batched(bags, device())
    E = None if eval_bags is None else _batched(eval_bags, device())
    ow = omega_plus(T["y"].cpu().numpy())
    gen = torch.Generator().manual_seed(seed)
    curve = []
    for _ in range(epochs):
        run_epoch(m, opt, T, ow, gen)
        if E is not None:
            curve.append(val_mse(m, E, ow))
    return m, curve
