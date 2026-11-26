"""Batched training (T6b): one padded tensor forward with an instance mask per batch of bags, equivalent to the
per-bag models (frozen copies in oracle_*.py) in outputs, penalties, losses and parameter gradients, and invariant to
the amount of padding."""
import numpy as np
import pytest
import torch

import oracle_e2e5 as O5
import oracle_item_comparators as OC
import oracle_surrogate as OS
from TRAMIL import e1_arms, frontend, item_training as it
from test_frontend import _items
from test_item_comparators import _Q
from TRAMIL.item_comparators import project_queries

ORACLE = {
    "full": lambda k, v, Q: OS.SurrogateModel(k, 0.0, mu_R=1 / v),
    "permuted": lambda k, v, Q: OS.SurrogateModel(k, 0.0, mu_R=1 / v, permuted=True),
    "untargeted": lambda k, v, Q: OS.SurrogateModel(k, 0.0, mu_R=0.0),
    "r12": lambda k, v, Q: OS.SurrogateModel(k, 0.0, prior="text"),
    "prompt_only": lambda k, v, Q: OS.SurrogateModel(k, 0.0, mu_R=1 / v, log_n=True),
    "abmil8": lambda k, v, Q: OC.ABMIL8(k, 0.0),
    "gated_abmil8": lambda k, v, Q: OC.ABMIL8(k, 0.0, gated=True),
    "abmil_r8": lambda k, v, Q: OC.ABMIL8(k, 0.0, with_R=True),
    "r06_8": lambda k, v, Q: OC.R06Items(k, 0.0),
    "symptom_guided": lambda k, v, Q: OC.SymptomGuided(Q, k, 0.0),
    # E2 to E5 arms (per-bag oracles of oracle_e2e5.py; aliases and data variants share the E1 oracle)
    "graph_corr": lambda k, v, Q: O5.PerBagVariant(k, 0.0, mu_R=1 / v, graph="corr"),
    "graph_prior": lambda k, v, Q: O5.PerBagVariant(k, 0.0, mu_R=1 / v, graph="prior"),
    "free_desc": lambda k, v, Q: O5.PerBagVariant(k, 0.0, mu_R=1 / v, free="desc", xi0=project_queries(Q)),
    "free_zero": lambda k, v, Q: O5.PerBagVariant(k, 0.0, mu_R=1 / v, free="zero"),
    "binary_sup": lambda k, v, Q: O5.PerBagVariant(k, 0.0, mu_R=1 / v, binary=True),
    "permuted_quint": lambda k, v, Q: OS.SurrogateModel(k, 0.0, mu_R=1 / v, permuted=True),
    "full_nohist": lambda k, v, Q: OS.SurrogateModel(k, 0.0, mu_R=1 / v),
    "full_x": lambda k, v, Q: OS.SurrogateModel(k, 0.0, mu_R=1 / v),
    "permuted_x": lambda k, v, Q: OS.SurrogateModel(k, 0.0, mu_R=1 / v, permuted=True),
    "untargeted_x": lambda k, v, Q: OS.SurrogateModel(k, 0.0, mu_R=0.0),
}
ORACLE.update({a: ORACLE[b] for a, b in e1_arms.ALIAS.items() if a not in ORACLE})  # an alias is the E1 arm itself
ARMS = tuple(e1_arms.ARMS)  # every registered arm has a per-bag oracle above


def _setup(arm, n=10, seed=0):
    items = _items(n=n, seed=seed)  # bag sizes 3..8
    src = e1_arms.SOURCE.get(arm, "X")
    k = 8 if src == "R" else 6
    bags, _, info = frontend.fold_bags(items, range(n), [], k=k, source=src)
    for b, i in zip(bags, items):  # varied item scores so the loss is not trivial
        b["s"] = torch.linspace(0, 3, 8) * (1 + (b["subject"] % 3))
        b["y"] = int(b["s"].sum() >= 10)
    torch.manual_seed(seed)
    new = e1_arms.build(arm, k=k, p_drop=0.0, vbar=info["vbar"], Q=_Q())()
    old = ORACLE[arm](k, info["vbar"], _Q())
    old.load_state_dict(new.state_dict())
    for p in new.parameters():  # move W_o off zero so the corrector and graph paths carry gradient
        p.data.add_(0.05 * torch.randn_like(p))
    old.load_state_dict(new.state_dict())
    return new.eval(), old.eval(), bags


def _old_terms(m, b):
    return m.fit_terms(b) if hasattr(m, "fit_terms") else (m(b), 0.0)


@pytest.mark.parametrize("arm", ARMS)
def test_batched_forward_equals_the_per_bag_models(arm):
    new, old, bags = _setup(arm)
    T = it.collate(bags)
    with torch.no_grad():
        got = new(T)
        want = torch.stack([old(b) for b in bags])
    assert got.shape == (len(bags), 1 if arm == "binary_sup" else 8) and torch.allclose(got, want, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("arm", ARMS)
def test_batched_loss_and_gradients_equal_the_per_bag_sum(arm):
    new, old, bags = _setup(arm)
    ow = it.omega_plus([b["y"] for b in bags])
    idx = [7, 2, 5, 0, 9, 3, 1, 8]
    T = it.collate(bags)
    loss_new = it.batch_loss(new, it.take(T, idx), ow)
    loss_new.backward()
    loss_old = 0.0
    for j in idx:
        s_hat, pen = _old_terms(old, bags[j])
        if arm == "binary_sup":
            loss_old = loss_old + O5.bce(s_hat, float(bags[j]["y"]), ow) + pen
            continue
        loss_old = loss_old + (ow if bags[j]["y"] else 1.0) * (s_hat - bags[j]["s"]).pow(2).mean() + pen
    loss_old = loss_old / len(idx)
    loss_old.backward()
    assert torch.allclose(loss_new, loss_old, atol=1e-5, rtol=1e-5)
    for (name, a), b in zip(new.named_parameters(), old.parameters()):
        ga = a.grad if a.grad is not None else torch.zeros_like(a)
        gb = b.grad if b.grad is not None else torch.zeros_like(b)
        assert torch.allclose(ga, gb, atol=1e-5, rtol=1e-4), (arm, name)


@pytest.mark.parametrize("arm", ARMS)
def test_extra_padding_never_changes_outputs_or_gradients(arm):
    new, _, bags = _setup(arm)
    T = it.collate(bags)
    P = it.collate(bags, n_max=20)
    assert P["Y"].shape[1] == 20
    outs = []
    for batch in (T, P):
        new.zero_grad()
        loss = it.batch_loss(new, batch, 1.5)
        loss.backward()
        outs.append((new(batch).detach(), [p.grad.clone() for p in new.parameters() if p.grad is not None]))
    assert torch.allclose(outs[0][0], outs[1][0], atol=1e-6)
    assert all(torch.allclose(a, b, atol=1e-6) for a, b in zip(outs[0][1], outs[1][1]))
    assert all(torch.isfinite(g).all() for g in outs[1][1])


def test_a_batch_is_an_index_slice_trimmed_to_its_longest_bag():
    _, _, bags = _setup("full")
    T = it.collate(bags)
    idx = [1, 4]
    b = it.take(T, idx)
    n = max(len(bags[i]["Y"]) for i in idx)
    assert b["Y"].shape == (2, n, bags[0]["Y"].shape[1]) and b["G"].shape == (2, n, n)
    assert b["mask"].sum(1).tolist() == [len(bags[i]["Y"]) for i in idx]
    assert torch.equal(b["Y"][0, : len(bags[1]["Y"])], bags[1]["Y"])


def test_training_on_batches_matches_the_per_bag_loop_without_dropout():
    _, _, bags = _setup("abmil8")
    make = lambda: e1_arms.build("abmil8", k=6, p_drop=0.0, vbar=1.0)()  # noqa: E731
    m, curve = it.train(make, bags[:8], 3, 11, eval_bags=bags[8:])
    torch.manual_seed(11)
    o = OC.ABMIL8(6, 0.0)
    opt = it.make_optimizer(o)
    ow = it.omega_plus([b["y"] for b in bags[:8]])
    gen = torch.Generator().manual_seed(11)
    want = []
    for _ in range(3):
        o.train()
        order = torch.randperm(8, generator=gen).tolist()
        for i in range(0, 8, it.BATCH):
            loss = sum((ow if bags[j]["y"] else 1.0) * (o(bags[j]) - bags[j]["s"]).pow(2).mean()
                       for j in order[i : i + it.BATCH]) / len(order[i : i + it.BATCH])
            opt.zero_grad()
            loss.backward()
            opt.step()
        o.eval()
        with torch.no_grad():
            P = torch.stack([o(b) for b in bags[8:]]).double()
        want.append(float(it.weighted_item_mse(P, torch.stack([b["s"] for b in bags[8:]]).double(),
                                               [b["y"] for b in bags[8:]], ow)))
    assert np.allclose(curve, want, rtol=1e-4, atol=1e-6)
