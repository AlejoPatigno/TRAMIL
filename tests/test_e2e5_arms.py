"""E3/E4/E5 arms of v07: graph ablations, free-query variants, binary head, the registry, parameter counts."""
import numpy as np
import pytest
import torch

from TRAMIL import e1_arms, frontend, item_training as it
from TRAMIL.item_comparators import project_queries
from TRAMIL.surrogate import KAPPA, TAU_E, SurrogateModel, masked_softmax
from test_frontend import _items
from test_item_comparators import _Q

K = 6


def _np(m):
    return sum(p.numel() for p in m.parameters())


def _bags(n=8):
    items = _items(n=n)
    bags, _, info = frontend.fold_bags(items, range(n), [], k=K)
    return bags, info["vbar"]


def test_every_new_arm_is_registered_and_the_e1_set_is_unchanged():
    assert len(e1_arms.E1_ARMS) == 10 and set(e1_arms.E1_ARMS) <= set(e1_arms.ARMS)
    assert set(e1_arms.E2E5_ARMS) == {"graph_corr", "graph_prior", "binary_sup", "free_desc", "free_zero",
                                      "permuted_quint", "full_nohist", "full_x", "permuted_x", "untargeted_x"}
    assert set(e1_arms.ARMS) == set(e1_arms.E1_ARMS) | set(e1_arms.E2E5_ARMS)


def test_parameter_counts_follow_the_r22_section_11_table():
    full = _np(SurrogateModel(K, mu_R=1.0))
    assert full == 32 * K + 3657 + 264  # r22 count plus the surrogate map W_r (264)
    assert _np(e1_arms.build("free_desc", K, 0.0, 1.0, _Q())()) == 32 * K + 3656 + 264  # lambda removed
    assert _np(e1_arms.build("free_zero", K, 0.0, 1.0, _Q())()) == 32 * K + 3656 + 264  # lambda removed
    assert _np(e1_arms.build("graph_prior", K, 0.0, 1.0)()) == full - 1024  # no W_a
    assert _np(e1_arms.build("graph_corr", K, 0.0, 1.0)()) == full  # W_a kept, prior term of Ec. 72 dropped
    assert _np(e1_arms.build("binary_sup", K, 0.0, 1.0)()) == full - 264 + 257  # item heads -> r06 classifier


def test_free_queries_start_at_the_descriptor_projection_or_zero_and_drift_is_the_cosine():
    d = e1_arms.build("free_desc", K, 0.0, 1.0, _Q())()
    z = e1_arms.build("free_zero", K, 0.0, 1.0, _Q())()
    assert torch.equal(d.W_o.weight, project_queries(_Q())) and not z.W_o.weight.any()
    assert torch.allclose(d.W_o.weight.norm(dim=1).mean(), torch.tensor(1.0), atol=1e-5)  # s_0 of Ec. 81
    assert torch.allclose(d.query_drift(), torch.ones(8), atol=1e-6)
    d.W_o.weight.data.neg_()
    assert torch.allclose(d.query_drift(), -torch.ones(8), atol=1e-6)
    assert d.rho is None and z.rho is None and "xi0" in d.state_dict()


def test_free_query_attention_is_the_softmax_of_the_correction_alone():
    bags, v = _bags()
    m = e1_arms.build("free_desc", K, 0.0, v, _Q())().eval()
    T = it.collate(bags)
    with torch.no_grad():
        out = m._parts(T)
        want = masked_softmax(out["D"], T["mask"][:, :, None], dim=1)
    assert torch.allclose(out["A"], want, atol=1e-6)
    full = e1_arms.build("full", K, 0.0, v)().eval()
    with torch.no_grad():
        assert not torch.allclose(full._parts(T)["L"], full._parts(T)["D"])  # the full model keeps the prior term


def test_graph_variants_remove_exactly_one_term_of_the_edge_logits():
    bags, v = _bags()
    T = it.collate(bags)
    torch.manual_seed(0)
    ref = e1_arms.build("full", K, 0.0, v)().eval()
    for p in ref.parameters():
        p.data.add_(0.05 * torch.randn_like(p))
    outs = {}
    for arm in ("full", "graph_corr", "graph_prior"):
        m = e1_arms.build(arm, K, 0.0, v)().eval()
        m.load_state_dict({k: w for k, w in ref.state_dict().items() if k in m.state_dict()})
        with torch.no_grad():
            outs[arm] = m._parts(T)
    m1 = T["mask"][:, None, :]
    E, Dl = outs["full"]["E"], outs["full"]["Delta"]
    assert torch.allclose(outs["graph_prior"]["Pi"], masked_softmax(KAPPA / TAU_E * E, m1, 2), atol=1e-6)
    assert not outs["graph_prior"]["Delta"].any()
    assert torch.allclose(outs["graph_corr"]["Pi"], masked_softmax(Dl, m1, 2), atol=1e-6)
    assert torch.allclose(outs["full"]["Pi"], masked_softmax(Dl + KAPPA / TAU_E * E, m1, 2), atol=1e-6)
    assert not torch.allclose(outs["graph_corr"]["Pi"], outs["full"]["Pi"], atol=1e-4)


def test_binary_head_trains_on_the_weighted_bce_and_predicts_one_logit():
    bags, v = _bags(10)
    for b in bags[:5]:
        b["y"] = 1
    for b in bags[5:]:
        b["y"] = 0
    make = e1_arms.build("binary_sup", K, 0.0, v)
    m, curve = it.train(make, bags[:8], 2, 11, eval_bags=bags[8:])
    P = it.predict(m, bags)
    assert P.shape == (10, 1) and len(curve) == 2 and np.isfinite(curve).all() and np.isfinite(P).all()
    ow = it.omega_plus([b["y"] for b in bags[:8]])
    T = it.collate(bags[8:])
    want = torch.nn.functional.binary_cross_entropy_with_logits(
        torch.from_numpy(it.predict(m, T))[:, 0], T["y"].double().cpu(), pos_weight=torch.tensor(ow, dtype=torch.float64))
    assert np.isclose(it.val_mse(m, T, ow), float(want), atol=1e-6)


@pytest.mark.parametrize("arm", e1_arms.E2E5_ARMS)
def test_every_new_arm_trains_one_epoch_with_finite_predictions(arm):
    bags, v = _bags(10)
    for b in bags:
        b["s"] = torch.linspace(0, 3, 8)
        b["y"] = int(b["subject"] % 2)
    m, curve = it.train(e1_arms.build(arm, K, 0.0, v, _Q()), bags[:8], 1, 11, eval_bags=bags[8:])
    assert np.isfinite(it.predict(m, bags)).all() and np.isfinite(curve).all()
