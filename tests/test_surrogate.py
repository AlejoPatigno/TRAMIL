"""Acoustic surrogate model (r22 Ec. 14, 15-21, 70-77, 104-106, 114-116) and its flags (permuted, prompt-only, r12)."""
import math

import numpy as np
import pytest
import torch

from TRAMIL import frontend, item_training, model, surrogate
from test_frontend import _items


def _bags(k=6, source="X", n_inst=None):
    items = _items(n=8, n_inst=n_inst)
    bags, _, info = frontend.fold_bags(items, range(8), [], k=k, source=source)
    return bags, info


def test_shapes_finite_outputs_and_attention_on_the_simplex():
    bags, info = _bags()
    m = surrogate.SurrogateModel(d_in=6, mu_R=1 / info["vbar"]).eval()
    gamma = 1 / surrogate.TAU_A + surrogate.KAPPA / surrogate.TAU_E  # Ec. 105 band
    for b in bags:
        out = m.parts(b)
        N = len(b["Y"])
        assert out["s_hat"].shape == (8,) and torch.isfinite(out["s_hat"]).all()
        assert torch.allclose(out["A"].sum(0), torch.ones(8), atol=1e-5) and (out["A"] > 0).all()
        assert torch.allclose(out["Pi"].sum(1), torch.ones(N), atol=1e-5)
        assert (out["Pi"] > math.exp(-2 * gamma) / N).all() and (out["Pi"] < math.exp(2 * gamma) / N).all()
        assert torch.all(out["Delta"].diagonal() == 0) and (out["Delta"].abs() < 1 / surrogate.TAU_A).all()


def test_fixed_constants_of_the_v07_table():
    assert (surrogate.TAU_E, surrogate.KAPPA, surrogate.TAU_A, surrogate.EPS, surrogate.TAU) == (1.0, 5.0, 0.2, 1e-8, 0.10)
    m = surrogate.SurrogateModel(d_in=6)
    assert torch.all(m.W_o.weight == 0) and torch.nn.functional.softplus(m.rho).item() == pytest.approx(1.0)


def test_zero_surrogate_penalty_is_one_under_mu_r_one_over_vbar():
    bags, info = _bags(n_inst=5)
    m = surrogate.SurrogateModel(d_in=6, mu_R=1 / info["vbar"])
    torch.nn.init.zeros_(m.W_r.weight)
    torch.nn.init.zeros_(m.W_r.bias)
    pens = [m.fit_terms(b)[1].item() for b in bags]
    assert np.mean(pens) == pytest.approx(1.0, rel=1e-5)


def test_penalty_gradient_reaches_only_the_surrogate_map():
    bags, info = _bags()
    m = surrogate.SurrogateModel(d_in=6, mu_R=1 / info["vbar"])
    _, pen = m.fit_terms(bags[0])
    names, params = zip(*m.named_parameters())
    grads = torch.autograd.grad(pen, params, allow_unused=True)
    hit = {n for n, g in zip(names, grads) if g is not None and g.abs().sum() > 0}
    assert hit == {"W_r.weight", "W_r.bias"}


def test_item_loss_still_trains_the_surrogate_map_through_the_forward_path():
    bags, _ = _bags()
    m = surrogate.SurrogateModel(d_in=6, mu_R=0.0)
    s_hat, pen = m.fit_terms(bags[0])
    assert pen == 0.0
    g = torch.autograd.grad(s_hat.sum(), m.W_r.weight)[0]
    assert g.abs().sum() > 0


def test_inference_reads_audio_only():
    bags, info = _bags()
    m = surrogate.SurrogateModel(d_in=6, mu_R=1 / info["vbar"]).eval()
    b = bags[0]
    assert torch.allclose(m(b), m({"Y": b["Y"]}))


def test_model_is_invariant_to_joint_permutation_of_instances():
    bags, _ = _bags()
    m = surrogate.SurrogateModel(d_in=6).eval()
    with torch.no_grad():
        m.W_o.weight.normal_()
    b = bags[0]
    p = torch.randperm(len(b["Y"]))
    assert torch.allclose(m(b), m({"Y": b["Y"][p]}), atol=1e-5)


@pytest.mark.parametrize("k", [16, 32, 64, 128, 768])
def test_parameter_count_is_r22_full_model_plus_the_surrogate_map(k):
    # r22 Section 11 states 32k + 3657 (28,233 at k = 768); that count omits W_r, b_r of Ec. 114 (32*8 + 8 = 264).
    assert model.n_params(surrogate.SurrogateModel(d_in=k)) == 32 * k + 3657 + 264
    assert model.n_params(surrogate.SurrogateModel(d_in=k, prior="text")) == 32 * k + 3657


def test_permuted_target_flag_trains_against_the_permuted_rows():
    bags, _ = _bags()
    b = bags[0]
    a = surrogate.SurrogateModel(d_in=6, mu_R=1.0).eval()
    p = surrogate.SurrogateModel(d_in=6, mu_R=1.0, permuted=True).eval()
    p.load_state_dict(a.state_dict())
    s_a, pen_a = a.fit_terms(b)
    s_p, pen_p = p.fit_terms(b)
    Rc_hat = a.parts(b)["Rc_hat"]
    assert torch.allclose(s_a, s_p)  # the forward pass is identical
    assert pen_p.item() == pytest.approx(((Rc_hat - b["Rc_perm"]) ** 2).sum().item() / (8 * len(Rc_hat)), rel=1e-5)
    assert pen_a.item() == pytest.approx(((Rc_hat - b["Rc"]) ** 2).sum().item() / (8 * len(Rc_hat)), rel=1e-5)


def test_prompt_only_model_reads_prompt_rows_and_log_bag_size_in_every_head():
    bags, info = _bags(k=8, source="R")
    m = surrogate.SurrogateModel(d_in=8, mu_R=1 / info["vbar"], log_n=True).eval()
    assert model.n_params(m) == 32 * 8 + 3657 + 264 + 8
    b = bags[0]
    with torch.no_grad():
        m.g.fill_(1.0)
    shifted = dict(b, logN=b["logN"] + 1.0)
    assert torch.allclose(m(shifted) - m(b), torch.ones(8), atol=1e-5)


def test_r12_reference_reads_the_prompt_prior_and_graph_at_inference():
    bags, _ = _bags()
    m = surrogate.SurrogateModel(d_in=6, prior="text").eval()
    b = bags[0]
    with pytest.raises(KeyError):
        m({"Y": b["Y"]})
    assert m.fit_terms(b)[1] == 0.0
    assert torch.allclose(m.parts(b)["E"], b["G"])


def test_every_flag_trains_one_epoch_with_finite_predictions():
    bags, info = _bags()
    for kw in ({}, {"permuted": True}, {"mu_R": 0.0}, {"prior": "text"}, {"p_drop": 0.3}):
        kw = {"d_in": 6, "mu_R": 1 / info["vbar"], **kw}
        m, curve = item_training.train(lambda: surrogate.SurrogateModel(**kw), bags, 1, 11, eval_bags=bags)
        assert np.isfinite(curve).all() and np.isfinite(item_training.predict(m, bags)).all()
