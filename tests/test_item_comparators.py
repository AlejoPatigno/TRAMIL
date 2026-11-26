"""Supervision-matched comparators with eight item heads (r22 Section 11, Ec. 33, 34, 75, 79), Symptom-Guided, and
the E1 arm registry."""
import numpy as np
import pytest
import torch

from TRAMIL import e1_arms, frontend, item_comparators as ic, item_training, model
from test_frontend import _items


def _Q(seed=0):
    q = np.random.default_rng(seed).standard_normal((8, 768)).astype(np.float32)
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def _bags(k=6):
    return frontend.fold_bags(_items(n=8), range(8), [], k=k)[0]


@pytest.mark.parametrize("k", [16, 768])
def test_parameter_counts_match_the_r22_section_11_table(k):
    assert model.n_params(ic.ABMIL8(d_in=k)) == 32 * k + 1384
    assert model.n_params(ic.ABMIL8(d_in=k, gated=True)) == 32 * k + 2440
    assert model.n_params(ic.ABMIL8(d_in=k, with_R=True)) == 32 * k + 1392
    assert model.n_params(ic.R06Items(d_in=k)) == 32 * k + 1609
    # Symptom-Guided is not in the r22 table: W_x, b_x + queries 256 + 3 LayerNorms 192 + 8 temperatures + heads 264
    assert model.n_params(ic.SymptomGuided(_Q(), d_in=k)) == 32 * k + 752


def test_single_attention_models_weight_instances_on_the_simplex_and_feed_one_vector_to_all_heads():
    b = _bags()[0]
    for kw in ({}, {"gated": True}, {"with_R": True}):
        m = ic.ABMIL8(d_in=6, **kw).eval()
        a, z = m.pool(b)
        assert a.shape == (len(b["Y"]),) and torch.isclose(a.sum(), torch.tensor(1.0)) and (a > 0).all()
        assert torch.allclose(m(b), (m.heads.weight * z).sum(1) + m.heads.bias)


def test_per_item_attention_models_have_one_simplex_column_per_item():
    b = _bags()[0]
    for m in (ic.R06Items(d_in=6), ic.SymptomGuided(_Q(), d_in=6)):
        A = m.eval().attention(b)
        assert A.shape == (len(b["Y"]), 8) and torch.allclose(A.sum(0), torch.ones(8), atol=1e-5) and (A > 0).all()


def test_prompt_reading_comparators_read_r_at_inference_and_the_others_do_not():
    b = _bags()[0]
    audio = {"Y": b["Y"]}
    for m in (ic.ABMIL8(d_in=6), ic.ABMIL8(d_in=6, gated=True), ic.SymptomGuided(_Q(), d_in=6)):
        assert torch.allclose(m.eval()(b), m(audio))
    abr, r06 = ic.ABMIL8(d_in=6, with_R=True).eval(), ic.R06Items(d_in=6).eval()
    with torch.no_grad():
        abr.u.weight.fill_(1.0)
        r06.corrector.out.weight.normal_()
    other = dict(b, R=torch.rand_like(b["R"]))
    assert not torch.allclose(abr(b), abr(other)) and not torch.allclose(r06(b), r06(other))


def test_symptom_guided_queries_start_from_projected_item_embeddings_with_unit_mean_norm():
    Q = _Q()
    m = ic.SymptomGuided(Q, d_in=6)
    xi0 = ic.project_queries(Q)
    assert torch.allclose(m.queries.data, xi0) and xi0.norm(dim=1).mean().item() == pytest.approx(1.0, rel=1e-5)
    assert torch.allclose(m.temperature(), torch.ones(8)) and m.queries.requires_grad
    P = ic.query_projection()
    assert np.allclose(P @ P.T, np.eye(32), atol=1e-5)


def test_models_are_invariant_to_joint_permutation_of_instances():
    b = _bags()[0]
    p = torch.randperm(len(b["Y"]))
    pb = {"Y": b["Y"][p], "R": b["R"][p]}
    for m in (ic.ABMIL8(d_in=6, with_R=True), ic.R06Items(d_in=6), ic.SymptomGuided(_Q(), d_in=6)):
        assert torch.allclose(m.eval()(b), m(pb), atol=1e-5)


def test_every_e1_arm_builds_trains_one_epoch_and_predicts_finite_items():
    items = _items(n=8)
    for arm in e1_arms.ARMS:
        k = 8 if e1_arms.SOURCE.get(arm, "X") == "R" else 6
        tr, ev, info = frontend.fold_bags(items, range(6), [6, 7], k=k, source=e1_arms.SOURCE.get(arm, "X"))
        make = e1_arms.build(arm, k=k, p_drop=0.15, vbar=info["vbar"], Q=_Q())
        m, curve = item_training.train(make, tr, 1, 11, eval_bags=ev)
        assert len(curve) == 1 and np.isfinite(curve).all(), arm
        assert item_training.predict(m, ev).shape == (2, 1 if arm == "binary_sup" else 8), arm  # binary head: one logit
    assert set(e1_arms.ARMS) >= {"full", "permuted", "untargeted", "r12", "prompt_only", "abmil8", "gated_abmil8",
                                 "abmil_r8", "r06_8", "symptom_guided"}


def test_comparators_and_untargeted_arm_train_without_the_penalty():
    b = _bags()[0]
    for arm in ("abmil8", "gated_abmil8", "abmil_r8", "r06_8", "symptom_guided", "untargeted"):
        m = e1_arms.build(arm, k=6, p_drop=0.0, vbar=0.01, Q=_Q())()
        pen = m.fit_terms(b)[1] if hasattr(m, "fit_terms") else 0.0
        assert float(pen) == 0.0, arm
    full = e1_arms.build("full", k=6, p_drop=0.0, vbar=0.01, Q=_Q())()
    assert full.mu_R == pytest.approx(100.0)


def test_label_permutation_moves_the_eight_item_scores_as_a_unit():
    S = np.arange(40, dtype=float).reshape(5, 8)
    P = item_training.permute_items(S, seed=3)
    assert sorted(map(tuple, P)) == sorted(map(tuple, S)) and not np.array_equal(P, S)
    assert np.array_equal(P, item_training.permute_items(S, seed=3))
