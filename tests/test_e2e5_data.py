"""E5 data controls: the quintile derangement of the penalty target (third control, d) and the history-removed bags (e)."""
import numpy as np
import pytest
import torch

from TRAMIL import frontend
from test_frontend import _items


def _sizes(n=60, seed=0):
    return np.random.default_rng(seed).integers(3, 200, n)


def test_derangement_has_no_fixed_point_stays_in_its_size_quintile_and_is_a_bijection():
    N = _sizes()
    donor = frontend.quintile_derangement(N, seed=5)
    assert sorted(donor) == list(range(len(N))) and (donor != np.arange(len(N))).all()
    order = np.argsort(N, kind="stable")
    q = np.empty(len(N), int)
    for g, idx in enumerate(np.array_split(order, 5)):
        q[idx] = g
    assert (q[donor] == q).all()
    assert np.array_equal(donor, frontend.quintile_derangement(N, seed=5))
    assert not np.array_equal(donor, frontend.quintile_derangement(N, seed=6))


def test_a_quintile_of_one_subject_is_merged_so_no_subject_keeps_its_own_target():
    donor = frontend.quintile_derangement(np.array([3, 4, 5, 6, 7, 8, 9]), seed=1)  # array_split gives groups of 2 and 1
    assert (donor != np.arange(7)).all()


def test_donor_rows_are_taken_in_order_truncated_or_cycled_to_the_recipient_count_and_centered():
    items = _items(n=6)
    bags, _, _ = frontend.fold_bags(items, range(6), [], k=5)
    donor = frontend.quintile_derangement([len(i["R"]) for i in items], seed=2)
    before = [b["Rc"].clone() for b in bags]
    frontend.retarget_quint(bags, items, donor)
    for b, i, d in zip(bags, range(6), donor):
        n = len(items[i]["R"])
        rows = items[d]["R"][np.arange(n) % len(items[d]["R"])]  # order, truncated or cycled
        assert torch.allclose(b["Rc_perm"], torch.from_numpy(rows - rows.mean(0)), atol=1e-6)
        assert torch.allclose(b["Rc_perm"].mean(0), torch.zeros(8), atol=1e-6)
    assert all(torch.equal(a, b["Rc"]) for a, b in zip(before, bags))  # only the penalty target moves
    assert {len(items[d]["R"]) == len(items[i]["R"]) for i, d in enumerate(donor)} >= {False}  # sizes differ somewhere


def _with_prompts(n=4):
    items = _items(n=n)
    for it in items:
        N = len(it["R"])
        it["prompts"] = ["Have you ever been diagnosed with depression?" if i == 0 else f"Tell me about topic {i}."
                         for i in range(N)]
    return items


def test_history_removal_drops_the_listed_prompts_in_every_aligned_array_and_keeps_every_subject():
    items = _with_prompts()
    out = frontend.drop_history(items, ["have you ever been diagnosed with depression"])
    assert [o["subject"] for o in out] == [i["subject"] for i in items]
    for o, i in zip(out, items):
        n = len(i["R"]) - 1
        assert len(o["X"]) == len(o["R"]) == len(o["prompts"]) == n
        assert np.array_equal(o["X"], i["X"][1:]) and np.array_equal(o["R"], i["R"][1:])
    assert len(items[0]["R"]) == len(out[0]["R"]) + 1  # inputs untouched


def test_history_removal_refuses_missing_prompts_an_empty_match_and_an_emptied_bag():
    items = _items(n=3)
    with pytest.raises(ValueError, match="prompts"):
        frontend.drop_history(items, ["x"])
    items = _with_prompts()
    with pytest.raises(ValueError, match="no segment"):
        frontend.drop_history(items, ["a prompt that never occurs"])
    items[0]["prompts"] = ["Have you ever been diagnosed with depression?"] * len(items[0]["R"])
    with pytest.raises(ValueError, match="empties"):
        frontend.drop_history(items, ["have you ever been diagnosed with depression"])
