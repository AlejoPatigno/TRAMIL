from pathlib import Path

import pandas as pd
import pytest

from TRAMIL import bags
from conftest import EDAIC, LABELS, needs_edaic, needs_labels

ITEMS = [f"PHQ8_{i}" for i in range(1, 9)]


def _frame(rows):
    cols = ["Participant", *ITEMS, "Depression_label", "split"]
    return pd.DataFrame(rows, columns=cols)


def test_label_is_phq8_total_at_least_10():
    # total 9 -> 0, total 10 -> 1; the distributed flag must not decide it
    df = _frame([[1, 2, 1, 1, 1, 1, 1, 1, 1, 1, "train"], [2, 2, 2, 1, 1, 1, 1, 1, 1, 0, "train"]])
    out = bags.load_labels_frame(df)
    assert out.set_index("Participant").label.to_dict() == {1: 0, 2: 1}
    assert out.set_index("Participant").label_distributed.to_dict() == {1: 1, 2: 0}


def test_bag_is_one_subject_with_answer_instances_in_order():
    inst = pd.DataFrame(
        {"id": [7, 7, 7, 8], "idx": [1, 0, 2, 0], "a_text": ["b", "a", "c", "z"], "a_dur": [1.0, 1.0, 0.1, 2.0]}
    )
    lab = pd.DataFrame({"Participant": [7, 8], "label": [1, 0], "label_distributed": [1, 0], "split": ["train"] * 2})
    out = bags.build_bags(inst, Path("audio"), lab, split="train")
    assert [b.subject for b in out] == [7, 8]
    b7 = out[0]
    assert b7.texts == ["a", "b"]  # idx order; the 0.1 s answer is below the 0.25 s floor
    assert [p.name for p in b7.wavs] == ["7_000_a.wav", "7_001_a.wav"]
    assert b7.y == 1 and len(b7) == 2


def test_bag_requires_at_least_one_instance():
    inst = pd.DataFrame({"id": [7], "idx": [0], "a_text": ["a"], "a_dur": [0.1]})
    lab = pd.DataFrame({"Participant": [7], "label": [1], "label_distributed": [1], "split": ["train"]})
    assert bags.build_bags(inst, Path("audio"), lab, split="train") == []


@needs_labels
def test_items_sum_to_total_for_all_275_rows():
    df = pd.read_csv(LABELS)
    assert len(df) == 275
    items = [c for c in df.columns if c.startswith("PHQ8_")]
    assert len(items) == 8
    assert (df[items].sum(axis=1) == df["Depression_severity"]).all()


@needs_labels
def test_recomputed_label_disagrees_with_flag_in_20_sessions():
    out = bags.load_labels(LABELS)
    bad = out[out.label != out.label_distributed]
    assert len(bad) == 20
    assert bad.split.value_counts().to_dict() == {"train": 13, "dev": 3, "test": 4}


@needs_labels
@needs_edaic
def test_e1_training_bags_are_141_subjects():
    lab = bags.load_labels(LABELS)
    inst = pd.read_csv(EDAIC / "instances_clean.csv")
    out = bags.build_bags(inst, EDAIC / "instances_audio", lab, split="train")
    assert len(out) == 141
    assert sum(b.y for b in out) == 44  # proposal Section 3: 44 positive training subjects
    assert all(w.exists() for b in out for w in b.wavs)


def test_stratified_subset_keeps_class_ratio_is_reproducible_and_sorted():
    from TRAMIL.bags import Bag

    mk = lambda i: Bag(subject=i, y=int(i < 30), y_distributed=0, texts=["a"], wavs=[Path("x")], idx=[0])
    allb = [mk(i) for i in range(100)]
    s1, s2 = bags.stratified_subset(allb, 20, seed=0), bags.stratified_subset(allb, 20, seed=0)
    assert [b.subject for b in s1] == [b.subject for b in s2] and len(s1) == 20
    assert sum(b.y for b in s1) == 6  # 30% of 20
    assert [b.subject for b in s1] == sorted(b.subject for b in s1)
    assert [b.subject for b in s1] != [b.subject for b in bags.stratified_subset(allb, 20, seed=1)]
