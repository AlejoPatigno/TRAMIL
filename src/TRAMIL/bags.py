"""Bags: one subject is one bag, its answer segments are the instances (proposal Ec. 1-4)."""
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

__provenance__ = {
    "revision": "research-concept-residual-semantic-prior-bipartite-mil-r06.md",
    "sections": ["2", "3"],
    "equations": [1, 2, 3, 4, 68],
    "invariants": [
        "label_is_phq8_total_at_least_10",
        "bag_is_one_subject_with_answer_instances_in_order",
        "bag_requires_at_least_one_instance",
        "items_sum_to_total_for_all_275_rows",
        "recomputed_label_disagrees_with_flag_in_20_sessions",
        "e1_training_bags_are_141_subjects",
        "stratified_subset_keeps_class_ratio_is_reproducible_and_sorted",
    ],
}

MIN_ANSWER_S = 0.25  # Section 3: responses shorter than 0.25 s are excluded
THRESHOLD = 10  # PHQ-8 total >= 10


@dataclass(frozen=True)
class Bag:
    """B_b = {(a_bi, t_bi)}: audio files and texts of one subject, y_b in {0, 1} (Ec. 1, 2)."""

    subject: int
    y: int
    y_distributed: int
    texts: list
    wavs: list
    idx: list

    def __len__(self):
        return len(self.texts)


def load_labels_frame(df):
    """Recompute the binary label from the eight item scores; keep the distributed flag as a sensitivity column."""
    items = [c for c in df.columns if c.startswith("PHQ8_")]
    out = df[["Participant", "split"]].copy()
    out["phq8_total"] = df[items].sum(axis=1)
    out["label"] = (out["phq8_total"] >= THRESHOLD).astype(int)
    out["label_distributed"] = df["Depression_label"].astype(int)
    return out


def load_labels(path):
    return load_labels_frame(pd.read_csv(path))


def build_bags(instances, audio_dir, labels, split="train"):
    """One bag per labelled subject of `split`; answer instances ordered by idx, floor of 0.25 s on the answer duration."""
    audio_dir = Path(audio_dir)
    lab = labels[labels["split"] == split].set_index("Participant")
    out = []
    for sid, g in instances.sort_values(["id", "idx"]).groupby("id"):
        if sid not in lab.index:
            continue
        g = g[g["a_dur"] >= MIN_ANSWER_S]
        if g.empty:
            continue
        idx = [int(i) for i in g["idx"]]
        out.append(
            Bag(
                subject=int(sid),
                y=int(lab.loc[sid, "label"]),
                y_distributed=int(lab.loc[sid, "label_distributed"]),
                texts=[str(t) if isinstance(t, str) else "" for t in g["a_text"]],
                wavs=[audio_dir / str(sid) / f"{sid}_{i:03d}_a.wav" for i in idx],
                idx=idx,
            )
        )
    return out


def stratified_subset(bags, n, seed):
    """Label-stratified subject subset of size ~n (rehearsal scale), reproducible, returned in subject order."""
    import numpy as np

    g = np.random.default_rng(seed)
    out = []
    for yy in (0, 1):
        grp = [b for b in bags if b.y == yy]
        k = round(n * len(grp) / len(bags))
        out += [grp[i] for i in g.choice(len(grp), k, replace=False)]
    return sorted(out, key=lambda b: b.subject)
