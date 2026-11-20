"""One-off measurement: spread of R_bij (interviewer-prompt embeddings vs PHQ-8 descriptors) and of G_b,ik on the 141 training bags."""
import numpy as np, pandas as pd
from pathlib import Path
from TRAMIL import bags as bagmod, features, e1_run

f = e1_run._find(e1_run._default_roots("wavlm"), acoustic="wavlm")
edaic = f["instances"].parent
inst = pd.read_csv(edaic / "instances_clean.csv")
bags = bagmod.build_bags(inst, edaic / "instances_audio", bagmod.load_labels(f["labels"]), "train")
qtext = {(int(r.id), int(r.idx)): str(r.q_text) if isinstance(r.q_text, str) else "" for r in inst.itertuples()}
Q = features.embed_texts(features.DESCRIPTORS)
uniq = sorted({qtext[(b.subject, i)] for b in bags for i in b.idx})
emb = dict(zip(uniq, features.embed_texts(uniq)))
cache = Path(f["features"])
Rs = []
for b in bags:
    W = np.load(cache / f"{b.subject}_W.npy"); ok = np.isfinite(W).all(1)
    P = np.stack([emb[qtext[(b.subject, i)]] for i in b.idx])[ok]
    if len(P): Rs.append((np.clip(P @ Q.T, -1, 1) + 1) / 2)
print("bags", len(Rs), "instances", sum(len(r) for r in Rs), "empty-prompt instances", sum(qtext[(b.subject, i)] == "" for b in bags for i in b.idx))

def q(v): v = np.asarray(v); return [v.min(), *np.percentile(v, [5, 50, 95]), v.max()]
def offdiag(rs, eps=1e-8):
    out = []
    for r in rs:
        if len(r) < 2: continue
        n = r / (np.linalg.norm(r, axis=1, keepdims=True) + eps); G = n @ n.T
        out.append(G[~np.eye(len(r), dtype=bool)])
    return np.concatenate(out)
variants = {
  "raw": Rs,
  "centered per item across segments": [r - r.mean(0) for r in Rs],
  "row z-scored": [(r - r.mean(1, keepdims=True)) / (r.std(1, keepdims=True) + 1e-8) for r in Rs],
}
md = ["# Diagnostic: spread of the semantic prior R and of G (r08 Ec. 10, 70)", "",
      "Source: interviewer prompt text `q_text` from `instances_clean.csv` (the r08 definition, not the response path), "
      "embedded with all-mpnet-base-v2 against the 8 PHQ-8 descriptors; 141 training bags, instances kept = finite WavLM vectors "
      f"({sum(len(r) for r in Rs)} instances). Percentiles are pooled over all entries; G uses off-diagonal pairs within a bag (cosine, eps 1e-8).", "",
      "| variant | quantity | min | p5 | median | p95 | max |", "| --- | --- | --- | --- | --- | --- | --- |"]
for n, rs in variants.items():
    ent = np.concatenate([r.ravel() for r in rs]); spread = np.concatenate([r.max(1) - r.min(1) for r in rs])
    for lab, v in ((f"R_ij entries", ent), ("row spread (max-min)", spread), ("G_ik off-diagonal", offdiag(rs))):
        md.append(f"| {n} | {lab} | " + " | ".join(f"{x:.4f}" for x in q(v)) + " |")
# column-wise spread across segments (does R discriminate segments within an item?)
colsd = np.concatenate([r.std(0) for r in Rs if len(r) > 1])
md += ["", f"Per-bag, per-item std of R_ij across segments (raw): min/p5/median/p95/max = " + " / ".join(f"{x:.4f}" for x in q(colsd)) + "."]
Path("TRAMIL/Results/diag_prior_spread.md").write_text("\n".join(md) + "\n"); print("\n".join(md))
