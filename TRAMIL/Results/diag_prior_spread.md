# Diagnostic: spread of the semantic prior R and of G (r08 Ec. 10, 70)

Source: interviewer prompt text `q_text` from `instances_clean.csv` (the r08 definition, not the response path), embedded with all-mpnet-base-v2 against the 8 PHQ-8 descriptors; 141 training bags, instances kept = finite WavLM vectors (7985 instances). Percentiles are pooled over all entries; G uses off-diagonal pairs within a bag (cosine, eps 1e-8).

| variant | quantity | min | p5 | median | p95 | max |
| --- | --- | --- | --- | --- | --- | --- |
| raw | R_ij entries | 0.4359 | 0.4886 | 0.5438 | 0.6414 | 0.8179 |
| raw | row spread (max-min) | 0.0178 | 0.0485 | 0.0894 | 0.1768 | 0.3187 |
| raw | G_ik off-diagonal | 0.9772 | 0.9946 | 0.9984 | 0.9996 | 1.0000 |
| centered per item across segments | R_ij entries | -0.1065 | -0.0565 | -0.0075 | 0.0807 | 0.2549 |
| centered per item across segments | row spread (max-min) | 0.0187 | 0.0330 | 0.0651 | 0.1517 | 0.2753 |
| centered per item across segments | G_ik off-diagonal | -0.9943 | -0.8276 | 0.0089 | 0.8427 | 1.0000 |
| row z-scored | R_ij entries | -2.3726 | -1.5147 | -0.0779 | 1.7449 | 2.5174 |
| row z-scored | row spread (max-min) | 2.3155 | 2.6005 | 3.1091 | 3.6167 | 3.9146 |
| row z-scored | G_ik off-diagonal | -0.9325 | -0.2446 | 0.5309 | 0.8921 | 1.0000 |

Per-bag, per-item std of R_ij across segments (raw): min/p5/median/p95/max = 0.0232 / 0.0308 / 0.0409 / 0.0586 / 0.0698.
