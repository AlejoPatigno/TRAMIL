# Diagnostic: permutation null for bag_svm

Subject labels permuted once per replicate (seeds 1000..1199), outer folds kept from the true labels (3x5, subject-disjoint), inner-CV (kernel, C) selection unchanged, mean val AUROC over 15 units. 141 training bags only (test split untouched). 200 permutations each, ~3 s/run.

| acoustic | observed | null mean | null sd | null 95th pct | n | p (one-sided) |
| --- | --- | --- | --- | --- | --- | --- |
| wavlm | 0.619 | 0.502 | 0.060 | 0.613 | 200 | 0.0348 |
| egemaps | 0.451 | 0.499 | 0.064 | 0.608 | 200 | 0.7761 |
