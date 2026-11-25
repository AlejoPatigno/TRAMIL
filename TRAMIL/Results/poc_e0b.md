# PoC E0b: redesigned text-distilled acoustic prior on synthetic bags

**Existence proof only; synthetic data.**

Readings of the pre-registered E0b spec, fixed before any E0b run (changes from E0 marked CHANGED):
- Generator and model: E0's, unchanged (poc_e0 docstring and READINGS). Bags are paired across strengths.
- CHANGED strengths: rho in {0, 0.5, 1, 2}; primary rho = 1 (P1, P2, P3); P4 at rho = 0; all strengths reported.
- CHANGED penalty (Ec. 77): mu_R ||Rhat_c - Rc||_F^2 / (8 N_b vbar), vbar = mean of Rc^2 over every TRAIN segment and
  item (does not depend on rho), so predicting Rhat_c = 0 scores about 1. Surrogate error on test is normalized the same
  way (1 = the zero predictor).
- CHANGED training: up to 400 epochs, early stop after 40 epochs without a val improvement; best val epoch restored
  (val unweighted item MSE without the penalty, ties -> earliest).
- CHANGED selection: lr is chosen on val per (arm, rho, seed) from {1e-3, 3e-3, 1e-2}, jointly with mu_R in
  {0.1, 1, 10} for the full arm (ties -> smaller lr, then smaller mu_R). W_o init comes from gate 0.
- Gate 0: graph-free model at rho = 1, lr x W_o grid, 5 seeds, val only; keeps W_o init; PASS iff the best seed-mean
  val MSE is below the constant's. FAIL -> INVALID.
- Arms: full (surrogate, mu_R > 0), untargeted (surrogate, mu_R = 0), r12 (text at inference), constant.
- Metrics: SS^w (Ec. 93); normalized surrogate error; AUROC; per-item Spearman of Rhat_c vs Rc pooled over the test
  segments of a seed, and its mean over the 8 items (spearman_mean).
- CHANGED tests (seed = unit, n = 10, alpha 0.05, Holm over P1, P2a, P2b, P2c): P1 one-sample Wilcoxon full SS^w > 0;
  P2a paired Wilcoxon full > untargeted SS^w; P2b full < untargeted surrogate error; P2c full > untargeted
  spearman_mean (prior recovery); rank-biserial effect size; percentile bootstrap CI (B = 10000). P4: bootstrap CI of
  full - untargeted SS^w at rho = 0 contains zero. P3: estimate of full - r12 SS^w at rho = 1. Dose response: full
  spearman_mean by rho, descriptive.
- Verdicts: P1 supported iff adjusted p < alpha; P2 supported iff P2a, P2b and P2c pass, partially supported iff one or
  two pass; H8 carried forward iff P1, P2, P4 supported.
- smoke mode (rehearsal only): 2 seeds, 32/16/16 bags, 3 epochs, reduced grids; runs the arms even if gate 0 fails.

## Gate 0 (graph-free, rho 1, val only)

| Configuration | Seed-mean val MSE | Constant val MSE | Best epochs | Kept |
|---|---|---|---|---|
| lr=0.001, W_o=zero | 1.2088 | 1.2642 | [34, 25, 29, 32, 15] |  |
| lr=0.001, W_o=normal | 1.2427 | 1.2642 | [20, 24, 25, 27, 22] |  |
| lr=0.003, W_o=zero | 1.1952 | 1.2642 | [12, 15, 11, 11, 12] | yes |
| lr=0.003, W_o=normal | 1.2256 | 1.2642 | [7, 7, 8, 11, 8] |  |
| lr=0.01, W_o=zero | 1.1980 | 1.2642 | [4, 5, 6, 3, 3] |  |
| lr=0.01, W_o=normal | 1.2287 | 1.2642 | [4, 9, 6, 9, 5] |  |

**Gate 0: PASS**

## Arms (test bags, seed mean)

| Arm, rho | SS^w mean | SS^w sd | Norm. surrogate error | Mean per-item Spearman | AUROC | Median best epoch |
|---|---|---|---|---|---|---|
| full, 0 | 0.0388 | 0.0206 | 1.0988 | -0.004 | 0.8010 | 16 |
| untargeted, 0 | 0.0388 | 0.0224 | 52.1034 | 0.000 | 0.8024 | 9 |
| r12, 0 | 0.1360 | 0.0387 | - | - | 0.8132 | 16 |
| constant, 0 | 0.0000 | 0.0000 | - | - | 0.5000 | - |
| full, 0.5 | 0.0419 | 0.0172 | 1.0687 | 0.045 | 0.8028 | 13 |
| untargeted, 0.5 | 0.0380 | 0.0275 | 65.5396 | 0.001 | 0.8030 | 8 |
| r12, 0.5 | 0.1401 | 0.0715 | - | - | 0.8164 | 17 |
| constant, 0.5 | 0.0000 | 0.0000 | - | - | 0.5000 | - |
| full, 1 | 0.0489 | 0.0391 | 1.0128 | 0.139 | 0.8015 | 14 |
| untargeted, 1 | 0.0442 | 0.0291 | 62.8815 | 0.021 | 0.8074 | 13 |
| r12, 1 | 0.2046 | 0.0462 | - | - | 0.8140 | 11 |
| constant, 1 | 0.0000 | 0.0000 | - | - | 0.5000 | - |
| full, 2 | 0.2419 | 0.0414 | 0.6653 | 0.317 | 0.8194 | 18 |
| untargeted, 2 | 0.1247 | 0.0487 | 185.9807 | 0.173 | 0.7711 | 13 |
| r12, 2 | 0.3121 | 0.0295 | - | - | 0.8018 | 12 |
| constant, 2 | 0.0000 | 0.0000 | - | - | 0.5000 | - |

## Comparisons (seed = unit)

| Comparison | Test | Statistic | Raw p | Adjusted p | Correction | Effect size | CI | Mean diff | n | alpha |
|---|---|---|---|---|---|---|---|---|---|---|
| P1 | one-sample Wilcoxon, full SS^w > 0 | 52.00 | 0.0049 | 0.0098 | Holm | 0.891 | [0.0262, 0.0717] | 0.0489 | 10 | 0.05 |
| P2a | paired Wilcoxon, full > untargeted SS^w | 32.00 | 0.3477 | 0.3477 | Holm | 0.164 | [-0.0194, 0.0296] | 0.0047 | 10 | 0.05 |
| P2b | paired Wilcoxon, full < untargeted normalized surrogate error | 55.00 | 0.0010 | 0.0039 | Holm | 1.000 | [-72.7556, -51.5245] | -61.8687 | 10 | 0.05 |
| P2c | paired Wilcoxon, full > untargeted mean per-item Spearman | 55.00 | 0.0010 | 0.0039 | Holm | 1.000 | [0.0995, 0.1366] | 0.1184 | 10 | 0.05 |
| P4 | bootstrap CI of paired full - untargeted SS^w at rho 0 | - | - | - | none | - | [-0.0093, 0.0090] | -0.0000 | 10 | 0.05 |
| P3 | estimate, full - r12 SS^w at rho 1 (no threshold) | - | - | - | none | - | [-0.1931, -0.1246] | -0.1557 | 10 | 0.05 |

## Verdicts

- P1: supported
- P2: partially supported
- P4: supported
- P3: estimate only
- H8_carried_forward: False
- gate0: PASS
