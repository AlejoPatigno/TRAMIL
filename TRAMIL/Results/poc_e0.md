# PoC E0: text-distilled acoustic prior on synthetic bags

**Existence proof only; synthetic data; PoC settings, not proposal values.**

Readings of the pre-registered E0 spec, fixed before any E0 run:
- Generator: see the module docstring. Paired across strengths: bags at rho = 0.5 and rho = 0 share every draw.
- Surrogate (Ec. 114-115): Rhat = sigmoid(W_r H + b_r), Rhat_c = Rhat - column mean over the bag; Ghat = cosine of the
  centered rows of Rhat (prior_graph, eps 1e-8). Graph (Ec. 72): Delta = (1/tau_a) prior_graph(W_a H) * offdiag,
  Pi = rowsoftmax(Delta + (kappa/tau_e) E), E = Ghat (surrogate) or G (text arm), H~ = H + tanh(W_g Pi H); 1/tau_a =
  kappa/tau_e = 5. Prior "none" is the graph-free Gate-0 model (H~ = H, no prior anywhere).
- Readout (Ec. 18): L = W_o H~ + (lam/tau) P with P = Rhat_c (surrogate) or Rc (text), tau = 0.10, lam learnable scalar
  initialised at 0.1; A = colsoftmax(L); s_j = head_j . (A[:, j]^T H~) + c_j. W_o starts at zero or N(0, sd 0.1).
  The surrogate forward never reads R, Rc or G.
- Loss (Ec. 77): omega_b mean_j (s^ - s)^2 + mu_R ||Rhat_c - Rc||_F^2 / (8 N_b), batch mean, omega_plus = M0/M1 from the
  TRAIN bags, omega_b = omega_plus if y_b else 1. mu_R only for the surrogate arms.
- Model selection: best epoch on val unweighted item MSE without the penalty (ties -> earliest), weights restored.
  Up to 400 epochs, Adam, batch 8, with the Gate-0 chosen lr and W_o init for every arm.
- Gate 0: prior none, rho = 0.5, lr x W_o-init grid, 5 seeds, val only (test bags never generated). Chosen = lowest
  seed-mean best val MSE; PASS iff below the seed-mean constant val MSE (unweighted train item mean). FAIL -> INVALID, no
  arm trained.
- Arms per (rho, seed): full = surrogate with mu_R* chosen on val over {0.1, 1, 10} (ties -> smaller); untargeted =
  surrogate mu_R = 0; r12 = text arm; constant = omega-weighted train item mean. Test metrics of a run are computed
  when it ends but no choice ever reads them; each run is cached and resumable.
- SS^w (Ec. 93) on test with omega and sbar_w from train. Surrogate error = mean over test bags of
  ||Rhat_c - Rc||_F^2 / (8 N). AUROC of sum_j s^_j against y. Per-item Spearman of Rhat_c vs Rc pooled over the test
  segments of one seed.
- Tests (seed = unit, n = 10, alpha 0.05): P1 one-sample Wilcoxon (greater) of full SS^w at rho 0.5 against 0; P2a paired
  Wilcoxon full > untargeted SS^w; P2b paired Wilcoxon full < untargeted surrogate error; Holm over the three; effect
  size = rank-biserial (positive = hypothesis direction); CI = percentile bootstrap (B = 10000, seeded) of the seed-mean
  difference (P1: of the mean). P4: bootstrap CI of full - untargeted SS^w at rho 0, supported iff it contains zero. P3:
  estimate and CI of full - r12 SS^w at rho 0.5, no threshold and no verdict.
- Verdicts: P1 supported iff adjusted p < alpha; P2 supported iff both P2a and P2b adjusted p < alpha, partial iff one,
  else not supported; inconclusive if gate 0 fails or a unit is missing. H8 carried forward iff P1, P2, P4 supported.
- smoke mode (rehearsal only): 2 seeds, 32/16/16 bags, 3 epochs, one gate seed and config; it runs the arms even when
  the gate verdict is FAIL so the whole pipeline is exercised, and the report says so.

## Gate 0 (graph-free model, rho 0.5, val only)

| Gate 0 configuration | Seed-mean validation MSE | Constant validation MSE | Kept |
|---|---|---|---|
| lr=0.001, W_o=zero | 1.2287 | 1.2642 |  |
| lr=0.001, W_o=normal | 1.2523 | 1.2642 |  |
| lr=0.003, W_o=zero | 1.2226 | 1.2642 |  |
| lr=0.003, W_o=normal | 1.2387 | 1.2642 |  |
| lr=0.01, W_o=zero | 1.2199 | 1.2642 | yes |
| lr=0.01, W_o=normal | 1.2486 | 1.2642 |  |

**Gate 0: PASS** (chosen lr=0.01, W_o=zero: 1.2199 vs constant 1.2642)

## Arms (test bags, seed mean +- sd)

| Arm at strength | SS omega mean | SS omega sd | Surrogate error mean | AUROC mean |
|---|---|---|---|---|
| Full, rho 0.5 | 0.0244 | 0.0316 | 0.0015 | 0.8099 |
| Untargeted, rho 0.5 | 0.0159 | 0.0397 | 0.0102 | 0.8025 |
| r12 reference, rho 0.5 | 0.1401 | 0.0715 | - | 0.8164 |
| Constant, rho 0.5 | 0.0000 | 0.0000 | - | 0.5000 |
| Full, rho 0 | 0.0089 | 0.0392 | 0.0047 | 0.7963 |
| Untargeted, rho 0 | 0.0133 | 0.0344 | 0.0087 | 0.7962 |
| r12 reference, rho 0 | 0.1360 | 0.0387 | - | 0.8132 |
| Constant, rho 0 | 0.0000 | 0.0000 | - | 0.5000 |

## Comparisons (seed is the unit; CI = percentile bootstrap of the seed-mean difference)

| Comparison | Test | Statistic | Raw p | Adjusted p | Correction | Effect size | CI | Mean diff | n | alpha |
|---|---|---|---|---|---|---|---|---|---|---|
| P1 | one-sample Wilcoxon, full SS^w > 0 | 45.00 | 0.0420 | 0.0488 | Holm | 0.636 | [0.0038, 0.0401] | 0.0244 | 10 | 0.05 |
| P2a | paired Wilcoxon, full > untargeted SS^w | 47.00 | 0.0244 | 0.0488 | Holm | 0.709 | [-0.0013, 0.0202] | 0.0084 | 10 | 0.05 |
| P2b | paired Wilcoxon, full < untargeted surrogate error | 55.00 | 0.0010 | 0.0029 | Holm | 1.000 | [-0.0118, -0.0059] | -0.0087 | 10 | 0.05 |
| P4 | bootstrap CI of paired full - untargeted SS^w at rho 0 | - | - | - | none | - | [-0.0219, 0.0119] | -0.0045 | 10 | 0.05 |
| P3 | estimate, full - r12 SS^w at rho 0.5 (no threshold) | - | - | - | none | - | [-0.1630, -0.0708] | -0.1157 | 10 | 0.05 |

## Verdicts

- P1: supported
- P2: supported
- P4: supported
- P3: estimate only
- H8_carried_forward: True
- gate0: PASS
