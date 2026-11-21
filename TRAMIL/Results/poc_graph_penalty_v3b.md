# Synthetic PoC v3b: favorable state with a validity gate (r12 bounded acoustic residual)

**Existence proof only.** Passing says nothing about E-DAIC performance. Synthetic data; PoC settings are not proposal values.

Readings of the pre-registered spec, fixed before any v3 run:
- Encoder: one linear 16->32 + ReLU (non-negative H as in Ec. 14), trained jointly; W_a, W_g 32x32 without bias.
- Item task: severities s_bj uniform on {0,1,2,3}; the segment signal is beta (s - 1.5) along the unit direction of
  its acoustic cluster mean, beta = 0.5 (same scale as s); the 25% disagreeing segments carry the severity of their
  acoustic item. Readout: learned per-item attention pooling A = colsoftmax(H~ W_o) (W_o = 0 at init, as in r12)
  plus one linear head per item; the prompt prior enters only through G (no prior term in the readout), so the
  graph is the only place the text can act. MSE over the 8 items, unweighted (no binary label, omega_b = 1).
- Optimizer and length reused from v1/v2: Adam lr 1e-2, 40 epochs, batch 8; no epoch selection.
- Losses compared (val and test) are the item MSE without the mu term (as Ec. 39 selects without it).
- mu* per seed = argmin val loss over the mu grid (ties -> smaller mu); F1/F2 use that seed's test value at mu*.
- Arms: full model per mu; prior-only = Delta fixed to 0 (W_a unused); acoustic-only = kappa = 0 with mu = 0
  (r12: a graph without the prior must not receive it through the penalty).
- F2: on test segments with z_text != z_ac, row mass of Pi on true acoustic neighbours = sum over k != i with
  z_ac(k) = z_ac(i) of Pi_ik; pooled over all disagreeing segments of the test bags, then mean over seeds.
- F3: ||G^a - G|| = ||(G^a - G) * (11^T - I)||_F / N per test bag, mean over bags; Spearman over the 6-point mu grid
  of the seed means. For prior-only the value uses the untrained W_a and is shown for completeness only.
- F4: every arm and seed, every test bag: max|Delta| < 1/tau_a, diag(Delta) = 0, and
  exp(-2 gamma) < N Pi_ik < exp(2 gamma) with the arm's gamma (full 1/tau_a + kappa/tau_e; prior-only kappa/tau_e;
  acoustic-only 1/tau_a). Evaluation is in float64: in float32, eps = 1e-8 is below the resolution of ||p||^2, so a
  cosine of two nearly parallel rows can round to 1 and the strict bound would fail by rounding, not by the math.
- Overall PASS iff F1, F2, F3, F4 all hold on seed means.

Readings of the pre-registered v3b spec, fixed before any v3b run (v3 readings above still apply):
- Gates are evaluated on VAL (seed means over 5 seeds), as the spec selects "the smallest a passing both gates on
  val"; the spec's V1/V2 wording says "test MSE", read as the same ratios computed on val for the decision. Test
  values of the gate quantities are reported alongside and decide nothing.
- Only a is calibrated (grid {0.5, 1, 2, 4}); the cluster-mean sd stays s = 0.5 (no grid was given for it).
- Constant predictor: per-item training mean of the severities, applied to every bag.
- ORACLE-NB: per bag and item j, the mean of the raw segment features X (16 dims) over segments with z_ac = j
  (true acoustic same-item neighbours; zero vector when the item has no segment in the bag); one ordinary
  least-squares linear head (with intercept) per item, fit on train in closed form (gelsd, relative rcond 1e-6,
  so float32 rounding directions of a noiseless design are not fitted; it does not bind on noisy data).
- ORACLE-SEG: the v3 model with the graph removed (H~ = H; encoder, attention pool W_o and item heads only),
  trained on train with the v3 loop (Adam lr 1e-2, 40 epochs, batch 8, same seed). It knows no neighbours.
- V1: NB <= 0.8 x constant; V2: NB <= 0.9 x SEG (both inclusive). No passing a -> INVALID; no model arm is trained.
- Model arms are trained only at the chosen a; the oracles never see a model arm and a model arm never informs a.
- F1' = v3 F1 and test MSE at mu* < constant test MSE (seed means). F2b: F2 mass at mu* > uniform-row expectation
  = mean over disagreeing test segments (pooled) of #(k != i with z_ac(k) = z_ac(i)) / (N - 1), seed mean.
- Overall: INVALID if no a passes; else PASS iff F1', F2, F2b, F3, F4 (seed means).

## Validity gate (oracles only; decided on val, seed means; test shown, decides nothing)

| a | constant val | ORACLE-NB val | ORACLE-SEG val | NB/const (V1 <= 0.8) | NB/SEG (V2 <= 0.9) | V1 | V2 | constant test | NB test | SEG test |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.5 | 1.2566 | 0.6785 | 1.7242 | 0.540 | 0.394 | pass | pass | 1.2525 | 0.6500 | 1.7768 |
| 1 | 1.2566 | 0.2754 | 1.1697 | 0.219 | 0.235 | pass | pass | 1.2525 | 0.2692 | 1.2931 |
| 2 | 1.2566 | 0.0903 | 0.7791 | 0.072 | 0.116 | pass | pass | 1.2525 | 0.0931 | 0.8535 |
| 4 | 1.2566 | 0.0345 | 0.2687 | 0.027 | 0.128 | pass | pass | 1.2525 | 0.0389 | 0.2762 |

**Chosen a: 0.5**

## Per-arm results at a = 0.5 (test bags, mean +- sd over 5 seeds)

| arm | test loss (item MSE) | val loss | F2 nbr mass, disagreeing | ‖G^a − G‖ | max\|Delta\| | Pi band ok |
|---|---|---|---|---|---|---|
| mu = 0 | 1.7309 +- 0.1067 | 1.7772 +- 0.1290 | 0.0602 +- 0.0081 | 0.5581 +- 0.0217 | 4.9454 | 5/5 |
| mu = 0.01 | 1.7846 +- 0.1129 | 1.8421 +- 0.1364 | 0.0546 +- 0.0046 | 0.5522 +- 0.0160 | 4.9836 | 5/5 |
| mu = 0.1 | 1.7681 +- 0.1101 | 1.7074 +- 0.0492 | 0.0515 +- 0.0085 | 0.5068 +- 0.0139 | 4.8308 | 5/5 |
| mu = 1 | 1.7668 +- 0.1010 | 1.7553 +- 0.1321 | 0.0410 +- 0.0046 | 0.4465 +- 0.0017 | 4.7875 | 5/5 |
| mu = 10 | 1.6884 +- 0.0800 | 1.7203 +- 0.1074 | 0.0391 +- 0.0046 | 0.4129 +- 0.0029 | 4.4856 | 5/5 |
| mu = 1000 | 1.7549 +- 0.0643 | 1.7400 +- 0.1052 | 0.0370 +- 0.0034 | 0.4100 +- 0.0027 | 4.6377 | 5/5 |
| mu* (val-selected) | 1.7477 +- 0.0874 | 1.6505 +- 0.0833 | 0.0478 +- 0.0130 | 0.4760 +- 0.0594 | 4.9304 | 5/5 |
| prior-only (Delta = 0) | 1.7808 +- 0.1492 | 1.7329 +- 0.2157 | 0.0280 +- 0.0016 | 0.4632 +- 0.0040 | 0.0000 | 5/5 |
| acoustic-only (kappa = 0) | 1.7734 +- 0.1161 | 1.7068 +- 0.2054 | 0.1618 +- 0.0103 | 0.5492 +- 0.0096 | 4.9537 | 5/5 |
| constant (train item mean) | 1.2525 +- 0.0301 | 1.2566 +- 0.0273 | - | - | - | - |
| ORACLE-NB | 0.6500 +- 0.0493 | 0.6785 +- 0.0370 | - | - | - | - |
| ORACLE-SEG | 1.7768 +- 0.1759 | 1.7242 +- 0.1269 | - | - | - | - |
| uniform row (F2b reference) | - | - | 0.1232 +- 0.0018 | - | - | - |

‖G^a − G‖ of prior-only uses the untrained W_a (not part of any criterion).

## Verdicts (pre-registered F1', F2, F2b, F3, F4)

- F1' combination: mu* per seed [10.0, 0.0, 1.0, 1.0, 0.1]; test loss at mu* 1.7477 vs prior-only 1.7808, acoustic-only 1.7734 and constant 1.2525 (needs below all three). **FAIL**
- F2 correction: neighbour mass on disagreeing segments at mu* 0.0478 vs prior-only 0.0280 (needs higher). **PASS**
- F2b above uniform: 0.0478 vs uniform-row expectation 0.1232 (needs higher). **FAIL**
- F3 alignment: ‖G^a − G‖ per mu [0.5581, 0.5522, 0.5068, 0.4465, 0.4129, 0.41], Spearman -1.000 (needs <= -0.9). **PASS**
- F4 bounded: max|Delta| 4.983563 < 5, max|diag Delta| 0, Pi band holds in 40/40 runs. **PASS**

**Overall: FAIL**

Pre-registered reading: PASS -> experiments protocol.
