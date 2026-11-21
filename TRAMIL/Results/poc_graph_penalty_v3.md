# Synthetic PoC v3: favorable-state existence proof of the r12 bounded acoustic residual

**Existence proof only.** Passing says nothing about E-DAIC performance; failing means the r12 mechanism is broken in its best case. Synthetic data; PoC settings (1/tau_a = kappa/tau_e = 5, mu grid, 5 seeds, 256/64/64 bags) are not proposal values.

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

## Per-arm results (test bags, mean +- sd over 5 seeds)

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

‖G^a − G‖ of prior-only uses the untrained W_a (not part of any criterion).

## Verdicts (pre-registered F1-F4)

- F1 combination: mu* per seed [10.0, 0.0, 1.0, 1.0, 0.1]; test loss at mu* 1.7477 vs prior-only 1.7808 and acoustic-only 1.7734 (needs below both). **PASS**
- F2 correction: neighbour mass on disagreeing segments at mu* 0.0478 vs prior-only 0.0280 (needs higher). **PASS**
- F3 alignment: ‖G^a − G‖ per mu [0.5581, 0.5522, 0.5068, 0.4465, 0.4129, 0.41], Spearman -1.000 (needs <= -0.9). **PASS**
- F4 bounded: max|Delta| 4.983563 < 5, max|diag Delta| 0, Pi band holds in 40/40 runs. **PASS**

**Overall: PASS**

Pre-registered reading: PASS -> experiments protocol; FAIL -> revise the mechanism before any real run.

## Post-hoc observations (computed after the run; not criteria; verdicts unchanged)

- No arm learns the item task: test MSE of every arm (1.69-1.78) is above that of the constant training-mean predictor, 1.2525 +- 0.0301 (per seed [1.2842, 1.2795, 1.2221, 1.2552, 1.2214]). The F1 comparison is therefore between models that all predict worse than a constant.
- F1 margins are within seed noise: paired test loss mu* - prior-only per seed [-0.0453, -0.2741, 0.1087, -0.0522, 0.0975] (3/5 negative); mu* - acoustic-only [-0.0734, -0.017, -0.0146, 0.0405, -0.064] (4/5 negative); seed sd of each arm ~0.06-0.15.
- F2 neighbour mass is low in absolute terms: a uniform row would give 0.1201; mu* gives 0.0478, prior-only 0.0280, acoustic-only 0.1618. The correction moves mass toward acoustic neighbours relative to the prior, but stays below uniform, and neighbour mass falls monotonically with mu.
- F4: max|Delta| reached 4.9836 of 5 (nearly parallel centered acoustic rows); the strict bound rests on eps, which is why evaluation is in float64.
