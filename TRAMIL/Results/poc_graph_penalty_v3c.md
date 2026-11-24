# Synthetic PoC v3c (running)

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

Readings of the pre-registered T7 spec, fixed before any T7 run:
- Step 1 uses a = 0.5 and the graph-free model (H~ = H: encoder, attention pool W_o, item heads), batch 8, Adam.
  Test is never read in Step 1 (only train and val bags are generated).
- D1: the first 16 train bags of seed 0; v3 config (lr 1e-2, W_o = 0), 400 epochs; train MSE on those 16 bags after
  every epoch. "Well below the constant" read as min train MSE <= 0.5 x the constant's train MSE on the same bags.
  If D1 fails, D2 is not run and the readout defect is reported.
- D2: grid lr {1e-3, 3e-3, 1e-2} x W_o init {zero, normal}; 5 seeds, full 256 train bags; up to 400 epochs, val
  item MSE after every epoch, best epoch on val (ties -> earliest) per run. Config score = seed mean of the best
  val MSE; the config with the lowest score is chosen. "N(0, 0.01)" read in the document's N(mean, variance)
  notation (as N(0, 0.005^2) for the prompt noise): sd 0.1.
- Gate V3: chosen config's score < constant val MSE (seed mean of the per-item training-mean predictor on val).
  V3 fails -> v3c is not run.
- v3c "training config" = chosen lr and W_o init, with the per-run rule "up to 400 epochs, best epoch on val (item
  MSE without the mu term)" for EVERY arm (full per mu, prior-only, acoustic-only). mu* = argmin over mu of the
  val item MSE at each run's best epoch. ORACLE-SEG is not computed in v3c (V2 now uses ORACLE-BAG).
- ORACLE-BAG: per bag, the mean of all raw segment features X (no item routing), the same vector for all 8 items;
  one OLS head (with intercept) per item, fit on train (gelsd, relative rcond 1e-6, as ORACLE-NB).
- Runs are independent (each seeds itself) and are executed in parallel worker processes with 1 torch thread each;
  results do not depend on the parallelism.
