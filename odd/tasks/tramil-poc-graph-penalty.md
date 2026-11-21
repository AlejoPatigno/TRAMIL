# TRAMIL synthetic PoC: does the Delta penalty pull the learned segment graph toward the text prior?

## Objective
Before full E-DAIC runs, test on synthetic bags whether the penalty mu*||Delta_b||^2/N_b^2 (r11 Ec. 72, 73, 77)
moves the learned segment attention Pi_b toward the prompt-prior graph G_b (Ec. 70), while at small mu the
graph still follows the acoustic structure (acoustic-centered). Full E-DAIC tests only if the PoC passes.

## Operator decisions (2026-11-20)
- Keep the r11 design (prior + penalized acoustic residual); no acoustic-first redesign.
- Judge r11 W1/W2 -> experiments protocol: add ABMIL-8 and Symptom-Guided as paired SS^omega comparators.
- Judge r11 W3-W7 -> mathematical fixes in a future proposal revision (r12).

## Scope
- New module `src/TRAMIL/poc_graph_penalty.py` + `tests/test_poc_graph_penalty.py`; report in
  `TRAMIL/Results/poc_graph_penalty.md` (+ .json). Synthetic data only; no E-DAIC, no Kaggle, test split untouched.
- mu, kappa/tau_e grids here are PoC sweep settings on synthetic data, NOT proposal values.

## Pre-registered pass criteria (fixed before any run)
- P1 pull: in the disagree scenario, mean row JS(Pi, Pi_G) decreases with mu (Spearman over the grid <= -0.9,
  mean over seeds) and at the largest mu JS(Pi, Pi_G) < JS(Pi, Pi_A).
- P2 acoustic-centered: in the disagree scenario, at mu = 0, JS(Pi, Pi_A) < JS(Pi, Pi_G) and the task loss is
  below the prior-only (Delta = 0) loss.
- P3 no harm: in the agree scenario, the task loss at every mu is no worse than prior-only + 5%.
PoC passes iff P1, P2, P3 all hold. Otherwise report which failed; do not tune criteria after seeing results.

## Tasks
- [x] T1 synthetic generator + graph model + alignment metrics, TDD (route: delegated writer, 2 files)
- [x] T2 run sweep (scenarios agree/partial/disagree x mu grid x 5 seeds), write report (route: same writer)
- [ ] T3 operator reads report; decide full E-DAIC tests

## Checks
TDD strict. Runner: `.venv/Scripts/python.exe -m pytest -q --basetemp=.pytest_tmp` (from implementations/TRAMIL).

## Progress / evidence
- T1: `src/TRAMIL/poc_graph_penalty.py`, `tests/test_poc_graph_penalty.py` (5 tests). RED observed (ImportError on
  collection), then GREEN 5 passed. Full suite: 92 passed, 3 skipped. TDD strict, runner as in Checks.
  Delta parametrized as Ec. 71 (scaled dot product, W_q, W_k), D = 16 toy encoder.
- T2: sweep 3 scenarios x mu {0, 1e-3, ..., 1e3} + prior-only x 5 seeds, kappa/tau_e = 5 (PoC setting), ~11 min CPU.
  Task: per-segment 8-way classification of z_ac (10% train label noise, clean held-out labels), head on H~.
  Report `TRAMIL/Results/poc_graph_penalty.md` (+ .json).
  - P1 PASS: Spearman(mu, JS(Pi,Pi_G)) = -1.000 every seed; at mu = 1e3 JS to G 0.0000 < JS to A 0.4374.
  - P2 FAIL: disagree, mu = 0: JS to A 0.4362 is NOT < JS to G 0.2956, and loss 1.0805 is NOT below prior-only 1.0161.
  - P3 PASS: agree, max loss over mu 0.5259 <= 1.05 x prior-only 0.5361.
  - Overall: FAIL (P2). At mu = 0 the learned Delta moves Pi away from the prior but not toward the acoustic
    clusters (JS to A unchanged vs prior-only) and overfits; the penalty pull itself works as intended.
- Commit: 82db92f (feat: synthetic PoC of the Delta penalty ...).

## v2: acoustic SNR sweep (operator go 2026-11-20; criteria fixed before any v2 run)
Why: v1 P2 failed; unknown whether the toy was too noisy (cluster-mean sd 0.5 vs noise sd 1) or Delta cannot learn
acoustic structure. Same model, task, kappa/tau_e = 5, seeds, scenarios as v1. Only the cluster-mean sd s varies.
- Grid: s in {0.5, 1, 2, 4} (noise sd 1). mu in {0, 1e-2, 0.1, 1, 1e3} + prior-only. 5 seeds.
- Q1 mechanism: at some s, v1's P2 holds (disagree, mu = 0: JS(Pi,Pi_A) < JS(Pi,Pi_G) AND loss < prior-only),
  seed means. Report the smallest such s.
- Q2 dose-response: disagree, mu = 0, JS(Pi,Pi_A) decreases with s (Spearman over s, seed means, <= -0.8).
- Q3 v1's P1 and P3 hold at every s.
Reading: Q1 & Q2 & Q3 -> v1 failure was the scenario; proceed to plan full E-DAIC tests. Q1 fails -> Delta does not
learn acoustic structure even with strong signal -> mathematical fix in r12 before E-DAIC.
- [x] T4 v2 sweep + report `TRAMIL/Results/poc_graph_penalty_v2.md` (+ .json) (route: delegated writer)
  - Code: `make_data(..., s=0.5)` (default reproduces v1, fingerprint test) + `run_v2()` in the same module.
    TDD: RED (TypeError, 2 new tests) -> GREEN. Metric bug found by the first v2 run: `row_js` gave NaN when Pi and
    Pi_A both have exact zeros (mu = 0, s >= 2); RED test -> guard -> GREEN; full sweep rerun, non-NaN cells
    bit-identical. Tests: module 8 passed; full suite 95 passed, 3 skipped. ~30 min CPU per sweep.
  - Readings stated before running (in the report header): P1 Spearman over the 5-point v2 mu grid per seed, then
    averaged; P3 over every v2 mu; Q2 Spearman over 4 s values of seed-mean JS(Pi,Pi_A).
  - Q1 FAIL: P2 holds at no s. Disagree, mu = 0, JS A vs G / loss vs prior-only: s=0.5 0.436 vs 0.296 / 1.081 vs
    1.016; s=1 0.434 vs 0.336 / 0.310 vs 0.305; s=2 0.540 vs 0.470 / 0.137 vs 0.139; s=4 0.627 vs 0.568 / 0.1128 vs 0.1130.
  - Q2 FAIL: JS(Pi,Pi_A) per s [0.436, 0.434, 0.540, 0.627], Spearman +0.800 (needs <= -0.8; wrong direction).
  - Q3 FAIL: P1 passes at every s; P3 fails at s=1 (0.2308 > 0.2015) and s=2 (0.1449 > 0.1438).
  - Reading (pre-registered): Q1 fails -> Delta does not learn acoustic structure even with strong signal ->
    mathematical fix in r12 before E-DAIC. Observation: unpenalized Delta grows with s (delta_rms 5 -> 23) and
    sharpens Pi without acoustic alignment; prior-only loss falls with s, so aggregation carries little task signal.
  - Commit: 0e5fb92 (feat(poc): add acoustic SNR sweep (v2) ...).

## v3: favorable-state existence proof of the r12 mechanism (operator go 2026-11-20; fixed before any v3 run)
Existence proof only: passing says nothing about E-DAIC performance; failing means the r12 mechanism is broken in
its best case. Model = r12 exactly: Delta = (1/tau_a) cos(bag-centered W_a h rows), W_a 32x32, masked diagonal,
same eps as Ec. 70; Gamma = Delta + (kappa/tau_e) G; Pi row softmax; H~ = H + tanh(Pi H W_g); alignment penalty
mu ||(G^a - G) ⊙ (11^T - I)||_F^2 / N^2 with gradient to W_a only (Ec. 71, 77, 104-106 of r12).
Favorable scenario: s = 0.5, noise sd 1 (single segment weak, neighbour average informative); text/acoustic latent
agree on 75% of segments; cluster means in 4 of 16 dims, 12 nuisance dims; 256 train / 64 val / 64 test bags,
N ~ U{20..60}; each bag has 8 item severities, segments of item j carry noisy signal of severity j; item heads + MSE.
PoC settings (not proposal values): 1/tau_a = kappa/tau_e = 5; mu in {0, 0.01, 0.1, 1, 10, 1e3}; 5 seeds;
mu* chosen on val, all criteria on test.
- F1 combination: test loss at mu* < prior-only (Delta = 0) AND < acoustic-only (kappa = 0).
- F2 correction: on the 25% disagreeing segments, row mass of Pi(mu*) on true acoustic neighbours > that of
  prior-only Pi.
- F3 alignment: ||G^a - G|| decreases with mu (Spearman <= -0.9, seed means).
- F4 bounded: |Delta| < 1/tau_a and Pi_ik within the Ec. 105 band, checked numerically.
Pass iff F1-F4 (seed means). Pass -> experiments protocol; fail -> revise mechanism before any real run.
- [x] T5 v3 favorable-state run + report `TRAMIL/Results/poc_graph_penalty_v3.md` (+ .json) (route: delegated writer)
  - Code: sibling module `src/TRAMIL/poc_graph_penalty_v3.py` (reuses `prior_graph`/centering and SEEDS from v1; v1/v2
    untouched, fingerprint test still green). Tests in `tests/test_poc_graph_penalty.py` (+7). TDD: RED (ImportError on
    collection) -> GREEN. Module 15 passed; full suite 102 passed, 3 skipped. ~35 min CPU.
  - Readings written to the report header (and to disk) before the run: toy encoder 16->32 ReLU; severities uniform
    {0..3}, beta = 0.5 along the cluster-mean direction; readout = learned per-item attention pool (W_o = 0 init) +
    item heads, prior only via G; Adam lr 1e-2, 40 epochs (v1 loop); acoustic-only uses mu = 0; F2 = mass on k != i
    with same z_ac, pooled over disagreeing test segments; F3 Spearman over 6 mu of seed means; F4 per arm gamma,
    evaluated in float64 (float32 cannot resolve eps = 1e-8).
  - F1 PASS: mu* per seed [10, 0, 1, 1, 0.1]; test MSE 1.7477 < prior-only 1.7808 and < acoustic-only 1.7734.
  - F2 PASS: neighbour mass on disagreeing segments 0.0478 (mu*) > 0.0280 (prior-only).
  - F3 PASS: ||G^a - G|| 0.558 -> 0.410 over mu, Spearman -1.000.
  - F4 PASS: max|Delta| 4.9836 < 5, diag 0, Ec. 105 band in 40/40 runs.
  - Overall: PASS (pre-registered reading -> experiments protocol). Post-hoc caveats (not criteria, in the report):
    every arm's test MSE (1.69-1.78) is worse than the constant training-mean predictor (1.25), F1 paired margins
    change sign across seeds (3/5 vs prior-only, 4/5 vs acoustic-only), and F2 mass at mu* stays below a uniform
    row (0.120); acoustic-only reaches 0.162. The task head does not learn; the pass rests on graph quantities.
  - Run-time fix: md write failed on Windows cp1252 (U+2016); added utf-8 encoding and regenerated the md from the
    saved JSON (verdicts recomputed and asserted identical); no rerun.

## Next step
v3 (T5): F1-F4 PASS (existence proof); post-hoc, no arm beats a constant predictor on the item task. Operator decides
whether to proceed to the experiments protocol or first make the toy task learnable.

v2 (T4): Q1/Q2/Q3 all FAIL; the pre-registered reading routes to an r12 mathematical fix before E-DAIC. Operator decides.

v1 (historical): T3: operator reads the report and decides whether P2's failure (Delta does not learn acoustic structure on this toy)
blocks full E-DAIC tests. Criteria were not changed after the run.

Engram mirror: pending (session bound to project `loq`, write to `tcamil_extension` refused).
