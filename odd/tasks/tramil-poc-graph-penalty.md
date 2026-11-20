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
- Commit: see `git log` (feat: synthetic PoC ...); hash recorded in the follow-up docs commit.

## Next step
T3: operator reads the report and decides whether P2's failure (Delta does not learn acoustic structure on this toy)
blocks full E-DAIC tests. Criteria were not changed after the run.

Engram mirror: pending (session bound to project `loq`, write to `tcamil_extension` refused).
