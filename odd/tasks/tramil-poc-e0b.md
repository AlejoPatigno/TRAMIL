# Feature: tramil-poc-e0b — redesigned E0 (proposal r22, H8)

## Objective
Rerun the E0 proof of concept with a design that can test prior recovery. Read on synthetic bags only.

## Problem and why
E0 (odd/tasks/tramil-poc-e0.md, Results/poc_e0.*) has three problems:
- Its surrogate arms were restored at epoch 1-2. Gate 0 chose lr 0.01 on the graph-free model, and that rate does not suit the surrogate arms.
- Its penalty was negligible: Rc has variance ~1.2e-4, so the zero predictor scores 1.2e-4 while the full arm scored 1.5e-3.
- Per-item Spearman was ~0 for every arm, so P1/P2 "supported" were early-stopping artifacts.
Diagnosis on 2026-11-25: an oracle reaches Spearman 0.517. A linear ridge ceiling reaches 0.10 at rho 0.5, 0.19 at rho 1 and 0.31 at rho 2.

## Pre-registered design (fixed before any E0b run; full text in src/TRAMIL/poc_e0b.py READINGS)
- Generator and model are E0's. Strengths rho in {0, 0.5, 1, 2}: primary rho 1 for P1, P2 and P3; P4 at rho 0.
- Penalty normalized by vbar, the mean of Rc^2 over train, so the zero predictor scores ~1. Surrogate error is normalized the same way.
- lr is chosen per (arm, rho, seed) on val from {1e-3, 3e-3, 1e-2}, jointly with mu_R in {0.1, 1, 10} for the full arm. Early stop with patience 40, max 400 epochs.
- Holm over P1, P2a, P2b and P2c. P2c is new: full > untargeted on mean per-item Spearman. P2 is supported only when all three of P2a, P2b and P2c pass.
- Disclosure: before launch, two pipeline-timing runs (seed 0, rho 1, lr 1e-3 and 1e-2, mu 1) printed test metrics: Spearman 0.17 and 0.16, SS^w 0.074 and 0.048. No design choice was changed after seeing them.

## Tasks
- [x] T1 Module src/TRAMIL/poc_e0b.py and tests/test_poc_e0b.py. TDD: RED was a collection error (module missing); GREEN is 7 passed. The E0 suite still passes (14).
- [x] T2 Kaggle CPU job tools/kaggle/poc-e0b: smoke passed (smoke record pass, readiness true); full run submitted as alejopatio/papersmith-poc-e0b at pin 721f542, running. Smoke rehearsal, then the full run (authorized by the operator on 2026-11-25: "ponlo a ejecutar apenas lo tengas").
- [x] T3 Fetch, report, update Results/discussion.json, and record E0b in experiments v06. Run complete (~6.5 h), fetch current. P1 supported, P2 partially (P2a fails; P2b, P2c pass), P4 supported, H8 not carried forward. discussion check: complete.

## Checks
pytest tests/test_poc_e0b.py tests/test_poc_e0.py --basetemp=.pytest_tmp_e0b
- v06 published 2026-11-25 (experiments-...-v06.md, receipt PASS): E0 section replaced with the E0b design above.
