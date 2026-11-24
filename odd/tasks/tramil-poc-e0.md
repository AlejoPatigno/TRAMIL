# PoC E0: text-distilled acoustic prior on synthetic bags

Objective: implement and run E0 of experiments v05 (proposal r22, H8) on synthetic bags, placed on Kaggle.
Problem: the r12 generator leaks item identity into audio (cluster means, severity direction, 75% prompt/audio
agreement), so the rho = 0 negative control (P4) would not be a null.
Accepted change (operator, 2026-11-23): E0 generator variant -- item identity in audio only through rho * p_z
(orthonormal p_z in the 12 nuisance dims), severity along one shared direction w, no prompt/audio disagreement.
Deviation from v05 ("r12 generator extended") to be recorded in experiments v06.
TDD: strict (session config), runner `.venv/Scripts/python -m pytest`.
Scope: `src/TRAMIL/poc_e0.py`, `tests/test_poc_e0.py`, results `TRAMIL/Results/poc_e0.{json,md}`, timing probe.
Constraints: Gate 0 re-run on the new generator (T7 result does not carry over); test read once after val choices.

## Tasks
- [ ] T1 generator + model (Ec. 114-116, 72 with G^, 18, 77) + SS^w (Ec. 93) with tests -- route: delegated writer (2 non-trivial files)
- [ ] T2 runner: gate 0, arms x rho x seeds, stats (Wilcoxon one-sided, Holm, rank-biserial, bootstrap), report, resumable per-run cache, smoke mode, timing probe -- route: same writer
- [ ] T3 Kaggle job generate + preflight + rehearsal (operator authorizes launch)
- [ ] T4 Kaggle run, fetch, report, discussion.json

## Evidence
