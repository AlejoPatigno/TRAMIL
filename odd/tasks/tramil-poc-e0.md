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
- [x] T1 generator + model (Ec. 114-116, 72 with G^, 18, 77) + SS^w (Ec. 93) with tests -- route: delegated writer (2 non-trivial files)
- [x] T2 runner: gate 0, arms x rho x seeds, stats (Wilcoxon one-sided, Holm, rank-biserial, bootstrap), report, resumable per-run cache, smoke mode, timing probe -- route: same writer
- [x] T3 Kaggle job generate + preflight + rehearsal (operator authorizes launch)
- [ ] T4 Kaggle run, fetch, report, discussion.json

## Evidence
- T1+T2 commit fe28b8e (`src/TRAMIL/poc_e0.py`, `tests/test_poc_e0.py`). Route: delegated writer. TDD strict, runner `.venv/Scripts/python -m pytest`.
  RED: collection ImportError (module absent). GREEN: 14 passed in tests/test_poc_e0.py; full suite 126 passed, 3 skipped.
  Env note: pytest needs `--basetemp=<writable dir>` here (default %TEMP%/pytest-of-LOQ gives PermissionError for every tmp_path test).
- Timing probe (`TRAMIL/Results/timing_e0.json`, 1 thread, warmed up): s/epoch none 0.151, surrogate 0.488, text 0.373.
  Extrapolated full design (30 + 80 + 20 runs): serial 20402 s at 400 epochs (5100 s at 100); /14 workers ideal 1457 s (364 s).
  Ideal gate estimate 129 s; MEASURED gate wall time on 16 cores was about 12-15 min (contention; linear speedup is optimistic ~5x), so budget the full run at roughly 2 h locally-equivalent; Kaggle sizing to be re-derived in T3.
- Gate 0 (generator E0, rho 0.5, 5 seeds, `TRAMIL/Results/poc_e0_gate.json`): chosen lr=0.01, W_o=zero, seed-mean best val MSE 1.2199 vs constant 1.2642 -> PASS (margin 3.5%; best epochs 3-6, i.e. early overfit).
- Smoke (2 seeds, scratch dir): pipeline runs end to end through the process pool; smoke ignores the gate verdict by design.
- Spec readings / gaps: timing lower bound = 100 epochs (stand-in, no observed best-epoch data for E0 arms yet); smoke runs arms even when gate FAILs; test metrics are computed inside each run but no choice reads them.
- Next: T3 Kaggle job generation + preflight + rehearsal (operator authorizes launch).
- T3 (2026-11-24): job `tools/kaggle/poc-e0` (CPU, pin 347cc76, carry Results/e0_runs, gate caches cloned).
  Preflight: 6 pass, warns = local/deadline rehearsal skipped (Windows venv vs WSL tooling), session-fit unmeasured.
  Remote smoke on alejopatio: complete, `smoke record` pass, `readiness` ready=true. Smoke verdicts are 3-epoch
  pipeline checks only, not results. Route: inline.
