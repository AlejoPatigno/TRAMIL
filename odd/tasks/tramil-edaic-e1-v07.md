# TRAMIL E-DAIC E1 pipeline per experiments v07

## Objective
Implement the E-DAIC E1 pipeline of experiments v07 (proposal r22) on frozen WavLM features, with a GPU timing
probe, ready for the operator's G-EXP-POC and a Kaggle GPU launch (2 parallel GPU sessions, run in parts).

## Problem / why
The E1 code is a v04/r06 pilot: binary label, 4 arms, no item supervision, no PCA/HPO, no Ec. 91 null, no DeLong,
per-fold AUROC only, and one small fit per process (a GPU would sit idle). v07 (published 2026-11-25) fixes the
protocol; the acoustic surrogate model exists only as the E0 toy (`poc_e0.py`, `poc_e0b.py`).

## Fixed settings (operator, 2026-11-25)
- Features: WavLM base+ layer 6 cache `{subject}_W.npy` (already extracted on Kaggle GPU, 184 sessions;
  near-equal windows <= 20 s with frame mean == v07 duration-weighted mean; no gain, do_normalize=false).
  Test split (56 agent-led) stays sealed until E7.
- mu_R = 1/vbar (vbar = mean (R^c)^2 on the training fold; E0b normalized mu=1), fixed, not tuned.
- PCA grid K={16,32,64,128} + k=768 reference arm; p_drop grid {0,0.15,0.3,0.45} (T=4) same for all arms;
  HPO with one seed; shared k*, per-arm p* (Ec. 102), ties to smaller k then smaller p.
- Nested CV 3 reps x 5 outer x 3 inner, init seeds {11,23,37}; epoch by inner class-weighted item MSE.
- R_pi=23; reduced per-permutation protocol (1 rep, 1 seed, frozen hyperparameters, inner epoch selection kept);
  item scores move as a unit. B=9999. Ec. 91: bootstrap lower bound > max of 23 nulls; p=(b+1)/24 descriptive.
- Comparators: ABMIL-8, Gated ABMIL-8, Symptom-Guided, r06-8, ABMIL+R-8 (8 item heads, item MSE), bag SVM on
  mean+sd (1536-d) on the binary label. delta_NI=0.10.

## Scope (authorized)
Repo `implementations/TRAMIL`, branch `feat/edaic-e1-v07` (off `feat/poc-e0`). Local implementation and tests.
Remote launches need the operator's explicit go; G-EXP-POC is the operator's approval.

## Constraints
- Target venv only. Strict TDD (RED -> GREEN -> REFACTOR). Runner:
  `.venv/Scripts/python.exe -m pytest -q --basetemp=.pytest_tmp`.
- New modules keep `__provenance__` + invariant tests. Artifacts in English. Conventional Commits, no AI attribution.
- Delivery strategy: ask-on-risk (forecast > 400 lines; PR slicing decided before opening any PR).

## Tasks
- [x] T1 Item supervision: s_bj targets in Bag/load_items, class weights and weighted item MSE (Ec. 75-77),
      decision sum_j s_hat >= 10 (Ec. 76); epoch selection on inner class-weighted item MSE. Route: delegated (writer).
- [x] T2 Models on real bags: acoustic surrogate (Ec. 114-116) with normalized penalty mu_R=1/vbar; 8-item-head
      ABMIL-8, Gated ABMIL-8, r06-8, ABMIL+R-8 (Ec. 79), Symptom-Guided; PCA front end (Ec. 94) + dropout,
      k=768 arm. Route: delegated (writer).
- [x] T3 Shared HPO (Ec. 102) frozen per outer fold to JSON. Route: delegated.
- [x] T4 Statistics: pooled OOF AUROC (Ec. 82/83), vectorized shared bootstrap B=9999 (Ec. 85), NI (Ec. 86),
      DeLong (Ec. 84), Ec. 91 rule, Ec. 100/87; SS^omega (Ec. 93). Route: delegated.
- [x] T5 Permutation runner (Ec. 91) with reduced protocol, unit (perm, fold, arm). Route: delegated.
- [x] T6 GPU throughput: tensors on device, N worker processes per GPU over independent units, cuda.synchronize.
      Route: delegated.
- [x] T6b Batched padded training/prediction for every neural arm (instance mask, masked softmax/centering,
      per-bag penalty), folds collated once on the device, one worker per GPU by default. Route: delegated.
- [x] T7 Timing probe (local CPU smoke only; GPU run pending): per-fit and concurrent throughput (1,2,4,8 workers) for every arm -> Results/timing.json.
      Route: delegated; GPU measurement runs on Kaggle after operator go.
- [ ] T8 (folders generated, launch pending operator go) Kaggle jobs e1-v07-{search,real,perm} with the WavLM dataset, smoke unit, merge/report scripts.
      Route: inline (remote-execution).
- [ ] T9 G-EXP-POC evidence pack for the operator.

## Acceptance
- Full suite green under the target venv; new invariants tested (Ec. 91 rule, label permutation as a unit, DeLong
  against a known value, penalty normalization, shared k* ties).
- Timing probe reports per-fit seconds and concurrent throughput per arm on GPU.

## Progress / evidence
- 2026-11-25: gap analysis (read-only mapper) done; branch created.
- T1 done, commit b4ecbcd (route: delegated writer, writer trigger). Bag.s / load_items "s"; item_training.py
  (omega_+, weighted item MSE Ec. 77, score/decide Ec. 76, select_epoch Ec. 39 argmin earliest, AdamW 1e-3/1e-4,
  batch 8, 80 epochs, model penalty via fit_terms). TDD: RED 2 failed + 1 collection error (10 tests) -> GREEN 23 passed.
- T2 done, commit 65832e0 (route: delegated writer). frontend.py (scaler + PCA Ec. 94, fold_bags, rc_scale),
  surrogate.py (full model Ec. 114-116/104-106/72 + r22 corrector; flags permuted, mu_R=0, log_n prompt-only,
  prior="text" r12), item_comparators.py (ABMIL-8, Gated ABMIL-8, ABMIL+R-8, r06-8, Symptom-Guided), e1_arms.py
  (registry), item_training.permute_items. TDD: RED 3 collection errors -> GREEN 32 passed (one float32 tolerance
  fix in a test). Full suite: 176 passed, 3 skipped (label/E-DAIC data absent locally); test_poc_e0b.py green.
- Findings: r22 Section 11 count 32k+3657 omits W_r,b_r (264); implemented full model = 32k+3921 (28,497 at
  k=768 vs stated 28,233). Symptom-Guided = 32k+752. Prompt-only uses PCA k=8 rotation per v07 E5.
  Gotcha: concurrent pytest runs sharing --basetemp wipe each other's tmp dirs.

- T3 done, commit 1030e7f (route: delegated writer). hpo.py: units (rep, fold, arm, k, p, inner) -> pure run_unit ->
  deterministic reduce (Ec. 39 objective, k* = argmax_k mean over E1 neural arms of max_p, per-arm p*, ties smaller k
  then p; prompt_only at k=8; p768 for the full model at k=768) -> hpo_r{rep}_f{fold}.json + load_frozen.
  TDD: RED collection error (8 tests) -> GREEN 8 passed.
- T4 done, commit c9ee0e8 (route: delegated writer). pooled_stats.py: Ec. 82/83, shared stratified bootstrap with
  vectorized rank AUROC (B=9999 x 8 models x 3 reps in ~3.3 s CPU), lower bounds, Ec. 86 verdict, DeLong (hand
  example Var=0.625), Ec. 91 rule + p=(b+1)/24, Ec. 100, Ec. 87, SS^omega + bootstrap (cross-checked with poc_e0.ss_w).
  TDD: RED collection error (13 tests) -> GREEN 13 passed. Full suite 197 passed, 3 skipped; test_poc_e0b.py green.

- E1 runner + T5 + T6 + T7 done, commit 2f48065 (route: delegated writer, writer trigger). e1v07_run.py: phases
  search/real/perm/report, each unit through a pure run_unit -> <out>/<phase>/<uid>.json, resume = skip existing,
  shard "i/n" or --units; search freezes hpo_r{rep}_f{fold}.json via new hpo.freeze; real = (rep, fold, arm, seed
  11/23/37) + full_k768 + bag SVM, inner epoch selection then refit; perm (T5) = R_pi=23, donor seed 2026+r, rep-0
  partition of the true labels, seed 11, frozen k*/p*, SVM kernel/C frozen from the real run; report = pooled stats,
  Ec. 91 table, NI/DeLong/Ec. 87, descriptive + Holm table, SS^omega -> E1_v07.json/.md. dispatch.py (T6): spawn
  pool, per-worker device-resident fold cache, default N from timing.json best fits/s. timing_v07.py (T7): per-fit
  seconds (epochs measured, linear extrapolation to the budget, flagged) and N in {1,2,4,8} rows appended to
  Results/timing.json (schema 2). TDD: RED 2 collection errors (9 tests) -> GREEN 9 passed (e2e synthetic ~40 s).
  Full suite 206 passed, 3 skipped; test_poc_e0b.py 7 passed. Local real timing probe not run (needs data/GPU).

- T8 folders generated (route: delegated writer), commits 44a3cea (remote callables e1v07_run.run / timing_v07.run,
  kernel-output WavLM discovery, upstream import; TDD RED 2 failed -> GREEN, full suite 208 passed, 3 skipped) and
  the job-folder commit after it: tools/kaggle/e1v07-{timing,search-a,search-b,real-a,real-b,perm-a,perm-b,report}
  + tools/kaggle/e1v07/README.md, pinned to 44a3cea, generated with jobfolder.generate_job offline (only the
  pin-published probe bypassed: branch not pushed). Offline validation: all 8 read + validate_run_config OK, staleness
  fresh, runner.ipynb equals current assets. Nothing submitted; push + operator go pending.

- T6b done (operator-approved after the Kaggle T4 probe at 15f438b: per-bag loop launch-bound, 2-8 workers 7-12x
  worse), commit 5348f7b (route: delegated writer). item_training.collate/take/batch_loss; surrogate masked_center,
  masked_graph, masked_softmax; every arm batched; dispatcher default 1 worker; timing warm-up per worker, in-worker
  seconds. TDD: RED 31 failed (tests/test_batched.py, oracle = frozen per-bag copies of the old models) -> GREEN 32
  passed (forward, loss and gradients vs per-bag sum at 1e-5, padding invariance, train curve vs per-bag loop).
  Full suite 240 passed, 3 skipped; test_poc_e0b.py 7 passed. Local CPU per fit (80 epochs extrapolated, 141
  synthetic bags 32-88 inst, k=128, machine shared): see report; e.g. abmil8 57.6 -> 7.6 s, permuted 173 -> 65 s.
  Job folders repinned to 5348f7b (commit after it), timing job workers [1,2], epochs 3. Push denied by the local
  permission classifier: branch not pushed, so the pin is not yet reachable.

## Next step
Push feat/edaic-e1-v07 (needs operator permission), then rehearse e1v07-timing on the T4 and re-read per-fit times.
