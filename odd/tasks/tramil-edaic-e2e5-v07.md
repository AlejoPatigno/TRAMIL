# TRAMIL E-DAIC: zip outputs, multi-account jobs and E2 to E5 of experiments v07

## Objective
(U1) One zip of outputs per Kaggle job so `fetch` never pulls thousands of JSONs. (U2) Make the e1v07/e2e5 jobs
runnable from other Kaggle accounts (dataset input mode, versioned job generator, operator guide). (U3) Implement E2 to
E5 of experiments v07 on top of the E1 runner: new arms, search/real units, per-job arm sets, analyses and report.

## Problem / why
`fetch` dies at 1800 s on 3420 loose JSONs (search-b). Private kernels cannot be read from another account, so jobs
cannot be spread over accounts. E2 to E5 need arms, extras (attention, surrogate fidelity, graph diagnostics) and
analyses that E1's runner does not have.

## Scope (authorized, operator)
Worktree `implementations/TRAMIL-worktrees/e2e5`, branch `feat/edaic-e2e5-v07` off `feat/edaic-e1-v07`. Offline only:
no Kaggle submit, no `implementation_cli gate/offer`, no `--consent`, no credentials, no dataset creation, e1v07 folders
in use (real/perm/report at ebf6647) are not regenerated.

## Constraints
- Route per task: delegated writer (this agent). TDD mode: strict (source: orchestrator CLAUDE.md / sdd-init).
  Runner: `PYTHONPATH=src ../../TRAMIL/.venv/Scripts/python.exe -m pytest -q --basetemp=.pytest_tmp` from the worktree
  (verified `TRAMIL.__file__` resolves inside the worktree).
- Fixed protocol (v07): nested CV 3x5x3, seeds {11,23,37}, epoch by inner class-weighted item MSE, AdamW 1e-3/1e-4,
  batch 8, 80 epochs, mu_R = 1/vbar, shared k* frozen by E1 (new arms search p only at k*).
- Do not stage `TRAMIL/AGREED.md`, `TRAMIL/.remote-execution/`, `kaggle-inbox/`. Conventional Commits, no attribution.
- ~400 changed lines per task is a heuristic only.

## Design decisions (U3)
- E1 real/perm already run permuted, untargeted, r12 and prompt_only (default arm set = all registry arms) and the
  prompt-only Ec. 91 null (PERM_MODELS). So E2 arms (b)(c)(d), E5 (a)(b)(c) and the E5 null need no new fits; no
  e2e5-perm job is generated.
- E1 units store only item predictions. E2 fidelity, E3 diagnostics, E4/E5 attention need extras, so extras are stored
  by the e2e5 runner and `full_x`, `permuted_x`, `untargeted_x` are aliases of the E1 arms (same model, same k*, same
  p*) re-run only to store extras. Their scores equal E1's in distribution; the report reads scores from E1 arms.
- New arms: graph_corr (E3 b), graph_prior (E3 c), binary_sup (E4 H3), free_desc and free_zero (E4 H3b),
  permuted_quint (E5 d), full_nohist (E5 e, needs the frozen history-prompt list).
- Separate module `e2e5_run.py` so E1 behavior and tests are untouched; zip/import helpers are shared.

## Checklist (all ticked items observed; route = delegated writer, TDD strict, runner above)
- [x] U1 One zip of outputs per job (commit 999ba53)
- [x] U2 Jobs from other Kaggle accounts (gen_jobs.py, MULTI_ACCOUNT.md, layout tests)
- [x] U3.1 model flags (graph corr/prior, free desc/zero, binary head) + extras + batched equivalence/padding tests
- [x] U3.2 registry + quintile derangement + history removal (function) + permuted_quint
- [x] U3.3 statistics (BH, profile metrics, stability, variance ratio, F/BF, edge diagnostics)
- [x] U3.4 e2e5 runner (commit 7a15507)
- [x] U3.5 report E2 to E5 (commit e10a5f8)
- [x] U3.6 job folders pinned to 289d68d, pushed
- [x] U3.7 CPU timing estimate per new arm
- [x] U4 Persist frozen models: a real unit stores its state_dict and the fitted front end (zip members real/<uid>.pt,
  frontend/<name>.pt, SVM real/<uid>.pkl) and frozen.Store/load/predict read them back for inference only (E6, E7)

## Operator decisions (2026-11-26)
1. Open question 1 ACCEPTED: keep the full_x, permuted_x, untargeted_x re-fits (~540 extra fits). Operator: "this kind of
   thing must not happen again; avoid spending GPU unnecessarily".
2. New task U4: every real unit (E1 and e2e5 runners) must persist everything a later experiment can need, so no
   downstream experiment re-fits a trained model (state_dict, frozen front end, OOF attention and diagnostics, all inside
   the job's verified zip), with loaders so E6 and E7 run inference only.
3. Run the full suite once before committing and report the counts.
4. Option A chosen for the E1 arms without a persisted model (gap 1 of U4): add abmil8_x, gated_abmil8_x,
   symptom_guided_x, r06_8_x, abmil_r8_x, prompt_only_x, r12_x to e2e5-real (same frozen E1 k* and p*; prompt_only_x at
   k = 8) and svm_x (the E1 bag-level SVM again, same deterministic inner kernel/C search, pipeline persisted). Done (U4b).

## Acceptance evidence
- Full suite with worktree src: 332 passed, 3 skipped (331 passed + the one fixed test, re-run alone: test_item_comparators +
  test_poc_e0b 17 passed). The 3 skips need absent E-DAIC labels (expected). One failure found by the full run
  (binary_sup predicts 1 logit, not 8) was a test expectation, fixed in 14f4e94.
- Batched-vs-per-bag equivalence and padding invariance: tests/test_batched.py parametrizes over every registered arm
  (graph_corr, graph_prior, binary_sup, free_desc, free_zero, permuted_quint, full_nohist, full_x, permuted_x,
  untargeted_x included); forward, loss, gradients at 1e-5 against frozen per-bag oracles (tests/oracle_e2e5.py); passed.
- Offline validation: validate_run_config OK and staleness fresh for all 5 e2e5 folders, commit 289d68d. Nothing submitted.

## Local CPU timing per fit (80 epochs extrapolated from 3, 141 synthetic bags of 32-88 segments, 768-d, k=128, p=0.15,
10 threads, shared machine, min of 3 interleaved passes; indicative only, GPU numbers come from the Kaggle probe)
full 62 s | graph_corr 51 s | graph_prior 37 s | binary_sup 50 s | free_desc 48 s | free_zero 48 s | permuted_quint 58 s | full_x 52 s
Real unit = 3 inner fits + 1 refit = 4 fits. Search unit = 1 fit.

## U4 frozen models: what is persisted, size, gaps (spec [exp:E6] line 481, [exp:E7] line 555)
Persisted per real unit (E1 and e2e5 runners, phase real only; searches and permutation nulls store nothing):
`<uid>.pt` = state_dict (float32, cpu), arm, k, p, vbar, selected epoch, held-out subject order, front end name;
`frontend/fe_r{rep}_f{fold}_k{k}_{source}[_nohist].pt` = scaler (mu, sd) and V_k fitted on the outer training subjects
only, written once per (rep, fold, k, source); SVM units `<uid>.pkl` (pickled pipeline). Already stored in the unit JSON:
OOF item predictions, and for the e2e5 arms the per-subject attention columns, ranks, F_eff, surrogate fidelity, Al,
RMS(Delta) and drift. The files are written before the JSON (JSON present means the unit is complete), travel in the
job zip, survive resume, and are never copied by report imports. `frozen.Store(roots)` indexes loose files and zips
anywhere; `frozen.load` rebuilds model, held-out bags and items; test: stored models reproduce their stored OOF
predictions and extras to 1e-5 by inference only, and the front end equals one fitted on the training subjects.
Size (k = 128, measured with real shapes, random weights, compressed): state 30 KB per full-like model (8017 params,
float32; 16 to 31 KB for the others), front end 0.37 MB per (rep, fold, k) at k <= 128 and 2.1 MB for the k = 768
rotation ablation. e2e5-real-a/b (about 200 units each, k* <= 128 front ends): about 12 MB per job. E1 real jobs
(about 500 units, plus the k = 768 front ends): about 15 MB of states plus up to 25 MB of front ends, ~30 MB per job
if the E1 jobs are regenerated from this branch.

Inputs E6 and E7 need that are STILL NOT persisted or not available:
- E1 real jobs in use (pinned ebf6647) predate U4 and store no models: the E6 arms abmil8, gated_abmil8, symptom_guided,
  r06_8, abmil_r8 (and E7 prompt_only, r12, abmil8, gated_abmil8, abmil_r8) have NO frozen models. Full and permuted are
  covered by full_x and permuted_x of e2e5-real. Fix options (operator decision, not done): regenerate e1v07-real-a/b
  from this branch before they run, or add `*_x` aliases for those arms to e2e5-real (225 units for the five E6 arms).
  The SVM pipelines of E1 are likewise absent.
- E6: response texts per segment and the NLI relevance matrix S_b (new pass, checkpoint and templates to pre-register); the
  loader gives no segment text. The deletion-pass, LiftSem (Ec. 49-60, 108-110) and Ec. 55 code do not exist yet
  (inference needs only the stored models and the bags, which Store.load provides).
- E7: the spec says "trained on the Wizard-of-Oz training subjects only ... frozen models of E1" but no model refit on all
  141 subjects exists (E1 trains per outer fold only). Either E7 ensembles the 45 fold models per arm (persisted, each
  with its own front end) or a refit-on-all job is needed: operator decision. Also absent: features and bags of the 56
  test sessions and 30 agent-led sessions, the prompt reconstruction, the Ec. 111 thresholds and the hand transcriptions.

U4b status: the gap 1 above is CLOSED for e2e5-real once it runs: 7 arm aliases + svm_x. Extra units vs the first e2e5
plan: 15 outer folds x (7 arms x 3 seeds + 1 svm) = 330 units (315 neural = 1260 fits, 15 svm). Local CPU per fit
(80 epochs, 141 bags): abmil8 3.8 s, gated 4.3, symptom_guided 4.9, r06_8 4.0, abmil_r8 4.0, prompt_only 9.6, r12 7.8;
about 58 min CPU per shard of two (GPU not measured for these arms; the batched path is expected to be no slower).
Neural *_x units carry no attention extras unless the model is a surrogate (prompt_only_x, r12_x do).

## Open questions for the operator
1. ACCEPTED by the operator (see decisions). E1 real units lack attention, fidelity and graph diagnostics, so full, permuted, untargeted must be re-run as `*_x` aliases (extra fits, E1 k* and p*).
2. binary_sup: spec ambiguous on whether the surrogate penalty (mu_R = 1/vbar) stays. Kept; epoch selection uses class-weighted BCE.
3. Percentile ranks for the H3 variance are computed over all out-of-fold subjects of each (rep, seed) run; spec ambiguous between per-fold and per-run.
4. Weighted kappa = quadratic kappa averaged over the items.
5. full_nohist (opt-in, excluded from default arms) needs the frozen list of history prompts and per-segment prompts; `exp_e1.load_items` provides no prompt text, so the arm is implemented and tested but NOT wired to real data (run raises without `history_prompts`). The e2e5 jobs do not include full_nohist.
6. LiftSem^S (needs E6 deletion influence), the E3 edge-attention panel and the E5 stacked bars are not implemented.
7. E3 (a) no-graph arm is r06_8 (no Al/RMS: n/a).
8. Ec. 91 null columns only for full and prompt_only (spec runs the null only for prompt-only).
9. Delivery strategy ask-on-risk; the forecast exceeds 400 changed lines (about 1900 added over 7 commits); the branch is one stack on feat/edaic-e1-v07.

## Honest TDD notes
gen_jobs.py and e2e5_stats.py were written before their tests (stats tests passed without a mutation check); layout tests
are characterization (no RED possible). All other units: RED observed first (e.g. test_e2e5_run collection error, then GREEN).

## Next step
Operator: review, then run e2e5-search-a/b (k* from e1v07-real-a/b), then e2e5-real-a/b, then e2e5-report. Decide question 5.
