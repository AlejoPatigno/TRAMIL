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

## Checklist
- [ ] U1 One zip of outputs per job (e1v07_run)
  - [ ] U1.1 write zip E1_v07_<phase>[_<shard>].zip at the end of a real run (temp name, testzip + count, rename, then delete loose units)
  - [ ] U1.2 smoke evidence stays loose
  - [ ] U1.3 import_upstream reads zips (os.walk) + loose JSON, never overwrites
- [ ] U2 Jobs from other Kaggle accounts
  - [ ] U2.1 find_inputs/import_upstream layout-agnostic (kernel-output and dataset layouts)
  - [ ] U2.2 tools/kaggle/gen_jobs.py (owner + inputs mode kernel|dataset)
  - [ ] U2.3 tools/kaggle/MULTI_ACCOUNT.md
- [ ] U3 E2 to E5
  - [ ] U3.1 model flags (graph, free queries, binary head) + extras + batched equivalence/padding tests
  - [ ] U3.2 registry + derangement (quintile) + history removal + permuted_quint
  - [ ] U3.3 statistics (BH, profile metrics, stability, variance ratio)
  - [ ] U3.4 e2e5 runner (units, search at k*, freeze, real with extras, phases, remote entry)
  - [ ] U3.5 report E2 to E5
  - [ ] U3.6 job folders e2e5-search-{a,b}, real-{a,b}, report pinned to the pushed commit
  - [ ] U3.7 CPU timing estimate per new arm

## Acceptance
Full suite green with worktree src, tests/test_poc_e0b.py green, batched-vs-per-bag equivalence and padding tests for
every new arm, offline job validation fresh, pushed branch.

## Open questions for the operator
(filled during the work)

## Progress / evidence
(filled during the work)

## Next step
U1.
