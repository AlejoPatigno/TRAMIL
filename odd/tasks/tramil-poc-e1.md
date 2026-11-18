# TRAMIL proof of concept: residual model + E1 rehearsal

## Objective
Implement the residual semantic-prior bipartite MIL model (proposal r06) and run experiment E1 of
experiments v04 at rehearsal scale on E-DAIC, with module provenance the experimental-implementation
engine can verify.

## Why
The user authorized code (2026-11-18) after experiments v04 fixed the label source. A rehearsal proves the
pipeline end to end before spending Kaggle quota on the 900-run protocol.

## Scope (authorized)
- Repo: `implementations/TRAMIL` (new, package `TRAMIL`, product folder `TRAMIL/`).
- Documents: `proposals/research-concept-residual-semantic-prior-bipartite-mil-r06.md`,
  `experiments/experiments-experimental-plan-for-the-residual-semantic-prior-bipartite-v04.md`.
- Approved object-to-module map (bags, features, proximity, standardize, encoder, corrector, aggregation,
  model, comparators, training, validation, statistics, exp_e1). `sgca.py`, `interpret.py`, exp_e2..e9 later.
- Premises: prediction = binary PHQ-8 >= 10; statisticalUnit = subject; metric = AUROC; direction = higher.
- Labels: `detailed_lables.csv` (PHQ-8 >= 10 recomputed from items; distributed flag as sensitivity).
- Out of scope now: Kaggle launch (needs explicit user authorization), Symptom-Guided comparator, E2-E9.

## Constraints
- Target venv only (`.venv/Scripts/python.exe`); never the forge interpreter.
- Strict TDD: observed RED before GREEN. Runner: `.venv/Scripts/python.exe -m pytest -q`.
- Every module: `__provenance__` with `revision`, `sections`, `equations` and/or `experiments`, `invariants`;
  each invariant has `tests/test_<id>`.
- Artifacts in English. Work-unit commits on branch `feat/poc-e1`, Conventional Commits, no AI attribution.
- ~400 changed lines per task is advisory only.

## Tasks
- [ ] T1 Scaffold + data: pyproject (pytest pythonpath, deps), `__implementation__`, product folders,
      label loader, bag builder (Ec. 1-4, 68). Route: delegated (writer trigger, 2+ non-trivial files).
- [ ] T2 Feature cache: eGeMAPS 88-d split 7+81 (Ec. 5, 6, 68), all-mpnet-base-v2 embeddings (Ec. 7, 8),
      cached under `TRAMIL/Data/`. Route: delegated.
- [ ] T3 Model: proximity (9, 10), standardize (11, 12), encoder (13, 14), corrector (15-17, 22, 23, 57),
      aggregation (18-21, 24-31), model wiring; invariant tests incl. permutation invariance (Ec. 27).
      Route: delegated.
- [ ] T4 Comparators (ABMIL, Gated ABMIL, learned-only, permuted prior, SVM Ec. 32-37, 66), training (38),
      validation (39-41). Route: delegated.
- [ ] T5 Statistics: Fisher ratio (62), corrected resampled t (63), TOST equivalence (69), Holm. Route: delegated.
- [ ] T6 `exp_e1` rehearsal: 141 train bags, 1 repetition x 2 folds x seed 11, few epochs; residual,
      learned-only, ABMIL, Gated ABMIL, SVM; report table to `TRAMIL/Results/`. Route: delegated.
- [ ] T7 Engine checks: `verify --revision <v04>`, then `agree`. Route: inline.

## Acceptance
- Suite green under the target interpreter; `verify` fidelity clean; `agree` passes without acknowledgements.
- E1 rehearsal completes and writes its table (numbers are plumbing checks, not results).

## Progress / evidence
(none yet)

## Next step
T1.
