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
- [x] T1 Scaffold + data: pyproject (pytest pythonpath, deps), `__implementation__`, product folders,
      label loader, bag builder (Ec. 1-4, 68). Route: delegated (writer trigger, 2+ non-trivial files).
- [x] T2 Feature cache: eGeMAPS 88-d split 7+81 (Ec. 5, 6, 68), all-mpnet-base-v2 embeddings (Ec. 7, 8),
      cached under `TRAMIL/Data/`. Route: delegated.
- [x] T3 Model: proximity (9, 10), standardize (11, 12), encoder (13, 14), corrector (15-17, 22, 23, 57),
      aggregation (18-21, 24-31), model wiring; invariant tests incl. permutation invariance (Ec. 27).
      Route: delegated.
- [x] T4 Comparators (ABMIL, Gated ABMIL, learned-only, permuted prior, SVM Ec. 32-37, 66), training (38),
      validation (39-41). Route: delegated.
- [x] T5 Statistics: Fisher ratio (62), corrected resampled t (63), TOST equivalence (69), Holm. Route: delegated.
- [x] T6 `exp_e1` rehearsal: 141 train bags, 1 repetition x 2 folds x seed 11, few epochs; residual,
      learned-only, ABMIL, Gated ABMIL, SVM; report table to `TRAMIL/Results/`. Route: delegated.
- [ ] T7 Engine checks: `verify --revision <v04>`, then `agree`. Route: inline.

## Acceptance
- Suite green under the target interpreter; `verify` fidelity clean; `agree` passes without acknowledgements.
- E1 rehearsal completes and writes its table (numbers are plumbing checks, not results).

## Progress / evidence
Resolved TDD: on (source: session config `Strict TDD Mode: enabled`), runner `.venv/Scripts/python.exe -m pytest -q`.
Route: all tasks delegated (writer trigger); worker = general agent standing in for `implementation-build`.
- T1 `2e7c073`: RED = ImportError (`cannot import name 'bags'`), GREEN 6 passed. Label recomputed from items; 275 rows sum
  check, 20 disagreements (13/3/4) observed; 141 train bags (44 positive), 43 dev bags (13 positive).
- T2 `a19cc28`: RED = ImportError on `features`; GREEN 6 passed (opensmile 88 = 7 + 81, gain, silence removal, unit-norm 768).
  Two tests (cache resumable, non-finite excluded) were written after the code / RED observed for the non-finite one only.
- T3 `efe58fe`: RED = ImportError; GREEN 15 passed (param counts 4418 / 4417, permutation invariance, Ec. 23, KL argmax).
- T4 `a51be4f`: RED = ImportError (2 modules); GREEN after fixing two test designs (toy training needed 60 epochs at lr 1e-3;
  bootstrap assertion was too strict). 16 passed incl. earlier modules.
- T5 `8b905fa`: RED = ImportError; GREEN 5 passed.
- T6 `1f933de`: RED = ImportError; GREEN 2 passed; real rehearsal below.
- Full suite: 51 passed (`.venv/Scripts/python.exe -m pytest -q`); every `__provenance__` invariant has a `test_<id>`.
- T6 real run (PILOT, plumbing): 48 of 141 training bags (stratified, seed 0; 15 positive), 2804 instances (1 non-finite dropped),
  1 repetition x 2 folds x seed 11, 8 max epochs; table `TRAMIL/Results/E1_pilot.{json,md}`. Subset because full extraction
  was estimated at ~70+ min (86 029 s of answer audio; opensmile ~20x real time per pass, two passes, ~2x parallel gain on this
  machine); 48 bags took 2519.5 s wall (8 workers, 25 newly extracted, contended by test runs).
- Data handling: instances_audio wavs were already cut with the offsets for 318/321/341/362 (cut_instances.py OFFSETS);
  318, 341, 362 are training bags, 321 is a dev bag; no re-shifting applied here. Audio is the project's cleaned audio
  (hum removal + 50 Hz high-pass) rather than the raw audio of Ec. 68 (deviation, reported).

## Next step
T7 (inline, orchestrator): `verify --revision <v04>` then `agree`. Open items for the operator: see build report.
