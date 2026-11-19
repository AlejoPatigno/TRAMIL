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
- [x] T7 Engine checks: `verify --revision <v04>`, then `agree`. Route: inline.
- [x] T8 Kaggle feature extraction job: `TRAMIL.extract_run.run` (train+dev sessions, folder or zip input, `limit` for smoke),
      `__steps__["extract-features"]` (remote, kaggle), generated job `tools/kaggle/extract-features`. Route: inline
      (one new module + test; writer trigger not fired).
- [x] T9 Full E1 remote entry `TRAMIL.e1_run`: 4 neural arms + SVM (Symptom-Guided deferred by the operator, 2026-11-18),
      3 repetitions x 5 folds x seeds 11/23/37, max 80 epochs, inner 3-fold epoch selection; one result file per
      (repetition, fold) written as soon as it finishes, existing ones skipped (resume across 12 h sessions); neural training
      on CUDA when available; `merge` builds the E1 table from the partition files. Route: delegated (writer trigger:
      new module + training device change + tests).
- [x] T10 Kaggle job `e1-full` (GPU, inputs: features + instances datasets): smoke with GPU-vs-CPU epoch timing shown to
      the operator, then position -> propose -> offer -> gate -> submit (operator authorized, 2026-11-18); resume chunks until
      all 15 partitions exist. Route: inline.

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

- T7 (inline): `verify` fidelity ok for both documents (experiments v04, proposal r06) after rebinding
  `statistics.py` to r06 section 18.1 (Ec. 62, 63, 69); it had declared equations under the experiments
  revision, which drifted the proposal lineage. `agree` status `agreed` (54 crossed, 15 out of scope, untested
  empty) with `--acknowledge absent:N`: v04 line 54 quotes the literal `[claims:N]` in prose. Suite 51 passed
  with data env vars, 48 passed + 3 skipped without. Test fixtures now read TRAMIL_LABELS / TRAMIL_EDAIC
  (defaults under gitignored TRAMIL/Data/) instead of machine paths, since the repo is public.
  Still reported by verify: scaffold gaps (kit files: test_smoke, findings, audit, verification notebook) and
  stray-module listing; audit `none`.

- T8 `a8faff0`, `2be408d`, job `a0acc6f`: RED = ImportError (`cannot import name 'extract_run'`); GREEN 7 passed (zip and folder
  inputs, train+dev coverage, limit, mount-nested root, stamp under runner only); full suite 55 passed, 3 skipped. Job generated
  under WSL (Windows child env cannot resolve DNS for the git probe): cpu accelerator (opensmile is CPU-bound, mpnet on ~10k
  short texts is minutes on CPU; GPU quota saved), input dataset `alejopatio/tramil-edaic-instances`, pip: opensmile,
  sentence-transformers. First attempt refused on an unresolved read (source digest in the stamp); fixed by dropping the digest
  (commit identifies the code), not by waiving. Preflight under WSL: pin, diagrams, code-reachability, inputs, accelerator pass;
  `interpreter` fails only because WSL python lacks kagglesdk (passes under the Windows workspace interpreter);
  warns: account needs its own smoke, session-fit unmeasured. Windows local-rehearsal fails for environment reasons (restricted
  child env breaks opensmile DLL and process pool); equivalent manual rehearsal with FORGE_RUN_MODE=smoke on a synthetic input
  (real mpnet) wrote `clone/TRAMIL/Data/features/*` and a valid `shard.json`. Expected 1-3 h on 4 Kaggle CPU cores, < 12 h limit,
  no carry path (per-subject cache is resumable only within a session).

- T8 run (2026-11-18): fixes `881765e` (audio inside Kaggle-unpacked `instances_audio/instances_audio/`, RED observed),
  `cc9fa33` (mpnet on CUDA when available, RED observed), `6a26031` (`__levels__ = ["pilot", "full"]`). Smoke passed on a
  Tesla T4 at `6a26031` (107 instances, 2 sessions). Launch chain: position (`TRAMIL/AGREED.md`) -> PoC diagram validated
  by the operator in chat (audit override for a 5.9 pt right margin) -> propose -> offer -> gate -> submit. Gotcha: offer/gate
  run on Windows record `tools\kaggle\...`; the WSL submit needs them recorded under WSL. Campaign: 2470 s on T4,
  184/184 sessions, 10476 instances, X 88-d / E 768-d, 6 non-finite acoustic rows (one each in 306, 327, 338, 362, 421, 459;
  dropped by `bag_features`). Fetched clone sources match `6a26031`. Cache published as private Kaggle dataset
  `alejopatio/tramil-edaic-features` (status ready, anonymous API 403).

- Decision (operator, 2026-11-18, "lo acepto como estan"): the four T2 deviations are accepted as they are -- cleaned audio
  instead of the raw audio of Ec. 68, session gain -26 dBFS, silence threshold 35 dB, pre-cut wavs for 318/321/341/362.
  The published feature cache stands; no re-extraction.

- T9 `c4e4abd` (delegated writer): RED = ImportError (`cannot import name 'e1_run'`); GREEN 67 passed, 3 skipped (parent spot
  check re-ran the suite: 67 passed). Bags build from the CSVs + cached features without wavs. Resume by partition file;
  merge pools the 15 units via `exp_e1._comparisons` (statistics reading to confirm with the operator). CUDA path covered only
  by a monkeypatched device test until the T10 smoke. SVM stays inside the GPU job (pilot: 1.2 s for all SVM folds; a separate
  CPU job would cost more in setup than the work).

- T10: `be64e5d` fixed `bag_features` building the opensmile extractor before checking the cache (RED observed; 68 passed).
  T4 smoke 0.10-0.16 s/epoch (local CPU 1.0-1.4 s). Campaign on T4 at `be64e5d`: ~5 h 20 min, 15/15 units, fetched sources
  match the pin. Result (`TRAMIL/Results/E1_full.md`): mean AUROC residual 0.474, learned-only 0.455, ABMIL 0.462, Gated
  ABMIL 0.456, SVM 0.451; no paired difference (p 0.67-0.78, TOST intervals cross -0.05). Checks: Kaggle features equal the
  local pilot features (median rel. diff 5e-8); SVM folds reproduce exactly on local CPU; SVM permutation null (200 perms,
  same partitions): null mean 0.501, 5-95% [0.399, 0.601], p(null >= observed) 0.765 -> no detectable signal at full scale.
  The 48-bag pilot (0.58-0.76) was optimistic small-sample noise.

## Next step
Operator: read the E1 result (no detectable signal) and decide the next move (G-EXP-POC discussion, Symptom-Guided,
feature or model diagnosis); confirm the statistics reading of pooling 15 units through the corrected resampled t.
