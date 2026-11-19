# TRAMIL E1 with WavLM acoustic features

## Objective
Re-run experiment E1 with frozen WavLM instance features in place of eGeMAPS, keeping every other
element of the E1 protocol (labels, 141 WoZ training bags, 3x5 outer partitions, 3 seeds, arms, statistics).

## Problem / why
Full E1 with eGeMAPS put every arm at chance (AUROC 0.451-0.474, inside the SVM permutation null).
The instance-level SVM diagnostic (2026-11-19, `TRAMIL/Results/diag_instance_svm.md`) showed the bag SVM
overfits (train 0.955 / val 0.451) and instance-level eGeMAPS reaches only 0.525 val: eGeMAPS carries at most
weak signal. The operator chose WavLM as the acoustic extractor (2026-11-19).

## Operator decisions (2026-11-19)
- Extractor: `microsoft/wavlm-base-plus`, frozen, hidden layer 6 (of 12), mean pooling over time -> 768-d per instance.
- This is a PoC deviation from experiments v04 (Ec. 5, 6, 68 define eGeMAPS 7+81); recorded here and in the
  discussion so G-EXP-POC sees it. The 7+81 pause split does not apply to WavLM.

## Scope (authorized)
- Repo `implementations/TRAMIL`, branch `feat/poc-e1`.
- Text embeddings (mpnet, `E`), labels, bags, partitions, arms, statistics unchanged.
- Official test split (56 agent-led) stays sealed (reserved for the single H7 reading).
- Kaggle account `alejopatio`. Downloads authorized. Each remote launch (extraction, full E1) needs the
  operator's explicit go before submit.

## Constraints
- Target venv only (`.venv/Scripts/python.exe`).
- Strict TDD: observed RED before GREEN. Runner: `.venv/Scripts/python.exe -m pytest -q --basetemp=.pytest_tmp`
  (`--basetemp` avoids the known WinError 5 on the user temp dir; 17 tmp_path errors without it are environmental).
- Every new module keeps `__provenance__` + invariant tests, as in `tramil-poc-e1.md`.
- Artifacts in English. Work-unit commits on `feat/poc-e1`, Conventional Commits, no AI attribution.
- A local run whose estimate exceeds 20 minutes goes to Kaggle.

## Tasks
- [x] T1 WavLM extractor: `features.wavlm_vector` (16 kHz, layer 6, mean pool, 768-d, finite), per-subject
      cache `{subject}_W.npy` aligned row-for-row with `_X.npy`; `extract_run` acoustic switch; tests on a tiny
      random-init WavLM config. Route: delegated (writer trigger).
- [x] T2 Acoustic source switch in loaders/model: `load_items(..., acoustic="egemaps"|"wavlm")`, model input dim
      from data instead of the hard-coded 88, SVM bag stats on 1536-d; eGeMAPS path unchanged. Route: delegated.
- [ ] T3 Kaggle extraction job (GPU) -> private dataset `alejopatio/tramil-edaic-wavlm`; smoke first; launch only
      after operator go. Route: inline (remote-execution).
- [ ] T4 Local check: instance-level SVM diagnostic on WavLM features (train/val AUROC) before the full E1.
      Route: inline.
- [ ] T5 Full E1 (WavLM) on Kaggle: timing probe, operator go, run, fetch, merge. Route: inline.
- [ ] T6 Results: `TRAMIL/Results/E1_wavlm.md`, discussion.json update (deviation + verdicts), train/val metrics.

## Acceptance
- Existing eGeMAPS E1 reproduces unchanged (SVM fold AUROC identical).
- WavLM cache covers all 184 sessions, one row per instance, 768 finite values.
- E1 WavLM reports per-arm train and validation AUROC over 15 units plus paired comparisons.

## Progress / evidence
- 397ab9f `feat: add instance-level SVM diagnostic` (pre-work: eGeMAPS diag, train/val AUROC).
- T1 done: 4cf9589 `feat: add frozen WavLM layer-6 instance features`. RED 8 failed -> GREEN 8 passed
  (`tests/test_wavlm.py`, tiny random-init WavLM; no real weights or audio yet). One segment per forward pass
  (wavlm-base-plus has no attention mask, so padded batches would change vectors).
- T2 done: 036d47f `feat: select acoustic features (egemaps|wavlm) in E1 loaders and model`. RED 5 failed ->
  GREEN 7 passed (`tests/test_acoustic_switch.py`). `d_in` from data in model/ABMIL; 7+81 split lives only in
  the eGeMAPS extractor. Switch: `acoustic=` kwarg or `TRAMIL_ACOUSTIC=wavlm`.
- Regression: eGeMAPS diag reproduced every AUROC/per-fold value exactly (only runtime differed).
- Parent spot check: full suite `84 passed, 3 skipped` (92.7 s).
- Route evidence: T1+T2 delegated to one writer (writer trigger: 2+ non-trivial files).

## Next step
T3: Kaggle extraction job for WavLM (needs `transformers` in requirements and internet for HF weights);
smoke, then launch only after operator go.
