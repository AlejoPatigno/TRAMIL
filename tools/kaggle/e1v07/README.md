# E1 v07 Kaggle jobs

Job folders: `tools/kaggle/e1v07-{timing,search-a,search-b,real-a,real-b,perm-a,perm-b,report}`. Each one is a
callable job: `TRAMIL.timing_v07.run` or `TRAMIL.e1v07_run.run`. Settings: T4 (`sm_75`), internet on, repo ref
`feat/edaic-e1-v07`, and the commit pin written in each `run-config.json`. `submit` adds the 12 h session limit.

## Inputs
- Datasets: `alejopatio/tramil-edaic-features` (text `_E.npy`) and `alejopatio/tramil-edaic-instances` (labels and
  instances CSV).
- WavLM `_W.npy` cache: this is the output of the kernel `alejopatio/papersmith-extract-wavlm`, attached through
  `kernel_sources`. `e1v07_run.find_inputs` locates it with os.walk below `/kaggle/input`, at
  `notebooks/<owner>/<slug>/clone/TRAMIL/Data/features`.
- Results from earlier kernels: every `E1_v07/{search,hpo,real,perm}/*.json` found under `/kaggle/input` is copied
  in before the run (`import_upstream`). Existing files are never overwritten.

## Launch order and dependencies
| Step | Jobs (run in parallel) | Kernel sources it needs | Writes |
|---|---|---|---|
| 1 | `e1v07-timing` | extract-wavlm | `TRAMIL/Results/timing.json` |
| 2 | `e1v07-search-a`, `e1v07-search-b` (shards 1/2, 2/2) | extract-wavlm | `E1_v07/search/*.json` |
| 3 | `e1v07-real-a`, `e1v07-real-b` | extract-wavlm, search-a, search-b | `E1_v07/hpo/*.json` (frozen from both search shards), `E1_v07/real/*.json` |
| 4 | `e1v07-perm-a`, `e1v07-perm-b` | extract-wavlm, real-a, real-b | `E1_v07/perm/*.json` |
| 5 | `e1v07-report` | extract-wavlm, real-a, real-b, perm-a, perm-b | `E1_v07/E1_v07.{json,md}` |

How to run each step:
- Rehearse every job before its real run: `submit --smoke`. A smoke run chains the needed phases on 40 bags with one
  arm and one epoch, does not import upstream results, and writes `shard.json` with `evidence.commit` and
  `environment.device.kind`.
- Read `timing.json` from step 1 before step 2:
  - The real runs use `workers: 4` by default. Change `run.kwargs.workers` if the timing shows a better N.
  - If one shard does not fit in the usable session (about 9.6 h), regenerate with more shards. The search grid is
    4680 fits of 80 epochs in total.
- Units are resumable, but a session cut at the limit returns nothing: there is no carry path.

## Kernel refs
The `kernel_sources` refs assume the jobs run under the account `alejopatio`, with the slug `papersmith-<job>`. The
Kaggle adapter derives the ref from the title. Carrying a private kernel across two accounts was never verified, so
run the whole chain on the account that owns `papersmith-extract-wavlm`.
