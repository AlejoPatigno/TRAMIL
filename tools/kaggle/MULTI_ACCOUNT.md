# Running the TRAMIL jobs from several Kaggle accounts

A private kernel cannot be read from another account, so a job that runs on account B cannot attach the WavLM cache
kernel or the outputs of a job that ran on account A. Dataset mode replaces both by private datasets that the operator
shares with B. `tools/kaggle/gen_jobs.py` writes the job folders in either mode:

```
python tools/kaggle/gen_jobs.py --jobs e2e5 --owner alejopatio                       # kernel mode (default, same account)
python tools/kaggle/gen_jobs.py --jobs e2e5 --owner alejopatio --inputs dataset      # dataset mode, folders for other accounts
python tools/kaggle/gen_jobs.py --jobs e2e5 --inputs dataset --data-owner alejopatio --ref feat/edaic-e2e5-v07
```

Nothing is regenerated unless named in `--jobs` (default `e2e5`): the e1v07 folders already in use are not touched.
Generate the dataset-mode folders into a copy of the repository (or rename the folders) if both modes are needed at once,
because a folder is named after the job.

## What the runner finds

`find_inputs` and `import_upstream` walk `/kaggle/input` whatever the mount layout: the WavLM cache is any folder holding
`<subject>_W.npy`; upstream results are any `E1_v07*.zip` / `E2E5_v07*.zip` (a job writes one zip per phase and shard,
members `E1_v07/<phase>/<unit>.json`) or loose files inside a folder named `E1_v07` / `E2E5_v07`. Existing files are
never overwritten. Upload the zips as they are; do not rename the members.

## Once per extra account (operator)

1. Drop the account's credentials in `TCAMIL_extension/kaggle-inbox/` as `kaggle.json`, or as a text list with one
   `username key` per line, and run `/kaggle-accounts validate`. It checks that the token authenticates and takes the
   credential in; remove an account with the same skill. The workspace flow for accounts tied to distinct e-mails
   (`smith team add ... --attest`) applies as for the first account.
2. Share the private datasets with that account in the Kaggle web UI (dataset page, Sharing, add the username as a
   viewer). Datasets the operator creates (the agent never creates or uploads them):
   - `<owner>/tramil-edaic-wavlm`: the `<subject>_W.npy` cache (the output of the `papersmith-extract-wavlm` kernel).
   - `<owner>/tramil-edaic-features` and `<owner>/tramil-edaic-instances`: already used by every job.
   - `<owner>/tramil-e1v07-upstream`: the E1 results the jobs read: the `hpo/` files (`hpo_r*_f*.json`) and the real and
     permutation zips or loose units of E1.
   - `<owner>/tramil-e2e5-upstream`: the zips of the e2e5 search and real jobs (for the e2e5 real and report jobs).
   Version a dataset again after each upstream job finishes (fetch the job on the account that ran it, copy its zips
   into the dataset folder, create a new dataset version).
3. Data governance: E-DAIC is credentialed data under a data use agreement. The WavLM features and the item scores are
   derived from it, so sharing those datasets only with accounts covered by the agreement is the operator's
   responsibility; the runner does not check it.

## Which job goes to which account

The jobs of one phase are independent halves (shard `1/2` and `2/2`), so the two halves of a phase can run on two
accounts at the same time. Phases are sequential because each reads the previous one.

| Order | Job | Reads | Account |
|---|---|---|---|
| 1 | `e2e5-search-a` | E1 hpo (k*, p*) | account A |
| 1 | `e2e5-search-b` | E1 hpo | account B |
| 2 | `e2e5-real-a` | E1 hpo, both e2e5 search zips | account A |
| 2 | `e2e5-real-b` | E1 hpo, both e2e5 search zips | account B |
| 3 | `e2e5-report` | E1 real and perm results, both e2e5 real zips | any one account |

Between steps 1 and 2 (and 2 and 3) the zips of both halves have to be in `tramil-e2e5-upstream` (dataset mode), so the
operator fetches each half from its account and publishes the new dataset version before the next phase starts. In
kernel mode on a single account the same chain works through kernel outputs with no dataset step.
The e1v07 jobs follow the same pattern (`search-a/b` -> `real-a/b` -> `perm-a/b` -> `report`) with
`tramil-e1v07-upstream` in place of the kernel outputs.
