"""Generate the Kaggle job folders of TRAMIL (e1v07 and e2e5) with the workspace's jobfolder.generate_job, offline.

    python tools/kaggle/gen_jobs.py --jobs e2e5 [--owner alejopatio] [--inputs kernel|dataset] [--data-owner X]
                                    [--ref feat/edaic-e2e5-v07] [--target PATH] [--skills PATH]

--owner       the Kaggle account whose kernels are read as inputs in kernel mode (and the default data owner).
--inputs      kernel  : the WavLM cache is the kernel source <owner>/papersmith-extract-wavlm and the upstream results
                        are kernel outputs of <owner>/papersmith-<job>; readable only from the same account.
              dataset : the cache and the upstream result zips come from private datasets the operator shares with the
                        account that runs the job (<data-owner>/tramil-edaic-wavlm, <data-owner>/tramil-e1v07-upstream,
                        and for E2 to E5 <data-owner>/tramil-e2e5-upstream); no kernel sources.
--jobs        e2e5 (default), e1v07, all, or a comma list of job names. Nothing is regenerated unless named: the e1v07
              folders in use are never touched by the default.

Only the `pin-published` condition (the one network probe) is bypassed: generate before pushing and submit/preflight
re-run it. Kernel sources are added to kernel-metadata.json, which the Kaggle adapter keeps on push."""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

REPO_URL = "https://github.com/AlejoPatigno/TRAMIL.git"
EV = ["evidence.commit", "environment.device.kind"]
TINY = {"limit_bags": 40, "max_epochs": 1, "reps": 1, "seeds": [11], "K": [16], "P": [0.0], "r_pi": 1,
        "arms": ["full"], "kstar_arms": ["full"], "workers": 1, "import_from_inputs": False}
E2E5_TINY = {"limit_bags": 40, "max_epochs": 1, "reps": 1, "seeds": [11], "P": [0.0], "workers": 1,
             "arms": ["graph_corr", "binary_sup"], "import_from_inputs": False, "k_star": 16, "p_alias": 0.0}
R1, R2 = "TRAMIL.e1v07_run", "TRAMIL.e2e5_run"


def job_table(upstream):
    """name -> (module, kwargs, smoke kwargs, upstream job names). `upstream` names the E1 jobs feeding E2 to E5."""
    jobs = {"e1v07-timing": ("TRAMIL.timing_v07", {"epochs": 3, "budget": 80, "workers": [1, 2]},
                             {"arms": ["full"], "workers": [1], "epochs": 1, "budget": 80, "limit_bags": 40}, [])}
    for half, i in (("a", 1), ("b", 2)):
        sh = f"{i}/2"
        jobs[f"e1v07-search-{half}"] = (R1, {"phases": "search", "shard": sh, "workers": 2},
                                        {**TINY, "phases": ["search"]}, [])
        jobs[f"e1v07-real-{half}"] = (R1, {"phases": "real", "shard": sh, "workers": 2},
                                      {**TINY, "phases": ["search", "real"]}, ["e1v07-search-a", "e1v07-search-b"])
        jobs[f"e1v07-perm-{half}"] = (R1, {"phases": "perm", "shard": sh, "workers": 2},
                                      {**TINY, "phases": ["search", "real", "perm"]}, ["e1v07-real-a", "e1v07-real-b"])
    jobs["e1v07-report"] = (R1, {"phases": "report", "workers": 1}, {**TINY, "phases": ["search", "real", "perm", "report"]},
                            [f"e1v07-{p}-{h}" for p in ("real", "perm") for h in ("a", "b")])
    # E2 to E5 (e2e5_run): p* of the new arms at the E1 k*, then real units with extras, then the E2-E5 report.
    e1_hpo = [f"{upstream}-real-a", f"{upstream}-real-b"]  # the real jobs carry the frozen hpo files
    for half, i in (("a", 1), ("b", 2)):
        sh = f"{i}/2"
        jobs[f"e2e5-search-{half}"] = (R2, {"phases": "search", "shard": sh, "workers": 2},
                                       {**E2E5_TINY, "phases": ["search"]}, e1_hpo)
        jobs[f"e2e5-real-{half}"] = (R2, {"phases": "real", "shard": sh, "workers": 2},
                                     {**E2E5_TINY, "phases": ["search", "real"]},
                                     [*e1_hpo, "e2e5-search-a", "e2e5-search-b"])
    jobs["e2e5-report"] = (R2, {"phases": "report", "workers": 1}, {**E2E5_TINY, "phases": ["search", "real", "report"]},
                           [*(f"{upstream}-{p}-{h}" for p in ("real", "perm") for h in ("a", "b")),
                            "e2e5-real-a", "e2e5-real-b"])
    return jobs


def _find_skill_dir(start):
    for p in [Path(start).resolve(), *Path(start).resolve().parents]:
        c = p / ".claude" / "skills" / "remote-execution" / "scripts" / "remote_cli.py"
        if c.exists():
            return c.parent
    raise SystemExit("remote-execution scripts not found above the target; pass --skills")


def _load_remote_cli(skills):
    spec = importlib.util.spec_from_file_location("remote_cli", Path(skills) / "remote_cli.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["remote_cli"] = mod
    spec.loader.exec_module(mod)
    mod._load_backend_module("kaggle")
    return mod


def inputs_for(owner, data_owner, mode, job, upstream_jobs):
    """(input datasets, kernel sources) of one job. Dataset mode replaces the cache kernel and the upstream kernel
    outputs by private datasets; the e2e5 jobs also read the e2e5 results through their own dataset."""
    base = [f"{owner}/tramil-edaic-features", f"{owner}/tramil-edaic-instances"]
    if mode == "kernel":
        return base, [f"{owner}/papersmith-extract-wavlm", *(f"{owner}/papersmith-{u}" for u in upstream_jobs)]
    ds = [f"{data_owner}/tramil-edaic-wavlm"]
    if upstream_jobs:
        ds.append(f"{data_owner}/tramil-e1v07-upstream")
        if any(u.startswith("e2e5") for u in upstream_jobs):
            ds.append(f"{data_owner}/tramil-e2e5-upstream")
    return [*base, *ds], []


def select(table, jobs):
    if jobs == "all":
        return list(table)
    if jobs in ("e2e5", "e1v07"):
        return [j for j in table if j.startswith(jobs)]
    names = jobs.split(",")
    bad = [n for n in names if n not in table]
    if bad:
        raise SystemExit(f"unknown job(s) {bad}; known: {sorted(table)}")
    return names


def main(argv=None):
    ap = argparse.ArgumentParser(prog="gen_jobs.py", description=__doc__.split("\n\n")[0])
    ap.add_argument("--owner", default="alejopatio")
    ap.add_argument("--data-owner")
    ap.add_argument("--inputs", choices=("kernel", "dataset"), default="kernel")
    ap.add_argument("--jobs", default="e2e5")
    ap.add_argument("--ref", default="feat/edaic-e2e5-v07")
    ap.add_argument("--target", default=str(Path(__file__).resolve().parents[2]))
    ap.add_argument("--skills")
    ap.add_argument("--upstream-prefix", default="e1v07", help="name prefix of the E1 jobs feeding E2 to E5")
    a = ap.parse_args(argv)
    table = job_table(a.upstream_prefix)
    names = select(table, a.jobs)
    remote_cli = _load_remote_cli(a.skills or _find_skill_dir(a.target))
    jobfolder = remote_cli.JOBFOLDER
    jobfolder._PIN_CONDITION_CHECKS["pin-published"] = lambda **kw: None
    for name in names:
        module, kwargs, smoke, up = table[name]
        datasets, sources = inputs_for(a.owner, a.data_owner or a.owner, a.inputs, name, up)
        d = jobfolder.generate_job(
            target=Path(a.target), service="kaggle", job_name=name, product="TRAMIL", repo_url=REPO_URL,
            repo_ref=a.ref, clone_paths=["src/TRAMIL"], run_module=module, run_function="run", run_kwargs=kwargs,
            smoke_module=module, smoke_function="run", smoke_kwargs=smoke, smoke_required_evidence=EV,
            accelerator_kind="cuda", accelerator_architectures=["sm_75"],
            environment_requirements=["sentence-transformers"], input_datasets=datasets,
            accept_unresolved_reads=True, regenerate=True)
        meta = json.loads((d / "kernel-metadata.json").read_text())
        meta["kernel_sources"] = sources
        (d / "kernel-metadata.json").write_text(json.dumps(meta))
        print(name, d, a.inputs)


if __name__ == "__main__":
    main()
