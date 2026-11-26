"""U2: the job generator's pure parts: the job table, the two input modes and the job selection."""
import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location("gen_jobs", Path(__file__).resolve().parents[1] / "tools/kaggle/gen_jobs.py")
gj = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gj)


def test_kernel_mode_keeps_the_cache_kernel_and_upstream_kernel_outputs():
    ds, src = gj.inputs_for("alejopatio", "alejopatio", "kernel", "e1v07-real-a", ["e1v07-search-a", "e1v07-search-b"])
    assert ds == ["alejopatio/tramil-edaic-features", "alejopatio/tramil-edaic-instances"]
    assert src == ["alejopatio/papersmith-extract-wavlm", "alejopatio/papersmith-e1v07-search-a",
                   "alejopatio/papersmith-e1v07-search-b"]


def test_dataset_mode_reads_the_cache_and_the_upstream_zips_from_shared_datasets():
    ds, src = gj.inputs_for("alejopatio", "main", "dataset", "e2e5-real-a", ["e1v07-real-a", "e2e5-search-a"])
    assert src == [] and ds[:2] == ["alejopatio/tramil-edaic-features", "alejopatio/tramil-edaic-instances"]
    assert ds[2:] == ["main/tramil-edaic-wavlm", "main/tramil-e1v07-upstream", "main/tramil-e2e5-upstream"]
    ds, _ = gj.inputs_for("a", "a", "dataset", "e1v07-search-a", [])
    assert ds[2:] == ["a/tramil-edaic-wavlm"]  # a search job has no upstream results


def test_job_table_and_selection_never_regenerate_e1v07_by_default():
    table = gj.job_table("e1v07")
    e2 = gj.select(table, "e2e5")
    assert sorted(e2) == ["e2e5-real-a", "e2e5-real-b", "e2e5-report", "e2e5-search-a", "e2e5-search-b"]
    assert not any(j.startswith("e1v07") for j in e2) and "e1v07-report" in gj.select(table, "all")
    assert gj.select(table, "e1v07-timing,e2e5-report") == ["e1v07-timing", "e2e5-report"]
    with pytest.raises(SystemExit):
        gj.select(table, "nope")
    assert table["e2e5-real-a"][1]["shard"] == "1/2" and table["e2e5-real-b"][1]["shard"] == "2/2"
