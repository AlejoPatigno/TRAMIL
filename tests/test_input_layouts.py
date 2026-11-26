"""U2: inputs are located by walking /kaggle/input whatever the mount layout (kernel sources or datasets)."""
import json
import zipfile

import pytest

from TRAMIL import e1v07_run as run

KERNEL_W = ("notebooks", "alejopatio", "papersmith-extract-wavlm", "clone", "TRAMIL", "Data", "features")
LAYOUTS = {
    "kernel": (KERNEL_W, ("notebooks", "alejopatio", "papersmith-e1v07-real-a", "clone", "TRAMIL", "Results", "E1_v07")),
    "dataset_flat": (("tramil-edaic-wavlm",), ("tramil-e1v07-upstream",)),
    "dataset_owner": (("datasets", "otheracct", "tramil-edaic-wavlm"), ("datasets", "otheracct", "tramil-e1v07-upstream")),
    "dataset_nested": (("datasets", "otheracct", "tramil-edaic-wavlm", "features", "W"),
                       ("datasets", "otheracct", "tramil-e1v07-upstream", "E1_v07")),
}


def _mount(tmp_path, w, up):
    root = tmp_path / "kaggle_input"
    (root / "datasets" / "alejopatio" / "tramil-edaic-features").mkdir(parents=True)
    (root / "datasets" / "alejopatio" / "tramil-edaic-features" / "300_E.npy").write_bytes(b"")
    inst = root / "datasets" / "alejopatio" / "tramil-edaic-instances"
    inst.mkdir(parents=True)
    (inst / "detailed_lables.csv").write_text("x")
    (inst / "instances_clean.csv").write_text("x")
    wd = root.joinpath(*w)
    wd.mkdir(parents=True)
    (wd / "300_W.npy").write_bytes(b"")
    ud = root.joinpath(*up)
    ud.mkdir(parents=True)
    (ud / "real").mkdir()
    (ud / "real" / "loose.json").write_text(json.dumps({"v": 1}))
    with zipfile.ZipFile(ud / "E1_v07_real_s2of2.zip", "w") as f:
        f.writestr("E1_v07/real/zipped.json", json.dumps({"v": 2}))
        f.writestr("E1_v07/hpo/hpo_r0_f0.json", json.dumps({"v": 3}))
    return root, wd


@pytest.mark.parametrize("layout", LAYOUTS)
def test_wavlm_cache_and_result_zips_are_found_in_any_mount_layout(tmp_path, layout):
    root, wd = _mount(tmp_path, *LAYOUTS[layout])
    assert run.find_inputs([root])["features"] == wd
    out = tmp_path / "out"
    run.import_upstream([root], out)
    got = {p.relative_to(out).as_posix() for p in out.rglob("*.json")}
    assert {"real/zipped.json", "hpo/hpo_r0_f0.json"} <= got
    assert ("real/loose.json" in got) == (layout != "dataset_flat" and layout != "dataset_owner")  # loose needs E1_v07/
