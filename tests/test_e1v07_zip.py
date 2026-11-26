"""U1: one zip of unit results per job; import_upstream reads zips and loose JSON without overwriting."""
import json
import zipfile

import pytest

from TRAMIL import e1v07_run as run
from test_e1v07_run import SMALL, data  # noqa: F401  (data is a module-scoped fixture)


def _units(out, phase, n, tag="x"):
    for i in range(n):
        (out / phase).mkdir(parents=True, exist_ok=True)
        (out / phase / f"u{i}.json").write_text(json.dumps({"i": i, "tag": tag}))


def test_zip_round_trip_holds_every_unit_and_hpo_then_removes_the_loose_units(tmp_path):
    out = tmp_path / "E1_v07"
    _units(out, "real", 4)
    _units(out, "hpo", 2)
    z = run.zip_outputs(out, ["real"], shard="1/2")
    assert z[0].name == "E1_v07_real_s1of2.zip"
    with zipfile.ZipFile(z[0]) as f:
        assert sorted(f.namelist()) == sorted(f"E1_v07/real/u{i}.json" for i in range(4))
        assert json.loads(f.read("E1_v07/real/u3.json")) == {"i": 3, "tag": "x"}
    assert not list((out / "real").glob("*.json")) and len(list((out / "hpo").glob("*.json"))) == 2
    assert not list(out.glob("*.tmp")) and not list(out.glob("*.part"))


def test_search_zip_also_holds_the_frozen_hpo(tmp_path):
    out = tmp_path / "E1_v07"
    _units(out, "search", 3)
    _units(out, "hpo", 2)
    (z,) = run.zip_outputs(out, ["search"])
    assert z.name == "E1_v07_search.zip"
    with zipfile.ZipFile(z) as f:
        assert {n.split("/")[1] for n in f.namelist()} == {"search", "hpo"} and len(f.namelist()) == 5


def test_loose_units_are_deleted_only_after_a_verified_zip(tmp_path, monkeypatch):
    out = tmp_path / "E1_v07"
    _units(out, "real", 3)
    monkeypatch.setattr(zipfile.ZipFile, "testzip", lambda self: "u1.json")
    with pytest.raises(RuntimeError):
        run.zip_outputs(out, ["real"])
    assert len(list((out / "real").glob("*.json"))) == 3 and not list(out.glob("*.zip"))
    assert not list(out.glob("*.tmp")) and not list(out.glob("*.part"))


def test_a_rerun_merges_with_the_existing_zip_and_resume_sees_the_units(tmp_path, data):  # noqa: F811
    out = tmp_path / "E1_v07"
    _units(out, "real", 2)
    run.zip_outputs(out, ["real"])
    _units(out, "real", 1, tag="late")  # u0 reproduced after the zip, loose again
    (z,) = run.zip_outputs(out, ["real"])
    with zipfile.ZipFile(z) as f:
        assert sorted(f.namelist()) == ["E1_v07/real/u0.json", "E1_v07/real/u1.json"]
        assert json.loads(f.read("E1_v07/real/u0.json"))["tag"] == "late"
    assert not (out / "real" / "u1.json").exists()
    assert run.unpack_own_zips(out) == 2 and (out / "real" / "u1.json").exists()


def _upstream(tmp_path):
    root = tmp_path / "in" / "notebooks" / "a" / "job" / "E1_v07"
    (root / "real").mkdir(parents=True)
    (root / "real" / "loose.json").write_text('{"v": "loose"}')
    (root / "real" / "both.json").write_text('{"v": "loose-both"}')
    with zipfile.ZipFile(root / "E1_v07_real_s1of2.zip", "w") as f:
        f.writestr("E1_v07/real/zipped.json", '{"v": "zip"}')
        f.writestr("E1_v07/real/both.json", '{"v": "zip-both"}')
        f.writestr("E1_v07/hpo/hpo_r0_f0.json", '{"v": "hpo"}')
        f.writestr("E1_v07/../evil.json", "{}")
        f.writestr("E1_v07/notes/x.json", "{}")
    return tmp_path / "in"


def test_import_reads_zips_and_loose_json_and_never_overwrites(tmp_path):
    out = tmp_path / "out"
    (out / "real").mkdir(parents=True)
    (out / "real" / "zipped.json").write_text('{"v": "mine"}')
    up = _upstream(tmp_path)
    n = run.import_upstream([up], out)
    got = {p.relative_to(out).as_posix(): json.loads(p.read_text())["v"] for p in out.rglob("*.json")}
    assert got["real/zipped.json"] == "mine" and got["real/loose.json"] == "loose"
    assert got["real/both.json"] in {"loose-both", "zip-both"} and got["hpo/hpo_r0_f0.json"] == "hpo"
    assert set(got) == {"real/zipped.json", "real/loose.json", "real/both.json", "hpo/hpo_r0_f0.json"}
    assert n == 3 and run.import_upstream([up], out) == 0


def test_smoke_runs_keep_loose_units_and_a_real_run_zips(tmp_path, monkeypatch, data):  # noqa: F811
    monkeypatch.setattr(run, "load_data", lambda input_root=None, limit_bags=None: (data, {}))
    smoke = tmp_path / "smoke"
    run.run("search", smoke, limit_bags=5, **SMALL)
    assert list((smoke / "search").glob("*.json")) and not list(smoke.glob("*.zip"))
    real = tmp_path / "real"
    run.run("search", real, **SMALL)
    assert list(real.glob("E1_v07_search*.zip")) and not list((real / "search").glob("*.json"))
    assert len(list((real / "hpo").glob("hpo_r0_f*.json"))) == 5
    again = run.run("search", real, **SMALL)  # resumed: the zipped units count as done
    assert again["search"]["ran"] == 0
