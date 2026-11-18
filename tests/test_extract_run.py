import json
import wave
import zipfile

import numpy as np
import pandas as pd
import pytest

from TRAMIL import extract_run, features

ITEMS = [f"PHQ8_{i}" for i in range(1, 9)]


def _wav_bytes(path, seconds=1.0, f0=150.0):
    t = np.arange(int(seconds * features.SR)) / features.SR
    x = (0.2 * np.sin(2 * np.pi * f0 * t) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(features.SR)
        w.writeframes(x.tobytes())


def _make_input(root, as_zip):
    """Mirrors the private dataset: labels, instances CSV and audio under instances_audio/<id>/<id>_<idx>_a.wav."""
    root.mkdir(parents=True)
    rows = [[s, *([2] * 8), 1, split] for s, split in [(301, "train"), (302, "dev"), (303, "train")]]
    pd.DataFrame(rows, columns=["Participant", *ITEMS, "Depression_label", "split"]).to_csv(
        root / "detailed_lables.csv", index=False
    )
    inst = [[s, i, f"answer {s} {i}", 1.0] for s in (301, 302, 303) for i in (0, 1)]
    pd.DataFrame(inst, columns=["id", "idx", "a_text", "a_dur"]).to_csv(root / "instances_clean.csv", index=False)
    stage = root / "stage"
    for s in (301, 302, 303):
        (stage / "instances_audio" / str(s)).mkdir(parents=True)
        for i in (0, 1):
            _wav_bytes(stage / "instances_audio" / str(s) / f"{s}_{i:03d}_a.wav", f0=120 + 10 * i)
    if as_zip:
        with zipfile.ZipFile(root / "instances_audio.zip", "w") as z:
            for p in sorted((stage / "instances_audio").rglob("*.wav")):
                z.write(p, p.relative_to(stage).as_posix())
        import shutil

        shutil.rmtree(stage)
    else:
        (stage / "instances_audio").rename(root / "instances_audio")
        stage.rmdir()
    return root


@pytest.fixture(autouse=True)
def fake_embeddings(monkeypatch):
    def embed(texts, batch_size=32):
        e = np.zeros((len(list(texts)), 768), dtype=np.float32)
        e[:, 0] = 1.0
        return e

    monkeypatch.setattr(features, "embed_texts", embed)


@pytest.mark.parametrize("as_zip", [True, False])
def test_input_root_resolves_folder_or_zip(tmp_path, as_zip):
    root = _make_input(tmp_path / "in", as_zip)
    out = tmp_path / "out"
    extract_run.run(root, out, workers=1)
    for s in (301, 302, 303):
        assert (out / f"{s}_X.npy").exists() and (out / f"{s}_E.npy").exists()
    X = np.load(out / "301_X.npy")
    assert X.shape == (2, 88) and np.isfinite(X).all()


def test_extraction_covers_train_and_dev_sessions(tmp_path):
    root = _make_input(tmp_path / "in", True)
    summary = extract_run.run(root, tmp_path / "out", workers=1)
    assert summary["sessions"] == 3 and summary["splits"] == {"train": 2, "dev": 1}


def test_limit_restricts_sessions_for_smoke(tmp_path):
    root = _make_input(tmp_path / "in", True)
    out = tmp_path / "out"
    summary = extract_run.run(root, out, workers=1, limit=2)
    assert summary["sessions"] == 2
    assert sorted(p.name for p in out.glob("*_X.npy")) == ["301_X.npy", "302_X.npy"]


def test_input_root_found_below_a_mount_directory(tmp_path):
    root = _make_input(tmp_path / "mount" / "datasets" / "owner" / "tramil", False)
    out = tmp_path / "out"
    extract_run.run(tmp_path / "mount", out, workers=1, limit=1)
    assert (out / "301_X.npy").exists()


def test_smoke_stamp_carries_commit_digest_device(tmp_path, monkeypatch):
    monkeypatch.setenv("FORGE_RUN_MODE", "smoke")
    monkeypatch.setenv("FORGE_CLONE_COMMIT", "a" * 40)
    monkeypatch.chdir(tmp_path)
    root = _make_input(tmp_path / "in", True)
    extract_run.run(root, tmp_path / "out", workers=1, limit=2)
    stamp = json.loads((tmp_path / "shard.json").read_text())
    assert stamp["evidence"]["commit"] and len(stamp["evidence"]["sourcesDigest"]) == 64
    assert stamp["environment"]["device"]["kind"] in {"cpu", "cuda"}
    assert stamp["runs"]["sessions"] == 2


def test_no_stamp_outside_the_runner(tmp_path, monkeypatch):
    monkeypatch.delenv("FORGE_RUN_MODE", raising=False)
    monkeypatch.chdir(tmp_path)
    extract_run.run(_make_input(tmp_path / "in", False), tmp_path / "out", workers=1, limit=1)
    assert not (tmp_path / "shard.json").exists()
