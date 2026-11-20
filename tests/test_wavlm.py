"""Invariants of the frozen WavLM layer-6 acoustic path. Uses a tiny random-init WavLM: no weights are downloaded."""
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from TRAMIL import extract_run, features
from TRAMIL.bags import Bag

SR = 16000


@pytest.fixture(scope="module")
def tiny():
    from transformers import Wav2Vec2FeatureExtractor, WavLMConfig, WavLMModel

    torch.manual_seed(0)
    cfg = WavLMConfig(
        hidden_size=32, num_hidden_layers=8, num_attention_heads=2, intermediate_size=64,
        conv_dim=(16,) * 7, conv_stride=(5, 2, 2, 2, 2, 2, 2), conv_kernel=(10, 3, 3, 3, 3, 2, 2),
        num_conv_pos_embeddings=16, num_conv_pos_embedding_groups=2, num_buckets=32, max_bucket_distance=64,
    )
    model = WavLMModel(cfg).eval()
    ext = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=SR, do_normalize=True, return_attention_mask=False)
    return model, ext


def _x(seconds, seed=0):
    return (0.2 * np.random.default_rng(seed).standard_normal(int(seconds * SR))).astype(np.float32)


def test_wavlm_vector_is_hidden_size_float32_finite_and_deterministic(tiny):
    model, ext = tiny
    v = features.wavlm_vector(_x(1.0), SR, model, ext)
    assert v.shape == (32,) and v.dtype == np.float32 and np.isfinite(v).all()
    assert np.array_equal(v, features.wavlm_vector(_x(1.0), SR, model, ext))


def test_wavlm_vector_is_layer_6_mean_over_time(tiny):
    model, ext = tiny
    x = _x(1.0, seed=3)
    v = features.wavlm_vector(x, SR, model, ext)
    inp = torch.from_numpy(ext(x, sampling_rate=SR, return_tensors="np")["input_values"])
    with torch.no_grad():
        hs = model(inp, output_hidden_states=True).hidden_states
    assert len(hs) == 9
    assert np.allclose(v, hs[6].mean(1)[0].numpy(), atol=1e-5)
    assert not np.allclose(v, hs[8].mean(1)[0].numpy(), atol=1e-3)  # not the last layer


def test_long_segments_run_in_equal_windows_and_pool_all_frames(tiny, monkeypatch):
    """A segment longer than WAVLM_WINDOW is split into near-equal windows (bounded attention memory); the vector is
    the layer-6 mean over every frame of every window, with the feature extractor applied once to the whole segment."""
    model, ext = tiny
    monkeypatch.setattr(features, "WAVLM_WINDOW", SR)  # 1 s windows
    x = _x(2.5, seed=5)
    seen = []
    orig = model.forward

    def spy(input_values, **kw):
        seen.append(input_values.shape[-1])
        return orig(input_values, **kw)

    monkeypatch.setattr(model, "forward", spy)
    v = features.wavlm_vector(x, SR, model, ext)
    assert len(seen) == 3 and max(seen) <= SR and min(seen) >= SR // 2
    monkeypatch.setattr(model, "forward", orig)
    full = ext(x, sampling_rate=SR, return_tensors="np")["input_values"][0]
    frames = []
    with torch.no_grad():
        for c in np.array_split(full, 3):
            frames.append(model(torch.from_numpy(c[None]), output_hidden_states=True).hidden_states[6][0])
    assert np.allclose(v, torch.cat(frames).mean(0).numpy(), atol=1e-5)


def test_segments_within_one_window_are_unchanged(tiny, monkeypatch):
    model, ext = tiny
    x = _x(1.0, seed=3)
    monkeypatch.setattr(features, "WAVLM_WINDOW", SR)
    v = features.wavlm_vector(x, SR, model, ext)
    inp = torch.from_numpy(ext(x, sampling_rate=SR, return_tensors="np")["input_values"])
    with torch.no_grad():
        ref = model(inp, output_hidden_states=True).hidden_states[6].mean(1)[0].numpy()
    assert np.allclose(v, ref, atol=1e-5)


def test_short_segments_are_zero_padded_to_one_wavlm_frame(tiny):
    model, ext = tiny
    short = _x(0.005)  # 80 samples, below one frame
    v = features.wavlm_vector(short, SR, model, ext)
    assert v.shape == (32,) and np.isfinite(v).all()
    padded = np.concatenate([short, np.zeros(400 - len(short), np.float32)])
    assert np.allclose(v, features.wavlm_vector(padded, SR, model, ext))


def _bag(tmp_path, n=3):
    import wave

    wavs = []
    for k in range(n):
        p = tmp_path / f"7_{k:03d}_a.wav"
        with wave.open(str(p), "wb") as w:
            w.setnchannels(1), w.setsampwidth(2), w.setframerate(SR)
            w.writeframes((_x(0.5 + 0.1 * k, seed=k) * 20000).astype("<i2").tobytes())
        wavs.append(p)
    return Bag(subject=7, y=0, y_distributed=0, texts=list("abc")[:n], wavs=wavs, idx=list(range(n)))


def test_wavlm_cache_has_one_row_per_instance_in_bag_order_and_is_reused_without_the_model(tiny, tmp_path, monkeypatch):
    bag = _bag(tmp_path)
    W = features._wavlm_bag(bag, tmp_path / "c", tiny)
    assert W.shape == (3, 32) and (tmp_path / "c" / "7_W.npy").exists()
    for i, w in enumerate(bag.wavs):  # row i belongs to instance i, the order of _X.npy
        assert np.allclose(W[i], features.wavlm_vector(features.read_wav(w), SR, *tiny), atol=1e-5)
    monkeypatch.setattr(features, "load_wavlm", lambda: (_ for _ in ()).throw(AssertionError("model loaded on a cache hit")))
    assert np.array_equal(features._wavlm_bag(bag, tmp_path / "c"), W)


def test_bag_features_wavlm_aligns_with_text_rows_and_drops_nonfinite(tmp_path):
    W = np.ones((3, 8), np.float32)
    W[1, 2] = np.nan
    np.save(tmp_path / "9_W.npy", W)
    np.save(tmp_path / "9_E.npy", np.arange(3, dtype=np.float32)[:, None] * np.ones((3, 4), np.float32))
    bag = SimpleNamespace(subject=9, wavs=[], texts=[])
    X, E = features.bag_features(bag, tmp_path, acoustic="wavlm")
    assert X.shape == (2, 8) and E[:, 0].tolist() == [0.0, 2.0]


def test_extract_run_wavlm_writes_only_w_files(tiny, tmp_path, monkeypatch):
    from test_extract_run import _make_input

    monkeypatch.setattr(features, "load_wavlm", lambda: tiny)
    monkeypatch.setattr(features, "make_smile", lambda: (_ for _ in ()).throw(AssertionError("opensmile used")))
    monkeypatch.delenv("FORGE_RUN_MODE", raising=False)
    root = _make_input(tmp_path / "in", True)
    summary = extract_run.run(root, tmp_path / "out", workers=1, acoustic="wavlm")
    assert summary["sessions"] == 3 and summary["acoustic"] == "wavlm"
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == ["301_W.npy", "302_W.npy", "303_W.npy"]
    assert np.load(tmp_path / "out" / "301_W.npy").shape == (2, 32)


def test_acoustic_env_var_selects_wavlm(tiny, tmp_path, monkeypatch):
    from test_extract_run import _make_input

    monkeypatch.setattr(features, "load_wavlm", lambda: tiny)
    monkeypatch.setenv("TRAMIL_ACOUSTIC", "wavlm")
    monkeypatch.delenv("FORGE_RUN_MODE", raising=False)
    extract_run.run(_make_input(tmp_path / "in", False), tmp_path / "out", workers=1, limit=1)
    assert (tmp_path / "out" / "301_W.npy").exists() and not (tmp_path / "out" / "301_X.npy").exists()


def test_wavlm_deviation_is_recorded_in_provenance():
    assert "wavlm_layer6_mean_pool_replaces_ec68_acoustic_map" in features.__provenance__["invariants"]
    assert "wavlm_cache_rows_align_with_the_egemaps_instance_order" in features.__provenance__["invariants"]
    assert "wavlm_cache_hit_does_not_load_the_model" in features.__provenance__["invariants"]
    assert "deviation" in features.__provenance__
