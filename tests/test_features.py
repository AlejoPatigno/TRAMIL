import numpy as np
import pytest

from TRAMIL import features

SR = 16000


def _speech(seconds, seed=0, f0=140.0):
    t = np.arange(int(seconds * SR)) / SR
    rng = np.random.default_rng(seed)
    x = 0.2 * np.sin(2 * np.pi * f0 * t) * (1 + 0.3 * np.sin(2 * np.pi * 3 * t)) + 0.01 * rng.standard_normal(t.size)
    return x.astype(np.float32)


@pytest.fixture(scope="module")
def smile():
    return features.make_smile()


def test_egemaps_split_is_7_pause_dependent_plus_81_rest(smile):
    names = list(smile.feature_names)
    tmp, rest = features.split_indices(names)
    assert len(names) == 88 and len(tmp) == 7 and len(rest) == 81
    assert set(tmp).isdisjoint(rest) and sorted(tmp + rest) == list(range(88))
    assert "equivalentSoundLevel_dBp" in {names[i] for i in tmp}


def test_acoustic_vector_is_88_finite_and_temporal_first(smile):
    x = np.concatenate([_speech(1.5), np.zeros(SR), _speech(1.5, seed=1)])
    v = features.acoustic_vector(x, SR, gain=1.0, smile=smile)
    assert v.shape == (88,) and np.isfinite(v).all()


def test_temporal_block_sees_the_full_answer_and_rest_sees_silence_removed(smile):
    names = list(smile.feature_names)
    tmp, rest = features.split_indices(names)
    x = np.concatenate([_speech(1.5), np.zeros(2 * SR), _speech(1.5, seed=1)])
    g = 0.5
    v = features.acoustic_vector(x, SR, gain=g, smile=smile)
    full = smile.process_signal(g * x, SR).iloc[0].to_numpy()
    cut = smile.process_signal(features.remove_silence(g * x, SR), SR).iloc[0].to_numpy()
    assert np.allclose(v[:7], full[tmp], equal_nan=False)
    assert np.allclose(v[7:], cut[rest])
    # the two chains genuinely differ on this signal, so the split is observable
    assert not np.allclose(full[rest], cut[rest])


def test_remove_silence_drops_exact_zero_runs_and_keeps_speech():
    s = _speech(1.0)
    x = np.concatenate([s, np.zeros(SR), s])
    y = features.remove_silence(x, SR)
    assert 0.9 * s.size * 2 <= y.size <= 1.05 * s.size * 2
    assert np.abs(y).max() > 0.1


def test_session_gain_hits_target_level_without_clipping():
    quiet = [_speech(2.0, seed=i) * 0.05 for i in range(3)]
    g = features.session_gain(quiet, SR, target_db=-26.0)
    lvl = 20 * np.log10(np.sqrt(np.mean(np.concatenate(quiet) ** 2)) * g)
    assert abs(lvl - (-26.0)) < 0.5
    loud = [np.clip(_speech(2.0) * 20, -0.9, 0.9)]
    gl = features.session_gain(loud, SR, target_db=-6.0)
    assert gl * 0.9 <= 1.0 + 1e-6  # limited so that no sample clips


def test_text_embeddings_are_unit_norm_768():
    e = features.embed_texts(["i feel tired all the time", "good", ""])
    assert e.shape == (3, 768)
    assert np.allclose(np.linalg.norm(e, axis=1), 1.0, atol=1e-5)


def test_bag_cache_is_resumable():
    import tempfile
    import wave
    from pathlib import Path

    from TRAMIL.bags import Bag

    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        wavs = []
        for k in range(2):
            p = d / f"1_{k:03d}_a.wav"
            with wave.open(str(p), "wb") as w:
                w.setnchannels(1), w.setsampwidth(2), w.setframerate(SR)
                w.writeframes((_speech(1.0, seed=k) * 20000).astype("<i2").tobytes())
            wavs.append(p)
        bag = Bag(subject=1, y=0, y_distributed=0, texts=["a", "b"], wavs=wavs, idx=[0, 1])
        X1 = features._acoustic_bag(bag, d / "c", features.make_smile())

        class Boom:
            def __getattr__(self, name):
                raise AssertionError("extractor used although cached")

        X2 = features._acoustic_bag(bag, d / "c", Boom())
        assert X1.shape == (2, 88) and np.array_equal(X1, X2)


def test_nonfinite_instances_are_excluded_from_the_bag():
    import tempfile
    from pathlib import Path

    from TRAMIL.bags import Bag

    with tempfile.TemporaryDirectory() as d:
        X = np.ones((3, 88), np.float32)
        X[1, 4] = np.nan
        E = np.arange(3, dtype=np.float32)[:, None] * np.ones((3, 768), np.float32)
        np.save(Path(d) / "9_X.npy", X)
        np.save(Path(d) / "9_E.npy", E)
        bag = Bag(subject=9, y=1, y_distributed=1, texts=["a", "b", "c"], wavs=[Path("x")] * 3, idx=[0, 1, 2])
        Xv, Ev = features.bag_features(bag, d)
        assert Xv.shape == (2, 88) and np.isfinite(Xv).all()
        assert Ev[:, 0].tolist() == [0.0, 2.0]  # the text rows of the surviving instances


def test_text_encoder_uses_gpu_when_available(monkeypatch):
    import torch
    from TRAMIL import features as F

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert F._text_device() == "cuda"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert F._text_device() == "cpu"
