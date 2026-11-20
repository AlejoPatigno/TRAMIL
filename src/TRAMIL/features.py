"""Frozen feature extraction: eGeMAPSv02 split 7 + 81 (Ec. 5, 6, 68) and all-mpnet-base-v2 embeddings (Ec. 7, 8)."""
import os
import wave
from pathlib import Path

import numpy as np

__provenance__ = {
    "revision": "research-concept-residual-semantic-prior-bipartite-mil-r06.md",
    "sections": ["4.1", "4.2", "18.9"],
    "equations": [5, 6, 7, 8, 68],
    "invariants": [
        "egemaps_split_is_7_pause_dependent_plus_81_rest",
        "acoustic_vector_is_88_finite_and_temporal_first",
        "temporal_block_sees_the_full_answer_and_rest_sees_silence_removed",
        "remove_silence_drops_exact_zero_runs_and_keeps_speech",
        "session_gain_hits_target_level_without_clipping",
        "text_embeddings_are_unit_norm_768",
        "bag_cache_is_resumable",
        "nonfinite_instances_are_excluded_from_the_bag",
        "wavlm_layer6_mean_pool_replaces_ec68_acoustic_map",
        "wavlm_cache_rows_align_with_the_egemaps_instance_order",
        "wavlm_cache_hit_does_not_load_the_model",
    ],
    # PoC deviation chosen by the operator (2026-11-19): for the WavLM variant of E1, the acoustic instance map of
    # Ec. 68 (eGeMAPS 7+81) is replaced by frozen microsoft/wavlm-base-plus hidden layer 6, mean-pooled over time
    # (768-d). No silence removal, no session gain; the 7+81 pause split does not apply. Default stays eGeMAPS.
    "deviation": "wavlm variant replaces the Ec. 68 acoustic map; eGeMAPS path unchanged",
}

SR = 16000
WAVLM_MODEL = "microsoft/wavlm-base-plus"
WAVLM_LAYER = 6  # hidden_states[0] is the CNN/embedding output
WAVLM_MIN = 400  # one WavLM frame (25 ms at 16 kHz)
# Longest single forward pass. Self-attention memory grows with T^2: one ~180 s answer asked a T4 for 3.87 GiB in one
# softmax (OOM, 2026-11-20). 20 s is on the order of WavLM's pretraining crops; longer segments run in equal windows.
WAVLM_WINDOW = 20 * SR
TEXT_MODEL = "sentence-transformers/all-mpnet-base-v2"
# The six temporal descriptors of eGeMAPSv02 plus the equivalent sound level (Section 18.9).
TEMPORAL = (
    "loudnessPeaksPerSec",
    "VoicedSegmentsPerSec",
    "MeanVoicedSegmentLengthSec",
    "StddevVoicedSegmentLengthSec",
    "MeanUnvoicedSegmentLength",
    "StddevUnvoicedSegmentLength",
    "equivalentSoundLevel_dBp",
)
TARGET_DB = -26.0  # session active-speech RMS target in dBFS; the proposal says only "a fixed target" (choice, reported)
FRAME_S = 0.025
SILENCE_REL_DB = 35.0  # frames this far below the response's loud frames are non-speech
DESCRIPTORS = (  # Section 4.2 table, fixed order
    "Little interest or pleasure in doing things",
    "Feeling down, depressed, irritable or hopeless",
    "Trouble falling or staying asleep, or sleeping too much",
    "Feeling tired or having little energy",
    "Poor appetite or overeating",
    "Feeling bad about yourself – or that you are a failure or have let yourself or your family down",
    "Trouble concentrating on things, such as school work, reading or watching television",
    "Moving or speaking so slowly that other people could have noticed? Or the opposite – being so fidgety or restless "
    "that you have been moving around a lot more than usual",
)


def make_smile():
    import opensmile

    return opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02, feature_level=opensmile.FeatureLevel.Functionals
    )


def split_indices(names):
    tmp = [names.index(n) for n in TEMPORAL]
    rest = [i for i in range(len(names)) if i not in tmp]
    return tmp, rest


def read_wav(path):
    with wave.open(str(path)) as w:
        assert w.getframerate() == SR and w.getnchannels() == 1
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.0


def remove_silence(x, sr=SR):
    """S: drop exact-zero runs (scrubbed spans) and frames far below the response's loud frames."""
    n = int(FRAME_S * sr)
    k = len(x) // n
    if k == 0:
        return x
    fr = x[: k * n].reshape(k, n)
    rms = np.sqrt((fr**2).mean(1))
    db = 20 * np.log10(np.maximum(rms, 1e-10))
    keep = (rms > 0) & (db >= np.percentile(db, 90) - SILENCE_REL_DB)
    out = fr[keep].reshape(-1)
    return np.concatenate([out, x[k * n :]]) if keep[-1] else out


def session_gain(responses, sr=SR, target_db=TARGET_DB):
    """g_b: brings the active-speech RMS of the session's answers to target_db, limited so no sample clips."""
    voiced = np.concatenate([remove_silence(r, sr) for r in responses])
    rms = np.sqrt(np.mean(voiced**2))
    g = 10 ** (target_db / 20) / max(rms, 1e-10)
    peak = max(float(np.abs(r).max()) for r in responses)
    return min(g, 1.0 / max(peak, 1e-10))


def acoustic_vector(x, sr, gain, smile):
    """Ec. 68: [phi_temp(g a) ; phi_rest(S(g a))] in R^88."""
    tmp, rest = split_indices(list(smile.feature_names))
    gx = (gain * x).astype(np.float32)
    full = smile.process_signal(gx, sr).iloc[0].to_numpy()
    cut = smile.process_signal(remove_silence(gx, sr), sr).iloc[0].to_numpy()
    return np.concatenate([full[tmp], cut[rest]]).astype(np.float32)


def acoustic_from_env():
    """Acoustic source of the E1 entry points: $TRAMIL_ACOUSTIC (egemaps | wavlm), default egemaps."""
    return os.environ.get("TRAMIL_ACOUSTIC") or "egemaps"


def load_wavlm():
    """(model, feature extractor): frozen WavLM base+ in eval mode on the GPU when available, else the CPU."""
    import torch
    from transformers import AutoFeatureExtractor, WavLMModel

    model = WavLMModel.from_pretrained(WAVLM_MODEL).eval().to("cuda" if torch.cuda.is_available() else "cpu")
    return model, AutoFeatureExtractor.from_pretrained(WAVLM_MODEL)


def wavlm_vector(x, sr, model, extractor):
    """Frozen WavLM layer-6 hidden states mean-pooled over time: (hidden_size,) float32; 768-d for base+.

    One segment per forward pass: the base model has no attention mask, so batching padded segments would change
    the result. Segments below one frame are zero-padded to WAVLM_MIN samples. Segments longer than WAVLM_WINDOW are
    split into near-equal windows (each > WAVLM_WINDOW / 2), run one at a time, and pooled over all their frames; the
    extractor normalization is applied once to the whole segment."""
    import torch

    assert sr == SR
    x = np.asarray(x, dtype=np.float32)
    if len(x) < WAVLM_MIN:
        x = np.concatenate([x, np.zeros(WAVLM_MIN - len(x), np.float32)])
    full = extractor(x, sampling_rate=sr, return_tensors="np")["input_values"][0]
    device = next(model.parameters()).device
    frames = []
    with torch.no_grad():
        for chunk in np.array_split(full, -(-len(full) // WAVLM_WINDOW)):
            inp = torch.from_numpy(np.ascontiguousarray(chunk)[None]).to(device)
            frames.append(model.eval()(inp, output_hidden_states=True).hidden_states[WAVLM_LAYER][0].float().cpu())
    return torch.cat(frames).mean(0).numpy().astype(np.float32)


_wl = None


def _worker_wavlm():
    global _wl
    if _wl is None:
        _wl = load_wavlm()
    return _wl


def _wavlm_bag(bag, cache_dir, wavlm=None):
    """W_b (n_instances, hidden) in the row order of X_b; cached as {subject}_W.npy, the model loaded only on a miss."""
    path = Path(cache_dir) / f"{bag.subject}_W.npy"
    if path.exists():
        return np.load(path)
    model, extractor = wavlm or _worker_wavlm()
    W = np.stack([wavlm_vector(read_wav(w), SR, model, extractor) for w in bag.wavs])
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    np.save(path, W)
    return W


_model = None


def _text_device():
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def embed_texts(texts, batch_size=32):
    """phi_t (Ec. 7, 8): frozen all-mpnet-base-v2, 768-d, unit Euclidean norm."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(TEXT_MODEL, device=_text_device())
    return _model.encode(list(texts), batch_size=batch_size, normalize_embeddings=True).astype(np.float32)


def _acoustic_bag(bag, cache_dir, smile):
    path = Path(cache_dir) / f"{bag.subject}_X.npy"
    if path.exists():
        return np.load(path)
    smile = smile or _worker_smile()  # built only on a cache miss: a cached bag needs no opensmile
    xs = [read_wav(w) for w in bag.wavs]
    g = session_gain(xs)
    X = np.stack([acoustic_vector(x, SR, g, smile) for x in xs])
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    np.save(path, X)
    return X


def _acoustic_worker(args):
    bag, cache_dir = args
    _acoustic_bag(bag, cache_dir, _worker_smile())
    return bag.subject


_ws = None


def _worker_smile():
    global _ws
    if _ws is None:
        _ws = make_smile()
    return _ws


def extract_all(bags, cache_dir, workers=1, acoustic="egemaps"):
    """Resumable: acoustic X_b per subject (process pool) then text E_b per subject; cached under cache_dir.
    acoustic="wavlm": only the W_b caches, sequentially (one model on the device); no text embeddings."""
    from concurrent.futures import ProcessPoolExecutor

    if acoustic == "wavlm":
        for b in bags:
            _wavlm_bag(b, cache_dir)
        return
    todo = [b for b in bags if not (Path(cache_dir) / f"{b.subject}_X.npy").exists()]
    if todo and workers > 1:
        with ProcessPoolExecutor(workers) as ex:
            list(ex.map(_acoustic_worker, [(b, cache_dir) for b in todo]))
    for b in bags:
        bag_features(b, cache_dir)


def bag_features(bag, cache_dir, smile=None, acoustic="egemaps", text_dir=None):
    """(X_b, E_b) for one bag, cached per subject under cache_dir (resumable); non-finite instances dropped.
    acoustic="wavlm" reads {subject}_W.npy in place of {subject}_X.npy; E_b comes from text_dir when given."""
    X = _wavlm_bag(bag, cache_dir) if acoustic == "wavlm" else _acoustic_bag(bag, cache_dir, smile)
    path = Path(text_dir or cache_dir) / f"{bag.subject}_E.npy"
    if not path.exists():
        np.save(path, embed_texts(bag.texts))
    E = np.load(path)
    ok = np.isfinite(X).all(1)  # Section 3: non-finite vectors are excluded from the bag
    return X[ok], E[ok]
