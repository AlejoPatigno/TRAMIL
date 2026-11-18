"""Remote entry point: frozen feature extraction (eGeMAPS 7+81, mpnet) for every E-DAIC session with audio."""
import hashlib
import json
import os
import platform
import subprocess
import tempfile
import zipfile
from pathlib import Path

import pandas as pd

from . import bags as bagmod
from . import features

__provenance__ = {
    "revision": "research-concept-residual-semantic-prior-bipartite-mil-r06.md",
    "sections": ["4.1", "4.2"],
    "equations": [5, 6, 7, 8, 68],
    "invariants": [
        "input_root_resolves_folder_or_zip",
        "extraction_covers_train_and_dev_sessions",
        "limit_restricts_sessions_for_smoke",
        "input_root_found_below_a_mount_directory",
        "smoke_stamp_carries_commit_digest_device",
        "no_stamp_outside_the_runner",
    ],
}

INPUT_REF = "alejopatio/tramil-edaic-instances"
SPLITS = ("train", "dev")
_MAX_DEPTH = 6


def _find(root, depth=_MAX_DEPTH):
    """Locate labels, instances CSV and audio (folder preferred over zip) at most `depth` levels below root."""
    found, root = {}, Path(root)
    for here, dirs, files in os.walk(root):
        level = len(Path(here).relative_to(root).parts)
        names = {"detailed_lables.csv": "labels", "instances_clean.csv": "instances", "instances_audio.zip": "zip"}
        for f in files:
            if f in names:
                found.setdefault(names[f], Path(here) / f)
        if "instances_audio" in dirs:
            found.setdefault("audio", Path(here) / "instances_audio")
            dirs.remove("instances_audio")  # thousands of wavs: never descend
        if level >= depth:
            dirs[:] = []
    if "labels" not in found or "instances" not in found or not ({"audio", "zip"} & found.keys()):
        raise FileNotFoundError(f"labels, instances_clean.csv and instances_audio (folder or zip) not found under {root}")
    return found


def _default_input_root():
    mounted = json.loads(os.environ.get("FORGE_INPUT_DATASETS") or "{}")
    return mounted.get(INPUT_REF) or "TRAMIL/Data"


def _default_out_dir():
    return Path(os.environ.get("FORGE_CLONE_ROOT", ".")) / "TRAMIL" / "Data" / "features"


def _all_bags(found, audio_dir):
    lab = bagmod.load_labels(found["labels"])
    inst = pd.read_csv(found["instances"])
    per_split = {s: bagmod.build_bags(inst, audio_dir, lab, s) for s in SPLITS}
    return per_split


def _stamp(summary):
    """shard.json beside the results, only under the runner (FORGE_RUN_MODE): what the smoke verdict reads."""
    mode = os.environ.get("FORGE_RUN_MODE")
    if not mode:
        return None
    src = Path(__file__).resolve().parent
    digest = hashlib.sha256(b"".join(p.read_bytes() for p in sorted(src.glob("*.py")))).hexdigest()
    commit = None
    try:
        r = subprocess.run(["git", "-C", str(src), "rev-parse", "HEAD"], capture_output=True, text=True)
        commit = r.stdout.strip() if r.returncode == 0 and r.stdout.strip() else None
    except OSError:
        pass
    commit = commit or os.environ.get("FORGE_CLONE_COMMIT")
    try:
        import torch

        cuda = torch.cuda.is_available()
        name = torch.cuda.get_device_name(0) if cuda else None
    except ImportError:
        cuda, name = False, None
    device = {"kind": "cuda", "name": name} if cuda else {"kind": "cpu", "name": platform.processor() or "cpu"}
    payload = {"mode": mode, "evidence": {"commit": commit, "sourcesDigest": digest},
               "environment": {"device": device}, "runs": summary}
    path = Path.cwd() / "shard.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def run(input_root=None, out_dir=None, workers=None, limit=None):
    """Extract the per-subject feature cache for train + dev sessions; `limit` keeps the first N subjects (smoke)."""
    found = _find(input_root or _default_input_root())
    out = Path(out_dir) if out_dir else _default_out_dir()
    workers = workers or os.cpu_count() or 1
    tmp = None
    audio_dir = found.get("audio")
    if audio_dir is None:  # ponytail: extracted to the system temp (not the fetched working dir); needs the free disk of the full zip
        tmp = tempfile.TemporaryDirectory()
        audio_dir = Path(tmp.name) / "instances_audio"
    per_split = _all_bags(found, audio_dir)
    chosen = sorted((b for bs in per_split.values() for b in bs), key=lambda b: b.subject)
    if limit:
        chosen = chosen[:limit]
    if tmp is not None:
        need = {f"instances_audio/{w.parent.name}/{w.name}" for b in chosen for w in b.wavs}
        with zipfile.ZipFile(found["zip"]) as z:
            z.extractall(tmp.name, [m for m in z.namelist() if m in need])
    features.extract_all(chosen, out, workers)
    ids = {b.subject for b in chosen}
    summary = {
        "sessions": len(chosen),
        "splits": {s: sum(b.subject in ids for b in bs) for s, bs in per_split.items()},
        "instances": sum(len(b) for b in chosen),
        "out_dir": out.name,
    }
    if tmp is not None:
        tmp.cleanup()
    _stamp(summary)
    return summary
