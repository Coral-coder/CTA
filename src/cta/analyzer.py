"""Per-file analysis, executed inside worker processes."""

from __future__ import annotations

import os
import time
import traceback

import numpy as np

# Bump to force re-analysis of every track after an algorithm change.
ANALYSIS_VERSION = 1


def _limit_threads():
    # Each worker is single-threaded; parallelism comes from the process pool.
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(v, "1")


def worker_init():
    _limit_threads()


def analyze_file(path: str, min_bpm: float = 70.0, max_bpm: float = 180.0,
                 want_model: bool = True, max_patches: int = 32) -> dict:
    """Analyse one file. Never raises: errors are returned in the result."""
    from . import audio, dsp
    from .tags import read_tags

    res: dict = {"path": path, "ok": False}
    try:
        st = os.stat(path)
        res["mtime"] = st.st_mtime
        res["size"] = st.st_size
        res.update(read_tags(path))
        dec = audio.decode(path, want_model=want_model)
        res["duration"] = dec.duration
        env = dsp.onset_envelope(dec.y)
        t = dsp.estimate_tempo(dec.y, min_bpm, max_bpm, env=env)
        k = dsp.estimate_key(dec.y)
        res["bpm"] = t.bpm
        res["bpm_conf"] = t.confidence
        res["key"] = k.key
        res["key_conf"] = k.confidence
        res.update(dsp.energy_features(dec.y, env))
        if want_model and dec.y16.size >= dsp.MODEL_SR:
            res["patches"] = dsp.model_patches(dec.y16, max_patches=max_patches)
        res["ok"] = True
    except Exception as e:  # noqa: BLE001
        res["error"] = f"{type(e).__name__}: {e}"
        res["traceback"] = traceback.format_exc(limit=4)
    res["analyzed_at"] = time.time()
    res["version"] = ANALYSIS_VERSION
    return res


# Typical tempo ranges for styles where half/double-time errors are common.
# If the detected BPM is outside the range but its double/half is inside,
# the AI style decides the octave.
STYLE_TEMPO_RANGES = {
    "Drum n Bass": (155, 185), "Jungle": (150, 185), "Breakcore": (150, 240), "Neurofunk": (165, 180),
    "Dubstep": (130, 150), "Grime": (130, 150), "Footwork": (150, 165), "Juke": (150, 165),
    "Hip Hop": (70, 115), "Boom Bap": (80, 100), "Trap": (120, 160), "Gangsta": (75, 105),
    "Pop Rap": (70, 120), "Conscious": (70, 115), "Thug Rap": (70, 110),
    "Downtempo": (70, 115), "Trip Hop": (70, 110), "Reggae": (60, 100), "Dub": (60, 100),
    "House": (115, 132), "Deep House": (112, 128), "Tech House": (118, 132), "Techno": (120, 150),
    "Trance": (125, 145), "Hardstyle": (140, 160), "Hardcore": (150, 220), "Gabber": (150, 220),
    "Psy-Trance": (135, 150), "Goa Trance": (135, 150), "Hard Trance": (135, 155),
}


def style_tempo_fix(bpm: float, styles: list[tuple[str, float]]) -> float:
    """Resolve a half/double-time ambiguity using the predicted style."""
    if not bpm or not styles:
        return bpm
    votes: dict[float, float] = {}
    for label, p in styles[:3]:
        rng = STYLE_TEMPO_RANGES.get(label.split("---")[-1])
        if rng is None or p < 0.15:
            continue
        lo, hi = rng
        if lo <= bpm <= hi:
            return bpm
        for cand in (bpm * 2, bpm / 2):
            if lo <= cand <= hi:
                votes[cand] = votes.get(cand, 0.0) + p
    if votes:
        cand, w = max(votes.items(), key=lambda kv: kv[1])
        if w >= 0.25:
            return cand
    return bpm


def summarize_styles(probs: np.ndarray, label_list: list[str], top: int = 5):
    """Top styles as [(label, prob)], plus a coarse genre (the Discogs parent)."""
    idx = np.argsort(probs)[::-1][:top]
    styles = [(label_list[i], float(probs[i])) for i in idx]
    # parent genre = genre with the largest summed probability
    parents: dict[str, float] = {}
    for lab, p in zip(label_list, probs):
        par = lab.split("---")[0]
        parents[par] = parents.get(par, 0.0) + float(p)
    parent = max(parents, key=parents.get) if parents else ""
    return styles, parent
