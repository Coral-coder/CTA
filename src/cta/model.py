"""Discogs-EffNet style/genre model + audio embedding via ONNX Runtime.

The model (~18 MB, MTG/UPF Essentia models, CC BY-NC-SA 4.0) predicts 400
Discogs styles and produces a 1280-d embedding used for "sounds like"
similarity. It needs well under 1 GB of GPU memory. On Windows it runs on any
DirectX 12 GPU through DirectML, on macOS through CoreML, on Linux through
CUDA when available - and always falls back to the CPU.
"""

from __future__ import annotations

import hashlib
import logging
import os
import sys
import threading
import urllib.request
from importlib import resources

import numpy as np

from . import paths

log = logging.getLogger(__name__)

MODEL_FILE = "discogs-effnet-bsdynamic-1.onnx"
MODEL_URL = "https://essentia.upf.edu/models/feature-extractors/discogs-effnet/" + MODEL_FILE
MODEL_SHA256 = "a280825b334797cf677939db8cd5762c0392aedd0ca6415dbc1cd083f045e43c"
EMBED_DIM = 1280
N_STYLES = 400


def labels() -> list[str]:
    txt = resources.files("cta").joinpath("discogs400_labels.txt").read_text(encoding="utf-8")
    return [l for l in txt.splitlines() if l.strip()]


def _bundled_model() -> str | None:
    cands = []
    if getattr(sys, "frozen", False):
        base = getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))
        cands.append(os.path.join(base, "models", MODEL_FILE))
    cands.append(os.path.join(os.path.dirname(__file__), "models", MODEL_FILE))
    for c in cands:
        if os.path.isfile(c):
            return c
    return None


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def find_model() -> str | None:
    env = os.environ.get("CTA_MODEL_PATH")
    if env and os.path.isfile(env):
        return env
    p = _bundled_model()
    if p:
        return p
    p = os.path.join(paths.models_dir(), MODEL_FILE)
    return p if os.path.isfile(p) else None


def download_model(progress=None) -> str:
    """Download and verify the model into the user data dir."""
    dest = os.path.join(paths.models_dir(), MODEL_FILE)
    tmp = dest + ".part"
    req = urllib.request.Request(MODEL_URL, headers={"User-Agent": "CTA/1.0"})
    with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while True:
            chunk = r.read(1 << 16)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if progress and total:
                progress(done, total)
    if _sha256(tmp) != MODEL_SHA256:
        os.remove(tmp)
        raise RuntimeError("downloaded model failed checksum verification")
    os.replace(tmp, dest)
    return dest


def ensure_model(progress=None) -> str | None:
    p = find_model()
    if p:
        return p
    try:
        return download_model(progress)
    except Exception as e:  # noqa: BLE001
        log.warning("AI model unavailable (%s); continuing without AI genres", e)
        return None


class StyleModel:
    """Thread-safe wrapper around the ONNX session."""

    def __init__(self, path: str, device: str = "auto"):
        import onnxruntime as ort

        self._lock = threading.Lock()
        self.path = path
        avail = ort.get_available_providers()
        order = []
        if device != "cpu":
            for p in ("CUDAExecutionProvider", "DmlExecutionProvider", "CoreMLExecutionProvider"):
                if p in avail:
                    order.append(p)
        order.append("CPUExecutionProvider")

        self.session = None
        self.provider = "CPUExecutionProvider"
        last_err = None
        for prov in order:
            try:
                so = ort.SessionOptions()
                so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                # leave cores free for the audio (scans run while you DJ)
                so.intra_op_num_threads = max(1, min(4, (os.cpu_count() or 2) // 3))
                if prov == "DmlExecutionProvider":
                    so.enable_mem_pattern = False
                    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
                providers = [prov] if prov == "CPUExecutionProvider" else [prov, "CPUExecutionProvider"]
                sess = ort.InferenceSession(path, sess_options=so, providers=providers)
                # sanity check: the accelerated path must produce sane numbers
                probe = np.random.default_rng(0).random((2, 128, 96), dtype=np.float32) * 3
                act, emb = sess.run(None, {"melspectrogram": probe})
                if not (np.all(np.isfinite(act)) and np.all(np.isfinite(emb))):
                    raise RuntimeError("non-finite output")
                actual = sess.get_providers()[0]
                if actual != prov:
                    # ORT silently fell back (e.g. no compatible GPU present)
                    raise RuntimeError(f"{prov} unavailable, session uses {actual}")
                self.session = sess
                self.provider = prov
                break
            except Exception as e:  # noqa: BLE001
                last_err = e
                log.warning("ONNX provider %s failed: %s", prov, e)
        if self.session is None:
            raise RuntimeError(f"could not initialise model: {last_err}")
        self.batch = 64 if self.provider != "CPUExecutionProvider" else 32
        log.info("AI model running on %s", self.provider)

    @property
    def device_name(self) -> str:
        return {
            "CUDAExecutionProvider": "GPU (CUDA)",
            "DmlExecutionProvider": "GPU (DirectML)",
            "CoreMLExecutionProvider": "Apple Neural Engine / GPU (CoreML)",
        }.get(self.provider, "CPU")

    def run(self, patches: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """patches (n,128,96) -> (activations (n,400), embeddings (n,1280))."""
        acts, embs = [], []
        with self._lock:
            for s in range(0, patches.shape[0], self.batch):
                chunk = patches[s:s + self.batch]
                n = chunk.shape[0]
                if n < self.batch and self.provider != "CPUExecutionProvider":
                    # fixed batch shape avoids re-compilation on GPU backends
                    chunk = np.concatenate([chunk, np.zeros((self.batch - n, 128, 96), np.float32)])
                a, e = self.session.run(None, {"melspectrogram": chunk})
                acts.append(a[:n])
                embs.append(e[:n])
        return np.concatenate(acts), np.concatenate(embs)

    def track_vectors(self, patch_groups: list[np.ndarray]) -> list[tuple[np.ndarray, np.ndarray]]:
        """Run several tracks in one batched call; returns per-track mean
        (style probabilities, embedding)."""
        if not patch_groups:
            return []
        allp = np.concatenate(patch_groups)
        a, e = self.run(allp)
        out, s = [], 0
        for g in patch_groups:
            n = g.shape[0]
            out.append((a[s:s + n].mean(axis=0), e[s:s + n].mean(axis=0)))
            s += n
        return out
