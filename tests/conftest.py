import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
# no background device-scan helpers outliving the test that opened DJ mode
os.environ["CTA_NO_DEVICE_WATCH"] = "1"

SR = 22050


def _kick(sr):
    n = int(0.2 * sr)
    t = np.arange(n) / sr
    return np.sin(2 * np.pi * (45 * t + 80 / 30 * (1 - np.exp(-t * 30)))) * np.exp(-t * 15)


def synth_track(bpm: float, seconds: float = 45, chord=(57, 60, 64), pattern="4x4", sr=SR, seed=0) -> np.ndarray:
    """Simple drum loop + sustained chord; enough to test tempo/key logic."""
    rng = np.random.default_rng(seed)
    x = np.zeros(int(seconds * sr))
    beat = 60.0 / bpm
    k = _kick(sr)
    hat = rng.standard_normal(int(0.03 * sr)) * np.exp(-np.arange(int(0.03 * sr)) / sr * 60) * 0.3
    snare = rng.standard_normal(int(0.15 * sr)) * np.exp(-np.arange(int(0.15 * sr)) / sr * 60) * 0.6

    def place(t, sig, g=1.0):
        s = int(t)
        e = min(x.size, s + sig.size)
        if s < x.size:
            x[s:e] += g * sig[: e - s]

    for b in range(int(seconds / beat)):
        t = b * beat * sr
        if pattern == "4x4":
            place(t, k)
            place(t + beat * sr / 2, hat)
            if b % 2 == 1:
                place(t, snare, 0.5)
        elif pattern == "hiphop":
            if b % 4 == 0:
                place(t, k)
            if b % 4 == 2:
                place(t + beat * sr * 0.75, k)
            if b % 2 == 1:
                place(t, snare)
            place(t, hat)
            place(t + beat * sr * 0.66, hat, 0.6)
        else:  # breakbeat (drum & bass feel)
            if b % 4 == 0:
                place(t, k)
            if b % 4 == 2:
                place(t + beat * sr / 2, k)
            if b % 2 == 1:
                place(t, snare)
            place(t, hat)
            place(t + beat * sr / 2, hat)
    tt = np.arange(x.size) / sr
    for m in chord:
        f = 440 * 2 ** ((m - 69) / 12)
        x += 0.05 * (np.sin(2 * np.pi * f * tt) + 0.5 * np.sin(4 * np.pi * f * tt) + 0.3 * np.sin(6 * np.pi * f * tt))
    return (x / np.abs(x).max() * 0.9).astype(np.float32)


def write_audio(path: str, y: np.ndarray, sr: int = SR, codec: str = "flac"):
    import av

    pcm = (np.clip(y, -1, 1) * 32767).astype(np.int16)
    with av.open(path, "w") as c:
        st = c.add_stream(codec, rate=sr, layout="mono")
        if codec == "flac":
            fr = av.AudioFrame.from_ndarray(pcm[None, :], format="s16", layout="mono")
            fr.sample_rate = sr
            frames = [fr]
        else:
            # lossy encoders need fixed size planar float frames
            fs = st.codec_context.frame_size or 1152
            yf = y.astype(np.float32)
            frames = []
            for s in range(0, yf.size, fs):
                chunk = yf[s:s + fs]
                if chunk.size < fs:
                    chunk = np.pad(chunk, (0, fs - chunk.size))
                fr = av.AudioFrame.from_ndarray(chunk[None, :], format="fltp", layout="mono")
                fr.sample_rate = sr
                fr.pts = s
                frames.append(fr)
        for fr in frames:
            for p in st.encode(fr):
                c.mux(p)
        for p in st.encode(None):
            c.mux(p)


@pytest.fixture
def tmp_data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("CTA_DATA_DIR", str(tmp_path / "data"))
    return tmp_path


@pytest.fixture(autouse=True)
def no_modal_dialogs(monkeypatch):
    """A modal dialog would block a headless test run forever: fail fast and
    show its text instead."""
    try:
        from PySide6.QtWidgets import QMessageBox
    except Exception:  # noqa: BLE001
        yield
        return

    def boom(*args, **kwargs):
        texts = [a for a in args if isinstance(a, str)]
        raise AssertionError(f"unexpected dialog: {texts}")

    for name in ("warning", "question", "information", "critical"):
        monkeypatch.setattr(QMessageBox, name, staticmethod(boom))
    monkeypatch.setattr(QMessageBox, "exec", lambda self: boom(self.text()))
    yield


def structured_track(bpm=126.0, layout=(("intro", 16), ("drop", 32), ("breakdown", 16), ("drop", 32), ("outro", 16)),
                     sr=44100, chord=(57, 60, 64), seed=0):
    """Synthetic dance track with real sections:
    intro/outro = kick + hats, drop = kick + bass + chords + hats,
    breakdown = chords only, build = kick + hats + rising noise."""
    rng = np.random.default_rng(seed)
    beat = 60.0 / bpm
    total_bars = sum(b for _, b in layout)
    n = int(total_bars * 4 * beat * sr) + sr
    x = np.zeros(n)
    k = _kick(sr)
    hat = rng.standard_normal(int(0.03 * sr)) * np.exp(-np.arange(int(0.03 * sr)) / sr * 60) * 0.25
    t_all = np.arange(n) / sr
    bar = 0
    for kind, bars in layout:
        s0 = int(bar * 4 * beat * sr)
        s1 = int((bar + bars) * 4 * beat * sr)
        seg_t = t_all[s0:s1]
        for b in range(bars * 4):
            pos = s0 + int(b * beat * sr)
            if kind in ("intro", "outro", "drop", "build"):
                e = min(n, pos + k.size)
                x[pos:e] += k[: e - pos]
                hp = pos + int(beat * sr / 2)
                e = min(n, hp + hat.size)
                if hp < n:
                    x[hp:e] += hat[: e - hp]
        if kind in ("drop", "breakdown"):
            for m in chord:
                f = 440 * 2 ** ((m - 69) / 12)
                x[s0:s1] += (0.08 if kind == "drop" else 0.12) * np.sin(2 * np.pi * f * seg_t)
        if kind == "drop":
            f = 440 * 2 ** ((chord[0] - 12 - 69) / 12)
            x[s0:s1] += 0.25 * np.sign(np.sin(2 * np.pi * f * seg_t)) * 0.6
        bar += bars
    return (x / np.abs(x).max() * 0.9).astype(np.float32)


# Qt and native extensions (PySide6, Rubber Band) can crash while Python tears
# them down at interpreter exit ("shared QObject was deleted directly") - after
# every test has passed. Once pytest has reported, leave without that teardown.
_EXIT_STATUS = {"code": 0}


def pytest_sessionfinish(session, exitstatus):
    _EXIT_STATUS["code"] = int(exitstatus)


def pytest_unconfigure(config):
    import os
    import sys
    import threading

    alive = [t.name for t in threading.enumerate() if t is not threading.main_thread()]
    sys.stdout.write(f"\n[cta] pytest exit status {_EXIT_STATUS['code']}; threads still running: {alive}\n")
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(_EXIT_STATUS["code"])
