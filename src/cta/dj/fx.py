"""Deck filter and beat-synced FX units (Traktor-style inserts).

All effects process (2, n) float32 blocks, keep their own state per deck,
and are written with numpy/scipy vector operations (no per-sample Python).
"""

from __future__ import annotations

import numpy as np

SR = 44100
FX_TYPES = ("Delay", "Reverb", "Flanger", "Gater", "Crush")
BEAT_DIVS = (1 / 8, 1 / 4, 1 / 2, 3 / 4, 1, 2)       # selectable effect timing in beats


class DeckFilter:
    """One-knob filter: -1..0 = low-pass sweeping down, 0..1 = high-pass
    sweeping up, centre = bypass. Butterworth 2nd order with a little
    resonance-free slope so sweeps stay musical."""

    def __init__(self):
        self.value = 0.0
        self.sos = None
        self.zi = None

    def _design(self, v: float):
        from scipy.signal import butter

        if v < 0:
            fc = 20000.0 * (100.0 / 20000.0) ** (-v)       # 20 kHz -> 100 Hz, exponential
            return butter(2, fc, "low", fs=SR, output="sos")
        fc = 20.0 * (8000.0 / 20.0) ** v                     # 20 Hz -> 8 kHz
        return butter(2, fc, "high", fs=SR, output="sos")

    def process(self, x: np.ndarray, value: float) -> np.ndarray:
        from .engine import sos_filter

        if abs(value) < 0.02:
            if self.sos is not None and abs(self.value) >= 0.02:
                self.zi = None                                # fully open: drop state
            self.value = value
            self.sos = None
            return x
        if self.sos is None or abs(value - self.value) > 1e-3:
            self.sos = self._design(value)
            if self.zi is None or self.zi.shape[1] != self.sos.shape[0]:
                self.zi = np.zeros((2, self.sos.shape[0], 2))
            self.value = value
        return sos_filter(self.sos, x, self.zi).astype(np.float32)


class _Delay:
    """Feedback delay line, length = N beats (must be >= block size; it is
    for any musical tempo since the shortest setting is 1/8 beat ~ 117 ms)."""

    def __init__(self):
        self.buf = np.zeros((2, SR * 4), np.float32)
        self.pos = 0

    def process(self, x, wet, amount, beat_s, div, **_):
        n = x.shape[1]
        d = int(np.clip(beat_s * div * SR, n, self.buf.shape[1] - n - 1))
        L = self.buf.shape[1]
        ridx = (self.pos - d + np.arange(n)) % L
        delayed = self.buf[:, ridx]
        fb = 0.25 + 0.6 * amount                       # amount = feedback
        widx = (self.pos + np.arange(n)) % L
        self.buf[:, widx] = x + delayed * fb
        self.pos = (self.pos + n) % L
        return x + wet * delayed


class _Reverb:
    """Convolution with a synthetic decaying-noise impulse response, done as
    block-wise FFT overlap-add (stateful, click-free)."""

    def __init__(self):
        self.ir_len = 0
        self.tail = None
        self.ir_f = None
        self.nfft = 0
        self._amount = None

    def _make_ir(self, amount):
        secs = 0.6 + 2.4 * amount
        n = int(secs * SR)
        rng = np.random.default_rng(7)
        t = np.arange(n) / SR
        ir = rng.standard_normal((2, n)).astype(np.float32) * np.exp(-6.9 * t / secs)
        ir[:, : int(0.01 * SR)] *= np.linspace(0, 1, int(0.01 * SR))
        ir /= np.sqrt((ir ** 2).sum(axis=1, keepdims=True)) * 1.4
        return ir

    def process(self, x, wet, amount, **_):
        n = x.shape[1]
        if self._amount is None or abs(amount - self._amount) > 0.05 or self.nfft < n + self.ir_len:
            ir = self._make_ir(amount)
            self.ir_len = ir.shape[1]
            self.nfft = 1 << int(np.ceil(np.log2(n + self.ir_len)))
            self.ir_f = np.fft.rfft(ir, self.nfft, axis=1)
            old = self.tail
            self.tail = np.zeros((2, self.ir_len), np.float32)
            if old is not None:
                m = min(old.shape[1], self.tail.shape[1])
                self.tail[:, :m] = old[:, :m]
            self._amount = amount
        y = np.fft.irfft(np.fft.rfft(x, self.nfft, axis=1) * self.ir_f, self.nfft, axis=1)
        y[:, : self.ir_len] += self.tail          # overlap-add the previous blocks' tail
        out = y[:, :n]
        self.tail = y[:, n:n + self.ir_len].astype(np.float32)
        return x + wet * out.astype(np.float32)


class _Flanger:
    def __init__(self):
        self.hist = np.zeros((2, 2048), np.float32)
        self.phase = 0.0

    def process(self, x, wet, amount, beat_s, div, **_):
        n = x.shape[1]
        buf = np.concatenate([self.hist, x], axis=1)
        period = max(beat_s * div * 4, 0.05) * SR           # one sweep per 4x the division
        ph = self.phase + np.arange(n) / period
        depth = (0.5 + 4.0 * amount) * SR / 1000            # up to ~4.5 ms
        dly = 0.3 * SR / 1000 + depth * (0.5 - 0.5 * np.cos(2 * np.pi * ph))
        idx = self.hist.shape[1] + np.arange(n) - dly
        i0 = np.floor(idx).astype(np.int64)
        fr = (idx - i0).astype(np.float32)
        delayed = buf[:, i0] * (1 - fr) + buf[:, i0 + 1] * fr
        self.phase = float(ph[-1] + 1 / period) % 1.0
        self.hist = buf[:, -self.hist.shape[1]:]
        return x * (1 - 0.3 * wet) + wet * 0.7 * delayed


class _Gater:
    def process(self, x, wet, amount, beat_s, div, beat_pos, beat_inc, **_):
        n = x.shape[1]
        b = beat_pos + beat_inc * np.arange(n)
        ph = (b / max(div, 1e-3)) % 1.0
        duty = 0.15 + 0.6 * (1 - amount)                     # amount = harder gating
        gate = (ph < duty).astype(np.float32)
        # 2 ms smoothing to avoid clicks
        k = 88
        gate = np.convolve(np.concatenate([np.full(k, gate[0]), gate]), np.ones(k) / k, "valid")[: n]
        return x * (1 - wet + wet * gate[None, :])


class _Crush:
    def process(self, x, wet, amount, **_):
        bits = 16 - 13 * amount
        q = 2 ** (bits - 1)
        hold = int(1 + amount * 20)
        y = np.round(x * q) / q
        if hold > 1:
            y = np.repeat(y[:, ::hold], hold, axis=1)[:, : x.shape[1]]
        return x * (1 - wet) + wet * y


_FX = {"Delay": _Delay, "Reverb": _Reverb, "Flanger": _Flanger, "Gater": _Gater, "Crush": _Crush}


class FXUnit:
    """One FX unit; each assigned deck gets its own effect state (insert)."""

    def __init__(self, kind: str = "Delay"):
        self.kind = kind
        self.on = False
        self.wet = 0.5            # dry/wet 0..1
        self.amount = 0.5         # effect parameter 0..1
        self.div_index = 2        # BEAT_DIVS index -> 1/2 beat
        self._state: dict[tuple[str, str], object] = {}
        self._off_frames: dict[tuple[str, str], int] = {}

    @property
    def div(self) -> float:
        return BEAT_DIVS[self.div_index]

    def process(self, deck: str, x: np.ndarray, beat_s: float, beat_pos: float, beat_inc: float) -> np.ndarray:
        key = (deck, self.kind)
        st = self._state.get(key)
        if st is None:
            st = _FX[self.kind]()
            self._state[key] = st
        kw = dict(beat_s=beat_s, div=self.div, beat_pos=beat_pos, beat_inc=beat_inc)
        if self.on:
            self._off_frames[key] = 0
            return st.process(x, self.wet, self.amount, **kw).astype(np.float32)
        if self.kind in ("Delay", "Reverb"):
            # after switching off, let echoes/reverb tails ring out (8 s), feeding silence
            off = self._off_frames.get(key, 10 ** 9)
            if off < 8 * SR:
                self._off_frames[key] = off + x.shape[1]
                y = st.process(np.zeros_like(x), 1.0, self.amount, **kw)
                return (x + y * self.wet).astype(np.float32)
        return x
