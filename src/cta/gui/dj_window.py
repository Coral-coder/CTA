"""DJ Mode: Traktor-style decks docked above the library.

Layout:   [ Deck C ]   [         ]   [ Deck D ]   (optional)
          [ Deck A ]   [  MIXER  ]   [ Deck B ]
          [ FX 1 ] [ FX 2 ]  [ Auto-mix / Assist ]

The mixer in the middle holds every level control per channel (gain, EQ
hi/mid/low, filter, FX assign, headphone cue, level meter, channel fader)
plus crossfader, master, headphones and recording.
"""

from __future__ import annotations

import logging
import os
import threading
import time
import uuid

import numpy as np
from PySide6.QtCore import QPointF, QRectF, QSettings, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDial,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLayout,
    QListWidget,
    QMenu,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..dj.automix import AutoMixSettings
from ..dj.engine import SR, TrackInfo, list_output_devices
from ..dj.engine_client import EngineClient
from ..dj.fx import BEAT_DIVS, FX_TYPES
from ..dj.loader import load_track
from ..dj.midi import DECK_ACTIONS, GLOBAL_ACTIONS, Binding, Controller, Learner, learn_steps
from ..dj.recorder import Recorder
from ..keys import camelot_str, key_name
from ..mixer import suggest_next
from .models import TRACK_MIME, key_color

log = logging.getLogger(__name__)

DECK_COLORS = {"A": QColor(64, 156, 255), "B": QColor(255, 140, 60), "C": QColor(120, 200, 120),
               "D": QColor(200, 120, 220)}
CUE_COLORS = [QColor(c) for c in ("#e53935", "#fb8c00", "#fdd835", "#43a047",
                                  "#00acc1", "#1e88e5", "#8e24aa", "#d81b60")]


def fmt_time(s: float) -> str:
    s = max(0.0, s)
    return f"{int(s // 60)}:{int(s % 60):02d}.{int((s * 10) % 10)}"


class Knob(QDial):
    """Rotary with a default position; double-click resets."""

    def __init__(self, default: int = 50, size: int = 38, tip: str = ""):
        super().__init__()
        self.default = default
        self.setRange(0, 100)
        self.setValue(default)
        self.setNotchesVisible(True)
        self.setFixedSize(size, size)
        self.setToolTip(tip + ("\n" if tip else "")
                        + "Drag up / down (Shift: fine) · double-click or right-click to reset")

    # drag like a hardware knob: up = more, down = less, from where it is now
    # (a click never jumps it to the mouse); Shift = fine
    PIXELS_FULL_TURN = 200.0

    def mousePressEvent(self, ev):
        if ev.button() == Qt.LeftButton:
            self._drag = (ev.position().y(), float(self.value()))
            self._fine = bool(ev.modifiers() & Qt.ShiftModifier)
            self.setSliderDown(True)
            ev.accept()
            return
        if ev.button() == Qt.RightButton:          # right-click: back to the default (like double-click)
            self._drag = None
            self.setValue(self.default)
            ev.accept()
            return
        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev):
        drag = getattr(self, "_drag", None)
        if drag is None:
            return                                # only a drag that started on the knob turns it
        fine = bool(ev.modifiers() & Qt.ShiftModifier)
        if fine != self._fine:
            # Shift pressed / let go mid-drag: carry on from here, no jump
            self._fine = fine
            self._drag = (ev.position().y(), float(self.value()))
            return
        y0, v0 = drag
        scale = (self.maximum() - self.minimum()) / self.PIXELS_FULL_TURN * (0.2 if fine else 1.0)
        v = v0 + (y0 - ev.position().y()) * scale
        self.setValue(int(round(min(max(v, self.minimum()), self.maximum()))))
        ev.accept()

    def mouseReleaseEvent(self, ev):
        if getattr(self, "_drag", None) is not None:
            self._drag = None
            self.setSliderDown(False)
            ev.accept()
            return
        super().mouseReleaseEvent(ev)

    def mouseDoubleClickEvent(self, _ev):
        self._drag = None
        self.setValue(self.default)


class _TitleLabel(QLabel):
    """The deck's title: double-click = what mixes next from this track."""

    doubleClicked = Signal()

    def mouseDoubleClickEvent(self, ev):
        self.doubleClicked.emit()
        ev.accept()


class ResetSlider(QSlider):
    """Fader / crossfader / tempo: double-click puts it back to its default."""

    def __init__(self, orientation, default: int = 0, tip: str = ""):
        super().__init__(orientation)
        self.default = default
        if tip:
            self.setToolTip(tip + "\nDrag from where it is (Shift: fine) · double-click or right-click to reset")

    # drag from where it is - clicking the track never jumps the handle to the
    # mouse (a fader you grab mid-mix must not leap); Shift = fine
    def mousePressEvent(self, ev):
        if ev.button() == Qt.RightButton:          # right-click: back to the default (like double-click)
            self._drag = None
            self.setValue(self.default)
            ev.accept()
            return
        if ev.button() == Qt.LeftButton:
            self._drag = (self._along(ev), float(self.value()))
            self._fine = bool(ev.modifiers() & Qt.ShiftModifier)
            self.setSliderDown(True)
            ev.accept()
            return
        super().mousePressEvent(ev)

    def _along(self, ev) -> float:
        """Mouse position along the slider, growing in the direction of 'more'."""
        p = ev.position()
        if self.orientation() == Qt.Horizontal:
            v = p.x()
        else:
            v = -p.y()                                  # up = more
        return -v if self.invertedAppearance() else v

    def mouseMoveEvent(self, ev):
        drag = getattr(self, "_drag", None)
        if drag is None:
            return
        fine = bool(ev.modifiers() & Qt.ShiftModifier)
        if fine != self._fine:                          # Shift mid-drag: carry on from here
            self._fine = fine
            self._drag = (self._along(ev), float(self.value()))
            return
        a0, v0 = drag
        length = max((self.width() if self.orientation() == Qt.Horizontal else self.height()) - 16, 20)
        scale = (self.maximum() - self.minimum()) / length * (0.2 if fine else 1.0)
        v = v0 + (self._along(ev) - a0) * scale
        self.setValue(int(round(min(max(v, self.minimum()), self.maximum()))))
        ev.accept()

    def mouseReleaseEvent(self, ev):
        if getattr(self, "_drag", None) is not None:
            self._drag = None
            self.setSliderDown(False)
            ev.accept()
            return
        super().mouseReleaseEvent(ev)

    def mouseDoubleClickEvent(self, _ev):
        self._drag = None
        self.setValue(self.default)


def knob_eq(v: int) -> float:
    x = v / 100.0
    if x <= 0.5:
        return (x / 0.5) ** 2 if x > 0.01 else 0.0
    return 1.0 + (x - 0.5) * 2.0


def eq_knob(g: float) -> int:
    return int(round(50 * np.sqrt(max(g, 0.0)))) if g <= 1 else int(round(50 + (g - 1) * 50))


class LevelMeter(QWidget):
    def __init__(self, width=10):
        super().__init__()
        self.setFixedWidth(width)
        self.setMinimumHeight(60)
        self.value = 0.0

    def paintEvent(self, _ev):
        p = QPainter(self)
        w, h = self.width(), self.height()
        p.fillRect(0, 0, w, h, QColor(20, 20, 22))
        v = float(np.clip(self.value, 0, 1.2)) / 1.2
        segs = 16
        for i in range(segs):
            frac = (i + 1) / segs
            if frac > v + 1e-6:
                break
            c = QColor(70, 200, 90) if frac < 0.7 else (QColor(240, 200, 40) if frac < 0.85 else QColor(235, 60, 50))
            y = h - (i + 1) * h / segs
            p.fillRect(QRectF(1, y + 1, w - 2, h / segs - 2), c)


# --------------------------------------------------------------------------
# waveforms (rendered with numpy into QImages: cheap at 30 fps)
# --------------------------------------------------------------------------

# Waveform colour themes (Options -> Waveform colours). Bands are (n, 3):
# low, mid, high energy per pixel column.
WAVE_THEMES = ["Rainbow", "Spectrum", "Three-band", "Fire", "Ice", "Classic"]
WAVE_THEME_TIPS = {
    "Rainbow": "The whole rainbow - red, orange, yellow, green, cyan, blue, violet - from the darkest to "
               "the brightest sound of each track (kicks red, vocals green, hats violet)",
    "Spectrum": "Bass red, mids green, highs blue (strong primaries)",
    "Three-band": "Bass, mids and highs drawn as layers: blue lows, amber mids, white highs",
    "Fire": "Loudness as heat: dark red to orange to yellow-white",
    "Ice": "Loudness as cold: navy to cyan to white",
    "Classic": "The original soft colours",
}
_theme = {"name": "Rainbow", "sat": 1.0}


def set_wave_theme(name: str):
    _theme["name"] = name if name in WAVE_THEMES else "Rainbow"


def set_wave_saturation(sat: float):
    """0 = grey, 1 = natural, up to 2 = extra vivid."""
    _theme["sat"] = float(np.clip(sat, 0.0, 2.0))


def _brightness(b: np.ndarray) -> np.ndarray:
    """Where each column's sound sits between bass (0) and treble (1)."""
    b = np.asarray(b, np.float64)
    return (0.5 * b[:, 1] + 1.0 * b[:, 2]) / (b.sum(axis=1) + 1e-9)


def colour_range(bands: np.ndarray | None) -> tuple[float, float]:
    """The darkest and brightest sound of a track: the rainbow is spread
    between them, so every track uses all of its colours."""
    if bands is None or len(bands) == 0:
        return 0.0, 1.0
    b = np.asarray(bands[:: max(1, len(bands) // 20000)], np.float64)
    loud = b.max(axis=1) > 0.05 * max(float(b.max()), 1e-9)
    x = _brightness(b[loud]) if loud.any() else _brightness(b)
    lo, hi = np.percentile(x, [3, 97]) if len(x) else (0.0, 1.0)
    return float(lo), float(max(hi, lo + 0.05))


def _hsv(h: np.ndarray, s: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Vectorised HSV (0..1) -> RGB float (n, 3)."""
    i = np.floor(h * 6.0).astype(np.int64) % 6
    f = h * 6.0 - np.floor(h * 6.0)
    p, q, t = v * (1 - s), v * (1 - f * s), v * (1 - (1 - f) * s)
    r = np.choose(i, [v, q, p, p, t, v])
    g = np.choose(i, [t, v, v, q, p, p])
    b = np.choose(i, [p, p, t, v, v, q])
    return np.stack([r, g, b], axis=1)


def _ramp(x: np.ndarray, stops: list[tuple[float, tuple[int, int, int]]]) -> np.ndarray:
    xs = [a for a, _ in stops]
    cols = np.array([c for _, c in stops], np.float64) / 255.0
    return np.stack([np.interp(x, xs, cols[:, k]) for k in range(3)], axis=1)


def _band_colors(b: np.ndarray, rng: tuple[float, float] | None = None) -> np.ndarray:
    """(n,3) band levels -> (n,3) uint8 RGB in the chosen theme. rng: the
    track's colour range (colour_range) for the rainbow."""
    name = _theme["name"]
    b = np.asarray(b, np.float64)
    tot = b.sum(axis=1) + 1e-9
    level = np.clip(b.max(axis=1), 0, 1)
    if name == "Classic":
        mix = b / tot[:, None]
        rgb = np.stack([0.25 + 0.75 * mix[:, 0] + 0.2 * mix[:, 1],
                        0.25 + 0.7 * mix[:, 1] + 0.15 * mix[:, 0],
                        0.3 + 0.9 * mix[:, 2]], axis=1)
    elif name == "Fire":
        rgb = _ramp(level, [(0.0, (70, 0, 10)), (0.35, (200, 20, 10)), (0.65, (255, 120, 0)),
                            (0.85, (255, 210, 40)), (1.0, (255, 250, 210))])
    elif name == "Ice":
        rgb = _ramp(level, [(0.0, (8, 20, 90)), (0.4, (0, 90, 230)), (0.7, (0, 210, 255)),
                            (0.9, (150, 245, 255)), (1.0, (245, 255, 255))])
    elif name == "Three-band":
        rgb = np.tile(np.array([[0.12, 0.42, 1.0]]), (len(b), 1))     # outline colour = bass layer
    elif name == "Rainbow":
        lo, hi = rng or (0.0, 1.0)
        x = np.clip((_brightness(b) - lo) / (hi - lo), 0.0, 1.0)
        hue = 0.80 * x                           # red, orange, yellow, green, cyan, blue, violet
        val = 0.62 + 0.38 * np.sqrt(level)
        rgb = _hsv(hue, np.ones_like(hue), val)
    else:  # Spectrum: the "centre of mass" of the spectrum picks the hue
        # emphasise the dominant band so colours stay vivid instead of muddy mixes
        w = b ** 2
        x = (w[:, 1] * 0.5 + w[:, 2] * 1.0) / (w.sum(axis=1) + 1e-12)
        hue = 0.0 + 0.78 * x                     # 0 red -> 0.13 orange -> 0.33 green -> 0.66 blue -> 0.78 violet
        sat = np.full_like(hue, 1.0)
        val = 0.55 + 0.45 * np.sqrt(level)
        rgb = _hsv(hue, sat, val)
    sat = _theme["sat"]
    if abs(sat - 1.0) > 1e-3:
        grey = rgb.mean(axis=1, keepdims=True)
        rgb = grey + (rgb - grey) * sat
        if sat > 1.0:                             # keep the brightest channel where it was
            peak = np.maximum(rgb.max(axis=1, keepdims=True), 1e-9)
            rgb = np.where(peak > 1.0, rgb / peak, rgb)
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)


THREE_BAND = ((31, 107, 255), (255, 154, 31), (245, 245, 245))   # lows, mids, highs


def _wave_image(b: np.ndarray, h: int, past: int | None = None, rng=None) -> QImage:
    """Waveform picture for per-column bands in the current theme."""
    b = np.asarray(b, np.float32)
    if _theme["name"] == "Three-band":
        return _render_layers(np.clip(b, 0, 1), h, past)
    return _render(np.clip(b.max(axis=1), 0, 1), _band_colors(b, rng), h, past)


def xy_scope_image(x: np.ndarray, size: int, color: QColor) -> QImage:
    """XY (goniometer) picture of a (2, n) stereo block: L-R turned 45 degrees,
    so mono is a vertical line and wide stereo a broad cloud."""
    size = max(int(size), 8)
    img = np.zeros((size, size), np.float32)
    if x is not None and x.size:
        left, right = x[0], x[1]
        side = (left - right) * 0.7071
        mid = (left + right) * 0.7071
        peak = max(float(np.abs(mid).max()), float(np.abs(side).max()), 0.05)
        half = (size - 1) / 2
        px = np.clip(half + side / peak * half * 0.95, 0, size - 1).astype(np.int64)
        py = np.clip(half - mid / peak * half * 0.95, 0, size - 1).astype(np.int64)
        np.add.at(img, (py, px), 1.0)
        img = np.minimum(img / max(float(img.max()) * 0.35, 1.0), 1.0)
    a = (img * 230).astype(np.uint32)
    rgb = (color.red() << 16) | (color.green() << 8) | color.blue()
    argb = (a << 24) | np.uint32(rgb)
    argb = np.where(a > 0, argb, np.uint32(0)).astype(np.uint32)
    q = QImage(argb.data, size, size, 4 * size, QImage.Format_ARGB32)
    q._buf = argb
    return q


def _grid_columns(img: np.ndarray, xs: np.ndarray, alpha: float, wide: bool = False):
    """Lighten whole pixel columns of a packed RGB32 image towards white (beat lines)."""
    w = img.shape[1]
    cols = np.round(xs).astype(np.int64)
    if wide:
        cols = np.concatenate([cols, cols + 1])
    cols = cols[(cols >= 0) & (cols < w)]
    if cols.size == 0:
        return
    sub = img[:, cols]
    a = int(alpha * 256)
    r = (sub >> 16) & 0xFF
    g = (sub >> 8) & 0xFF
    b = sub & 0xFF
    r = r + (((255 - r) * a) >> 8)
    g = g + (((255 - g) * a) >> 8)
    b = b + (((255 - b) * a) >> 8)
    img[:, cols] = np.uint32(0xFF000000) | (r << 16) | (g << 8) | b


def _render_layers(b: np.ndarray, h: int, past: int | None = None) -> QImage:
    """Three-band view: the bass outline with the mids and highs drawn over it."""
    w = b.shape[0]
    ys = _YS.get(h)
    if ys is None:
        ys = _YS[h] = np.abs(np.arange(h, dtype=np.float32)[:, None] + 0.5 - h / 2)
    img = np.full((h, w), _BG32, np.uint32)
    scale = [1.0, 0.85, 0.6]                 # highs are quieter: keep them visible but inside
    for k, (r, g, bl) in enumerate(THREE_BAND):
        c = np.full(w, (0xFF << 24) | (r << 16) | (g << 8) | bl, np.uint32)
        if past is not None and past > 0:
            c[:past] = (0xFF << 24) | ((r * 115 >> 8) << 16) | ((g * 115 >> 8) << 8) | (bl * 115 >> 8)
        hk = np.clip(b[:, k] * scale[k] if k else b.max(axis=1), 0, 1)
        mask = ys <= (hk * (h / 2 - 1))[None, :]
        img = np.where(mask, c[None, :], img)
    img = img.astype(np.uint32)
    q = QImage(img.data, w, h, 4 * w, QImage.Format_RGB32)
    q._buf = img
    return q


_BG32 = np.uint32(0xFF000000 | (18 << 16) | (19 << 8) | 22)
_YS: dict[int, np.ndarray] = {}


def _render(heights: np.ndarray, colors: np.ndarray, h: int, past: int | None = None) -> QImage:
    """Mirror-waveform image from per-column heights (0..1) and colours.
    Packed 32-bit pixels: one numpy pass per frame (the UI thread must stay
    free so buttons react at once)."""
    w = heights.size
    ys = _YS.get(h)
    if ys is None:
        ys = _YS[h] = np.abs(np.arange(h, dtype=np.float32)[:, None] + 0.5 - h / 2)
    mask = ys <= (heights.astype(np.float32) * (h / 2 - 1))[None, :]
    c = colors.astype(np.uint32)
    if past is not None and past > 0:
        c[:past] = (c[:past] * 115) >> 8
    packed = np.uint32(0xFF000000) | (c[:, 0] << 16) | (c[:, 1] << 8) | c[:, 2]
    img = np.where(mask, packed[None, :], _BG32).astype(np.uint32)
    q = QImage(img.data, w, h, 4 * w, QImage.Format_RGB32)
    q._buf = img                        # keep the pixels alive as long as the image
    return q


class Waveform(QWidget):
    seek = Signal(float)          # seconds
    marker = Signal(str, float)   # kind ("in", "out", "cue", "clear"), seconds

    def __init__(self, deck: str, zoom: bool, parent=None):
        super().__init__(parent)
        self.deck = deck
        self.zoom = zoom
        self.setMinimumHeight(84 if zoom else 30)
        if not zoom:
            self.setMaximumHeight(40)
        self.bands = None          # (frames,3) @ 256-sample hop
        self.peaks = None          # overview (W,3)
        self.crange = (0.0, 1.0)   # the track's colour range (rainbow)
        self.duration = 1.0
        self.pos = 0.0
        self.grid = None
        self.cue = 0.0
        self.loop = None
        self.hotcues = [None] * 8
        self.span = 8.0
        self.mix_in = None         # seconds
        self.mix_out = None
        self.mix_out_end = None    # seconds: the blend ends here when set
        self.mix_beats = 64.0      # blend length shown after the outro marker
        self._over = None          # cached (width, band levels per column)
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._menu)
        self._drag_x = None
        if zoom:
            self.setCursor(Qt.OpenHandCursor)
            self.setToolTip("Drag to scratch · double-click to jump · − / + to zoom · right-click for markers")

    def set_track(self, peaks, bands, duration, grid):
        self.peaks, self.bands, self.duration, self.grid = peaks, bands, max(duration, 1e-3), grid
        self.crange = colour_range(bands if bands is not None else peaks)
        self._over = None
        self._over_img = (None, None)
        self.update()

    def _time_at(self, x: float) -> float:
        f = x / max(self.width(), 1)
        return self.pos + (f - 0.5) * self.span if self.zoom else f * self.duration

    def _bar_snap(self, t: float) -> float:
        """Start (downbeat) of the bar that contains t."""
        if self.grid is None:
            return t
        b = self.grid.beat_at(t)
        return max(0.0, self.grid.time_of_beat(np.floor(b / 4.0 + 1e-6) * 4.0))

    def _menu(self, pos):
        if self.peaks is None:
            return
        from PySide6.QtWidgets import QMenu

        t = self._bar_snap(self._time_at(pos.x()))
        m = QMenu(self)
        stamp = f"{int(t // 60)}:{t % 60:04.1f}"
        m.addAction(f"Set mix-in (intro) here  [{stamp}]", lambda: self.marker.emit("in", t))
        m.addAction(f"Set mix-out (outro) here  [{stamp}]", lambda: self.marker.emit("out", t))
        m.addAction(f"Set outro end here  [{stamp}]", lambda: self.marker.emit("out_end", t))
        m.addAction(f"Set cue point here  [{stamp}]", lambda: self.marker.emit("cue", t))
        m.addSeparator()
        m.addAction("Edit markers (zoom in)…", lambda: self.marker.emit("edit", 0.0))
        if self.mix_in is not None or self.mix_out is not None:
            m.addSeparator()
            m.addAction("Clear mix-in / mix-out markers", lambda: self.marker.emit("clear", 0.0))
        m.exec(self.mapToGlobal(pos))

    scratch = Signal(float)       # seconds of audio moved by the hand since the last event
    scratch_end = Signal()

    def mousePressEvent(self, ev):
        if self.peaks is None or ev.button() != Qt.LeftButton:
            return
        if self.zoom:
            # grab the record: dragging the waveform scratches (Serato / Traktor style)
            self._drag_x = ev.position().x()
            self.setCursor(Qt.ClosedHandCursor)
            self.scratch.emit(0.0)
            return
        x = ev.position().x() / max(self.width(), 1)
        self.seek.emit(x * self.duration)

    def mouseMoveEvent(self, ev):
        if not self.zoom and self.peaks is not None and ev.buttons() & Qt.LeftButton:
            # the whole-track strip: drag to scrub through the track
            x = min(max(ev.position().x() / max(self.width(), 1), 0.0), 1.0)
            self.seek.emit(x * self.duration)
            return
        if self.zoom and getattr(self, "_drag_x", None) is not None:
            x = ev.position().x()
            dx = x - self._drag_x
            self._drag_x = x
            # pulling the waveform left moves the audio forward
            self.scratch.emit(-dx / max(self.width(), 1) * self.span)

    def mouseReleaseEvent(self, ev):
        if self.zoom and getattr(self, "_drag_x", None) is not None:
            self._drag_x = None
            self.setCursor(Qt.OpenHandCursor)
            self.scratch_end.emit()

    def mouseDoubleClickEvent(self, ev):
        if self.zoom and self.peaks is not None:
            x = ev.position().x() / max(self.width(), 1)
            self.seek.emit(self.pos + (x - 0.5) * self.span)
        elif self.peaks is not None:
            self.marker.emit("edit", 0.0)        # the whole track, zoomable, with its markers

    def wheelEvent(self, ev):
        ev.ignore()                  # zoom with the - / + buttons; the wheel scrolls the page

    def set_span(self, span: float):
        self.span = float(np.clip(span, 2.0, 32.0))
        self.update()

    def _zoom_image(self, w: int, h: int):
        """The scrolling view's picture. Its columns are fixed to the track, so
        levels and colours are worked out once per zoom level for the whole
        track; each frame only slices them (no per-frame colour maths)."""
        col = self.span / w
        t0 = self.pos - self.span / 2
        c0 = int(np.floor(t0 / col))
        # one page = 4 screen-widths of columns around the view: a few ms to build,
        # rebuilt only when the view leaves it (never the whole track at once - that
        # held the interpreter for 20-30 ms on a long track, enough to drop audio)
        page_w = 4 * w
        page = (c0 + w) // page_w
        key = (round(self.span, 6), w, _theme["name"], _theme["sat"], self.crange, id(self.bands), page)
        cache = getattr(self, "_zcache", None)
        if cache is None or cache[0] != key:
            hop = 256 / SR
            n = self.bands.shape[0]
            first = page * page_w - 2 * w - 2
            cidx = np.arange(first, first + 2 * page_w + 3)
            edges = np.floor(cidx * col / hop).astype(np.int64)
            padded = np.vstack([self.bands, np.zeros((1, 3), self.bands.dtype)])
            b = np.maximum.reduceat(padded, np.clip(edges, 0, n))[:-1].astype(np.float32)
            inside = (edges[1:] > 0) & (edges[:-1] < n)
            b[~inside] = 0
            heights = np.clip(b.max(axis=1), 0, 1)
            colors = _band_colors(b, self.crange)
            cache = self._zcache = (key, first, b, heights, colors)
        _key, first, b, heights, colors = cache
        frac = float(t0 / col - c0)
        i = c0 - first
        lo, hi = max(i, 0), min(i + w + 1, len(heights))
        pad_l, pad_r = lo - i, (i + w + 1) - hi
        hs = heights[lo:hi]
        if _theme["name"] == "Three-band":
            bb = np.pad(b[lo:hi], ((pad_l, max(pad_r, 0)), (0, 0)))
            return _render_layers(np.clip(bb, 0, 1), h), frac
        cs = colors[lo:hi]
        if pad_l or pad_r > 0:
            hs = np.pad(hs, (pad_l, max(pad_r, 0)))
            cs = np.pad(cs, ((pad_l, max(pad_r, 0)), (0, 0)))
        return _render(hs, cs, h), frac

    def _x(self, t, w):
        if self.zoom:
            return (t - (self.pos - self.span / 2)) / self.span * w
        return t / self.duration * w

    def paintEvent(self, _ev):
        p = QPainter(self)
        w, h = self.width(), self.height()
        if self.peaks is None or w < 2:
            p.fillRect(0, 0, w, h, QColor(18, 19, 22))
            p.setPen(QColor(90, 90, 96))
            if self.zoom:
                p.drawText(self.rect(), Qt.AlignCenter, "Drag a track here, or right-click a track → Load to Deck")
            return
        if self.zoom and self.bands is not None:
            img, frac = self._zoom_image(w, h)
            if self.grid is not None:
                # beat / bar lines go straight into the picture (one numpy pass, not a line call each)
                t0 = self.pos - self.span / 2
                b0 = int(np.floor(self.grid.beat_at(t0)))
                beats = np.arange(b0, int(np.ceil(self.grid.beat_at(t0 + self.span))) + 1)
                xs = (np.array([self.grid.time_of_beat(int(bt)) for bt in beats]) - t0) / self.span * w + frac
                _grid_columns(img._buf, xs[beats % 4 == 0], 0.55, wide=True)
                _grid_columns(img._buf, xs[beats % 4 != 0], 0.16)
            p.drawImage(QPointF(-frac, 0), img)
        else:
            if self._over is None or self._over[0] != w:
                n = self.peaks.shape[0]
                b = self.peaks[np.linspace(0, n - 1, w).astype(int)]
                self._over = (w, b)
            past = int(self.pos / self.duration * w)
            key = (w, h, past, _theme["name"], _theme["sat"])
            if getattr(self, "_over_img", (None, None))[0] != key:
                self._over_img = (key, _wave_image(self._over[1], h, past, rng=self.crange))
            p.drawImage(0, 0, self._over_img[1])
        if self.loop:
            x0, x1 = self._x(self.loop[0] / SR, w), self._x(self.loop[1] / SR, w)
            p.fillRect(QRectF(x0, 0, max(x1 - x0, 2), h), QColor(80, 220, 120, 60))
        for i, c in enumerate(self.hotcues):
            if c is None:
                continue
            x = self._x(c / SR, w)
            if -5 <= x <= w + 5:
                p.setPen(QPen(CUE_COLORS[i], 2))
                p.drawLine(int(x), 0, int(x), h)
                p.fillRect(QRectF(x, 0, 12, 12), CUE_COLORS[i])
                p.setPen(QColor(0, 0, 0))
                p.drawText(QRectF(x, 0, 12, 12), Qt.AlignCenter, str(i + 1))
        # mix markers: the intro (start .. IN, green), the outro blend (OUT .. END, red), labelled flags
        if self.mix_in is not None:
            x1 = self._x(self.mix_in, w)
            if x1 > 0:
                p.fillRect(QRectF(max(self._x(0.0, w), 0), 0, x1 - max(self._x(0.0, w), 0), h), QColor(67, 160, 71, 38))
        if self.mix_out is not None and self.grid is not None:
            x0 = self._x(self.mix_out, w)
            x1 = self._x(self.mix_out_end if self.mix_out_end is not None
                         else self.mix_out + self.mix_beats * self.grid.beat_len, w)
            p.fillRect(QRectF(x0, 0, max(x1 - x0, 2), h), QColor(229, 57, 53, 55))
        for t, label, colr in ((self.mix_in, "IN", QColor(67, 160, 71)), (self.mix_out, "OUT", QColor(229, 57, 53)),
                               (self.mix_out_end, "END", QColor(255, 152, 0))):
            if t is None:
                continue
            x = self._x(t, w)
            if -30 <= x <= w + 5:
                p.setPen(QPen(colr, 2))
                p.drawLine(int(x), 0, int(x), h)
                fh = min(14, h)
                fx = x - 30 if label == "END" else x               # END's flag sits before its line
                p.fillRect(QRectF(fx, 0, 30, fh), colr)
                p.setPen(QColor(255, 255, 255))
                p.drawText(QRectF(fx, 0, 30, fh), Qt.AlignCenter, label)
        cx = self._x(self.cue, w)
        p.setPen(QPen(QColor(255, 190, 0), 1, Qt.DashLine))
        p.drawLine(int(cx), 0, int(cx), h)
        px = w // 2 if self.zoom else int(self._x(self.pos, w))
        p.setPen(QPen(QColor(255, 60, 60) if self.zoom else QColor(255, 255, 255), 2))
        p.drawLine(px, 0, px, h)


class Oscilloscope(QWidget):
    """The master output as a trace, triggered on a rising zero crossing so a
    steady sound stands still (drawn with numpy into one image)."""

    def __init__(self, w: int = 84, h: int = 46):
        super().__init__()
        self.setFixedSize(w, h)
        self.setToolTip("Oscilloscope: the master output")
        self.data: np.ndarray | None = None      # (2, n) the newest master audio
        self.data_id = None

    def paintEvent(self, _ev):
        w, h = self.width(), self.height()
        img = np.full((h, w), np.uint32(0xFF101114), np.uint32)
        img[h // 2, :] = np.uint32(0xFF2A2C33)           # zero line
        x = self.data
        if x is not None and x.shape[1] > 64:
            mono = (x[0] + x[1]) * 0.5
            span = min(1024, mono.size // 2)
            seg = mono[: mono.size - span]
            up = np.nonzero((seg[:-1] <= 0) & (seg[1:] > 0))[0]
            i0 = int(up[-1]) if up.size else mono.size - span
            k = max(span // w, 1)
            win = mono[i0:i0 + k * w]
            if win.size < k * w:
                win = np.pad(win, (0, k * w - win.size))
            win = win.reshape(w, k)
            lo, hi = win.min(axis=1), win.max(axis=1)
            gain = 0.9 / max(float(max(hi.max(), -lo.min())), 0.15)   # auto-scale: quiet passages stay readable
            lo, hi = lo * gain, hi * gain
            ys = np.arange(h)[:, None]
            y_lo = np.clip((1 - hi) * (h - 1) / 2, 0, h - 1)
            y_hi = np.clip((1 - lo) * (h - 1) / 2, 0, h - 1)
            hit = (ys >= np.floor(y_lo)[None, :]) & (ys <= np.ceil(y_hi)[None, :])
            img = np.where(hit, np.uint32(0xFF5CE08A), img).astype(np.uint32)
        q = QImage(img.data, w, h, 4 * w, QImage.Format_RGB32)
        p = QPainter(self)
        p.drawImage(0, 0, q)
        p.setPen(QColor(60, 62, 70))
        p.drawRect(0, 0, w - 1, h - 1)


class PhaseMeter(QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedHeight(8)
        self.err = None

    def paintEvent(self, _ev):
        p = QPainter(self)
        w, h = self.width(), self.height()
        p.fillRect(0, 0, w, h, QColor(30, 31, 35))
        p.setPen(QColor(90, 90, 90))
        p.drawLine(w // 2, 0, w // 2, h)
        if self.err is None:
            return
        x = w / 2 + self.err * w
        good = abs(self.err) < 0.02
        p.fillRect(QRectF(min(x, w / 2), 1, abs(x - w / 2) + 3, h - 2),
                   QColor(80, 220, 120) if good else QColor(255, 120, 60))


def _btn(text, checkable=False, tip="", w=None, h=26):
    b = QPushButton(text)
    b.setCheckable(checkable)
    if tip:
        b.setToolTip(tip)
    if w:
        b.setFixedWidth(w)
        b.setObjectName("mini")          # narrow buttons: no padding so the label fits
    b.setFixedHeight(h)
    return b


def _sync_check(b, on):
    if b.isChecked() != on:
        b.blockSignals(True)
        b.setChecked(on)
        b.blockSignals(False)


def _set_text(w, t):
    """setText only on change: every change re-lays out the panel."""
    if w.text() != t:
        w.setText(t)


def _set_ss(w, st):
    if w.styleSheet() != st:
        w.setStyleSheet(st)


def _sync_value(w, v, tol=2):
    if abs(w.value() - v) > tol and not getattr(w, "isSliderDown", lambda: False)():
        w.blockSignals(True)
        w.setValue(v)
        w.blockSignals(False)


# --------------------------------------------------------------------------
# deck
# --------------------------------------------------------------------------

class JogPlatter(QWidget):
    """A virtual turntable: it spins with the music; grab and drag it around
    to scratch (or to search when the deck is paused)."""

    scratch = Signal(float)       # revolutions moved by the hand since the last event
    scratch_end = Signal()

    def __init__(self, color: QColor, size: int = 150):
        super().__init__()
        self.color = color
        self.setFixedSize(size, size)
        self.angle = 0.0
        self.held = False
        self.art: QPixmap | None = None      # album art: the record label
        self.scope: np.ndarray | None = None  # (2, n) recent samples: XY scope over the record (optional)
        self._last = None
        self.setCursor(Qt.OpenHandCursor)
        self.setToolTip("Grab and drag around to scratch")

    def _ang(self, pos) -> float:
        c = self.rect().center()
        return float(np.degrees(np.arctan2(pos.y() - c.y(), pos.x() - c.x())))

    def mousePressEvent(self, ev):
        if ev.button() == Qt.LeftButton:
            self.held = True
            self._last = self._ang(ev.position())
            self.setCursor(Qt.ClosedHandCursor)
            self.scratch.emit(0.0)
            self.update()

    def mouseMoveEvent(self, ev):
        if self.held and self._last is not None:
            a = self._ang(ev.position())
            d = (a - self._last + 180.0) % 360.0 - 180.0
            self._last = a
            self.scratch.emit(d / 360.0)

    def mouseReleaseEvent(self, _ev):
        if self.held:
            self.held = False
            self._last = None
            self.setCursor(Qt.OpenHandCursor)
            self.scratch_end.emit()
            self.update()

    def _static(self) -> QPixmap:
        """The record itself (body, grooves, deck-coloured rim): drawn once."""
        key = (self.width(), self.height(), self.held)
        if getattr(self, "_bg", (None, None))[0] == key:
            return self._bg[1]
        pm = QPixmap(self.size())
        pm.fill(Qt.transparent)
        p = QPainter(pm)
        p.setRenderHint(QPainter.Antialiasing)
        r = QRectF(self.rect().adjusted(4, 4, -4, -4))
        p.setPen(QPen(QColor(10, 11, 14), 2))
        p.setBrush(QColor(22, 23, 27))
        p.drawEllipse(r)
        p.setBrush(Qt.NoBrush)
        for k in range(1, 7):
            p.setPen(QPen(QColor(44, 46, 54), 1))
            inset = k * r.width() / 40
            p.drawEllipse(r.adjusted(inset, inset, -inset, -inset))
        ring = QColor(self.color)
        ring.setAlpha(255 if self.held else 170)
        p.setPen(QPen(ring, 5))
        p.drawEllipse(r.adjusted(3, 3, -3, -3))
        p.end()
        self._bg = (key, pm)
        return pm

    def _label(self, lr: float) -> QPixmap:
        """The record label (round album art, or the deck colour), drawn once
        per track and size; each frame only rotates this picture."""
        key = (id(self.art), int(lr))
        if getattr(self, "_lab", (None, None))[0] == key:
            return self._lab[1]
        d = int(2 * lr)
        pm = QPixmap(d, d)
        pm.fill(Qt.transparent)
        p = QPainter(pm)
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        clip = QPainterPath()
        clip.addEllipse(QRectF(0, 0, d, d))
        p.setClipPath(clip)
        if self.art is not None and not self.art.isNull():
            aw, ah = self.art.width(), self.art.height()
            sq = min(aw, ah)
            p.drawPixmap(QRectF(0, 0, d, d), self.art, QRectF((aw - sq) / 2, (ah - sq) / 2, sq, sq))
        else:
            p.fillRect(QRectF(0, 0, d, d), self.color.darker(170))
            p.setPen(QPen(self.color, 2))
            p.drawEllipse(QPointF(d / 2, d / 2), d * 0.3, d * 0.3)
        p.setClipping(False)
        p.setPen(QPen(QColor(0, 0, 0, 160), 1.5))
        p.drawEllipse(QRectF(0.75, 0.75, d - 1.5, d - 1.5))
        p.end()
        self._lab = (key, pm)
        return pm

    def paintEvent(self, _ev):
        p = QPainter(self)
        p.drawPixmap(0, 0, self._static())
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        r = self.rect().adjusted(4, 4, -4, -4)
        c = QPointF(r.center())
        a = np.radians(self.angle - 90)
        lr = r.width() * 0.34
        lab = self._label(lr)
        p.save()
        p.translate(c)
        p.rotate(self.angle)
        p.drawPixmap(QPointF(-lab.width() / 2, -lab.height() / 2), lab)
        p.restore()
        # spinning marker on the vinyl, outside the label
        rad0, rad1 = lr + 3, r.width() / 2 - 9
        p.setPen(QPen(QColor(240, 240, 245), 4, Qt.SolidLine, Qt.RoundCap))
        p.drawLine(c + QPointF(np.cos(a) * rad0, np.sin(a) * rad0), c + QPointF(np.cos(a) * rad1, np.sin(a) * rad1))
        if self.scope is not None:
            # XY scope: left against right, turned 45 degrees (mono = a vertical line)
            img = xy_scope_image(self.scope, int(r.width() * 0.62), self.color)
            p.drawImage(QPointF(c.x() - img.width() / 2, c.y() - img.height() / 2), img)
        p.setBrush(QColor(20, 20, 24))
        p.setPen(QPen(QColor(200, 200, 205), 1))
        p.drawEllipse(c, 3, 3)


class PressButton(QPushButton):
    """Fires the moment it is pressed (like a hardware button), not on release."""

    hit = Signal()

    def mousePressEvent(self, ev):
        if ev.button() == Qt.LeftButton:
            self.setDown(True)
            self.hit.emit()
            ev.accept()
            return
        super().mousePressEvent(ev)

    def mouseDoubleClickEvent(self, ev):
        self.mousePressEvent(ev)

    def mouseReleaseEvent(self, ev):
        if ev.button() == Qt.LeftButton:
            self.setDown(False)
            ev.accept()
            return
        super().mouseReleaseEvent(ev)


class CueButton(QPushButton):
    """CUE: acts on the press (no waiting for a click). Let go on the button
    and the preview stops; drag off it and let go, and the track plays on."""

    down = Signal()
    up = Signal(bool)          # True: released away from the button -> keep playing

    def __init__(self, text):
        super().__init__(text)
        self._held = False

    def mousePressEvent(self, ev):
        if ev.button() == Qt.LeftButton:
            self._held = True
            self.setDown(True)
            self.down.emit()
            ev.accept()
            return
        super().mousePressEvent(ev)

    def mouseDoubleClickEvent(self, ev):
        self.mousePressEvent(ev)        # rapid taps: every one counts

    def mouseMoveEvent(self, ev):
        if self._held:
            self.setDown(self.rect().contains(ev.position().toPoint()))
            ev.accept()
            return
        super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev):
        if self._held and ev.button() == Qt.LeftButton:
            self._held = False
            self.setDown(False)
            self.up.emit(not self.rect().contains(ev.position().toPoint()))
            ev.accept()
            return
        super().mouseReleaseEvent(ev)


class _Scratcher:
    """Turns hand movement into a scratch rate, with a hold when the hand stops."""

    def __init__(self, eng, deck: str):
        self.eng = eng
        self.deck = deck
        self.active = False
        self.last_t = 0.0
        self.rate = 0.0
        self.timer = QTimer()
        self.timer.setInterval(25)
        self.timer.timeout.connect(self._idle)

    def move(self, seconds: float):
        d = self.eng.decks[self.deck]
        if not d.loaded:
            return
        now = time.perf_counter()
        if not self.active:
            self.active = True
            self.last_t = now
            self.eng.scratch(self.deck, 0.0, True)               # hand on the record: it stops
            self.timer.start()
            return
        dt = max(1e-3, now - self.last_t)
        self.last_t = now
        target = float(np.clip(seconds / dt, -8, 8))
        self.rate = 0.5 * self.rate + 0.5 * target              # smooth jittery mouse events
        self.eng.scratch(self.deck, self.rate, True)

    def _idle(self):
        if self.active and time.perf_counter() - self.last_t > 0.035:
            self.rate *= 0.3
            self.eng.scratch(self.deck, self.rate, True)

    def end(self):
        if self.active:
            self.active = False
            self.rate = 0.0
            self.timer.stop()
            self.eng.jog_release(self.deck)


def _relayout(widget: QWidget) -> int:
    """Forget every cached size below widget (nested layouts and child
    widgets included) and return its minimum height for the rows now shown."""
    def walk(lay):
        for i in range(lay.count()):
            it = lay.itemAt(i)
            if it.layout() is not None:
                walk(it.layout())
            w = it.widget()
            if w is not None and w.layout() is not None and not w.isHidden():
                walk(w.layout())
                w.updateGeometry()
        lay.invalidate()
    walk(widget.layout())
    return widget.layout().totalMinimumSize().height()


def _min_width(widget: QWidget) -> int:
    return widget.layout().totalMinimumSize().width()


class QueueList(QListWidget):
    """Auto-mix queue: drag to reorder, drag out / ✕ / Delete to remove."""

    removeRequested = Signal(int)
    reordered = Signal()

    def __init__(self):
        super().__init__()
        from PySide6.QtWidgets import QAbstractItemView, QStyledItemDelegate

        self.setDragDropMode(QAbstractItemView.InternalMove)
        self.setDefaultDropAction(Qt.MoveAction)
        self.setSelectionMode(QAbstractItemView.SingleSelection)
        self.model().rowsMoved.connect(lambda *_: self.reordered.emit())

        class Delegate(QStyledItemDelegate):
            def paint(self, p, opt, ix):
                super().paint(p, opt, ix)
                r = opt.rect
                p.save()
                p.setPen(QColor(230, 90, 100))
                p.drawText(QRectF(r.right() - 24, r.top(), 22, r.height()), Qt.AlignCenter, "✕")
                p.restore()
        self.setItemDelegate(Delegate(self))

    def mousePressEvent(self, ev):
        it = self.itemAt(ev.position().toPoint())
        if it is not None and ev.button() == Qt.LeftButton and ev.position().x() > self.viewport().width() - 26:
            self.removeRequested.emit(self.row(it))         # the ✕
            return
        super().mousePressEvent(ev)

    def keyPressEvent(self, ev):
        if ev.key() in (Qt.Key_Delete, Qt.Key_Backspace) and self.currentRow() >= 0:
            self.removeRequested.emit(self.currentRow())
            return
        super().keyPressEvent(ev)

    def startDrag(self, actions):
        from PySide6.QtGui import QCursor, QDrag

        row = self.currentRow()
        it = self.currentItem()
        if it is None:
            return
        drag = QDrag(self)
        drag.setMimeData(self.mimeData([it]))
        res = drag.exec(Qt.MoveAction)
        inside = self.viewport().rect().contains(self.viewport().mapFromGlobal(QCursor.pos()))
        if res == Qt.IgnoreAction and not inside:
            self.removeRequested.emit(row)                  # dropped outside the list: remove it


class MarkerWave(QWidget):
    """Zoomable full waveform for placing the mix markers precisely.

    Wheel: zoom around the mouse · drag a flag: move it (snaps to the bar) ·
    drag elsewhere: scroll · double-click: play from there."""

    moved = Signal(str, float)      # marker kind, seconds (bar-snapped)
    seek = Signal(float)
    placed = Signal()               # a one-shot placement happened (the editor un-arms its button)

    COLORS = {"in": QColor(67, 160, 71), "out": QColor(229, 57, 53), "out_end": QColor(255, 152, 0)}
    LABELS = {"in": "IN", "out": "OUT", "out_end": "OUT END"}
    SECTION_COLORS = {"intro": QColor(67, 160, 71), "groove": QColor(90, 120, 200), "build": QColor(255, 193, 7),
                      "drop": QColor(229, 57, 53), "breakdown": QColor(120, 90, 200), "outro": QColor(255, 152, 0)}
    STRIP = 16                      # the section strip under the bar numbers

    def __init__(self):
        super().__init__()
        self.setMinimumHeight(200)
        self.setMouseTracking(True)
        self.bands = None
        self.duration = 1.0
        self.grid = None
        self.pos = 0.0
        self.marks: dict[str, float | None] = {"in": None, "out": None, "out_end": None}
        self.blend_beats = 64.0      # shown after OUT when there is no OUT END
        self.v0, self.v1 = 0.0, 1.0
        self._drag = None            # ("mark", kind) | ("pan", x0, v0, v1)
        self._preview: tuple[str, float] | None = None
        self.place: str | None = None          # armed: the next click puts this marker there
        self.sections: list[tuple[float, float, str]] = []   # (t0, t1, kind) from the structure analysis
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._menu)

    def set_track(self, bands, duration, grid):
        self.bands, self.duration, self.grid = bands, max(duration, 1e-3), grid
        self.crange = colour_range(bands)
        self._coarse = (None,)
        self.v0, self.v1 = 0.0, self.duration
        self.sections = []
        if bands is not None and grid is not None:
            try:
                from ..dj.structure import analyse

                st = analyse(bands, grid)
                self.sections = [(grid.time_of_beat(s.start_bar * 4), grid.time_of_beat(s.end_bar * 4), s.kind)
                                 for s in st.sections]
            except Exception:  # noqa: BLE001
                self.sections = []

    def _menu(self, pos):
        if self.bands is None:
            return
        from PySide6.QtWidgets import QMenu

        t = self._snap(self._t(pos.x()))
        m = QMenu(self)
        bar = f"bar {int(round(self.grid.beat_at(t) / 4)) + 1}" if self.grid else fmt_time(t)
        for k in ("in", "out", "out_end"):
            m.addAction(f"Put {self.LABELS[k]} here  ({bar})", lambda _=False, k=k, t=t: self.moved.emit(k, t))
        m.addSeparator()
        m.addAction(f"Play from here  ({bar})", lambda: self.seek.emit(t))
        m.exec(self.mapToGlobal(pos))

    # ---- view ------------------------------------------------------------------------
    def zoom(self, factor: float, around: float | None = None):
        span = self.v1 - self.v0
        c = around if around is not None else (self.v0 + self.v1) / 2
        min_span = (self.grid.beat_len * 8) if self.grid else 4.0
        ns = float(np.clip(span * factor, min_span, self._extent()))
        f = (c - self.v0) / max(span, 1e-9)
        self.v0 = c - f * ns
        self.v1 = self.v0 + ns
        self._clamp()
        self.update()

    def show_around(self, t: float, bars: float = 32):
        bl = self.grid.beat_len * 4 if self.grid else 2.0
        span = min(self.duration, bars * bl)
        self.v0, self.v1 = t - span * 0.3, t - span * 0.3 + span
        self._clamp()
        self.update()

    def _extent(self) -> float:
        """Track length, or further when OUT END lies past the end (a looped outro)."""
        ends = [t for t in self.marks.values() if t is not None]
        bl = self.grid.beat_len * 4 if self.grid else 2.0
        return max([self.duration] + [t + 2 * bl for t in ends])

    def fit(self):
        self.v0, self.v1 = 0.0, self._extent()
        self.update()

    def _clamp(self):
        span = self.v1 - self.v0
        lo, hi = 0.0, max(self._extent(), span)
        if self.v0 < lo:
            self.v0, self.v1 = lo, lo + span
        if self.v1 > hi + span * 0.2:
            self.v1 = hi + span * 0.2
            self.v0 = self.v1 - span

    def _t(self, x: float) -> float:
        return self.v0 + x / max(self.width(), 1) * (self.v1 - self.v0)

    def _x(self, t: float) -> float:
        return (t - self.v0) / max(self.v1 - self.v0, 1e-9) * self.width()

    def _snap(self, t: float) -> float:
        if self.grid is None:
            return t
        return max(0.0, self.grid.time_of_beat(np.floor(self.grid.beat_at(t) / 4.0 + 0.5) * 4.0))

    # ---- mouse --------------------------------------------------------------------------
    def _hit(self, x: float) -> str | None:
        best, bd = None, 9.0
        for k, t in self.marks.items():
            if t is not None and abs(self._x(t) - x) < bd:
                best, bd = k, abs(self._x(t) - x)
        return best

    def wheelEvent(self, ev):
        ev.ignore()                  # zoom with the - / + buttons

    def mousePressEvent(self, ev):
        if ev.button() != Qt.LeftButton or self.bands is None:
            return
        x = ev.position().x()
        if self.place:
            # armed from the editor: this click puts the marker here - the deck never jumps
            self.moved.emit(self.place, self._snap(self._t(x)))
            self.place = None
            self.placed.emit()
            self.setCursor(Qt.OpenHandCursor)
            return
        k = self._hit(x)
        self._drag = ("mark", k) if k else ("pan", x, self.v0, self.v1)
        self.setCursor(Qt.SizeHorCursor if k else Qt.ClosedHandCursor)

    def mouseMoveEvent(self, ev):
        x = ev.position().x()
        if self._drag is None:
            if self.place:
                self.setCursor(Qt.CrossCursor)
                self._preview = (self.place, self._snap(self._t(x)))
                self.update()
                return
            self.setCursor(Qt.SizeHorCursor if self._hit(x) else Qt.OpenHandCursor)
            return
        if self._drag[0] == "mark":
            self._preview = (self._drag[1], self._snap(self._t(x)))
        else:
            _, x0, v0, v1 = self._drag
            dt = (x0 - x) / max(self.width(), 1) * (v1 - v0)
            self.v0, self.v1 = v0 + dt, v1 + dt
            self._clamp()
        self.update()

    def mouseReleaseEvent(self, ev):
        if self._drag and self._drag[0] == "mark" and self._preview is not None:
            self.moved.emit(*self._preview)
        self._drag = None
        self._preview = None
        self.setCursor(Qt.OpenHandCursor)
        self.update()

    def mouseDoubleClickEvent(self, ev):
        ev.ignore()                  # a click never moves the deck: "Play from here" is in the right-click menu

    # ---- paint ------------------------------------------------------------------------------
    def paintEvent(self, _ev):
        p = QPainter(self)
        w, h = self.width(), self.height()
        p.fillRect(0, 0, w, h, QColor(16, 17, 20))
        if self.bands is None or w < 4:
            return
        wave_h = h - 22 - (self.STRIP if self.sections else 0)
        ts = self.v0 + (np.arange(w) + 0.5) / w * (self.v1 - self.v0)
        idx = (ts * SR / 256).astype(np.int64)
        valid = (idx >= 0) & (idx < self.bands.shape[0])
        # max over each pixel's span so short hits survive zooming out
        step = max(1, int((self.v1 - self.v0) / w * SR / 256))
        b = np.zeros((w, 3), np.float32)
        src = self.bands
        if step > 1:
            if getattr(self, "_coarse", (None,))[0] != step:
                n = -(-src.shape[0] // step) * step
                pad = np.zeros((n - src.shape[0], src.shape[1]), src.dtype)
                self._coarse = (step, np.concatenate([src, pad]).reshape(-1, step, src.shape[1]).max(axis=1))
            b[valid] = self._coarse[1][idx[valid] // step]
        else:
            b[valid] = src[idx[valid]]
        p.drawImage(0, 0, _wave_image(b, wave_h, rng=getattr(self, "crange", None)))
        # bars (phrase lines every 8 bars, numbers), beats when close enough
        if self.grid is not None:
            g = self.grid
            bar_px = 4 * g.beat_len / max(self.v1 - self.v0, 1e-9) * w
            every = next((k for k in (1, 4, 8, 16, 32) if bar_px * k >= 28), 64)
            b0 = int(np.floor(g.beat_at(self.v0) / 4)) - 1
            b1 = int(np.ceil(g.beat_at(self.v1) / 4)) + 1
            p.setFont(self.font())
            for bar in range(max(b0, 0), b1 + 1):
                x = self._x(g.time_of_beat(bar * 4))
                phrase = bar % 8 == 0
                if bar % every == 0 or phrase and every <= 8:
                    p.setPen(QPen(QColor(255, 255, 255, 150 if phrase else 70), 2 if phrase else 1))
                    p.drawLine(int(x), 0, int(x), wave_h)
                if bar % every == 0:
                    p.setPen(QColor(170, 172, 180))
                    p.drawText(QRectF(x + 2, wave_h + 2, 60, 18), Qt.AlignLeft | Qt.AlignVCenter, str(bar + 1))
                if bar_px >= 60:
                    p.setPen(QPen(QColor(255, 255, 255, 35), 1))
                    for k in range(1, 4):
                        xb = self._x(g.time_of_beat(bar * 4 + k))
                        p.drawLine(int(xb), 0, int(xb), wave_h)
        # the song's sections (intro / groove / build / drop / breakdown / outro) under the bar numbers
        if self.sections:
            y = h - self.STRIP
            p.setFont(self.font())
            for t0, t1, kind in self.sections:
                x0, x1 = self._x(t0), self._x(t1)
                if x1 < 0 or x0 > w:
                    continue
                c = self.SECTION_COLORS.get(kind, QColor(120, 120, 120))
                p.fillRect(QRectF(x0, y, max(x1 - x0 - 1, 1), self.STRIP - 2),
                           QColor(c.red(), c.green(), c.blue(), 150))
                if x1 - x0 > 50:
                    p.setPen(QColor(255, 255, 255, 220))
                    p.drawText(QRectF(x0 + 3, y, x1 - x0 - 6, self.STRIP - 2), Qt.AlignLeft | Qt.AlignVCenter,
                               kind.upper())
        marks = dict(self.marks)
        if self._preview:
            marks[self._preview[0]] = self._preview[1]
            if self._preview[0] == "out" and self.marks["out"] is not None and self.marks["out_end"] is not None:
                marks["out_end"] = self.marks["out_end"] + (self._preview[1] - self.marks["out"])
        # the intro the next track mixes into: start .. IN
        if marks["in"] is not None:
            x1 = self._x(marks["in"])
            if x1 > 0:
                p.fillRect(QRectF(0, 0, x1, wave_h), QColor(67, 160, 71, 32))
                p.setPen(QColor(67, 160, 71, 200))
                p.drawText(QRectF(max(x1 - 90, 4), wave_h - 18, 84, 16), Qt.AlignRight | Qt.AlignVCenter,
                           "intro ▶")
        # the outgoing blend: OUT .. OUT END (or the auto-mix length)
        if marks["out"] is not None:
            end = marks["out_end"]
            if end is None and self.grid is not None:
                end = marks["out"] + self.blend_beats * self.grid.beat_len
            if end is not None:
                x0, x1 = self._x(marks["out"]), self._x(end)
                p.fillRect(QRectF(x0, 0, max(x1 - x0, 2), wave_h), QColor(229, 57, 53, 40))
                p.setPen(QColor(229, 57, 53, 220))
                p.drawText(QRectF(x0 + 4, wave_h - 18, 160, 16), Qt.AlignLeft | Qt.AlignVCenter,
                           "◀ outro blend" + (" (auto length)" if marks["out_end"] is None else ""))
        for k in ("in", "out", "out_end"):
            t = marks[k]
            if t is None:
                continue
            x = self._x(t)
            c = self.COLORS[k]
            p.setPen(QPen(c, 3))
            p.drawLine(int(x), 0, int(x), wave_h)
            label = self.LABELS[k]
            if self.grid is not None:
                label += f"  bar {int(round(self.grid.beat_at(t) / 4)) + 1}"
            fw = p.fontMetrics().horizontalAdvance(label) + 10
            lx = x - fw if k == "out_end" else x                       # END's label sits before its line
            lx = min(max(lx, 0), w - fw)
            p.fillRect(QRectF(lx, 0, fw, 18), c)
            p.setPen(QColor(255, 255, 255))
            p.drawText(QRectF(lx, 0, fw, 18), Qt.AlignCenter, label)
        px = self._x(self.pos)
        p.setPen(QPen(QColor(255, 255, 255), 2))
        p.drawLine(int(px), 0, int(px), wave_h)


class MarkerEditor(QDialog):
    """Per-deck marker editor: zoom in, drag or bump IN / OUT / OUT END."""

    def __init__(self, dj: DJPanel, deck: str):
        super().__init__(dj)
        self.dj = dj
        self.deck = deck
        self.setWindowTitle(f"Mix markers - deck {deck}")
        self.resize(1100, 430)
        lay = QVBoxLayout(self)
        top = QHBoxLayout()
        self.title = QLabel("")
        self.title.setStyleSheet("font-size: 15px; font-weight: 700;")
        top.addWidget(self.title, 1)
        for txt, f, tip in (("−", 1.6, "Zoom out (or mouse wheel)"), ("+", 0.6, "Zoom in (or mouse wheel)")):
            b = _btn(txt, w=34, tip=tip)
            b.clicked.connect(lambda _=False, f=f: self.wave.zoom(f))
            top.addWidget(b)
        b = _btn("Whole track", tip="Show the whole track")
        b.clicked.connect(lambda: self.wave.fit())
        top.addWidget(b)
        lay.addLayout(top)
        self.wave = MarkerWave()
        self.wave.moved.connect(self._moved)
        self.wave.seek.connect(lambda t: self.dj.engine.seek(self.deck, t))
        self.wave.placed.connect(self._unarm)
        lay.addWidget(self.wave, 1)
        prow = QHBoxLayout()
        prow.addWidget(QLabel("<b>Add a marker:</b> press one, then click where it goes"))
        self.place_btns = {}
        for kind in ("in", "out", "out_end"):
            b = _btn(f"+ {MarkerWave.LABELS[kind]}", checkable=True, h=26,
                     tip=f"Then click on the waveform: {MarkerWave.LABELS[kind]} goes to the start of that bar")
            b.setStyleSheet(f"QPushButton:checked {{ background: {MarkerWave.COLORS[kind].name()}; color: white; }}")
            b.toggled.connect(lambda on, k=kind: self._arm(k, on))
            prow.addWidget(b)
            self.place_btns[kind] = b
        prow.addSpacing(16)
        prow.addWidget(QLabel("<b>Remove:</b>"))
        for kind in ("in", "out", "out_end"):
            b = _btn(f"− {MarkerWave.LABELS[kind]}", h=26, tip=f"Remove the {MarkerWave.LABELS[kind]} marker")
            b.clicked.connect(lambda _=False, k=kind: self._clear(k))
            prow.addWidget(b)
        prow.addStretch()
        lay.addLayout(prow)
        hint = QLabel("Clicking the waveform never moves the deck. Drag a flag to move it (it snaps to the start "
                      "of the bar) · drag elsewhere to scroll · right-click: put a marker here / play from here · "
                      "the strip under the bars shows the song's sections (intro, build, drop, breakdown, outro)")
        hint.setStyleSheet("color: #9a9ba3;")
        hint.setWordWrap(True)
        lay.addWidget(hint)
        self.rows = {}
        g = QGridLayout()
        for r, (kind, name) in enumerate((("in", "IN (next track starts)"), ("out", "OUT (start mixing out)"),
                                          ("out_end", "OUT END (outgoing gone)"))):
            lab = QLabel(name)
            lab.setStyleSheet(f"color: {MarkerWave.COLORS[kind].name()}; font-weight: 700;")
            val = QLabel("—")
            val.setMinimumWidth(170)
            g.addWidget(lab, r, 0)
            g.addWidget(val, r, 1)
            col = 2
            for txt, beats in (("−4 beats", -4), ("+4 beats", 4)):
                b = _btn(txt, tip=f"Move {kind.upper().replace('_', ' ')} {beats:+d} beats (one bar)")
                b.clicked.connect(lambda _=False, k=kind, bt=beats: self._do(self.dj.bump_marker, self.deck, k, bt))
                g.addWidget(b, r, col)
                col += 1
            b = _btn("Set at playhead", tip="Put it at the start of the bar that is playing")
            b.clicked.connect(lambda _=False, k=kind: self._at_playhead(k))
            g.addWidget(b, r, col)
            b = _btn("Show", tip="Zoom to it")
            b.clicked.connect(lambda _=False, k=kind: self._show(k))
            g.addWidget(b, r, col + 1)
            b = _btn("Remove", tip=f"Remove the {kind.upper().replace('_', ' ')} marker")
            b.clicked.connect(lambda _=False, k=kind: self._clear(k))
            g.addWidget(b, r, col + 2)
            self.rows[kind] = val
        lrow = QHBoxLayout()
        lrow.addWidget(QLabel("Outro length:"))
        for n in (8, 16, 32, 64, 128):
            b = _btn(f"{n} bars", tip=f"OUT END = OUT + {n} bars (the outgoing track is looped if it ends sooner)")
            b.clicked.connect(lambda _=False, n=n: self._do(self.dj.set_out_length, self.deck, n))
            lrow.addWidget(b)
        b = _btn("Auto", tip="No OUT END: auto-mix picks the length")
        b.clicked.connect(lambda: self._do(self.dj.set_out_length, self.deck, None))
        lrow.addWidget(b)
        lrow.addStretch()
        g.addLayout(lrow, 3, 0, 1, 8)
        lay.addLayout(g)
        self.status = QLabel("")
        lay.addWidget(self.status)
        self._path = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(50)
        self.refresh()

    def _do(self, fn, *a):
        try:
            self.status.setText(fn(*a) or "")
        except ValueError as e:
            self.status.setText(f"⚠ {e}")
        self.refresh()

    def _arm(self, kind: str, on: bool):
        if on:
            for k, b in self.place_btns.items():
                if k != kind and b.isChecked():
                    b.setChecked(False)
            self.wave.place = kind
            self.wave.setCursor(Qt.CrossCursor)
        elif self.wave.place == kind:
            self.wave.place = None
            self.wave._preview = None
            self.wave.update()

    def _unarm(self):
        for b in self.place_btns.values():
            b.setChecked(False)

    def _moved(self, kind, t):
        mi, mo, me = self.dj.markers_of(self.deck)
        if kind == "in":
            mi = t
        elif kind == "out":
            if me is not None and mo is not None:
                me = t + (me - mo)
            mo = t
        else:
            me = t
        self._do(self.dj.set_markers, self.deck, mi, mo, me)

    def _at_playhead(self, kind):
        d = self.dj.engine.decks[self.deck]
        self._moved(kind, self.dj._snap_bar(self.deck, d.time))

    def _clear(self, kind):
        mi, mo, me = self.dj.markers_of(self.deck)
        if kind == "in":
            mi = None
        elif kind == "out":
            mo = me = None
        else:
            me = None
        self._do(self.dj.set_markers, self.deck, mi, mo, me)

    def _show(self, kind):
        t = dict(zip(("in", "out", "out_end"), self.dj.markers_of(self.deck)))[kind]
        if t is not None:
            self.wave.show_around(t)

    def refresh(self):
        d = self.dj.engine.decks[self.deck]
        if d.info.path != self._path:
            self._path = d.info.path
            self.title.setText(f"{d.info.artist + ' - ' if d.info.artist else ''}{d.info.title}" if d.loaded
                               else "Deck is empty")
            if d.loaded:
                self.wave.marks = dict(zip(("in", "out", "out_end"), self.dj.markers_of(self.deck)))
                self.wave.set_track(getattr(d, "bands", None), d.length / SR, d.grid)
                self.wave.fit()
        if not d.loaded:
            return
        self.wave.pos = d.time
        self.wave.grid = d.grid
        self.wave.blend_beats = 4.0 * self.dj.automix.s.mix_bars
        marks = dict(zip(("in", "out", "out_end"), self.dj.markers_of(self.deck)))
        self.wave.marks = marks
        for k, t in marks.items():
            if t is None:
                self.rows[k].setText("—" if k != "out_end" else "— (auto length)")
            else:
                bar = int(round(d.grid.beat_at(t) / 4)) + 1 if d.grid else 0
                txt = f"{fmt_time(t)}  ·  bar {bar}"
                if k == "out_end" and marks["out"] is not None and d.grid is not None:
                    txt += f"  ·  {int(round((d.grid.beat_at(t) - d.grid.beat_at(marks['out'])) / 4))} bars"
                self.rows[k].setText(txt)
        self.wave.update()

    def done(self, r):
        self.timer.stop()
        super().done(r)


class DeckPanel(QFrame):
    """Controller-style deck: header, scrolling waveform (drag to scratch),
    overview, jog platter, performance pads and transport."""

    _art_ready = Signal(str, object)      # path, image bytes (from a worker thread)

    def __init__(self, name: str, dj: DJPanel, mirrored: bool = False):
        super().__init__()
        self._art_ready.connect(self._set_art)
        self.name = name
        self.dj = dj
        self.eng = dj.engine
        self.setAcceptDrops(True)
        self.setObjectName("deck")
        col = DECK_COLORS[name]
        self.col = col
        self.setStyleSheet(f"#deck {{ background: #1c1c1e; border: 1px solid #0b0b0b;"
                           f" border-top: 3px solid {col.name()}; border-radius: 3px; }}")
        self.scr = _Scratcher(self.eng, name)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(5, 4, 5, 5)
        outer.setSpacing(3)
        outer.setSizeConstraint(QLayout.SetNoConstraint)      # the view decides what fits

        # ---- header: deck letter, art, title / artist, key, tempo, time ----------------------
        head = QHBoxLayout()
        head.setSpacing(6)
        badge = QLabel(name)
        badge.setFixedSize(22, 22)
        badge.setAlignment(Qt.AlignCenter)
        badge.setStyleSheet(f"background: {col.name()}; color: #111; font-weight: 900; font-size: 13px;"
                            "border-radius: 3px;")
        head.addWidget(badge)
        self.art = QLabel()
        self.art.setFixedSize(32, 32)
        self.art.setScaledContents(True)
        self.art.setStyleSheet("background: #232325; border-radius: 2px;")
        head.addWidget(self.art)
        titles = QVBoxLayout()
        titles.setSpacing(0)
        self.title = _TitleLabel("—")
        self.title.setStyleSheet("font-size: 13px; font-weight: 700; color: #f2f2f5;")
        self.title.setToolTip("Double-click: what mixes next from this track")
        self.title.doubleClicked.connect(lambda: self.dj.next_from_deck(name))
        self.artist = QLabel("Drop a track here")
        self.artist.setStyleSheet("font-size: 11px; color: #8d8f97;")
        titles.addWidget(self.title)
        titles.addWidget(self.artist)
        head.addLayout(titles, 1)
        self.key_lbl = QLabel("")
        self.key_lbl.setAlignment(Qt.AlignCenter)
        self.key_lbl.setMinimumWidth(52)
        head.addWidget(self.key_lbl)
        bpmbox = QVBoxLayout()
        bpmbox.setSpacing(0)
        self.bpm_lbl = QLabel("")
        self.bpm_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.bpm_lbl.setStyleSheet("font-size: 19px; font-weight: 800; color: #ffffff;")
        self.pct_lbl = QLabel("")
        self.pct_lbl.setAlignment(Qt.AlignRight)
        self.pct_lbl.setStyleSheet("font-size: 10px; color: #8d8f97;")
        bpmbox.addWidget(self.bpm_lbl)
        bpmbox.addWidget(self.pct_lbl)
        head.addLayout(bpmbox)
        timebox = QVBoxLayout()
        timebox.setSpacing(0)
        self.remain_lbl = QLabel("")
        self.remain_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.remain_lbl.setMinimumWidth(70)
        self.remain_lbl.setStyleSheet("font-size: 16px; font-weight: 700; color: #f2f2f5;")
        self.time_lbl = QLabel("")
        self.time_lbl.setAlignment(Qt.AlignRight)
        self.time_lbl.setStyleSheet("font-size: 10px; color: #8d8f97;")
        timebox.addWidget(self.remain_lbl)
        timebox.addWidget(self.time_lbl)
        head.addLayout(timebox)
        outer.addLayout(head)

        def wrap(layout, spacing=3):
            w = QWidget()
            w.setObjectName("deckrow")          # tight buttons (theme)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(spacing)
            w.setLayout(layout)
            return w

        # ---- waveform, with the tempo fader beside it (jog platter optional) -----------------
        self.zoom = Waveform(name, True)
        self.zoom.setMinimumHeight(48)
        self.over = Waveform(name, False)
        for wv in (self.zoom, self.over):
            wv.seek.connect(lambda t, n=name: self.eng.seek(n, t))
            wv.marker.connect(lambda kind, t, n=name: self.dj.set_marker(n, kind, t))
        # grabbing the waveform or the on-screen platter with the mouse always
        # scrubs: that is a deliberate grab (Scratch mode only governs the
        # controller's jog top, which a hand rests on by accident)
        self.zoom.scratch.connect(lambda secs: (self.dj.manual_touch(name, "scratch"), self.scr.move(secs)))
        self.zoom.scratch_end.connect(self.scr.end)
        self.platter = JogPlatter(col, 112)
        self.platter.scratch.connect(lambda revs: (self.dj.manual_touch(name, "scratch"),
                                                   self.scr.move(revs * 1.8)))
        self.platter.scratch_end.connect(self.scr.end)
        tb = QVBoxLayout()
        self.tempo = ResetSlider(Qt.Vertical, 0, "Tempo")
        self.tempo.setRange(-1000, 1000)
        self.tempo.setInvertedAppearance(True)       # down = faster, like a turntable
        self.tempo.valueChanged.connect(lambda v: (self.dj.manual_touch(name, "tempo"),
                                                   self.eng.set(name, tempo=v / 1000.0)))
        tb.addWidget(self.tempo, 1, Qt.AlignHCenter)
        self.range = QComboBox()
        self.range.addItems(["±6%", "±8%", "±16%", "±50%"])
        self.range.setCurrentIndex(1)
        self.range.setFixedWidth(58)
        self.range.setToolTip("Tempo fader range")
        self.range.currentIndexChanged.connect(
            lambda i: self.eng.set(name, tempo_range=(0.06, 0.08, 0.16, 0.5)[i]))
        tb.addWidget(self.range, 0, Qt.AlignHCenter)
        # waveform zoom: - / + and sync zoom (both decks together), under the tempo fader
        zr = QHBoxLayout()
        zr.setSpacing(2)
        b_zo = _btn("−", tip="Zoom the waveform out", h=20, w=18)
        b_zi = _btn("+", tip="Zoom the waveform in", h=20, w=18)
        b_zo.clicked.connect(lambda: self.dj.zoom_waveforms(self.name, 1.25))
        b_zi.clicked.connect(lambda: self.dj.zoom_waveforms(self.name, 0.8))
        self.b_zsync = _btn("⛓", True, "Sync zoom: both decks zoom together", h=20, w=20)
        self.b_zsync.clicked.connect(lambda on: self.dj.set_zoom_sync(on))
        for w_ in (b_zo, b_zi, self.b_zsync):
            zr.addWidget(w_)
        tb.addLayout(zr)
        self.w_tempo = wrap(tb, 2)
        body = QHBoxLayout()
        body.addWidget(self.platter, 0, Qt.AlignVCenter)
        body.addWidget(self.zoom, 1)
        body.addWidget(self.w_tempo)
        self.body_lay = body                       # the zoom moves out of here in the stacked layout
        self.w_body = wrap(body, 4)
        outer.addWidget(self.w_body, 1)

        # ---- SYNC  MASTER  KEY  [phase]  beat jump ------------------------------------------------
        self.b_sync = QPushButton("SYNC")
        self.b_sync.setObjectName("sync")
        self.b_sync.setCheckable(True)
        self.b_sync.setToolTip("Tempo + phase lock to the master deck")
        self.b_sync.clicked.connect(lambda on: (self.dj.manual_touch(name, "sync"), self.eng.set(name, sync=on)))
        self.b_master = QPushButton("MASTER")
        self.b_master.setObjectName("toggle")
        self.b_master.setCheckable(True)
        self.b_master.clicked.connect(lambda: self.eng.set_master(name))
        self.b_key = QPushButton("KEY")
        self.b_key.setObjectName("toggle")
        self.b_key.setCheckable(True)
        self.b_key.setChecked(True)
        self.b_key.setToolTip("Key lock: tempo changes keep the original key")
        self.b_key.clicked.connect(lambda on: self.eng.set(name, keylock=on))
        self.phase = PhaseMeter()
        b_jb = _btn("◂", tip="Beat jump back", h=22)
        b_jf = _btn("▸", tip="Beat jump forward", h=22)
        self.jump = QComboBox()
        self.jump.addItems(["1", "2", "4", "8", "16", "32"])
        self.jump.setCurrentText("4")
        self.jump.setToolTip("Beat jump size (beats)")
        b_jb.clicked.connect(lambda: (self.dj.manual_touch(name, "jump"),
                                      self.eng.beatjump(name, -int(self.jump.currentText()))))
        b_jf.clicked.connect(lambda: (self.dj.manual_touch(name, "jump"),
                                      self.eng.beatjump(name, int(self.jump.currentText()))))
        srow = QHBoxLayout()
        for b in (self.b_sync, self.b_master, self.b_key):
            b.setFixedHeight(22)
            srow.addWidget(b)
        srow.addWidget(self.phase, 1)
        for w_ in (b_jb, self.jump, b_jf):
            srow.addWidget(w_)
        self.w_jump = wrap(srow)
        outer.addWidget(self.w_jump)
        outer.addWidget(self.over)

        # ---- transport + loops:  ▶ CUE CUP FLX | ‹ 1 2 4 8 16 32 › | IN OUT ACTIVE | ⌃ ------------
        self.b_play = PressButton("▶")
        self.b_play.setObjectName("play")
        self.b_play.setCheckable(True)
        self.b_play.setToolTip("Play / pause")
        self.b_play.hit.connect(lambda: (self.dj.manual_touch(name, "play"), self.eng.toggle_play(name)))
        self.b_cue = CueButton("CUE")
        self.b_cue.setObjectName("cue")
        self.b_cue.setToolTip("Cue (hold to preview from the cue point; drag off the button or press "
                              "PLAY while holding to keep playing)")
        self.b_cue.down.connect(lambda: (self.dj.manual_touch(name, "cue"), self.eng.cue_press(name)))
        self.b_cue.up.connect(lambda keep: self.eng.cue_release(name, keep))
        b_cup = PressButton("CUP")
        b_cup.setToolTip("Cue-play: jump to the cue point and play")
        b_cup.hit.connect(self._cup)
        self.b_flux = _btn("FLX", True, "Flux: after a loop or jump the track continues where it would have been")
        self.b_flux.clicked.connect(lambda on: self.eng.set(name, flux=on))
        trow = QHBoxLayout()
        for b in (self.b_play, self.b_cue, b_cup, self.b_flux):
            b.setFixedHeight(26)
            trow.addWidget(b)
        trow.addSpacing(6)
        self.loop_beats_sel = 4
        self.loop_btns = {}
        b_half = _btn("‹", tip="Halve the loop", h=26)
        b_half.clicked.connect(lambda: self._loop_step(0.5))
        trow.addWidget(b_half)
        for n_ in (1, 2, 4, 8, 16, 32):
            b = _btn(str(n_), True, f"Loop {n_} beats (click again: off)", h=26)
            b.setObjectName("loopsize")
            b.clicked.connect(lambda _=False, n_=n_: self._loop_size(n_))
            self.loop_btns[n_] = b
            trow.addWidget(b)
        b_dbl = _btn("›", tip="Double the loop", h=26)
        b_dbl.clicked.connect(lambda: self._loop_step(2.0))
        trow.addWidget(b_dbl)
        trow.addSpacing(6)
        b_in = _btn("IN", tip="Loop in", h=26)
        b_in.clicked.connect(lambda: self.eng.loop_in(name))
        b_out = _btn("OUT", tip="Loop out", h=26)
        b_out.clicked.connect(lambda: self.eng.loop_out(name, self.loop_beats_sel))
        self.b_loop = _btn("LOOP", True, "Loop on / off (active)", h=26)
        self.b_loop.setObjectName("loop")
        self.b_loop.clicked.connect(self._toggle_loop)
        for b in (b_in, b_out, self.b_loop):
            trow.addWidget(b)
        b_mk = _btn("⚑", tip="Mix markers: intro (IN), outro (OUT), outro end - add, move, remove", h=26, w=28)
        b_mk.clicked.connect(lambda: self.dj.edit_markers(name))
        trow.addWidget(b_mk)
        self.b_adv = _btn("⌃", True, "More: hot cues, beat grid, key, tempo tools, mix markers", h=26)
        self.b_adv.setChecked(self.dj.settings.value(f"dj_adv_{name}", "true") == "true")
        self.b_adv.toggled.connect(self._adv_toggled)
        trow.addWidget(self.b_adv)
        self.w_loop = wrap(trow)
        outer.addWidget(self.w_loop)

        # ---- advanced: hot cues | grid · key · tempo tools · markers -----------------------------
        pads = QHBoxLayout()
        self.hc = []
        for i in range(8):
            b = QPushButton(str(i + 1))
            b.setObjectName("pad")
            b.setFixedHeight(22)
            b.setMinimumWidth(22)
            b.setToolTip("Hot cue: click = set / jump · right-click = delete")
            b.clicked.connect(lambda _=False, i=i: self._hotcue(i))
            b.setContextMenuPolicy(Qt.CustomContextMenu)
            b.customContextMenuRequested.connect(lambda _p, i=i: self._hotcue_delete(i))
            self.hc.append(b)
            pads.addWidget(b)
        self.w_pads = wrap(pads)
        outer.addWidget(self.w_pads)
        krow = QHBoxLayout()
        for txt, ms in (("◂", -5), ("▸", 5)):
            b = _btn(txt, w=22, tip=f"Shift the beat grid {ms:+d} ms", h=22)
            b.clicked.connect(lambda _=False, ms=ms: self._grid_shift(ms / 1000))
            krow.addWidget(b)
        b1 = _btn("1↓", w=28, tip="Set beat 1 (downbeat) at the playhead", h=22)
        b1.clicked.connect(self._grid_downbeat)
        krow.addWidget(b1)
        self.key_shift = QSpinBox()
        self.key_shift.setRange(-12, 12)
        self.key_shift.setPrefix("Key ")
        self.key_shift.setSuffix(" st")
        self.key_shift.setToolTip("Transpose in semitones (tempo unaffected)")
        self.key_shift.valueChanged.connect(lambda v: self.eng.set(name, key_shift=float(v)))
        krow.addWidget(self.key_shift)
        b_ks = _btn("KEY SYNC", tip="Shift to the nearest key that is harmonic with the master", h=22)
        b_ks.clicked.connect(lambda: self.eng.key_sync(name))
        krow.addWidget(b_ks)
        self.grid_bpm = QLabel("")
        self.grid_bpm.setAlignment(Qt.AlignCenter)
        self.grid_bpm.setMinimumWidth(62)
        self.grid_bpm.setToolTip("The track's own tempo (beat grid)")
        self.grid_bpm.setStyleSheet("background: #121213; border: 1px solid #0b0b0b; font-weight: 700;")
        krow.addWidget(self.grid_bpm, 1)
        for txt, tip, fn in (("/2", "Halve the track's BPM", lambda: self._grid_bpm_scale(0.5)),
                             ("x2", "Double the track's BPM", lambda: self._grid_bpm_scale(2.0)),
                             ("RESET", "Tempo fader back to 0", lambda: self.tempo.setValue(0)),
                             ("TAP", "Tap the beat to set the track's BPM", self._tap)):
            b = _btn(txt, tip=tip, h=22)
            b.clicked.connect(fn)
            krow.addWidget(b)
        b_mk = _btn("✎ MARKERS", tip="Zoom in and place the IN / OUT / OUT END mix markers precisely", h=22)
        b_mk.clicked.connect(lambda: self.dj.edit_markers(name))
        krow.addWidget(b_mk)
        self.w_key = wrap(krow)
        outer.addWidget(self.w_key)
        self._taps: list[float] = []
        self._loaded_path = None
        # narrow screens: long titles are clipped (full text in the tooltip) and
        # buttons may get narrower, so the decks never force the window wider
        from PySide6.QtWidgets import QSizePolicy

        for lb in (self.title, self.artist):
            lb.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
            lb.setMinimumWidth(40)
        for w_ in (self.w_loop, self.w_pads):     # symbols / short labels: safe to narrow
            for b in w_.findChildren(QPushButton):
                pol = QSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
                pol.setHorizontalStretch(len(b.text()) + 2)     # wordier labels get more of the row
                b.setSizePolicy(pol)
                b.setMinimumWidth(18)
        self.view = None
        self.apply_view()

    # ---- deck views (Traktor style): full / essential / micro -----------------------------------
    VIEWS = ("full", "essential", "micro")

    def _needs(self) -> dict:
        """Height each view really needs (measured with the current fonts and settings)."""
        if getattr(self, "_measuring", False):
            # Qt (on Windows) asks for size hints while rows are being shown or
            # hidden for the measurement: answer from the last result, never re-enter
            return getattr(self, "_need", None) or {k: 0 for v in self.VIEWS for k in (v, v + "_w")}
        key = (self.dj.show_platter(), self.font().pointSizeF())
        if getattr(self, "_need_key", None) != key:
            cur = self.view
            need = {}
            self._measuring = True              # showing/hiding rows must not re-enter resizeEvent
            try:
                for v in self.VIEWS:
                    self._show(v)
                    need[v] = _relayout(self)
                    need[v + "_w"] = _min_width(self)
                if cur:
                    self._show(cur)
            finally:
                self._measuring = False
            self._need, self._need_key = need, key
        return self._need

    def minimumSizeHint(self):
        # the smallest view: the splitter may shrink the deck that far, the view follows
        from PySide6.QtCore import QSize

        n = self._needs()
        return QSize(n["micro_w"], n["micro"])

    def wanted_view(self) -> str:
        pref = self.dj.deck_view_pref()
        if pref in self.VIEWS:
            return pref
        need, h, w = self._needs(), self.height(), self.width()
        for v in self.VIEWS:                     # the richest view that fits (height and width)
            if h >= need[v] and w >= need[v + "_w"]:
                return v
        return "micro"

    def apply_view(self, force: bool = False):
        if force:
            self._need_key = None
        v = self.wanted_view()
        if v == self.view and not force:
            return
        self.view = v
        self._show(v)
        self.updateGeometry()

    def _show(self, v: str):
        full = v == "full"
        adv = full and self.b_adv.isChecked()
        self.w_body.setVisible(v != "micro")
        self.platter.setVisible(full and self.dj.show_platter())
        self.w_jump.setVisible(v != "micro")
        self.w_pads.setVisible(adv)
        self.w_key.setVisible(adv)

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        if getattr(self, "_measuring", False):
            return
        if self.dj.deck_view_pref() == "auto":
            QTimer.singleShot(0, self.apply_view)

    def _toggle_loop(self, on):
        self.dj.manual_touch(self.name, "loop")
        self.eng.loop_beats(self.name, self.loop_beats_sel if on else None)

    def _loop_size(self, n: int):
        """Loop-size buttons: start a loop of that size; the same size again ends it."""
        self.dj.manual_touch(self.name, "loop")
        self.eng.set(self.name, loop_size=float(n))
        d = self.eng.decks[self.name]
        if d.loop is not None and self.loop_beats_sel == n:
            self.eng.loop_beats(self.name, None)
        else:
            self.loop_beats_sel = n
            self.eng.loop_beats(self.name, n)

    def _loop_step(self, f: float):
        d = self.eng.decks[self.name]
        self.loop_beats_sel = int(min(max(self.loop_beats_sel * f, 1), 32))
        if d.loop is not None:
            self.eng.loop_resize(self.name, f)
        self.eng.set(self.name, loop_size=float(self.loop_beats_sel))     # after the resize: exact

    def _cup(self):
        self.dj.manual_touch(self.name, "play")
        d = self.eng.decks[self.name]
        if d.loaded:
            self.eng.seek(self.name, d.cue_point / SR)
            self.eng.set_playing(self.name, True)

    def _adv_toggled(self, on):
        self.dj.settings.setValue(f"dj_adv_{self.name}", "true" if on else "false")
        self.apply_view(force=True)

    def _grid_bpm_scale(self, f: float):
        d = self.eng.decks[self.name]
        if d.grid is not None:
            self.eng.set_grid(self.name, bpm=d.grid.bpm * f)
            self.dj.grid_changed(self.name)

    def _tap(self):
        """Tap the beat: after 4 taps the track's BPM follows them."""
        import time as _t

        now = _t.monotonic()
        if self._taps and now - self._taps[-1] > 2.0:
            self._taps = []
        self._taps.append(now)
        if len(self._taps) >= 4:
            bpm = 60.0 / float(np.median(np.diff(self._taps[-8:])))
            d = self.eng.decks[self.name]
            if d.grid is not None and 50 <= bpm <= 220:
                self.eng.set_grid(self.name, bpm=bpm / max(d.rate, 1e-6))
                self.dj.grid_changed(self.name)

    def dragEnterEvent(self, ev):
        if ev.mimeData().hasFormat(TRACK_MIME):
            ev.acceptProposedAction()

    def dropEvent(self, ev):
        try:
            idx = int(bytes(ev.mimeData().data(TRACK_MIME)).decode())
        except ValueError:
            return
        self.dj.load_index(self.name, idx)
        ev.acceptProposedAction()

    def _hotcue(self, i):
        self.eng.hotcue(self.name, i)
        self.dj.cues_changed(self.name)

    def _hotcue_delete(self, i):
        self.eng.hotcue_delete(self.name, i)
        self.dj.cues_changed(self.name)

    def _grid_shift(self, secs):
        d = self.eng.decks[self.name]
        if d.grid is not None:
            self.eng.set_grid(self.name, first_beat=d.grid.first_beat + secs)
            self.dj.grid_changed(self.name)

    def _grid_downbeat(self):
        self.eng.grid_downbeat_here(self.name)
        self.dj.grid_changed(self.name)

    def _load_art(self, path: str):
        self.art.clear()
        self.platter.art = None
        if not path:
            return

        def job():
            try:
                from ..art import get_art

                data = get_art(path)
            except Exception:  # noqa: BLE001
                data = None
            self._art_ready.emit(path, data)
        threading.Thread(target=job, daemon=True, name="deck-art").start()

    def _set_art(self, path: str, data):
        if path != self._loaded_path or not data:
            return
        pm = QPixmap()
        if pm.loadFromData(data):
            # scaled once here: the platter redraws it rotated 30 times a second
            self.art.setPixmap(pm.scaled(64, 64, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation))
            self.platter.art = pm.scaled(200, 200, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
            self.platter.update()

    def refresh(self):
        d = self.eng.decks[self.name]
        if d.info.path != self._loaded_path:
            fresh = bool(d.info.path)
            self._loaded_path = d.info.path
            self.title.setText(d.info.title or "—")
            if fresh and d.info.lib_index >= 0:
                # a track loaded into a deck: it is the one the library shows and
                # the Next Track list is what mixes from it
                try:
                    self.dj.main.focus_track(int(d.info.lib_index))
                except Exception:  # noqa: BLE001
                    log.exception("focusing the loaded track failed")
            self.title.setToolTip(f"{d.info.artist + ' - ' if d.info.artist else ''}{d.info.title}\n"
                                  "Double-click: what mixes next from this track")
            lib = self.dj.main.lib
            played = lib is not None and 0 <= d.info.lib_index < lib.n and lib.play_count[d.info.lib_index] > 0
            sub = d.info.artist or ""
            if d.info.style:
                sub += f"  ·  {d.info.style}"
            if played:
                sub += "  ·  ✓ played before"
            self.artist.setText(sub if d.loaded else "Drop a track here")
            self._load_art(d.info.path if d.loaded else "")
            bands = getattr(d, "bands", None)
            for wv in (self.zoom, self.over):
                wv.set_track(d.peaks, bands, d.length / SR, d.grid)
            _sync_value(self.key_shift, int(d.key_shift), 0)
        if not d.loaded:
            for lb in (self.bpm_lbl, self.pct_lbl, self.key_lbl, self.time_lbl, self.remain_lbl):
                _set_text(lb, "")
            _set_ss(self.key_lbl, "")
            return
        eff = d.rate_override if d.rate_override is not None else d.base_rate   # what it really plays at
        pct = (eff - 1) * 100
        _set_text(self.bpm_lbl, f"{d.bpm:.2f}")
        _set_text(self.pct_lbl, f"{pct:+.2f}%  ·  orig {d.grid.bpm if d.grid else 0:.1f}")
        ek = d.effective_key
        if ek >= 0:
            kc = key_color(ek) or QColor(200, 200, 200)
            shifted = ek != d.info.key
            _set_text(self.key_lbl, f"{camelot_str(ek)}\n{key_name(ek)}")
            _set_ss(self.key_lbl, f"background: {kc.darker(260).name()}; color: {kc.name()};"
                                       f" border: 1px solid {kc.name()}; border-radius: 6px;"
                                       f" font-weight: 800; font-size: 14px; padding: 2px 6px;"
                                       + (" border-style: dashed;" if shifted else ""))
            self.key_lbl.setToolTip(f"original key {camelot_str(d.info.key)} {key_name(d.info.key)}"
                                    if shifted else "")
        # what you see matches what you hear: positions drawn as late as the output plays
        t_vis = max(0.0, d.time - self.eng.display_latency * d.rate) if d.playing else d.time
        # ... but never before where it (re)started: CUE / PLAY / a jump must not look like a step back
        s0 = d.start_pos / SR
        if d.playing and s0 <= d.time < s0 + self.eng.display_latency + 0.05:
            t_vis = max(t_vis, s0)
        remain = d.length / SR - t_vis
        _set_text(self.remain_lbl, f"-{fmt_time(remain)}")
        _set_text(self.time_lbl, fmt_time(t_vis))
        warn = d.playing and remain < 30
        _set_ss(self.remain_lbl, "font-size: 20px; font-weight: 700; color: "
                                      + ("#ff5050;" if warn else "#f2f2f5;"))
        for wv in (self.zoom, self.over):
            wv.pos = t_vis
            wv.cue = d.cue_point / SR
            wv.loop = d.loop
            wv.hotcues = d.hotcues
            wv.grid = d.grid
            wv.mix_in = d.mix_in / SR if d.mix_in is not None else None
            wv.mix_out = d.mix_out / SR if d.mix_out is not None else None
            wv.mix_out_end = d.mix_out_end / SR if d.mix_out_end is not None else None
            wv.mix_beats = 4.0 * self.dj.automix.s.mix_bars
        # redraw only what changed: the scrolling view whenever the track moves,
        # the overview only when its playhead moves a pixel or a marker / loop changes
        state = (round(t_vis, 3), d.cue_point, d.loop, tuple(d.hotcues), d.mix_in, d.mix_out, d.mix_out_end,
                 self.zoom.span, self.zoom.width(), self.zoom.height())
        if state != getattr(self, "_zoom_state", None):
            self._zoom_state = state
            self.zoom.update()
        ow = max(self.over.width(), 1)
        ostate = (int(t_vis / max(self.over.duration, 1e-9) * ow), d.cue_point, d.loop, tuple(d.hotcues),
                  d.mix_in, d.mix_out, d.mix_out_end, ow, self.over.height())
        if ostate != getattr(self, "_over_state", None):
            self._over_state = ostate
            self.over.update()
        if not self.platter.held:
            ang = (t_vis / 1.8) * 360.0 % 360.0
            if abs(ang - self.platter.angle) > 0.4:
                self.platter.angle = ang
                self.platter.update()
        err = self.eng.phase_error(self.name) if d.playing else None
        if err is None or self.phase.err is None or abs(err - self.phase.err) > 0.002:
            if err != self.phase.err:
                self.phase.err = err
                self.phase.update()
        for b, on in ((self.b_play, d.playing), (self.b_sync, d.sync), (self.b_key, d.keylock),
                      (self.b_master, self.eng.master_deck == self.name), (self.b_flux, d.flux),
                      (self.b_loop, d.loop is not None)):
            _sync_check(b, on)
        _set_text(self.b_play, "❚❚" if d.playing else "▶")
        for i, b in enumerate(self.hc):
            st = (f"background: {CUE_COLORS[i].name()}; color: #111; font-weight: 800;"
                  if d.hotcues[i] is not None else "")
            if b.styleSheet() != st:
                b.setStyleSheet(st)
        # the fader shows the offset the deck really plays at (following the lead, or its own)
        if d.rate_override is not None and d.tempo_range > 0:
            shown = float(np.clip((d.rate_override - 1.0) / d.tempo_range, -1.0, 1.0))
        else:
            shown = d.tempo
        _sync_value(self.tempo, int(shown * 1000))
        k = (0.06, 0.08, 0.16, 0.5)
        want = min(range(4), key=lambda i: abs(k[i] - d.tempo_range))
        if self.range.currentIndex() != want:
            self.range.blockSignals(True)
            self.range.setCurrentIndex(want)
            self.range.blockSignals(False)
        _sync_value(self.key_shift, int(round(d.key_shift)), 0)
        # loop sizes: the chosen size lights up while its loop plays
        lb = None
        if d.loop is not None and d.grid is not None:
            lb = round((d.loop[1] - d.loop[0]) / SR / d.grid.beat_len)
        for n_, b in self.loop_btns.items():
            _sync_check(b, d.loop is not None and lb == n_)
        if d.grid is not None:
            txt = f"{d.grid.bpm:.3f}"
            if self.grid_bpm.text() != txt:
                self.grid_bpm.setText(txt)


# --------------------------------------------------------------------------
# mixer (centre)
# --------------------------------------------------------------------------

class ChannelStrip(QWidget):
    def __init__(self, name: str, dj: DJPanel):
        super().__init__()
        self.name = name
        self.dj = dj
        self.eng = dj.engine
        lay = QVBoxLayout(self)
        lay.setContentsMargins(2, 0, 2, 0)
        lay.setSpacing(1)
        head = QLabel(name)
        head.setAlignment(Qt.AlignCenter)
        head.setStyleSheet(f"color: {DECK_COLORS[name].name()}; font-weight: bold; font-size: 14px;")
        lay.addWidget(head)
        self.knobs = {}
        self.labels = {}
        for key, lab, tip in (("trim", "GAIN", "Channel gain"), ("high", "HI", "EQ high (left = kill)"),
                              ("mid", "MID", "EQ mid (left = kill)"), ("low", "LOW", "EQ low (left = kill)"),
                              ("filter", "FILTER", "Left = low-pass, right = high-pass")):
            k = Knob(50, 24, tip)
            k.valueChanged.connect(self._knobs_changed)
            self.knobs[key] = k
            lbl = QLabel(lab)
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet("font-size: 8px; color: #8d8f97; font-weight: 800;")
            self.labels[key] = lbl
            lay.addWidget(lbl)
            lay.addWidget(k, 0, Qt.AlignHCenter)
        fxrow = QHBoxLayout()
        fxrow.setSpacing(1)
        self.fx1 = _btn("FX1", True, "Route this channel through FX unit 1", w=30, h=20)
        self.fx2 = _btn("FX2", True, "Route this channel through FX unit 2", w=30, h=20)
        for b in (self.fx1, self.fx2):
            b.toggled.connect(self._fx_changed)
            fxrow.addWidget(b)
        lay.addLayout(fxrow)
        self.cue = _btn("🎧", True, "Headphone cue (pre-listen)", h=24)
        self.cue.toggled.connect(lambda on: self.eng.set(name, cue_phones=on))
        lay.addWidget(self.cue)
        frow = QHBoxLayout()
        self.meter = LevelMeter(8)
        self.fader = ResetSlider(Qt.Vertical, 1000, f"Channel {name} volume")
        self.fader.setRange(0, 1000)
        self.fader.setValue(1000)
        self.fader.setMinimumHeight(72)
        self.fader.valueChanged.connect(self._fader)
        frow.addWidget(self.meter)
        frow.addWidget(self.fader)
        lay.addLayout(frow, 1)

    def set_level(self, level: int):
        """0: everything · 1: no GAIN / FILTER · 2: fader only (small screens)."""
        for key in self.knobs:
            show = level == 0 or (level == 1 and key in ("high", "mid", "low"))
            self.knobs[key].setVisible(show)
            self.labels[key].setVisible(show)
        self.fader.setMinimumHeight(72 if level < 2 else 36)
        self.meter.setMinimumHeight(60 if level < 2 else 30)

    def _knobs_changed(self, *_):
        self.dj.manual_touch(self.name)
        k = self.knobs
        self.eng.set(self.name, eq=(knob_eq(k["low"].value()), knob_eq(k["mid"].value()), knob_eq(k["high"].value())),
                     trim=float(10 ** ((k["trim"].value() / 100.0 - 0.5) * 24 / 20)),
                     filter=float((k["filter"].value() - 50) / 50.0))

    def _fx_changed(self, *_):
        s = set()
        if self.fx1.isChecked():
            s.add(0)
        if self.fx2.isChecked():
            s.add(1)
        self.eng.set(self.name, fx=s)

    def _fader(self, v):
        self.dj.manual_touch(self.name)
        self.eng.set(self.name, volume=v / 1000.0)

    def refresh(self):
        d = self.eng.decks[self.name]
        if abs(self.meter.value - d.level) > 0.004:
            self.meter.value = d.level
            self.meter.update()
        _sync_value(self.fader, int(d.volume * 1000))
        _sync_check(self.fx1, 0 in d.fx)
        _sync_check(self.fx2, 1 in d.fx)
        _sync_check(self.cue, d.cue_phones)
        # reflect moves made by the controller or by auto-mix (bass swap)
        for key, g in (("low", d.eq[0]), ("mid", d.eq[1]), ("high", d.eq[2])):
            _sync_value(self.knobs[key], eq_knob(g))
        _sync_value(self.knobs["filter"], int(50 + d.filter * 50))
        # gain knob follows the deck (it is reset when a new track loads)
        trim = float(getattr(d, "trim", 1.0))
        _sync_value(self.knobs["trim"], int(round(50 + 100 * np.log10(max(trim, 1e-3)) * 20 / 24)))


class MixerPanel(QFrame):
    """The channel strips between the decks. Its master level, recorder,
    crossfader and headphone knobs are separate widgets the DJ panel places
    in the global row and the crossfader bar."""

    def __init__(self, dj: DJPanel):
        super().__init__()
        self.setObjectName("mixer")
        self.setStyleSheet("#mixer { background: #1c1c1e; border: 1px solid #0b0b0b; border-radius: 3px; }")
        self.dj = dj
        eng = dj.engine
        lay = QVBoxLayout(self)
        lay.setContentsMargins(3, 3, 3, 3)
        lay.setSizeConstraint(QLayout.SetNoConstraint)
        strips = QHBoxLayout()
        strips.setSpacing(2)
        self.strips = {n: ChannelStrip(n, dj) for n in "CABD"}
        for n in "CABD":
            strips.addWidget(self.strips[n])
        lay.addLayout(strips, 1)
        self.level = 0
        # ---- master level (global row) ----
        self.w_master = QWidget()
        g = QHBoxLayout(self.w_master)
        g.setContentsMargins(0, 0, 0, 0)
        g.setSpacing(4)
        self.master = Knob(90, 30, "Main (master) level")
        self.master.valueChanged.connect(lambda v: setattr(eng, "master_volume", v / 100))
        self.master_meter = LevelMeter(8)
        self.master_meter.setMinimumHeight(30)
        mcol = QVBoxLayout()
        mcol.setSpacing(0)
        mcol.addWidget(self.master, 0, Qt.AlignHCenter)
        ml = QLabel("MAIN")
        ml.setStyleSheet("font-size: 9px; color: #8d8f97; font-weight: 800;")
        mcol.addWidget(ml, 0, Qt.AlignHCenter)
        self.limiter = QLabel("LIMITER")
        self.limiter.setToolTip("Lights up while the limiter is holding the master below clipping")
        self.limiter.setStyleSheet("font-size: 9px; font-weight: 800; color: #4a4c52;")
        mcol.addWidget(self.limiter, 0, Qt.AlignHCenter)
        g.addLayout(mcol)
        g.addWidget(self.master_meter)
        self.scope = Oscilloscope()
        g.addWidget(self.scope)
        # ---- crossfader bar (under the decks) ----
        self.xf_bar = QWidget()
        # three equal-stretch columns: the crossfader sits dead centre, between the decks
        grid_xf = QGridLayout(self.xf_bar)
        grid_xf.setContentsMargins(4, 0, 4, 0)
        grid_xf.setColumnStretch(0, 1)
        grid_xf.setColumnStretch(2, 1)
        xb = QHBoxLayout()
        xb.setSpacing(6)
        la = QLabel("A")
        la.setStyleSheet(f"color: {DECK_COLORS['A'].name()}; font-weight: 900;")
        xb.addWidget(la)
        self.xf = ResetSlider(Qt.Horizontal, 0, "Crossfader")
        self.xf.setRange(-100, 100)
        self.xf.setFixedWidth(220)
        self.xf.valueChanged.connect(self._xf)
        xb.addWidget(self.xf)
        lb = QLabel("B")
        lb.setStyleSheet(f"color: {DECK_COLORS['B'].name()}; font-weight: 900;")
        xb.addWidget(lb)
        grid_xf.addLayout(xb, 0, 1)
        hp = QHBoxLayout()
        hp.setSpacing(6)
        hp.addStretch(1)
        for attr, default, tip, txt in (("cue_mix", 0, "Headphones: cue ↔ master", "MIX"),
                                        ("phones", 80, "Headphone volume", "VOL")):
            k = Knob(default, 24, tip)
            setattr(self, attr, k)
            t = QLabel(txt)
            t.setStyleSheet("font-size: 9px; color: #8d8f97; font-weight: 800;")
            hp.addWidget(t)
            hp.addWidget(k)
        grid_xf.addLayout(hp, 0, 2)
        self.cue_mix.valueChanged.connect(lambda v: setattr(eng, "cue_mix", v / 100))
        self.phones.valueChanged.connect(lambda v: setattr(eng, "phones_volume", v / 100))
        # ---- recorder (global row) ----
        self.rec = _btn("● REC", True, "Record the master output to a WAV file", h=24)
        self.rec.setStyleSheet("QPushButton:checked { background: #c62828; color: white; }")
        self.rec.clicked.connect(lambda _on: dj.toggle_record())
        self.rec_lbl = QLabel("Rec off")
        self.rec_lbl.setAlignment(Qt.AlignCenter)
        self.rec_lbl.setStyleSheet("color: #8d8f97;")
        self.rec_lbl.setMinimumWidth(64)
        from PySide6.QtWidgets import QSizePolicy

        self.rec_lbl.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)

    def _set_level(self, lvl: int):
        self.level = lvl
        for st in self.strips.values():
            st.set_level(lvl)

    def _needs(self) -> list[int]:
        if getattr(self, "_measuring", False):          # re-entered by Qt while measuring
            return getattr(self, "_need", None) or [0, 0, 0]
        if getattr(self, "_need", None) is None:
            cur, need = self.level, []
            self._measuring = True
            try:
                for lvl in (0, 1, 2):
                    self._set_level(lvl)
                    need.append(_relayout(self))
                self._set_level(cur)
            finally:
                self._measuring = False
            self._need = need
        return self._need

    def minimumSizeHint(self):
        from PySide6.QtCore import QSize

        return QSize(super().minimumSizeHint().width(), self._needs()[2])

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        if getattr(self, "_measuring", False):
            return
        need, h = self._needs(), self.height()      # shed knobs as the decks get shorter
        lvl = next((i for i in (0, 1, 2) if h >= need[i]), 2)
        if lvl != self.level:
            self._set_level(lvl)

    def _xf(self, v):
        self.dj.automix.manual(None, "crossfader")
        self.dj.engine.crossfader = v / 100.0

    def refresh(self, show_cd: bool):
        eng = self.dj.engine
        for n, s in self.strips.items():
            s.setVisible(show_cd or n in "AB")
            if s.isVisible():
                s.refresh()
        cur = eng.scope
        if cur is not None and cur is not self.scope.data_id:
            prev = eng.scope_prev
            self.scope.data_id = cur
            self.scope.data = np.concatenate([prev, cur], axis=1) if prev is not None and prev.shape[0] == 2 else cur
            self.scope.update()
        if abs(self.master_meter.value - eng.peak_level) > 0.004:
            self.master_meter.value = eng.peak_level
            self.master_meter.update()
        _sync_value(self.xf, int(eng.crossfader * 100), 1)
        r = eng.recorder
        _sync_check(self.rec, r is not None)
        _set_text(self.rec_lbl, f"REC {fmt_time(r.seconds)}" if r is not None else "Rec off")
        lim = eng._lim_gain < 0.98
        if lim != getattr(self, "_lim_on", None):
            self._lim_on = lim
            _set_ss(self.limiter, "font-size: 9px; font-weight: 800; color: "
                    + ("#ff9a3c;" if lim else "#4a4c52;"))


class FXPanel(QGroupBox):
    def __init__(self, dj: DJPanel, unit: int):
        super().__init__(f"FX {unit + 1}")
        self.u = dj.engine.fx_units[unit]
        lay = QHBoxLayout(self)
        lay.setContentsMargins(4, 8, 4, 2)
        lay.setSpacing(3)
        self.on = _btn("ON", True, w=34, h=22)
        self.on.toggled.connect(lambda v: setattr(self.u, "on", v))
        lay.addWidget(self.on)
        self.kind = QComboBox()
        self.kind.addItems(FX_TYPES)
        self.kind.setCurrentText(self.u.kind)
        self.kind.currentTextChanged.connect(lambda t: setattr(self.u, "kind", t))
        # narrow combos: the global row must fit a 1280-pixel screen with Windows fonts
        self.kind.setFixedWidth(78)
        lay.addWidget(self.kind)

        def knob(k, cap):
            col = QVBoxLayout()
            col.setSpacing(0)
            col.addWidget(k, 0, Qt.AlignHCenter)
            lb = QLabel(cap)
            lb.setStyleSheet("font-size: 8px; color: #8d8f97; font-weight: 800;")
            col.addWidget(lb, 0, Qt.AlignHCenter)
            lay.addLayout(col)
        self.wet = Knob(50, 24, "Dry / wet")
        self.wet.valueChanged.connect(lambda v: setattr(self.u, "wet", v / 100))
        knob(self.wet, "D/W")
        self.amt = Knob(50, 24, "Effect amount (feedback / size / depth)")
        self.amt.valueChanged.connect(lambda v: setattr(self.u, "amount", v / 100))
        knob(self.amt, "AMT")
        self.div = QComboBox()
        self.div.setFixedWidth(50)
        self.div.addItems(["1/8", "1/4", "1/2", "3/4", "1", "2"])
        self.div.setCurrentIndex(self.u.div_index)
        self.div.setToolTip("Beat-synced timing")
        self.div.currentIndexChanged.connect(lambda i: setattr(self.u, "div_index", i))
        lay.addWidget(self.div)
        assert len(BEAT_DIVS) == self.div.count()

    def refresh(self):
        _sync_check(self.on, self.u.on)
        _sync_value(self.wet, int(self.u.wet * 100), 1)
        _sync_value(self.amt, int(self.u.amount * 100), 1)
        if self.kind.currentText() != self.u.kind:
            self.kind.blockSignals(True)
            self.kind.setCurrentText(self.u.kind)
            self.kind.blockSignals(False)


# --------------------------------------------------------------------------
# dialogs
# --------------------------------------------------------------------------

def _pairs(ch: int) -> list[str]:
    return [f"{k + 1}-{k + 2}" for k in range(0, max(ch, 2) - 1, 2)]


def _pair_tuple(txt: str):
    if not txt or txt.startswith("off"):
        return None
    a, b = txt.split("-")
    return (int(a) - 1, int(b) - 1)


class AudioSettings(QDialog):
    """Outputs: every connected device (scanned live by a helper process, the
    music keeps playing), master and headphones on the same or different
    cards, extra master outputs (more speakers, kept in sync) and each
    output's delay (driver, Bluetooth guess or measured with the tap test)."""

    _scanned = Signal(object)

    def __init__(self, settings: QSettings, parent=None, dj=None):
        super().__init__(parent)
        self.setWindowTitle("Audio output")
        self.s = settings
        self.dj = dj
        self.restart_needed = False      # main sound card / buffer changed
        self.reinit_needed = False       # ... to a device this process has not seen yet
        self.records: list[dict] = []
        self._scanned.connect(self._fill)
        import json as _json

        try:
            self.extras = list(_json.loads(settings.value("dj_extra_outputs", "[]") or "[]"))
        except ValueError:
            self.extras = []
        self._before = self._snapshot()
        lay = QVBoxLayout(self)
        form = QFormLayout()
        lay.addLayout(form)
        row = QHBoxLayout()
        self.dev = QComboBox()
        self.dev.setMinimumWidth(320)
        row.addWidget(self.dev, 1)
        b_rescan = _btn("↻ Rescan", tip="Look for devices plugged in or paired since Tempest started "
                        "(the music keeps playing)")
        b_rescan.clicked.connect(lambda: self.rescan(force=True))
        row.addWidget(b_rescan)
        form.addRow("Master device", row)
        self.master = QComboBox()
        form.addRow("Master outputs", self.master)
        self.pdev = QComboBox()
        form.addRow("Headphones device", self.pdev)
        self.phones = QComboBox()
        form.addRow("Headphone outputs", self.phones)
        self.block = QComboBox()
        self.block.addItems(["256", "512", "1024", "2048"])
        self.block.setCurrentText(str(settings.value("dj_block", "512")))
        form.addRow("Buffer (samples)", self.block)
        self.safety = QComboBox()
        for label, key in (("Auto - on the edge of safe (starts tight, backs off only at glitches)", "auto"),
                           ("Normal - tight and stable (25 ms cushion)", "normal"),
                           ("Safe - most headroom (50 ms: a slow or busy computer)", "safe"),
                           ("Tight - least delay, no safety net", "tight")):
            self.safety.addItem(label, key)
        self.safety.setCurrentIndex(max(self.safety.findData(settings.value("dj_safety", "auto")), 0))
        self.safety.setToolTip("How much audio the sound card holds in reserve. More = the music never "
                               "glitches when the computer is busy or Tempest is in the background; less = "
                               "buttons answer a few milliseconds sooner.")
        form.addRow("Output safety", self.safety)
        self.own_proc = QCheckBox("Audio engine in its own process (recommended)")
        self.own_proc.setToolTip("The decks, mixing and auto-mix run in a separate process, so the screen, "
                                 "library scans or a stuck window can never make the music glitch. "
                                 "Takes effect after a restart.")
        self.own_proc.setChecked(settings.value("dj_audio_process", "true") == "true")
        form.addRow("", self.own_proc)
        self.mdelay, self.m_auto, self.m_info = self._delay_row(form, "Master delay", "master")
        self.pdelay, self.p_auto, self.p_info = self._delay_row(form, "Headphones delay", "phones")
        # more speakers on the master (Bluetooth...), each in its own helper: they can come
        # and go while the music plays, and are delayed so every speaker plays in time
        self.extra_list = QListWidget()
        self.extra_list.setMaximumHeight(90)
        self.extra_list.currentRowChanged.connect(self._extra_selected)
        erow = QHBoxLayout()
        self.extra_add = QComboBox()
        self.extra_add.setMinimumWidth(240)
        b_add = _btn("＋ Add", tip="Play the master on this device too, in sync")
        b_add.clicked.connect(self._extra_add)
        b_rm = _btn("Remove")
        b_rm.clicked.connect(self._extra_remove)
        for w_ in (self.extra_add, b_add, b_rm):
            erow.addWidget(w_)
        drow = QHBoxLayout()
        self.edelay = QSpinBox()
        self.edelay.setRange(0, 1000)
        self.edelay.setSuffix(" ms")
        self.e_auto = QCheckBox("Auto")
        self.e_auto.setToolTip("Driver latency, about 200 ms for Bluetooth")
        b_em = _btn("Measure…", tip="Tap along with clicks on this speaker")
        b_em.clicked.connect(lambda: self._measure("extra"))
        self.edelay.valueChanged.connect(self._extra_delay)
        self.e_auto.toggled.connect(self._extra_delay)
        for w_ in (QLabel("Delay"), self.edelay, self.e_auto, b_em):
            drow.addWidget(w_)
        drow.addStretch()
        ebox = QVBoxLayout()
        ebox.addWidget(self.extra_list)
        ebox.addLayout(erow)
        ebox.addLayout(drow)
        form.addRow("More master outputs", ebox)
        self.scan_lbl = QLabel("")
        self.scan_lbl.setStyleSheet("color: #9a9ba3;")
        form.addRow("", self.scan_lbl)
        # ---- sync: which output keeps time, which follow it, fine adjustment ----
        self.clock = QComboBox()
        self.clock.setToolTip("The output everything else lines up with. Auto: the slowest one (usually a "
                              "Bluetooth speaker), so all of them play the same beat together.")
        form.addRow("Clock (keeps time)", self.clock)
        self.sync_list = QListWidget()
        self.sync_list.setMaximumHeight(84)
        self.sync_list.setToolTip("Ticked outputs wait for the clock so they play in time with it. "
                                  "Untick one to hear it without any added delay (e.g. your headphones).")
        self.sync_list.itemChanged.connect(lambda _it: self._clock_live())
        form.addRow("Sync to the clock", self.sync_list)
        arow = QHBoxLayout()
        self.adjust = QSpinBox()
        self.adjust.setRange(-1000, 1000)
        self.adjust.setSingleStep(5)
        self.adjust.setSuffix(" ms")
        self.adjust.setToolTip("Fine-tune by ear while the music plays: + holds the synced outputs back "
                               "(they were early), − brings them forward (they were late)")
        self.adjust.setValue(int(float(settings.value("dj_clock_adjust_ms", 0) or 0)))
        self.adjust.valueChanged.connect(lambda _v: self._clock_live())
        arow.addWidget(self.adjust)
        for txt, d in (("−10", -10), ("−1", -1), ("+1", 1), ("+10", 10)):
            b = _btn(txt, tip=f"{d:+d} ms", h=24)
            b.clicked.connect(lambda _=False, d=d: self.adjust.setValue(self.adjust.value() + d))
            arow.addWidget(b)
        self.clock_info = QLabel("")
        self.clock_info.setStyleSheet("color: #9a9ba3;")
        arow.addWidget(self.clock_info, 1)
        form.addRow("Clock adjust", arow)
        self._clock_before = (self.s.value("dj_clock", "auto"), self.s.value("dj_sync_outputs", ""),
                              self.s.value("dj_clock_adjust_ms", 0))
        self._fill_clock()
        self.clock.currentIndexChanged.connect(lambda _i: self._clock_live())
        self.rec_dir = QPushButton(settings.value("rec_dir", os.path.join(os.path.expanduser("~"), "Music",
                                                                          "Tempest Recordings")))
        self.rec_dir.clicked.connect(self._pick_dir)
        form.addRow("Recordings folder", self.rec_dir)
        self.port = QSpinBox()
        self.port.setRange(1024, 65535)
        self.port.setValue(int(settings.value("stream_port", 8000)))
        form.addRow("Phone stream port", self.port)
        note = QLabel("<small>Bluetooth plays audio 150-300 ms late. Set its delay (or <b>Measure</b> it by tapping "
                      "along) and Tempest keeps master and headphones in time and draws the waveforms that late, "
                      "so what you see is what you hear.<br>DDJ-T1: choose its sound card, master 1-2, headphones "
                      "3-4. Use a bigger buffer if you hear crackles.</small>")
        note.setWordWrap(True)
        lay.addWidget(note)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        self.dev.currentIndexChanged.connect(self._dev_changed)
        self.pdev.currentIndexChanged.connect(self._dev_changed)
        self.rescan(force=False)

    # ---- clock ------------------------------------------------------------------------------
    def _outputs(self) -> list[tuple[str, str]]:
        """(id, name) of every output in use."""
        outs = [("master", f"Main output ({self.dev.currentText() or 'default'})")]
        if self.pdev.currentData() or not self.phones.currentText().lower().startswith("off"):
            outs.append(("phones", "Headphones"))
        for e in self.extras:
            outs.append((f"extra:{e['label']}", e["label"]))
        return outs

    def _fill_clock(self):
        import json as _json

        cur = self.clock.currentData() or self.s.value("dj_clock", "auto")
        try:
            raw = self.s.value("dj_sync_outputs", "") or ""
            synced = None if raw in ("", None) else set(_json.loads(raw))
        except ValueError:
            synced = None
        prev: set = set()
        if self.sync_list.count():               # keep what was just ticked in this dialog
            prev = {self.sync_list.item(i).data(Qt.UserRole) for i in range(self.sync_list.count())}
            synced = {self.sync_list.item(i).data(Qt.UserRole) for i in range(self.sync_list.count())
                      if self.sync_list.item(i).checkState() == Qt.Checked}
        self.clock.blockSignals(True)
        self.sync_list.blockSignals(True)
        self.clock.clear()
        self.clock.addItem("Auto - the slowest output", "auto")
        self.sync_list.clear()
        from PySide6.QtWidgets import QListWidgetItem

        for oid, name in self._outputs():
            self.clock.addItem(name, oid)
            it = QListWidgetItem(name)
            it.setData(Qt.UserRole, oid)
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            new = bool(prev) and oid not in prev            # an output added just now follows the clock
            it.setCheckState(Qt.Checked if synced is None or oid in synced or new else Qt.Unchecked)
            self.sync_list.addItem(it)
        i = self.clock.findData(cur)
        self.clock.setCurrentIndex(max(i, 0))
        self.clock.blockSignals(False)
        self.sync_list.blockSignals(False)

    def _clock_values(self):
        oids = [self.sync_list.item(i).data(Qt.UserRole) for i in range(self.sync_list.count())]
        synced = [o for i, o in enumerate(oids) if self.sync_list.item(i).checkState() == Qt.Checked]
        return self.clock.currentData() or "auto", (None if len(synced) == len(oids) else synced), \
            self.adjust.value()

    def _clock_live(self):
        """Clock changes are heard at once (the music keeps playing)."""
        import json as _json

        clock, synced, adj = self._clock_values()
        self.s.setValue("dj_clock", clock)
        self.s.setValue("dj_sync_outputs", "" if synced is None else _json.dumps(synced))
        self.s.setValue("dj_clock_adjust_ms", adj)
        if self.dj is not None:
            self.dj.apply_clock()
            d = self.dj.engine.output_delay_s
            names = dict(self._outputs())
            self.clock_info.setText("  ".join(f"{names.get(k, k).split(' (')[0]}: +{v * 1000:.0f} ms"
                                              for k, v in d.items()))

    def reject(self):
        # put the clock back as it was
        c, so, adj = self._clock_before
        self.s.setValue("dj_clock", c)
        self.s.setValue("dj_sync_outputs", so)
        self.s.setValue("dj_clock_adjust_ms", adj)
        if self.dj is not None:
            self.dj.apply_clock()
        super().reject()

    def _delays(self) -> dict:
        """Measured / set output delays, per device label (a delay measured on
        Bluetooth must never be applied to the wired card you play on later)."""
        import json as _json

        try:
            d = _json.loads(self.s.value("dj_output_delays", "") or "{}")
            return d if isinstance(d, dict) else {}
        except ValueError:
            return {}

    def _load_delays(self):
        """Show each output's stored delay for the device chosen right now."""
        from ..dj.outputs import BLUETOOTH_GUESS_MS, is_bluetooth

        delays = self._delays()
        for combo, sp, auto, info in ((self.dev, self.mdelay, self.m_auto, self.m_info),
                                      (self.pdev if self.pdev.currentData() else self.dev, self.pdelay,
                                       self.p_auto, self.p_info)):
            label = combo.currentText() if combo.currentData() is not None else ""
            v = delays.get(label)
            bt = is_bluetooth(label)
            if v is None and bt:
                v = BLUETOOTH_GUESS_MS
            auto.setChecked(v is None)
            sp.setValue(int(round(v)) if v is not None else 0)
            info.setText("Bluetooth: about 200 ms late - measure it" if bt else
                         "measured on this device" if v is not None else "driver's figure")

    def _delay_row(self, form, label, which):
        row = QHBoxLayout()
        sp = QSpinBox()
        sp.setRange(0, 1000)
        sp.setSuffix(" ms")
        auto = QCheckBox("Auto")
        auto.setToolTip("Use what the driver reports")
        info = QLabel("")
        info.setStyleSheet("color: #9a9ba3;")
        b = _btn("Measure…", tip="Tap along with clicks on this output: Tempest works out how late it plays")
        b.clicked.connect(lambda: self._measure(which))
        auto.toggled.connect(lambda on: sp.setEnabled(not on))
        auto.setChecked(True)                   # _load_delays fills it in for the chosen device
        for w_ in (sp, auto, b, info):
            row.addWidget(w_)
        row.addStretch()
        form.addRow(label, row)
        return sp, auto, info

    # ---- devices ------------------------------------------------------------------------
    def rescan(self, force: bool = False):
        """Look for devices with a helper process: nothing that is playing stops."""
        if not self.records:
            try:                                      # what this process already knows, at once
                self._fill([{"label": lb, "name": lb.rsplit(" (", 1)[0], "channels": ch}
                            for _i, lb, ch in list_output_devices()], final=False)
            except Exception:  # noqa: BLE001
                self._fill([], final=False)
        self.scan_lbl.setText("Looking for devices…")

        def job():
            from ..dj import outputs

            self._scanned.emit(outputs.scan_devices())
        threading.Thread(target=job, daemon=True, name="device-scan").start()

    def _fill(self, records, final: bool = True):
        if final and not records and self.records:
            self.scan_lbl.setText("Could not look for new devices - showing the known ones")
            return
        self.records = list(records)
        if final:
            self.scan_lbl.setText(f"{len(self.records)} output devices connected")
        for combo, key, first in ((self.dev, "dj_device_name", "System default"),
                                  (self.pdev, "dj_phones_device_name", "Same as master")):
            cur = combo.currentText() if combo.count() else str(self.s.value(key, "") or "")
            combo.blockSignals(True)
            combo.clear()
            combo.addItem(first, None)
            for r in self.records:
                combo.addItem(r["label"], (r["label"], r["channels"]))
            ix = combo.findText(cur)
            combo.setCurrentIndex(ix if ix >= 0 else 0)
            combo.blockSignals(False)
        self.extra_add.clear()
        for r in self.records:
            self.extra_add.addItem(("🔵 " if r.get("bluetooth") else "") + r["label"], r["label"])
        self._extras_refresh()
        self._dev_changed()

    # ---- extra master outputs ------------------------------------------------------------------------
    def _extras_refresh(self):
        if hasattr(self, "sync_list"):
            self._fill_clock()
        row = self.extra_list.currentRow()
        self.extra_list.clear()
        present = {r["label"] for r in self.records}
        for e in self.extras:
            d = "auto" if e.get("delay_ms") is None else f"{e['delay_ms']} ms"
            here = "" if e["label"] in present or not self.records else "  (not connected - added when it is)"
            self.extra_list.addItem(f"{e['label']} · {d}{here}")
        if self.extras:
            self.extra_list.setCurrentRow(min(max(row, 0), len(self.extras) - 1))

    def _extra_add(self):
        lb = self.extra_add.currentData()
        if not lb or any(e["label"] == lb for e in self.extras):
            return
        from ..dj.outputs import is_bluetooth

        self.extras.append({"label": lb, "delay_ms": None if not is_bluetooth(lb) else None})
        self._extras_refresh()
        self.extra_list.setCurrentRow(len(self.extras) - 1)

    def _extra_remove(self):
        r = self.extra_list.currentRow()
        if 0 <= r < len(self.extras):
            self.extras.pop(r)
            self._extras_refresh()

    def _extra_selected(self, r):
        if not 0 <= r < len(self.extras):
            return
        e = self.extras[r]
        for w_ in (self.edelay, self.e_auto):
            w_.blockSignals(True)
        self.e_auto.setChecked(e.get("delay_ms") is None)
        self.edelay.setEnabled(e.get("delay_ms") is not None)
        from ..dj.outputs import BLUETOOTH_GUESS_MS, is_bluetooth

        self.edelay.setValue(int(e["delay_ms"]) if e.get("delay_ms") is not None
                             else int(BLUETOOTH_GUESS_MS if is_bluetooth(e["label"]) else 30))
        for w_ in (self.edelay, self.e_auto):
            w_.blockSignals(False)

    def _extra_delay(self, *_):
        r = self.extra_list.currentRow()
        if not 0 <= r < len(self.extras):
            return
        self.extras[r]["delay_ms"] = None if self.e_auto.isChecked() else self.edelay.value()
        self.edelay.setEnabled(not self.e_auto.isChecked())
        self.extra_list.item(r).setText(self.extra_list.item(r).text().split(" · ")[0] + " · "
                                        + ("auto" if self.e_auto.isChecked() else f"{self.edelay.value()} ms"))

    def _channels(self, combo) -> int:
        d = combo.currentData()
        return d[1] if d else 2

    def _dev_changed(self, *_):
        if hasattr(self, "sync_list"):
            self._fill_clock()

        for combo, chans, key, extra in ((self.master, self._channels(self.dev), "dj_master", []),
                                         (self.phones, self._channels(self.pdev if self.pdev.currentData()
                                                                      else self.dev), "dj_phones",
                                          ["off (master only)"])):
            cur = combo.currentText() or str(self.s.value(key, ""))
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(extra + _pairs(chans))
            ix = combo.findText(cur)
            combo.setCurrentIndex(ix if ix >= 0 else (0 if not extra or chans < 4 else combo.findText("3-4")))
            combo.blockSignals(False)
        self._load_delays()

    # ---- measuring ---------------------------------------------------------------------------
    def _measure(self, which: str):
        if self.dj is None:
            return
        self.save()                                   # measure on the outputs chosen here
        self.dj.apply_audio_settings(self)
        self._before = self._snapshot()
        self.restart_needed = self.reinit_needed = False
        target = which
        if which == "extra":
            r = self.extra_list.currentRow()
            if not 0 <= r < len(self.extras):
                QMessageBox.information(self, "Measure", "Add and select a speaker first.")
                return
            label = self.extras[r]["label"]
            if not any(o.label == label and o.alive for o in self.dj.engine.extra_outputs):
                QMessageBox.information(self, "Measure", f"{label} is not playing yet - is it connected?")
                return
            target = "extra:" + label
        if which == "phones" and self.dj.engine.phones_chans is None:
            QMessageBox.information(self, "Measure", "Choose headphone outputs first.")
            return
        dlg = TapCalibration(self.dj.engine, target, self)
        if dlg.exec() and dlg.result_ms is not None:
            if which == "extra":
                self.e_auto.setChecked(False)
                self.edelay.setValue(int(round(dlg.result_ms)))
                return
            sp, auto = (self.mdelay, self.m_auto) if which == "master" else (self.pdelay, self.p_auto)
            auto.setChecked(False)
            sp.setValue(int(round(dlg.result_ms)))

    def _pick_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Recordings folder", self.rec_dir.text())
        if d:
            self.rec_dir.setText(d)

    def _snapshot(self):
        g = lambda k, d="": str(self.s.value(k, d) or "")  # noqa: E731
        return (g("dj_device_name"), g("dj_phones_device_name"), g("dj_master", "1-2"), g("dj_phones", "3-4"),
                g("dj_block", "512"), g("dj_safety", "safe"))

    def save(self):
        import json as _json

        self.s.setValue("dj_extra_outputs", _json.dumps(self.extras))
        self.s.setValue("dj_device_name", self.dev.currentText() if self.dev.currentData() is not None else "")
        self.s.setValue("dj_phones_device_name", self.pdev.currentText() if self.pdev.currentData() else "")
        self.s.setValue("dj_master", self.master.currentText())
        self.s.setValue("dj_phones", self.phones.currentText())
        self.s.setValue("dj_block", self.block.currentText())
        self.s.setValue("dj_safety", self.safety.currentData())
        self.s.setValue("dj_audio_process", "true" if self.own_proc.isChecked() else "false")
        delays = self._delays()
        for combo, sp, auto in ((self.dev, self.mdelay, self.m_auto),
                                (self.pdev if self.pdev.currentData() else self.dev, self.pdelay, self.p_auto)):
            label = combo.currentText() if combo.currentData() is not None else ""
            if auto.isChecked():
                delays.pop(label, None)
            else:
                delays[label] = int(sp.value())
        self.s.setValue("dj_output_delays", _json.dumps(delays))
        self.s.remove("dj_master_delay_ms")      # old single values: a delay now belongs to its device
        self.s.remove("dj_phones_delay_ms")
        self.s.setValue("rec_dir", self.rec_dir.text())
        self.s.setValue("stream_port", self.port.value())

    def accept(self):
        self.save()
        # only the main sound card, its channels or the buffer need an audio restart;
        # delays and extra speakers are changed live
        self.restart_needed = self._snapshot() != self._before
        if self.restart_needed:
            try:
                known = {lb for _i, lb, _c in list_output_devices()}
            except Exception:  # noqa: BLE001
                known = set()
            chosen = [x for x in (self.dev.currentData(), self.pdev.currentData()) if x]
            self.reinit_needed = any(c[0] not in known for c in chosen)
        super().accept()


class TapCalibration(QDialog):
    """Tap along with clicks on one output; the gap between the clicks being
    made and your taps is how late that output plays."""

    def __init__(self, eng, which: str, parent=None):
        super().__init__(parent)
        import time as _t

        self.eng = eng
        self.which = which
        self.result_ms: float | None = None
        self.taps: list[float] = []
        self._t = _t
        self.setWindowTitle("Measure the output delay")
        lay = QVBoxLayout(self)
        where = {"master": "master", "phones": "headphone"}.get(which, which[6:])
        lay.addWidget(QLabel(f"Listen on the <b>{where}</b> output and "
                             "tap (button or space bar) <b>exactly when you hear each click</b>.<br>"
                             "About 20 taps. Keep steady - it is the average that counts."))
        self.b = QPushButton("TAP")
        self.b.setMinimumHeight(90)
        self.b.setStyleSheet("font-size: 28px; font-weight: 800;")
        self.b.pressed.connect(self.tap)
        lay.addWidget(self.b)
        self.lbl = QLabel("0 taps")
        lay.addWidget(self.lbl)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.ok = bb.button(QDialogButtonBox.Ok)
        self.ok.setEnabled(False)
        self.ok.setText("Use it")
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        # measure the raw output: no alignment delay while clicking
        eng.calibrate(which)              # clicks on that output, alignment delays off meanwhile

    def keyPressEvent(self, ev):
        if ev.key() == Qt.Key_Space and not ev.isAutoRepeat():
            self.tap()
            return
        super().keyPressEvent(ev)

    def tap(self):
        from ..dj.outputs import estimate_latency

        self.taps.append(self._t.monotonic())
        est = estimate_latency(self.taps, list(self.eng.click_times))
        if est is None:
            self.lbl.setText(f"{len(self.taps)} taps" + (" - keep going, steady" if len(self.taps) >= 8 else ""))
            return
        self.result_ms = est * 1000
        self.lbl.setText(f"{len(self.taps)} taps · this output plays <b>{self.result_ms:.0f} ms</b> late")
        self.lbl.setTextFormat(Qt.RichText)
        self.ok.setEnabled(len(self.taps) >= 12)

    def done(self, r):
        self.eng.calibrate(None)
        super().done(r)


class LearnWizard(QDialog):
    """Controller setup, one control at a time. Move or press the control for
    the step shown; Tempest shows what it heard; press SPACE to keep it and go on
    (Backspace: listen again, S or Space with nothing heard: keep what is set).
    Nothing changes until you confirm a step, so a fader that keeps sending
    while you move it can never run through the list."""

    got = Signal(object)

    def __init__(self, controller: Controller, decks=("A", "B"), parent=None):
        super().__init__(parent)
        self.setWindowTitle("Controller setup")
        self.ctl = controller
        self.decks = decks
        self.result = {(b.action, b.deck): b for b in controller.bindings}
        self.learner = Learner()
        self.candidate: Binding | None = None
        self._ignore: tuple[tuple, float] | None = None     # (control, until) just confirmed
        self._recent: dict[tuple, list] = {}
        lay = QVBoxLayout(self)
        self.adv = QCheckBox("Include hot cues, loops, beat jump, filter and FX controls")
        self.adv.setChecked(True)
        self.adv.toggled.connect(self._rebuild)
        self.adv.setFocusPolicy(Qt.NoFocus)
        lay.addWidget(self.adv)
        self.prog = QProgressBar()
        lay.addWidget(self.prog)
        self.prompt = QLabel()
        self.prompt.setStyleSheet("font-size: 16px; padding: 12px;")
        self.prompt.setWordWrap(True)
        lay.addWidget(self.prompt)
        self.heard = QLabel("")
        self.heard.setStyleSheet("font-size: 14px; padding: 4px 12px;")
        lay.addWidget(self.heard)
        keys = QLabel("<b>SPACE</b> keep it and go on  ·  <b>Backspace</b> listen again  ·  "
                      "<b>S</b> skip (keep what is set)  ·  <b>←</b> back")
        keys.setStyleSheet("color: #8d8f97; padding: 0 12px;")
        lay.addWidget(keys)
        row = QHBoxLayout()
        for txt, fn in (("← Back", self._back), ("Listen again", self._retry), ("Skip", self._next),
                        ("Keep && next (Space)", self._confirm)):
            b = QPushButton(txt)
            b.clicked.connect(fn)
            b.setFocusPolicy(Qt.NoFocus)        # Space belongs to the wizard, not a focused button
            b.setAutoDefault(False)
            row.addWidget(b)
        row.addStretch()
        for txt, fn in (("Reset to built-in", self._reset_builtin), ("Import Traktor (.tsi)…", self._import_tsi),
                        ("Import…", self._import),
                        ("Export…", self._export), ("Finish && save", self._finish)):
            b = QPushButton(txt)
            b.clicked.connect(fn)
            b.setFocusPolicy(Qt.NoFocus)
            b.setAutoDefault(False)
            row.addWidget(b)
        lay.addLayout(row)
        self.got.connect(self._on_event)
        controller.learn_cb = lambda ev: self.got.emit(ev)
        self.resize(640, 260)
        self._rebuild()

    def _rebuild(self, *_):
        self.steps = learn_steps(self.decks, self.adv.isChecked())
        self.i = 0
        self.prog.setRange(0, len(self.steps))
        self._show()

    def _show(self):
        self.candidate = None
        self.learner = Learner()
        if self.i >= len(self.steps):
            self.prompt.setText("All done - click <b>Finish & save</b> (then <b>Export…</b> to keep a copy).")
            self.heard.setText("")
            self.prog.setValue(len(self.steps))
            return
        action, deck, text = self.steps[self.i]
        if action == "tempo":
            extra = "Move it slowly from the TOP to the BOTTOM."
        elif action in ("jog", "browse"):
            extra = "Turn it a little clockwise."
        elif (DECK_ACTIONS.get(action) or GLOBAL_ACTIONS.get(action))[0] == "button":
            extra = "Press it once."
        else:
            extra = "Move it a little."
        have = self.result.get((action, deck))
        cur = f"<br><small>now set to: {have.type} ch{have.ch + 1} #{have.num}</small>" if have else \
            "<br><small>not set yet</small>"
        self.prompt.setText(f"<b>{self.i + 1}/{len(self.steps)} · {text}</b><br><small>{extra}</small>{cur}")
        self.heard.setText("Waiting for the control… (Space to skip)")
        self.prog.setValue(self.i)

    @staticmethod
    def _id(b) -> tuple:
        return (b.type if b.type != "cc14" else "cc", b.ch, b.num)

    def _on_event(self, ev):
        import time as _t

        if self.i >= len(self.steps):
            return
        now = _t.monotonic()
        cid = (ev.type if ev.type != "cc14" else "cc", ev.ch, ev.num)
        if self._ignore is not None and cid == self._ignore[0] and now < self._ignore[1]:
            return                                  # the control you just confirmed, still moving
        rec = self._recent.setdefault(cid, [])
        rec.append((now, ev.value))
        del rec[:-12]
        if self.candidate is not None:
            if self._id(self.candidate) == cid:
                self.heard.setText(self._describe(self.candidate) + f"  = {ev.value}   - press SPACE to keep it")
            return                                  # one control per step until you confirm or retry
        action, deck, _ = self.steps[self.i]
        kind = (DECK_ACTIONS.get(action) or GLOBAL_ACTIONS.get(action))[0]
        if kind == "button" and ev.type != "note":
            # a fader sends a stream of different values; a button only full on / off
            vals = {v for t, v in rec if now - t < 0.6}
            if ev.value != 127 or len(vals) > 2:
                return
        b = self.learner.offer(action, ev)
        if b is None:
            self.heard.setText(f"heard {ev.type} ch{ev.ch + 1} #{ev.num} = {ev.value} …")
            return
        b.deck = deck
        self.candidate = b
        clash = [k for k, v in self.result.items() if self._id(v) == self._id(b) and k != (action, deck)]
        warn = ""
        if clash:
            a2, d2 = clash[0]
            warn = f"<br><span style='color:#ffb547'>already used for {a2}{' ' + d2 if d2 else ''} - Space uses " \
                   f"it for both, Backspace to pick another</span>"
        self.heard.setText(f"<b>Got: {self._describe(b)}</b> - press <b>SPACE</b> to keep it{warn}")

    @staticmethod
    def _describe(b) -> str:
        kind = {"cc14": "14-bit fader", "cc": "control", "note": "button"}.get(b.type, b.type)
        extra = (" (relative)" if b.relative else "") + (" (inverted)" if b.invert else "")
        return f"{kind} ch{b.ch + 1} #{b.num}{extra}"

    def keyPressEvent(self, ev):
        k = ev.key()
        if ev.isAutoRepeat():
            return
        if k == Qt.Key_Space:
            self._confirm()
        elif k in (Qt.Key_Backspace, Qt.Key_Delete):
            self._retry()
        elif k == Qt.Key_S:
            self._next()
        elif k == Qt.Key_Left:
            self._back()
        elif k == Qt.Key_Escape:
            super().keyPressEvent(ev)

    def _confirm(self):
        import time as _t

        if self.i >= len(self.steps):
            return
        if self.candidate is not None:
            action, deck, _ = self.steps[self.i]
            self.result[(action, deck)] = self.candidate
            self._ignore = (self._id(self.candidate), _t.monotonic() + 1.0)
        self._next()

    def _retry(self):
        self.candidate = None
        self.learner = Learner()
        self.heard.setText("Listening again… move or press the control.")

    def _next(self):
        self.i = min(self.i + 1, len(self.steps))
        self._show()

    def _back(self):
        self.i = max(0, self.i - 1)
        self._show()

    def _finish(self):
        self.ctl.set_bindings(list(self.result.values()), learned=True)
        self.accept()

    def _export(self):
        from ..dj.midi import export_mapping

        path, _ = QFileDialog.getSaveFileName(self, "Export controller mapping",
                                              f"{self.ctl.device or 'controller'}.json", "Mapping (*.json)")
        if path:
            export_mapping(path, self.ctl.device, list(self.result.values()))
            self.heard.setText(f"Exported to {path}")

    def _import(self):
        from ..dj.midi import import_mapping

        path, _ = QFileDialog.getOpenFileName(self, "Import controller mapping", "", "Mapping (*.json)")
        if not path:
            return
        try:
            bs = import_mapping(path)
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, "Import", f"Could not read that mapping:\n{e}")
            return
        self.result = {(b.action, b.deck): b for b in bs}
        self.ctl.set_bindings(bs, learned=True)        # used from now on, every time it connects
        self.heard.setText(f"Imported {len(bs)} controls from {path} - saved for this controller")
        self._show()

    def _reset_builtin(self):
        from ..dj.midi import ddj_t1_default

        if "DDJ-T1" not in (self.ctl.device or "").upper():
            QMessageBox.information(self, "Controller setup", "Tempest has a built-in map for the Pioneer DDJ-T1 only; "
                                    "other controllers are set up with this guide.")
            return
        if QMessageBox.question(self, "Controller setup", "Replace this controller's mapping with Tempest's built-in "
                                "DDJ-T1 map (every control, from Pioneer's MIDI list)?") != QMessageBox.Yes:
            return
        self.ctl.reset_to_builtin()
        bs = ddj_t1_default()
        self.result = {(b.action, b.deck): b for b in bs}
        self.heard.setText(f"Built-in DDJ-T1 map restored: {len(bs)} controls (custom mapping forgotten)")
        self._show()

    def _import_tsi(self):
        from ..dj.tsi import TsiError, tsi_bindings

        path, _ = QFileDialog.getOpenFileName(self, "Import a Traktor mapping", "",
                                              "Traktor settings / mapping (*.tsi)")
        if not path:
            return
        try:
            name, bs, unused = tsi_bindings(path)
        except TsiError as e:
            QMessageBox.warning(self, "Import Traktor mapping", str(e))
            return
        self.result = {(b.action, b.deck): b for b in bs}
        self.ctl.set_bindings(bs, learned=True)        # used from now on, every time it connects
        msg = f"Imported the {name} mapping: {len(bs)} controls, saved for this controller."
        if unused:
            msg += "\n\nNot used by Tempest: " + ", ".join(unused)
        self.heard.setText(msg.split("\n")[0])
        QMessageBox.information(self, "Import Traktor mapping", msg)
        self._show()

    def done(self, r):
        self.ctl.learn_cb = None
        super().done(r)


class HistoryDialog(QDialog):
    def __init__(self, dj: DJPanel):
        super().__init__(dj)
        self.dj = dj
        self.setWindowTitle("Played tracks (history)")
        lay = QVBoxLayout(self)
        self.only_session = QCheckBox("This session only")
        self.only_session.setChecked(True)
        self.only_session.toggled.connect(self._fill)
        lay.addWidget(self.only_session)
        self.list = QListWidget()
        lay.addWidget(self.list, 1)
        row = QHBoxLayout()
        b = QPushButton("Export…")
        b.clicked.connect(self._export)
        row.addStretch()
        row.addWidget(b)
        lay.addLayout(row)
        self.resize(640, 480)
        self._fill()

    def _rows(self):
        rows = self.dj.main.db.history(self.dj.session if self.only_session.isChecked() else None)
        return list(reversed(rows))

    def _fill(self, *_):
        self.list.clear()
        for ts, path, deck, _s in self._rows():
            self.list.addItem(f"{time.strftime('%H:%M', time.localtime(ts))}  [{deck}]  {os.path.basename(path)}")

    def _export(self):
        lib = self.dj.main.lib
        idx = [lib.paths.index(p) for _ts, p, _d, _s in self._rows() if p in lib.paths]
        if not idx:
            return
        name = f"Tempest history {time.strftime('%Y-%m-%d')}"
        path, _ = QFileDialog.getSaveFileName(self, "Export history", f"{name}.m3u8",
                                              "M3U8 (*.m3u8);;Traktor NML (*.nml);;rekordbox XML (*.xml);;CSV (*.csv)")
        if path:
            from ..export import export
            export(lib, idx, path, name)


# --------------------------------------------------------------------------
# the DJ panel
# --------------------------------------------------------------------------

class QRCode(QWidget):
    def __init__(self, text: str, size: int = 220):
        super().__init__()
        self.setFixedSize(size, size)
        self.matrix = None
        try:
            import qrcode

            q = qrcode.QRCode(border=2, error_correction=qrcode.constants.ERROR_CORRECT_M)
            q.add_data(text)
            q.make(fit=True)
            self.matrix = q.get_matrix()
        except Exception:  # noqa: BLE001
            log.exception("QR code failed")

    def paintEvent(self, _ev):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(255, 255, 255))
        if not self.matrix:
            p.setPen(QColor(0, 0, 0))
            p.drawText(self.rect(), Qt.AlignCenter, "(no QR code)")
            return
        n = len(self.matrix)
        cell = self.width() / n
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(0, 0, 0))
        for y, row in enumerate(self.matrix):
            for x, on in enumerate(row):
                if on:
                    p.drawRect(QRectF(x * cell, y * cell, cell + 0.5, cell + 0.5))


class RemoteInfoDialog(QDialog):
    """How to connect the phone: QR code, address, PIN."""

    def __init__(self, dj: DJPanel):
        super().__init__(dj)
        self.dj = dj
        self.setWindowTitle("Phone remote")
        self.lay = QVBoxLayout(self)
        self._build()

    def _build(self):
        while self.lay.count():
            it = self.lay.takeAt(0)
            if it.widget():
                it.widget().deleteLater()
        srv = self.dj.stream
        if srv is None:
            self.close()
            return
        head = QLabel("<b>Scan with your phone's camera</b> (same Wi-Fi as this computer)")
        self.lay.addWidget(head)
        row = QHBoxLayout()
        row.addStretch()
        row.addWidget(QRCode(srv.pair_url()))
        row.addStretch()
        w = QWidget()
        w.setLayout(row)
        self.lay.addWidget(w)
        urls = "<br>".join(f"<b>{u}</b>" for u in srv.urls)
        pin = (f"PIN: <b style='font-size:20px'>{srv.pin}</b>" if srv.pin
               else "<i>No PIN (anyone on this network can control the mix)</i>")
        info = QLabel(f"or open {urls}<br>{pin}<br><br>"
                      "From the phone you can skip, go back, pause, pick a seed song, edit the queue and move the "
                      "fade markers - and tap <b>Listen</b> to hear the mix (a few seconds behind).")
        info.setWordWrap(True)
        info.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.lay.addWidget(info)
        bb = QHBoxLayout()
        req = QCheckBox("Require PIN")
        req.setChecked(srv.pin is not None)
        req.toggled.connect(self._require)
        b_new = QPushButton("New PIN")
        b_new.setToolTip("Phones that used the old PIN will have to enter the new one")
        b_new.clicked.connect(self._new_pin)
        b_new.setEnabled(srv.pin is not None)
        ok = QPushButton("Close")
        ok.clicked.connect(self.accept)
        bb.addWidget(req)
        bb.addWidget(b_new)
        bb.addStretch()
        bb.addWidget(ok)
        w = QWidget()
        w.setLayout(bb)
        self.lay.addWidget(w)

    def _require(self, on):
        self.dj.settings.setValue("remote_require_pin", "true" if on else "false")
        self.dj.stream.pin = self.dj.remote_pin()
        self._build()

    def _new_pin(self):
        from .remote import new_pin

        self.dj.settings.setValue("remote_pin", new_pin())
        self.dj.stream.pin = self.dj.remote_pin()
        self._build()


class DJPanel(QWidget):
    # worker/MIDI threads must reach the GUI through queued signals
    _error = Signal(str)
    _browse_sig = Signal(int)
    _load_sig = Signal(str)
    _devices_sig = Signal(object)

    def __init__(self, main_window):
        super().__init__()
        self.main = main_window
        self.settings = main_window.settings
        # the engine runs in its own process (audio never waits for the GUI); the
        # client has the engine's API. In-process when asked (tests) or if no child starts.
        from dataclasses import asdict

        inproc = os.environ.get("CTA_AUDIO_INPROC") == "1" or \
            self.settings.value("dj_audio_process", "true") != "true"
        block = int(self.settings.value("dj_block", 512))
        am_cfg = asdict(self._automix_settings())
        try:
            self.engine = EngineClient(block=block, inproc=inproc, automix_settings=am_cfg,
                                       load_fn=self._automix_load)
        except Exception:  # noqa: BLE001
            log.exception("audio process did not start - running the engine in-process")
            self.engine = EngineClient(block=block, inproc=True, automix_settings=am_cfg,
                                       load_fn=self._automix_load)
        self.automix = self.engine.automix
        from ..dj.learn import MixLearner

        self._downloading: dict[str, str] = {}
        self.learner = MixLearner(main_window.db, self.automix)
        self.automix.learner = self.learner
        self.learner.on_change = self._push_learned
        self._learned_n = -1
        from ..dj.rehearsal import RehearsalStore

        self.rehearsals = RehearsalStore(main_window.db)
        self.automix.rehearsals = self.rehearsals
        self.rehearsals.on_change = self._push_rehearsals
        self._push_rehearsals()
        self.automix.rehearsal_fill = self.settings.value("ts_fill", "true") == "true"
        self.controller: Controller | None = None
        self.session = time.strftime("%Y-%m-%d ") + uuid.uuid4().hex[:6]
        self._set_started: float | None = None         # the set's playlist is named after this moment
        self._logged: dict[str, tuple[str, float]] = {}
        self._played: set[int] = set()
        self.set_seed: int | None = None     # the song the set started from: auto-picks stay in its genre
        self._error.connect(lambda msg: QMessageBox.warning(self, "DJ Mode", msg))
        self._browse_sig.connect(lambda d: self.main.browse_library(d))
        self._load_sig.connect(lambda deck: self.load_selected(deck))

        outer = QVBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)
        bar = QHBoxLayout()
        self.audio_lbl = QLabel("Audio: off")
        b_audio = QPushButton("Audio…")
        b_audio.clicked.connect(self.audio_settings)
        self.midi_combo = QComboBox()
        self.midi_combo.setMinimumWidth(200)
        b_midi = QPushButton("Connect")
        b_midi.clicked.connect(self.connect_midi)
        b_learn = QPushButton("Controller setup…")
        b_learn.clicked.connect(self.learn)
        b_builtin = QPushButton("Built-in map")
        b_builtin.setToolTip("Forget any custom controller mapping (setup guide, Traktor import) and use "
                             "Tempest's built-in Pioneer DDJ-T1 map: every jog, fader, knob and button")
        b_builtin.clicked.connect(self.forget_mapping)
        self.quant = QCheckBox("Quantize")
        self.quant.setChecked(True)
        self.quant.setToolTip("Snap cues, loops and jumps to the beat grid")
        self.quant.toggled.connect(lambda on: setattr(self.engine, "quantize", on))
        self.show_cd = QCheckBox("Decks C/D")
        self.show_cd.toggled.connect(self._toggle_cd)
        b_hist = QPushButton("History…")
        b_hist.clicked.connect(lambda: HistoryDialog(self).exec())
        self.view_combo = QComboBox()
        self.view_combo.addItems(["Auto", "Full", "Essential", "Micro"])
        self.view_combo.setToolTip("Deck view (Traktor style). Auto picks what fits: drag the bar under the decks "
                                   "to give them more or less room.")
        self.view_combo.setCurrentText(str(self.settings.value("dj_view", "Auto")))
        self.view_combo.currentTextChanged.connect(self._view_changed)
        self.platter_chk = QCheckBox("Platter")
        self.platter_chk.setToolTip("Show the jog platters (Full view). You can always scratch on the waveform.")
        self.platter_chk.setChecked(self.settings.value("dj_platter", "true") == "true")
        self.platter_chk.toggled.connect(self._view_changed)
        self.scratch_chk = QCheckBox("Scratch mode (controller jog top)")
        self.scratch_chk.setToolTip("On: a hand on the controller's jog top holds the record and turning it "
                                    "scratches.\nOff (default): touching the jog never stops a playing track; "
                                    "turning it bends the pitch (nudge) on a playing deck and searches a stopped "
                                    "one.\nDragging the waveform or the on-screen platter always scrubs.")
        self.scratch_chk.setChecked(self.settings.value("dj_scratch", "false") == "true")
        self.scratch_chk.toggled.connect(self._scratch_toggled)
        self.phase_chk = QCheckBox("SYNC also locks the beat phase")
        self.phase_chk.setToolTip("On: SYNC matches tempo and keeps the beats lined up.\n"
                                  "Off: SYNC matches the tempo only - you line up the beats yourself "
                                  "(jog / nudge) and they stay where you put them.\n"
                                  "Auto-mix keeps its own blends on the beat either way.")
        self.phase_chk.setChecked(self.settings.value("dj_phase_lock", "true") == "true")
        self.engine.phase_lock = self.phase_chk.isChecked()

        def phase_toggled(on):
            self.engine.phase_lock = bool(on)
            self.settings.setValue("dj_phase_lock", "true" if on else "false")
        self.phase_chk.toggled.connect(phase_toggled)
        self.xy_chk = QCheckBox("XY scope on the platters")
        self.xy_chk.setToolTip("Show each deck's stereo image (a goniometer) over its jog platter")
        self.xy_chk.setChecked(self.settings.value("dj_xyscope", "false") == "true")
        self.xy_chk.toggled.connect(lambda on: (self.settings.setValue("dj_xyscope", "true" if on else "false"),
                                                self.engine.set_want_deck_scope(bool(on))))
        self.tools_chk = QCheckBox("Global section (FX, auto-mix, recorder)")
        self.tools_chk.setToolTip("Show the row above the decks: FX units, auto-mix, main level, recorder")
        self.tools_chk.setChecked(self.settings.value("dj_tools", "true") == "true")
        self.b_stream = QPushButton("📱 Phone remote")
        self.b_stream.setCheckable(True)
        self.b_stream.setToolTip("Mix server: control the mix and listen to it from your phone (same Wi-Fi)")
        self.b_stream.toggled.connect(self.toggle_stream)
        self.stream_lbl = QLabel("")
        self.stream_lbl.setTextFormat(Qt.RichText)
        self.stream_lbl.setTextInteractionFlags(Qt.TextBrowserInteraction)
        self.stream_lbl.linkActivated.connect(lambda _l: self.show_remote_info())
        self.stream = None
        # a toolbar: whatever does not fit goes behind its » button (narrow screens)
        from PySide6.QtWidgets import QSizePolicy, QToolBar

        bar = QToolBar()
        bar.setStyleSheet("QToolBar { border: 0; background: transparent; padding: 0; }")
        for w in (self.audio_lbl, b_audio, QLabel("  MIDI:"), self.midi_combo, b_midi, b_learn, b_builtin):
            bar.addWidget(w)
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        bar.addWidget(spacer)
        # less-used switches live in one menu so the bar fits a laptop screen
        from PySide6.QtWidgets import QToolButton, QWidgetAction

        b_opts = QToolButton()
        b_opts.setText("Options ▾")
        b_opts.setPopupMode(QToolButton.InstantPopup)
        menu = QMenu(b_opts)
        for chk in (self.phase_chk, self.platter_chk, self.xy_chk, self.scratch_chk, self.tools_chk, self.quant,
                    self.show_cd):
            act = QWidgetAction(menu)
            holder = QWidget()
            hl = QHBoxLayout(holder)
            hl.setContentsMargins(10, 3, 10, 3)
            hl.addWidget(chk)
            act.setDefaultWidget(holder)
            menu.addAction(act)
        menu.addSeparator()
        from .shortcuts import HELP, DeckKeys

        self.keys_chk = QCheckBox("Keyboard control (Traktor keys)")
        self.keys_chk.setChecked(self.settings.value("dj_keys", "true") == "true")
        act = QWidgetAction(menu)
        holder = QWidget()
        hl = QHBoxLayout(holder)
        hl.setContentsMargins(10, 3, 10, 3)
        hl.addWidget(self.keys_chk)
        act.setDefaultWidget(holder)
        menu.addAction(act)
        set_wave_theme(self.settings.value("dj_wave_theme2", "Rainbow"))
        set_wave_saturation(float(self.settings.value("dj_wave_sat", 1.0)))
        from PySide6.QtGui import QActionGroup

        lmenu = menu.addMenu("Waveform layout")
        lgrp = QActionGroup(lmenu)
        self.layout_acts = {}
        for key, label, tip in (("decks", "In each deck (classic)", "The zoomed waveform sits in its deck"),
                                ("stacked", "Stacked across the top (Serato style)",
                                 "All decks' waveforms ride in parallel above the decks, playheads in one line - "
                                 "beats line up visually; the Three-band colours look most like Serato")):
            a = lmenu.addAction(label, lambda key=key: self.set_wave_layout(key))
            a.setCheckable(True)
            a.setToolTip(tip)
            lgrp.addAction(a)
            self.layout_acts[key] = a
        lmenu.setToolTipsVisible(True)
        wmenu = menu.addMenu("Waveform colours")
        grp = QActionGroup(wmenu)
        for name in WAVE_THEMES:
            a = wmenu.addAction(name, lambda name=name: self.set_wave_theme(name))
            a.setCheckable(True)
            a.setChecked(name == _theme["name"])
            a.setToolTip(WAVE_THEME_TIPS[name])
            grp.addAction(a)
        wmenu.setToolTipsVisible(True)
        wmenu.addSeparator()
        sat_holder = QWidget()
        sl = QHBoxLayout(sat_holder)
        sl.setContentsMargins(10, 3, 10, 3)
        sl.addWidget(QLabel("Saturation"))
        self.sat_slider = ResetSlider(Qt.Horizontal, 100, "Colour saturation (100 % = natural)")
        self.sat_slider.setRange(0, 200)
        self.sat_slider.setValue(int(round(_theme["sat"] * 100)))
        self.sat_slider.setMinimumWidth(140)
        self.sat_lbl = QLabel(f"{self.sat_slider.value()} %")
        self.sat_lbl.setMinimumWidth(42)
        self.sat_slider.valueChanged.connect(self.set_wave_saturation)
        sl.addWidget(self.sat_slider)
        sl.addWidget(self.sat_lbl)
        sat_act = QWidgetAction(wmenu)
        sat_act.setDefaultWidget(sat_holder)
        wmenu.addAction(sat_act)
        menu.addAction("Keyboard shortcuts…", lambda: QMessageBox.information(
            self, "Keyboard shortcuts", f"<pre style='font-size:12px'>{HELP}</pre>"))
        b_opts.setMenu(menu)
        from PySide6.QtGui import QKeySequence, QShortcut

        QShortcut(QKeySequence("Ctrl+T"), self, activated=lambda: self.open_timeshift(),
                  context=Qt.WindowShortcut)
        self.deck_keys = DeckKeys(self)
        self.deck_keys.enabled = self.keys_chk.isChecked()
        QApplication.instance().installEventFilter(self.deck_keys)

        def keys_toggled(on):
            self.deck_keys.enabled = on
            self.settings.setValue("dj_keys", "true" if on else "false")
        self.keys_chk.toggled.connect(keys_toggled)
        self.audio_lbl.setMaximumWidth(230)
        self.stream_lbl.setMaximumWidth(210)
        self.midi_combo.setMinimumWidth(130)
        self.midi_combo.setMaximumWidth(220)
        b_ts = QToolButton()
        b_ts.setText("⏱ Time-Shift")
        b_ts.setToolTip("Mix the next transition now on your headphones; auto-mix plays it live later")
        b_ts.clicked.connect(lambda: self.open_timeshift())
        for w in (self.stream_lbl, self.b_stream, b_ts, QLabel("View:"), self.view_combo, b_opts, b_hist):
            bar.addWidget(w)
        outer.addWidget(bar)

        # ---- global row: FX 1 · clock / auto-mix · main level · FX 2 · recorder -------------
        self.decks = {n: DeckPanel(n, self, mirrored=n in "BD") for n in "ABCD"}
        self.zoom_sync = self.settings.value("dj_zoom_sync", "true") == "true"
        span0 = float(self.settings.value("dj_zoom_span", 8.0))
        for p in self.decks.values():
            p.zoom.set_span(span0)
            _sync_check(p.b_zsync, self.zoom_sync)
        self.mixer = MixerPanel(self)
        self.fx = [FXPanel(self, 0), FXPanel(self, 1)]
        self.global_w = QWidget()
        gl = QHBoxLayout(self.global_w)
        gl.setContentsMargins(0, 0, 0, 0)
        gl.setSpacing(3)
        gl.addWidget(self.fx[0], 3)
        gl.addWidget(self._automix_panel(), 4)
        gl.addWidget(self.mixer.w_master)
        gl.addWidget(self.fx[1], 3)
        recw = QFrame()
        recw.setObjectName("rec")
        recw.setStyleSheet("#rec { background: #1c1c1e; border: 1px solid #0b0b0b; border-radius: 3px; }")
        rl = QVBoxLayout(recw)
        rl.setContentsMargins(5, 3, 5, 3)
        rl.setSpacing(2)
        rl.addWidget(self.mixer.rec)
        rl.addWidget(self.mixer.rec_lbl)
        gl.addWidget(recw)
        outer.addWidget(self.global_w)
        # ---- decks and mixer ---------------------------------------------------------------------
        grid = QGridLayout()
        grid.setSpacing(3)
        grid.addWidget(self.decks["C"], 0, 0)
        grid.addWidget(self.decks["D"], 0, 2)
        grid.addWidget(self.decks["A"], 1, 0)
        grid.addWidget(self.mixer, 0, 1, 2, 1)
        grid.addWidget(self.decks["B"], 1, 2)
        grid.setColumnStretch(0, 5)
        grid.setColumnStretch(2, 5)
        grid.setContentsMargins(0, 0, 0, 0)
        decks_w = QWidget()
        decks_w.setLayout(grid)
        # Serato-style option: every deck's waveform in one stack across the top, playheads in line
        self.stack_w = QFrame()
        self.stack_w.setObjectName("stack")
        self.stack_w.setStyleSheet("#stack { background: #0e0f12; border: 1px solid #0b0b0b; border-radius: 3px; }")
        sl = QVBoxLayout(self.stack_w)
        sl.setContentsMargins(2, 2, 2, 2)
        sl.setSpacing(1)
        self.stack_rows = {}
        for n in "ABCD":
            row = QWidget()
            rl = QHBoxLayout(row)
            rl.setContentsMargins(0, 0, 0, 0)
            rl.setSpacing(2)
            tag = QLabel(n)
            tag.setFixedWidth(18)
            tag.setAlignment(Qt.AlignCenter)
            c = DECK_COLORS[n]
            tag.setStyleSheet(f"background: {c.name()}; color: #111; font-weight: 800; border-radius: 2px;")
            rl.addWidget(tag)
            self.stack_rows[n] = row
            sl.addWidget(row)
        self.stack_w.hide()
        outer.addWidget(self.stack_w)
        outer.addWidget(decks_w, 1)
        outer.addWidget(self.mixer.xf_bar)
        self.wave_layout = "decks"
        self.set_wave_layout(str(self.settings.value("dj_wave_layout", "decks")), save=False)
        self.automix.set_decks("ABCD" if self.show_cd.isChecked() else "AB")
        self.global_w.setVisible(self.tools_chk.isChecked())
        self.tools_chk.toggled.connect(self._tools_toggled)
        self._toggle_cd(False)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(33)
        # speakers coming and going (Bluetooth): checked in a helper process, music untouched
        self._devices_sig.connect(self._on_devices)
        self.offer_btn = QPushButton("")
        self.offer_btn.hide()
        self.offer_btn.clicked.connect(self._accept_offer)
        # auto-mix stopped and its queue empty: offer to clear the decks for the next set
        self.unload_btn = QPushButton("⏏ Unload decks")
        self.unload_btn.setToolTip("Empty every deck that is not playing and reset auto-mix, ready for the next set")
        self.unload_btn.hide()
        self.unload_btn.clicked.connect(lambda: self.unload_decks(False))
        try:
            self.main.statusBar().addPermanentWidget(self.offer_btn)
            self.main.statusBar().addPermanentWidget(self.unload_btn)
        except Exception:  # noqa: BLE001
            pass
        self.dev_timer = QTimer(self)
        self.dev_timer.timeout.connect(self._scan_devices_bg)
        self.dev_timer.start(10000)
        QTimer.singleShot(3000, self._scan_devices_bg)
        self._refresh_midi_ports()
        QTimer.singleShot(200, self.start_audio)
        QTimer.singleShot(400, self._auto_connect_midi)

    def _automix_panel(self) -> QWidget:
        """Centre of the global row: master tempo, auto-mix transport, blend
        length and status (settings in its ▾ menu). The queue lives in the
        browser ("Auto-mix queue" tab)."""
        from PySide6.QtWidgets import QToolButton, QWidgetAction

        box = QFrame()
        box.setObjectName("clock")
        box.setStyleSheet("#clock { background: #1c1c1e; border: 1px solid #0b0b0b; border-radius: 3px; }")
        lay = QHBoxLayout(box)
        lay.setContentsMargins(6, 3, 6, 3)
        lay.setSpacing(5)
        col = QVBoxLayout()
        col.setSpacing(0)
        cap = QLabel("MASTER BPM")
        cap.setStyleSheet("font-size: 9px; color: #8d8f97; font-weight: 800;")
        self.clock_bpm = QLabel("—")
        self.clock_bpm.setStyleSheet("font-size: 18px; font-weight: 800; color: #ffffff;")
        col.addWidget(cap)
        col.addWidget(self.clock_bpm)
        lay.addLayout(col)
        self.b_auto = _btn("▶ AUTO-MIX", True, "Auto-mix: plays the queue as a continuous, beat-matched mix", h=24)
        self.b_auto.setObjectName("sync")
        self.b_auto.setFixedWidth(96)
        self.b_auto.clicked.connect(self.toggle_automix)
        b_tr = _btn("⏭", tip="Mix into the other deck now (from the next bar)", h=24, w=30)
        b_tr.clicked.connect(lambda: self.automix.transition_now())
        self.b_pause = _btn("⏸", True, tip="Fade the whole mix out and hold it (also from the phone)", h=24, w=30)
        self.b_pause.clicked.connect(lambda on: self.engine.set_paused(on))
        grid = QGridLayout()
        grid.setSpacing(3)
        grid.addWidget(self.b_auto, 0, 0)
        grid.addWidget(b_tr, 0, 1)
        grid.addWidget(self.b_pause, 0, 2)
        self.auto_len = QCheckBox("Auto")
        self.auto_len.setToolTip("Auto: the blend length follows the music (up to the number of bars set) and "
                                 "never hangs on dead air.\nOff: always blend over exactly that many bars "
                                 "(the outgoing track is looped if it ends sooner).")
        self.auto_len.setChecked(self.settings.value("dj_auto_len", "true") == "true")
        self.auto_len.toggled.connect(self._update_am_settings)
        self.mix_bars = QSpinBox()
        self.mix_bars.setRange(1, 128)
        self.mix_bars.setSuffix(" bars")
        self.mix_bars.setFixedWidth(72)
        self.mix_bars.setToolTip("Blend length (the longest blend when Auto is on)")
        self.mix_bars.setValue(int(self.settings.value("dj_mix_bars", 16)))
        self.mix_bars.valueChanged.connect(self._update_am_settings)
        brow = QHBoxLayout()
        brow.setSpacing(3)
        bl = QLabel("BLEND")
        bl.setStyleSheet("font-size: 9px; color: #8d8f97; font-weight: 800;")
        brow.addWidget(bl)
        brow.addWidget(self.mix_bars)
        brow.addWidget(self.auto_len)
        grid.addLayout(brow, 1, 0, 1, 3)
        # mix ahead of the play point: rehearse the next transition now, it plays live later
        b_ahead = QPushButton("⏱\nMIX AHEAD")
        b_ahead.setToolTip("Time-Shift (Ctrl+T): mix the next transition now on your headphones, ahead of the "
                           "play point - auto-mix plays your mix to the room when the track gets there")
        b_ahead.setStyleSheet("QPushButton { font-size: 9px; font-weight: 900; color: #ffb547; padding: 1px 4px; }")
        b_ahead.setFixedWidth(64)
        b_ahead.clicked.connect(lambda: self.open_timeshift())
        grid.addWidget(b_ahead, 0, 3, 2, 1)
        lay.addLayout(grid)
        stc = QVBoxLayout()
        stc.setSpacing(0)
        self.am_status = QLabel("Auto-mix off")
        self.am_status.setStyleSheet("font-weight: 700;")
        self.am_status.setMinimumWidth(80)
        from PySide6.QtWidgets import QSizePolicy

        self.am_status.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.learn_lbl = QLabel("")
        self.learn_lbl.setStyleSheet("color: #8d8f97; font-size: 10px;")
        self.learn_lbl.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        stc.addWidget(self.am_status)
        stc.addWidget(self.learn_lbl)
        lay.addLayout(stc, 1)
        # the rest of the auto-mix options: one menu
        self.glide = QCheckBox("Glide tempo to each track's BPM")
        self.glide.setChecked(self.settings.value("dj_glide", "false") == "true")
        self.glide.toggled.connect(self._update_am_settings)
        self.auto_suggest = QCheckBox("Auto-load best next track")
        self.auto_suggest.setChecked(self.settings.value("dj_auto_suggest", "true") == "true")
        self.learn_chk = QCheckBox("Learn from my mixes")
        self.learn_chk.setToolTip("Every transition you mix by hand teaches auto-mix where and how long you blend")
        self.learn_chk.setChecked(self.settings.value("dj_learn", "true") == "true")
        self.learn_chk.toggled.connect(self._learn_toggled)
        mb = QToolButton()
        mb.setText("▾")
        mb.setToolTip("Auto-mix options")
        mb.setPopupMode(QToolButton.InstantPopup)
        menu = QMenu(mb)
        for chk in (self.glide, self.auto_suggest, self.learn_chk):
            act = QWidgetAction(menu)
            holder = QWidget()
            hl = QHBoxLayout(holder)
            hl.setContentsMargins(10, 3, 10, 3)
            hl.addWidget(chk)
            act.setDefaultWidget(holder)
            menu.addAction(act)
        menu.addSeparator()
        menu.addAction("⏱ Mix ahead (Time-Shift)…\tCtrl+T", lambda: self.open_timeshift())
        menu.addAction("⏏ Unload stopped decks", lambda: self.unload_decks(False))
        menu.addAction("⏏ Unload all decks (stops the music)…", lambda: self.unload_decks(True))
        self.resume_chk = QCheckBox("Auto-mix carries on after I take over")
        self.resume_chk.setToolTip("On: once your hand-mixed blend is done (one track playing alone), auto-mix "
                                   "picks up again by itself.\nOff: after a takeover it stays off until you "
                                   "press ▶ AUTO-MIX again.")
        self.resume_chk.setChecked(self.settings.value("dj_resume_takeover", "true") == "true")
        self.automix.resume_after_takeover = self.resume_chk.isChecked()

        def resume_toggled(on):
            self.automix.resume_after_takeover = bool(on)
            self.settings.setValue("dj_resume_takeover", "true" if on else "false")
        self.resume_chk.toggled.connect(resume_toggled)
        ract = QWidgetAction(menu)
        rholder = QWidget()
        rl = QHBoxLayout(rholder)
        rl.setContentsMargins(10, 3, 10, 3)
        rl.addWidget(self.resume_chk)
        ract.setDefaultWidget(rholder)
        menu.addAction(ract)
        menu.addAction("Queue = setlist", self.queue_from_setlist)
        menu.addAction("Learned mixes & templates… (edit, export, import)", self.open_learned)
        menu.addAction("Forget what was learned from my mixes", self._forget_learning)
        bm = menu.addMenu("Blend length")
        for n in (4, 8, 16, 32, 64, 128):
            bm.addAction(f"{n} bars", lambda n=n: self.mix_bars.setValue(n))
        mb.setMenu(menu)
        lay.addWidget(mb)
        # the queue, shown as a browser tab
        self.queue_panel = QWidget()
        ql = QVBoxLayout(self.queue_panel)
        ql.setContentsMargins(4, 4, 4, 4)
        qh = QHBoxLayout()
        qh.addWidget(QLabel("<b>Up next</b> · drag to reorder, drag out / ✕ / Delete to remove"))
        qh.addStretch()
        b_clear = _btn("Clear queue", h=22, tip="Remove every track from the auto-mix queue")
        b_clear.clicked.connect(self.clear_queue)
        qh.addWidget(b_clear)
        ql.addLayout(qh)
        self.queue_list = QueueList()
        self.queue_list.removeRequested.connect(self.remove_from_queue)
        self.queue_list.reordered.connect(self._queue_reordered)
        ql.addWidget(self.queue_list, 1)
        return box

    def deck_view_pref(self) -> str:
        v = getattr(self, "view_combo", None)
        return v.currentText().lower() if v is not None else "auto"

    def show_platter(self) -> bool:
        c = getattr(self, "platter_chk", None)
        return c is None or c.isChecked()

    def _view_changed(self, *_):
        self.settings.setValue("dj_view", self.view_combo.currentText())
        self.settings.setValue("dj_platter", "true" if self.platter_chk.isChecked() else "false")
        for p in self.decks.values():
            p.apply_view(force=True)

    def _tools_toggled(self, on):
        self.settings.setValue("dj_tools", "true" if on else "false")
        self.global_w.setVisible(on)

    def _toggle_cd(self, on):
        self.decks["C"].setVisible(on)
        self.decks["D"].setVisible(on)
        self.automix.set_decks("ABCD" if on else "AB")        # four decks: auto-mix goes round the table
        for n in "CD":
            self.stack_rows[n].setVisible(on and self.wave_layout == "stacked")

    # ---- audio ---------------------------------------------------------------
    def start_audio(self, quiet: bool = False):
        from ..dj import outputs

        outputs.raise_priority()
        s = self.settings
        self.engine.out_latency = {"tight": "low", "safe": 0.05, "normal": 0.025,
                                   "auto": "auto"}.get(str(s.value("dj_safety", "auto")), "auto")
        name = s.value("dj_device_name", "")
        dev = None
        try:
            devices = list_output_devices()
            for _i, n, _ch in devices:
                if n == name:
                    dev = n
            if dev is None:
                for _i, n, ch in devices:                # prefer the DDJ-T1 sound card
                    if "DDJ-T1" in n.upper() and ch >= 4:
                        dev = n
                        break
            pname = s.value("dj_phones_device_name", "") or ""
            pdev = next((n for _i, n, _c in devices if n == pname), None) if pname else None
            mc = _pair_tuple(str(s.value("dj_master", "1-2"))) or (0, 1)
            pc = _pair_tuple(str(s.value("dj_phones", "3-4")))
            if pc == mc and pdev is None:
                pc = None
            eng = self.engine
            eng.master_delay_s, eng.phones_delay_s = self.stored_delays()
            eng.start(dev, int(s.value("dj_block", 512)), mc, pc, phones_label=pdev)
            self.engine.set_want_deck_scope(self.xy_chk.isChecked())
            if not eng.extra_outputs:
                self.apply_extras()
            self.apply_clock()
            if eng.phones_chans is None:
                phones = "no separate headphones"
            elif eng._pstream is not None:
                phones = "headphones on " + pname.split(" (")[0]
            else:
                phones = f"headphones {eng.phones_chans[0] + 1}-{eng.phones_chans[1] + 1}"
            m, p = eng.output_delays()
            delay = f"{m * 1000:.0f} ms" + (f" / {p * 1000:.0f} ms" if eng.phones_chans is not None and p != m else "")
            delay += " measured" if eng.master_delay_s is not None else " (driver)"
            self.audio_lbl.setText(f"Audio: {delay}, {phones}")
            self.audio_lbl.setToolTip(f"{eng.out_channels} channels, buffer {eng.block}. Delay = how late the "
                                      "output plays (driver or measured); master and headphones are kept in time.")
        except Exception as e:  # noqa: BLE001
            log.exception("audio start failed")
            # keep the mix running for the phone stream / recording
            self.engine.start_virtual()
            self.audio_lbl.setText("Audio: no output device (phone stream only) - click Audio…")
            if self.main.isVisible() and not quiet:
                self._error.emit(f"Could not open the audio device:\n{e}\n\n"
                                 "The mix still runs for the phone remote and recordings.")

    def audio_settings(self):
        dlg = AudioSettings(self.settings, self, dj=self)
        if dlg.exec():
            self.apply_audio_settings(dlg)
        elif not self.engine.running:
            self.start_audio()

    def stored_delays(self) -> tuple[float | None, float | None]:
        """(master, phones) delay in seconds for the devices chosen now, from
        the per-device table - a delay measured on another card is never used."""
        import json as _json

        s = self.settings
        try:
            delays = _json.loads(s.value("dj_output_delays", "") or "{}")
        except ValueError:
            delays = {}
        if not isinstance(delays, dict):
            delays = {}
        master = str(s.value("dj_device_name", "") or "")
        phones = str(s.value("dj_phones_device_name", "") or "") or master
        f = lambda label: (float(delays[label]) / 1000.0) if delays.get(label) is not None else None  # noqa: E731
        return f(master), f(phones)

    def apply_clock(self):
        """Which output keeps time, which follow it, and the DJ's adjustment."""
        import json as _json

        eng = self.engine
        eng.clock = str(self.settings.value("dj_clock", "auto") or "auto")
        raw = self.settings.value("dj_sync_outputs", "") or ""
        try:
            eng.sync_outputs = None if raw in ("", None) else set(_json.loads(raw))
        except ValueError:
            eng.sync_outputs = None
        eng.clock_adjust_s = float(self.settings.value("dj_clock_adjust_ms", 0) or 0) / 1000.0
        eng.update_alignment()

    def apply_audio_settings(self, dlg=None):
        """Apply the audio settings. Only a new main sound card, channels or
        buffer restart the audio; delays and extra speakers change live."""
        eng = self.engine
        if dlg is None or dlg.restart_needed or not eng.running:
            eng.stop()
            if dlg is not None and dlg.reinit_needed:
                from ..dj.outputs import rescan_devices

                try:
                    rescan_devices()          # the chosen card appeared after start-up
                except Exception:  # noqa: BLE001
                    log.exception("device rescan failed")
            eng.block = int(self.settings.value("dj_block", 512))
            self.start_audio()
        else:
            eng.master_delay_s, eng.phones_delay_s = self.stored_delays()
            eng.update_alignment()
        self.apply_extras()
        self.apply_clock()

    # ---- extra master outputs (more speakers, in sync) -------------------------------------------
    def wanted_extras(self) -> list[dict]:
        import json as _json

        try:
            return list(_json.loads(self.settings.value("dj_extra_outputs", "[]") or "[]"))
        except ValueError:
            return []

    def apply_extras(self, available: set | None = None):
        """Start / stop / re-time the extra speakers. available: labels (and
        names) connected right now - unknown means "try them all"."""
        from ..dj.outputs import base_name

        eng = self.engine
        wanted = {e["label"]: e for e in self.wanted_extras()}
        for eo in tuple(eng.extra_outputs):
            if eo.label not in wanted:
                eng.remove_output(eo.label)
        started = []
        for label, e in wanted.items():
            delay = None if e.get("delay_ms") is None else float(e["delay_ms"]) / 1000
            cur = next((o for o in eng.extra_outputs if o.label == label), None)
            if cur is not None:
                if cur.delay_s != delay:
                    eng.set_output_delay(label, delay)
                continue
            if available is not None and label not in available and base_name(label) not in available:
                continue
            try:
                eng.add_output(label, delay_s=delay)
                started.append(base_name(label))
            except OSError as ex:
                log.warning("could not start %s: %s", label, ex)
        eng.update_alignment()
        if started:
            self.main.status_label.setText(f"Playing on {', '.join(started)} too - in sync "
                                           f"({eng.display_latency * 1000:.0f} ms)")

    def _scan_devices_bg(self):
        if getattr(self, "_scanning", False) or getattr(self, "_closing", False) \
                or os.environ.get("CTA_NO_DEVICE_WATCH"):
            return
        self._scanning = True

        def job():
            from ..dj import outputs

            try:
                recs = outputs.scan_devices()
                if not getattr(self, "_closing", False):
                    self._devices_sig.emit(recs)
            except RuntimeError:          # the panel went away meanwhile
                pass
            finally:
                self._scanning = False
        threading.Thread(target=job, daemon=True, name="device-watch").start()

    def _on_devices(self, records):
        """Every few seconds: speakers that left, came back, or are new."""
        from ..dj.outputs import base_name

        if not records:
            return
        eng = self.engine
        labels = {r["label"] for r in records}
        avail = labels | {base_name(lb) for lb in labels}
        for eo in tuple(eng.extra_outputs):
            if not eo.alive:
                eng.remove_output(eo.label)
                if eo.label not in avail and base_name(eo.label) not in avail:
                    self.main.status_label.setText(f"{base_name(eo.label)} disconnected - it rejoins in sync "
                                                   "when it is back")
        self.apply_extras(available=avail)
        # a Bluetooth speaker that just appeared: offer it for the master
        used = {e["label"] for e in self.wanted_extras()} | {
            str(self.settings.value("dj_device_name", "") or ""), str(self.settings.value("dj_phones_device_name", "")
                                                                     or "")}
        seen = getattr(self, "_seen", None)
        new = [r for r in records if r.get("bluetooth") and r["label"] not in used
               and (seen is not None and r["label"] not in seen)]
        self._seen = labels
        if new:
            self._offer = new[0]["label"]
            self.offer_btn.setText(f"＋ Play the master on {base_name(self._offer)} too")
            self.offer_btn.show()
        elif getattr(self, "_offer", None) and self._offer not in labels:
            self.offer_btn.hide()

    def _accept_offer(self):
        import json as _json

        lb = getattr(self, "_offer", None)
        self.offer_btn.hide()
        if not lb:
            return
        ex = self.wanted_extras()
        if not any(e["label"] == lb for e in ex):
            ex.append({"label": lb, "delay_ms": None})
            self.settings.setValue("dj_extra_outputs", _json.dumps(ex))
        self.apply_extras()

    def toggle_record(self):
        eng = self.engine
        if eng.recorder is not None:
            r = eng.recorder
            eng.recorder = None
            r.close()
            self.main.status_label.setText(f"Recording saved: {r.path}")
            return
        folder = self.settings.value("rec_dir", os.path.join(os.path.expanduser("~"), "Music", "Tempest Recordings"))
        try:
            os.makedirs(folder, exist_ok=True)
            path = os.path.join(folder, f"Tempest mix {time.strftime('%Y-%m-%d %H%M%S')}.wav")
            eng.recorder = Recorder(path)
            self.main.status_label.setText(f"Recording to {path}")
        except Exception as e:  # noqa: BLE001
            self._error.emit(f"Could not start recording:\n{e}")

    # ---- live stream ------------------------------------------------------------
    def _stream_status(self) -> dict:
        decks = []
        for n, d in self.engine.decks.items():
            if d.loaded:
                ek = d.effective_key
                decks.append({"deck": n, "title": d.info.title, "artist": d.info.artist, "playing": d.playing,
                              "bpm": round(d.bpm, 1), "key": f"{camelot_str(ek)} {key_name(ek)}".strip()})
        return {"decks": decks, "automix": self.automix.status}

    def remote_pin(self) -> str | None:
        if self.settings.value("remote_require_pin", "true") != "true":
            return None
        pin = str(self.settings.value("remote_pin", "") or "")
        if not pin:
            from .remote import new_pin

            pin = new_pin()
            self.settings.setValue("remote_pin", pin)
        return pin

    def toggle_stream(self, on: bool, show_info: bool = True):
        """Start/stop the mix server (live stream + phone remote)."""
        from ..dj.stream import StreamServer
        from .remote import RemoteBackend

        if not on:
            if self.stream is not None:
                self.stream.stop()
                self.stream = None
            self.stream_lbl.setText("")
            return
        if self.stream is not None:
            return
        port = int(self.settings.value("stream_port", 8000))
        try:
            self.stream = StreamServer(self.engine, port=port, status_fn=self._stream_status,
                                       backend=RemoteBackend(self), pin=self.remote_pin())
            self.stream.start()
        except OSError as e:
            self.stream = None
            _sync_check(self.b_stream, False)
            self._error.emit(f"Could not start the phone remote on port {port}:\n{e}\n\n"
                             "Another program (or another copy of Tempest) may be using it - change the port "
                             "under Audio….")
            return
        _sync_check(self.b_stream, True)
        self.stream_lbl.setText(f"Phone: <a href='#' style='color:#6fb3ff'>{self.stream.urls[0]}</a>")
        self.stream_lbl.setToolTip("Click for the QR code and PIN. Windows may ask to allow Tempest through the "
                                   "firewall - allow it on private networks.")
        if show_info:
            self.show_remote_info()

    def show_remote_info(self):
        if self.stream is not None:
            RemoteInfoDialog(self).exec()

    # ---- MIDI ------------------------------------------------------------------
    def _refresh_midi_ports(self):
        self.midi_combo.clear()
        ports = Controller.list_inputs()
        self.midi_combo.addItems(ports or ["(no MIDI devices)"])
        for i, p in enumerate(ports):
            if "DDJ" in p.upper():
                self.midi_combo.setCurrentIndex(i)

    def _auto_connect_midi(self):
        if "DDJ" in self.midi_combo.currentText().upper():
            self.connect_midi()

    def connect_midi(self):
        port = self.midi_combo.currentText()
        if not port or port.startswith("("):
            self._refresh_midi_ports()
            return
        dev = "DDJ-T1" if "DDJ-T1" in port.upper() else port
        if self.controller:
            self.controller.close()
        self.controller = Controller(self.engine, self.automix, self, device=dev)
        self.controller.scratch_enabled = self.scratch_on()
        try:
            if not self.controller.open(port):
                raise RuntimeError("port not found")
        except Exception as e:  # noqa: BLE001
            self.controller = None
            self._error.emit(f"Could not open {port}:\n{e}")
            return
        if self.controller.learned:
            self.main.status_label.setText(f"Controller connected: {port} - custom mapping in use "
                                           "(Built-in map forgets it)")
        else:
            self.main.status_label.setText(f"Controller connected: {port} - built-in mapping")
        if not self.controller.learned and "DDJ-T1" not in dev.upper():
            r = QMessageBox.question(
                self, "Controller setup",
                f"{port} is connected with a basic mapping (play, cue, jog, tempo, load, browse).\n\n"
                "Run the guided setup now so every fader, knob and button works exactly as on your unit?")
            if r == QMessageBox.Yes:
                self.learn()

    def forget_mapping(self):
        """Delete the custom controller mapping and go back to the built-in DDJ-T1 map."""
        from ..dj.midi import forget_mapping

        port = self.midi_combo.currentText()
        dev = "DDJ-T1" if "DDJ-T1" in port.upper() else (self.controller.device if self.controller else port)
        if "DDJ-T1" not in (dev or "").upper():
            QMessageBox.information(self, "Built-in map", "Tempest has a built-in map for the Pioneer DDJ-T1 only; "
                                    "other controllers are set up with Controller setup.")
            return
        if QMessageBox.question(self, "Built-in map", "Forget the custom mapping for this controller and use "
                                "Tempest's built-in DDJ-T1 map (every control, from Pioneer's MIDI list)?") \
                != QMessageBox.Yes:
            return
        if self.controller is not None and "DDJ-T1" in self.controller.device.upper():
            self.controller.reset_to_builtin()
        else:
            forget_mapping("DDJ-T1")
        self.main.status_label.setText("Controller: built-in DDJ-T1 map (custom mapping forgotten)")
        if self.controller is None and port and not port.startswith("("):
            self.connect_midi()

    def learn(self):
        if self.controller is None:
            self.connect_midi()
            if self.controller is None:
                return
        decks = ("A", "B", "C", "D") if self.show_cd.isChecked() else ("A", "B")
        LearnWizard(self.controller, decks, self).exec()

    # controller callbacks (MIDI thread -> GUI thread via signals)
    def pickup_hint(self, deck, action, app_x: float, ctl_x: float):
        """The controller's knob has not met the app's value yet (soft takeover)."""
        import time as _t

        now = _t.monotonic()
        if now - getattr(self, "_pickup_t", 0.0) < 0.5:
            return
        self._pickup_t = now
        what = action.replace("_", " ")
        where = f"deck {deck} " if deck else ""
        self.main.status_label.setText(f"{where}{what}: controller at {ctl_x * 100:.0f} %, the app is at "
                                       f"{app_x * 100:.0f} % - move it there to pick it up (no jumps)")

    def browse(self, delta: int):
        self._browse_sig.emit(int(delta))

    def load(self, deck: str):
        self._load_sig.emit(deck)

    def cues_changed(self, deck: str):
        QTimer.singleShot(60, lambda: self._save_cues(deck))

    def _save_cues(self, deck: str):
        d = self.engine.decks[deck]
        if d.loaded and d.info.path:
            try:
                self.main.db.set_cues(d.info.path, [None if c is None else c / SR for c in d.hotcues])
            except Exception:  # noqa: BLE001
                log.exception("saving cues failed")

    def set_marker(self, deck: str, kind: str, t: float):
        """Intro/outro/cue from the waveform's right-click menu (bar-snapped)."""
        d = self.engine.decks[deck]
        if not d.loaded:
            return
        if kind == "cue":
            self.engine.set_cue(deck, t)
            return
        if kind == "edit":
            self.edit_markers(deck)
            return
        mi = d.mix_in / SR if d.mix_in is not None else None
        mo = d.mix_out / SR if d.mix_out is not None else None
        me = d.mix_out_end / SR if d.mix_out_end is not None else None
        if kind == "in":
            mi = t
        elif kind == "out":
            if me is not None and mo is not None:
                me = t + (me - mo)              # moving OUT keeps the blend length
            mo = t
        elif kind == "out_end":
            me = t
        elif kind == "clear":
            mi = mo = me = None
        try:
            self.set_markers(deck, mi, mo, me)
        except ValueError as e:
            self._error.emit(str(e))
            return
        what = {"in": "Mix-in (intro)", "out": "Mix-out (outro)", "out_end": "Outro end",
                "clear": "Markers"}[kind]
        self.main.status_label.setText(f"{what} {'cleared' if kind == 'clear' else 'set'} for {d.info.title} "
                                       "(saved with the track)")

    def edit_markers(self, deck: str):
        if not self.engine.decks[deck].loaded:
            self.main.status_label.setText(f"Load a track on deck {deck} first")
            return
        eds = self.__dict__.setdefault("_editors", {})
        ed = eds.get(deck)
        if ed is None or not ed.isVisible():
            ed = MarkerEditor(self, deck)
            eds[deck] = ed
        ed.show()
        ed.raise_()

    def _snap_bar(self, deck: str, t: float | None) -> float | None:
        g = self.engine.decks[deck].grid
        if t is None or g is None:
            return t
        return max(float(g.time_of_beat(np.floor(g.beat_at(t) / 4.0 + 1e-6) * 4.0)), 0.0)

    def set_markers(self, deck: str, mi: float | None, mo: float | None, me: float | None = None) -> str:
        """Set the mix markers (seconds, snapped to the start of their bar):
        intro, outro start and (optional) outro end. Re-plans auto-mix with
        them and saves them with the track."""
        d = self.engine.decks[deck]
        if not d.loaded:
            raise ValueError("That deck is empty.")
        mi, mo, me = self._snap_bar(deck, mi), self._snap_bar(deck, mo), self._snap_bar(deck, me)
        dur = d.length / SR
        mi, mo = (None if t is None else min(max(t, 0.0), dur) for t in (mi, mo))
        # the outro end may lie past the end of the track: auto-mix loops the groove to last that long
        if mo is None:
            me = None
        if mi is not None and mo is not None and mo <= mi:
            raise ValueError("The mix-out (outro) marker must come after the mix-in (intro) marker.")
        if me is not None and me <= mo:
            raise ValueError("The outro end must come after the outro start.")
        self.engine.set_markers(deck, mi, mo, me)
        self.automix.refresh_markers(deck)
        try:
            self.main.db.set_markers(d.info.path, mi, mo, me)
        except Exception:  # noqa: BLE001
            log.exception("saving markers failed")
        msg = f"Markers saved for {d.info.title}"
        if mo is not None and me is not None and d.grid is not None:
            bars = round((d.grid.beat_at(me) - d.grid.beat_at(mo)) / 4)
            msg += f" (outro blend {bars} bars)"
        return msg

    def markers_of(self, deck: str) -> tuple[float | None, float | None, float | None]:
        d = self.engine.decks[deck]
        f = lambda v: v / SR if v is not None else None  # noqa: E731
        return f(d.mix_in), f(d.mix_out), f(d.mix_out_end)

    def set_out_length(self, deck: str, bars: int | None) -> str:
        """Outro end = outro start + bars (None: no end marker, auto length)."""
        d = self.engine.decks[deck]
        mi, mo, _me = self.markers_of(deck)
        if mo is None or d.grid is None:
            raise ValueError("Set the OUT marker first.")
        me = None if not bars else d.grid.time_of_beat(d.grid.beat_at(mo) + 4 * int(bars))
        return self.set_markers(deck, mi, mo, me)

    def bump_marker(self, deck: str, kind: str, beats: float) -> str:
        """Move a marker by a number of beats (4 = one bar)."""
        d = self.engine.decks[deck]
        if not d.loaded or d.grid is None:
            raise ValueError("That deck is empty.")
        mi, mo, me = self.markers_of(deck)
        g = d.grid
        mv = lambda t: max(0.0, g.time_of_beat(g.beat_at(t) + beats)) if t is not None else None  # noqa: E731
        if kind == "in":
            if mi is None:
                raise ValueError("Set the IN marker first.")
            mi = mv(mi)
        elif kind == "out":
            if mo is None:
                raise ValueError("Set the OUT marker first.")
            mo, me = mv(mo), mv(me)            # the outro moves as a whole
        elif kind == "out_end":
            if me is None:
                if mo is None:
                    raise ValueError("Set the OUT marker first.")
                me = mo
            me = mv(me)
        else:
            raise ValueError(f"unknown marker {kind!r}")
        return self.set_markers(deck, mi, mo, me)

    def grid_changed(self, deck: str):
        def save():
            d = self.engine.decks[deck]
            if d.grid is not None and d.info.path:
                self.main.db.set_grid(d.info.path, d.grid.first_beat, d.info.bpm, version=3,
                                      phrase_origin=getattr(d, "phrase_origin", 0))
        QTimer.singleShot(60, save)

    # ---- loading ------------------------------------------------------------------
    def deck_playing(self, deck: str) -> bool:
        return self.engine.decks[deck].playing

    def _track_info(self, idx: int) -> tuple[str, float, TrackInfo]:
        lib = self.main.lib
        t = lib.track(idx)
        loud = -12.0
        try:
            r = self.main.db.conn().execute("SELECT loudness FROM tracks WHERE id=?", (t.id,)).fetchone()
            loud = r[0] if r and r[0] is not None else -12.0
        except Exception:  # noqa: BLE001
            pass
        return t.path, float(lib.bpm[idx]), TrackInfo(path=t.path, title=t.title, artist=t.artist, key=t.key,
                                                      bpm=float(lib.bpm[idx]), loudness=loud, lib_index=idx,
                                                      genre=lib.parent_genres[idx] or lib.tag_genres[idx],
                                                      style=(lib.styles[idx][0][0].split("---")[-1]
                                                             if lib.styles[idx] else lib.tag_genres[idx]))

    def load_index(self, deck: str, idx: int):
        if self.engine.decks[deck].playing:
            self.main.status_label.setText(f"Deck {deck} is playing - stop it before loading a new track")
            return
        if deck in "CD" and not self.show_cd.isChecked():
            self.show_cd.setChecked(True)
        path, bpm, info = self._track_info(idx)
        self.decks[deck].title.setText(f"Loading {info.title}…")

        def job():
            try:
                self._downloading[deck] = path
                try:
                    self.main.bridge.ensure_local(path)        # a home-library track: fetched first
                finally:
                    self._downloading.pop(deck, None)
                lt = load_track(path, bpm, info, self.main.db)
                # never replaces a deck that started playing meanwhile
                self.engine.load_track(deck, lt, only_if_stopped=True)
            except Exception as e:  # noqa: BLE001
                log.exception("load failed")
                self._error.emit(f"Could not load {path}:\n{e}")
        threading.Thread(target=job, daemon=True, name=f"load-{deck}").start()

    def load_selected(self, deck: str):
        idx = self.main.selected_for_load()
        if idx is None:
            self.main.status_label.setText("Select a track first (right panel's open tab, or the library)")
            return
        self.load_index(deck, idx)

    def _automix_load(self, idx):
        path, bpm, info = self._track_info(idx)
        self.main.bridge.ensure_local(path)                     # a home-library track: fetched first
        return load_track(path, bpm, info, self.main.db)

    # ---- auto-mix --------------------------------------------------------------------
    def _automix_settings(self) -> AutoMixSettings:
        return AutoMixSettings(mix_bars=int(self.settings.value("dj_mix_bars", 16)),
                               auto_length=self.settings.value("dj_auto_len", "true") == "true",
                               glide=self.settings.value("dj_glide", "false") == "true")

    def _update_am_settings(self, *_):
        self.settings.setValue("dj_auto_len", "true" if self.auto_len.isChecked() else "false")
        self.automix.s.auto_length = self.auto_len.isChecked()
        self.settings.setValue("dj_mix_bars", self.mix_bars.value())
        self.settings.setValue("dj_glide", "true" if self.glide.isChecked() else "false")
        self.automix.s.mix_bars = self.mix_bars.value()
        self.automix.s.glide = self.glide.isChecked()

    def _learn_toggled(self, on):
        self.settings.setValue("dj_learn", "true" if on else "false")
        self.learner.enabled = on
        self.automix.learner = self.learner if on else None

    def _forget_learning(self):
        n_rec = len(self.rehearsals.items())
        box = QMessageBox(self)
        box.setWindowTitle("Forget learning")
        box.setText("Forget every transition auto-mix learned from your mixes?")
        if n_rec:
            box.setInformativeText(f"You also have {n_rec} transition{'s' if n_rec != 1 else ''} recorded with Mix "
                                   "Ahead - auto-mix plays those exactly as recorded.")
        b_learn = box.addButton("Forget learned mixes", QMessageBox.AcceptRole)
        b_all = box.addButton("Forget learned + Mix Ahead recordings", QMessageBox.DestructiveRole) if n_rec else None
        box.addButton(QMessageBox.Cancel)
        box.exec()
        clicked = box.clickedButton()
        if clicked not in (b_learn, b_all):
            return
        self.learner.forget()
        msg = "Learned transitions forgotten"
        if b_all is not None and clicked is b_all:
            for r in self.rehearsals.items():
                self.rehearsals.delete(r.out_path, r.in_path)
            msg += " · Mix Ahead recordings deleted"
        self.automix.replan()                  # planned with the old learning: plan again
        self._learned_n = -1
        self.main.status_label.setText(msg + " - the next transition is planned afresh")

    def queue_from_setlist(self):
        sl = list(self.main.setlist)
        if not sl:
            QMessageBox.information(self, "Auto-mix", "Build a setlist in the Setlist Builder first "
                                    "(or just start auto-mix: it picks the best next tracks itself).")
            return
        self.automix.set_queue(sl)
        self.set_seed = sl[0]
        self._show_queue()

    def queue_items(self, items: list[int]):
        """Auto-mix plays these next, in order (a playlist)."""
        self.automix.set_queue(list(items))
        if items:
            self.set_seed = items[0]
        self._show_queue()
        self.main.status_label.setText(f"{len(items)} tracks queued for auto-mix - press ▶ AUTO-MIX to play them")

    def _show_queue(self):
        from PySide6.QtWidgets import QListWidgetItem

        self.queue_list.clear()
        lib = self.main.lib
        if lib is None:
            return
        # home-library tracks in the queue are fetched now, not when the deck needs them
        self.main.bridge.download_many(int(i) for i in self.automix.queue
                                       if isinstance(i, (int, np.integer)) and 0 <= int(i) < lib.n)
        for i in list(self.automix.queue):
            if not isinstance(i, (int, np.integer)) or not 0 <= int(i) < lib.n:
                continue
            t = lib.track(int(i))
            it = QListWidgetItem(f"{t.bpm:6.1f}  {camelot_str(t.key):>3}  {t.artist} - {t.title}")
            it.setData(Qt.UserRole, int(i))
            it.setToolTip("Drag to reorder · drag out of the list, click ✕ or press Delete to remove")
            self.queue_list.addItem(it)

    def clear_queue(self):
        self.automix.queue_clear()
        self._show_queue()
        self.main.status_label.setText("Queue cleared (auto-mix picks the next track itself when it runs dry)"
                                        if self.auto_suggest.isChecked() else "Queue cleared")

    def remove_from_queue(self, pos: int):
        self.automix.queue_remove(pos)
        self._show_queue()

    def _queue_reordered(self):
        order = [self.queue_list.item(r).data(Qt.UserRole) for r in range(self.queue_list.count())]
        self.automix.set_queue([int(i) for i in order if i is not None])

    def unload_decks(self, all_decks: bool = False):
        """Empty the decks in preparation for the next set (stopped ones only,
        unless all_decks - which asks first, since it stops the music)."""
        if all_decks:
            if QMessageBox.question(self, "Unload decks", "Stop and empty every deck, including the one playing?") \
                    != QMessageBox.Yes:
                return
            if self.automix.active:
                self.automix.stop()
        am = self.automix
        n = 0
        self._set_started = None                   # the next track on air opens a new set playlist
        for name, d in self.engine.decks.items():
            if not d.loaded or (d.playing and not all_decks):
                continue
            if am.active and (name in am.meta or name in am._loading):
                continue                       # auto-mix is preparing this one: leave it
            self.engine.unload(name, even_if_playing=all_decks)
            am.forget_deck(name)
            n += 1
        reset = not am.active                  # (all_decks stopped it above)
        if reset:
            am.reset()
            self.set_seed = None
            self._show_queue()
        self.unload_btn.hide()
        msg = f"Unloaded {n} deck{'s' if n != 1 else ''}" if n else "No decks to unload"
        self.main.status_label.setText(msg + (" · auto-mix reset, ready for the next set" if reset else ""))

    def _offer_unload(self):
        """Show ⏏ Unload decks while auto-mix is off, its queue is empty and a
        stopped deck still holds a track."""
        am = self.automix
        want = (not am.active and not am.queue and not am._loading
                and any(d.loaded and not d.playing for d in self.engine.decks.values()))
        if want != self.unload_btn.isVisible():
            self.unload_btn.setVisible(want)

    def toggle_automix(self, on):
        if on:
            if not self.automix.queue and not any(d.loaded for d in self.engine.decks.values()):
                if self.main.setlist:
                    self.automix.set_queue(list(self.main.setlist))
                elif self.main.current is not None:
                    self.automix.set_queue([self.main.current])
                else:
                    QMessageBox.information(self, "Auto-mix", "Select a track or build a setlist first.")
                    self.b_auto.setChecked(False)
                    return
            if self.set_seed is None:
                m = self.engine._master()
                q = list(self.automix.queue)
                self.set_seed = m.info.lib_index if m is not None and m.info.lib_index >= 0 else (
                    int(q[0]) if q and isinstance(q[0], (int, np.integer)) else None)
            self.automix.start()
        else:
            self.automix.stop()
        self._show_queue()

    def manual_touch(self, deck, action: str = "fader"):
        """Anything done by hand (mouse, keyboard): a transition in progress
        becomes the DJ's own - auto-mix steps back and learns from it."""
        self.automix.manual(deck, action)

    def played_indices(self) -> set[int]:
        """Everything that is on a deck, queued, loading or already played -
        including other files of the same song (duplicates)."""
        lib = self.main.lib
        used = {d.info.lib_index for d in self.engine.decks.values() if d.info.lib_index >= 0}
        used |= {int(i) for i in self.automix.used_items if isinstance(i, (int, np.integer))}
        used |= {int(i) for i in self.automix.queue if isinstance(i, (int, np.integer))}
        used |= self._played
        used |= self.main.session_played()
        if lib is None or not used:
            return used
        codes = lib.dedup_code[[i for i in used if 0 <= i < lib.n]]
        return used | set(np.nonzero(np.isin(lib.dedup_code, codes))[0].tolist())

    def _keep_queue_filled(self):
        am = self.automix
        lib = self.main.lib
        if lib is None or am.queue or am._loading or not self.auto_suggest.isChecked():
            return
        m = self.engine._master()
        if m is None or m.info.lib_index < 0:
            return
        if any(not d.playing and d.loaded and n in am.meta for n, d in self.engine.decks.items()):
            return
        self._played |= {d.info.lib_index for d in self.engine.decks.values() if d.info.lib_index >= 0}
        seed = self.set_seed if self.set_seed is not None and 0 <= self.set_seed < lib.n else None
        sug = suggest_next(lib, m.info.lib_index, self.main.mix_settings(), n=20, exclude=self.played_indices(),
                           anchor=seed)
        if sug:
            am.set_queue([sug[0].idx])
            self._show_queue()

    def _log_history(self):
        """A track counts as played after 30 s on air."""
        for n, d in self.engine.decks.items():
            if not (d.playing and d.loaded and d.info.path):
                continue
            key = self._logged.get(n)
            if key is None or key[0] != d.info.path:
                self._logged[n] = (d.info.path, time.time())
            elif key[1] > 0 and time.time() - key[1] > 30 and d.volume > 0.3:
                try:
                    self.main.db.add_history(time.time(), d.info.path, n, self.session)
                    self.main.track_played(d.info.lib_index)
                    if self._set_started is None:
                        self._set_started = time.time() - 30.0        # when this set's first track went on
                    self.main.playlists.log_played(int(d.info.lib_index), self._set_started)
                except Exception:  # noqa: BLE001
                    log.exception("history failed")
                self._logged[n] = (d.info.path, -1.0)

    # ---- refresh -------------------------------------------------------------------------
    def _watchdog(self):
        """The sound card stopped asking for audio (device lost, driver
        hiccup, sleep): reopen it. The decks keep their place."""
        eng = self.engine
        now = time.monotonic()
        if not eng.process_alive:
            log.error("the audio process died - starting a new one")
            self.main.status_label.setText("Audio process restarted (it crashed) - see the log")
            try:
                eng.respawn()
                self.start_audio(quiet=True)
            except Exception:  # noqa: BLE001
                log.exception("audio process restart failed")
            return
        errs = eng.deck_errors
        if errs and errs != getattr(self, "_errs_seen", None):
            self._errs_seen = errs
            self.main.status_label.setText(errs[-1] + " - see the log")
        if not eng.running or eng.virtual:
            self._wd = (eng.frames_rendered, now)
            return
        fr, t = getattr(self, "_wd", (-1, now))
        if eng.frames_rendered != fr:
            self._wd = (eng.frames_rendered, now)
        elif now - t > 3.0:
            log.warning("audio output stalled for %.1f s (frames %d, underruns %d, callback errors %d) "
                        "- reopening the audio device", now - t, fr, eng.underruns, eng.callback_errors)
            self._wd = (fr, now)
            self.start_audio(quiet=True)
            self.main.status_label.setText("Audio device stopped responding - reconnected")

    def next_from_deck(self, deck: str):
        """Show the library's Next Track list for what this deck holds."""
        d = self.engine.decks[deck]
        if not d.loaded or d.info.lib_index < 0:
            self.main.status_label.setText(f"Deck {deck} holds no library track")
            return
        self.main.focus_track(int(d.info.lib_index))
        self.main.tabs.setCurrentIndex(0)
        self.main.status_label.setText(f"Next tracks for deck {deck}: {d.info.title}")

    def open_learned(self):
        from .learned import LearnedMixesDialog

        w = getattr(self, "_learned_dlg", None)
        if w is not None and w.isVisible():
            w.refresh()
            w.raise_()
            return w
        self._learned_dlg = LearnedMixesDialog(self)
        self._learned_dlg.show()
        return self._learned_dlg

    def open_timeshift(self, out_idx: int | None = None, in_idx: int | None = None):
        from .timeshift import TimeShiftWindow

        w = getattr(self, "_timeshift", None)
        if w is not None and w.isVisible():
            w.raise_()
            w.activateWindow()
            return w
        self._timeshift = TimeShiftWindow(self, out_idx, in_idx)
        self._timeshift.show()
        return self._timeshift

    def set_wave_saturation(self, v: int):
        set_wave_saturation(v / 100.0)
        self.settings.setValue("dj_wave_sat", _theme["sat"])
        self.sat_lbl.setText(f"{v} %")
        for w in self.findChildren(QWidget):
            if isinstance(w, (Waveform, MarkerWave)):
                w.update()

    def scratch_on(self) -> bool:
        return self.scratch_chk.isChecked()

    def _scratch_toggled(self, on: bool):
        self.settings.setValue("dj_scratch", "true" if on else "false")
        if self.controller is not None:
            self.controller.scratch_enabled = bool(on)
        if not on:
            for n in self.engine.decks:
                self.engine.jog_release(n)          # let go of anything held

    def zoom_waveforms(self, deck: str, factor: float):
        """- / + on a deck: zoom its waveform (both decks with sync zoom on)."""
        span = self.decks[deck].zoom.span * factor
        for n, p in self.decks.items():
            if n == deck or self.zoom_sync:
                p.zoom.set_span(span)
        self.settings.setValue("dj_zoom_span", self.decks[deck].zoom.span)

    def set_zoom_sync(self, on: bool):
        self.zoom_sync = bool(on)
        self.settings.setValue("dj_zoom_sync", "true" if on else "false")
        for p in self.decks.values():
            _sync_check(p.b_zsync, on)
        if on:                                    # line the others up with deck A
            for p in self.decks.values():
                p.zoom.set_span(self.decks["A"].zoom.span)

    def set_wave_layout(self, mode: str, save: bool = True):
        """"decks": the zoomed waveform in each deck; "stacked": all of them in
        one panel across the top, Serato style (the deck keeps its overview,
        platter and tempo fader)."""
        mode = "stacked" if mode == "stacked" else "decks"
        self.wave_layout = mode
        if save:
            self.settings.setValue("dj_wave_layout", mode)
        for n, p in self.decks.items():
            row = self.stack_rows[n]
            lay = row.layout()
            if mode == "stacked":
                if p.zoom.parent() is not row:
                    p.body_lay.removeWidget(p.zoom)
                    p.body_lay.insertStretch(1, 1)
                    lay.addWidget(p.zoom, 1)
                    p.zoom.setMinimumHeight(64)
                row.setVisible(n in "AB" or self.show_cd.isChecked())
            else:
                if p.zoom.parent() is row:
                    lay.removeWidget(p.zoom)
                    item = p.body_lay.takeAt(1)          # the stretch that filled the gap
                    del item
                    p.body_lay.insertWidget(1, p.zoom, 1)
                    p.zoom.setMinimumHeight(48)
                row.setVisible(False)
            p.zoom.show()
            p.apply_view(force=True)
        self.stack_w.setVisible(mode == "stacked")
        for key, a in getattr(self, "layout_acts", {}).items():
            a.setChecked(key == mode)

    def set_wave_theme(self, name: str):
        set_wave_theme(name)
        self.settings.setValue("dj_wave_theme2", _theme["name"])
        for w in self.findChildren(QWidget):
            if isinstance(w, (Waveform, MarkerWave)):
                w.update()

    def _audio_health(self):
        """Blocks that nearly missed their deadline mean stutter is close: say how to fix it."""
        import time as _t

        now = _t.monotonic()
        last_t, last_n = getattr(self, "_health", (now, self.engine.slow_blocks))
        if now - last_t < 5.0:
            return
        n = self.engine.slow_blocks
        self._health = (now, n)
        notes = getattr(self.engine, "audio_notes", None)
        if notes and notes[-1] != getattr(self, "_note_seen", None):
            self._note_seen = notes[-1]
            self.main.status_label.setText(f"Audio: {notes[-1]}")
            return
        if n - last_n >= 3 and self.engine.running and str(self.settings.value("dj_safety", "auto")) != "auto":
            self.main.status_label.setText(
                f"Audio is near its limit ({n - last_n} late blocks) - raise the buffer in Audio… "
                f"(now {self.engine.block}) to stop any stutter")

    def _push_learned(self):
        """What auto-mix learned lives here (the database); the engine process gets the numbers."""
        groups = set(self.learner.groups()) | {"long", "club", "short"}
        prefs = {g: self.learner.preferences(g) for g in groups}
        prefs["*"] = self.learner.preferences("")           # any other style: all of the DJ's mixes
        self.engine.push_learned(prefs)

    def _push_rehearsals(self):
        tpl = {g: r.to_dict() for g in self.rehearsals.templates()
               if (r := self.rehearsals.template(g)) is not None}
        self.engine.push_rehearsals([r.to_dict() for r in self.rehearsals.items() if r.enabled], tpl)

    def _tick(self):
        self._offer_unload()
        for deck, path in list(self._downloading.items()):        # a deck waiting for a home-library file
            txt = self.main.bridge.progress_text(path)
            if txt:
                self.decks[deck].title.setText(txt)
        # with a controller attached its knobs set the levels: loading a track keeps them
        want_reset = self.controller is None
        if want_reset != self.engine.reset_on_load:
            self.engine.reset_on_load = want_reset
        n_learned = len(self.learner.examples)
        if n_learned != self._learned_n:
            self._learned_n = n_learned
            self._push_learned()
        self._watchdog()
        self._audio_health()
        self.automix.maybe_resume()
        show_cd = self.show_cd.isChecked()
        xy = self.xy_chk.isChecked()
        for n, p in self.decks.items():
            if p.isVisible():
                p.refresh()
                if p.platter.isVisible():
                    d = self.engine.decks[n]
                    sc = d.scope if xy and d.playing else None
                    if sc is not p.platter.scope:
                        p.platter.scope = sc
                        p.platter.update()
        self.mixer.refresh(show_cd)
        for f in self.fx:
            f.refresh()
        am = self.automix
        _set_text(self.am_status, ("⏸ PAUSED · " if self.engine.paused else "") + am.status)
        m = self.engine._master()
        bpm = f"{m.bpm:.2f}" if m is not None else "—"
        if self.clock_bpm.text() != bpm:
            _set_text(self.clock_bpm, bpm)
        _sync_check(self.b_pause, self.engine.paused)
        _sync_check(self.b_auto, am.active)
        _set_text(self.b_auto, "■ STOP MIX" if am.active else "▶ AUTO-MIX")
        if am.active:
            self._keep_queue_filled()
            if self.queue_list.count() != len(am.queue):
                self._show_queue()
        self._log_history()
        try:
            self.learner.tick(self.engine)
        except Exception:  # noqa: BLE001
            log.exception("learner failed")
        n = self.learner.count()
        _set_text(self.learn_lbl, f"{n} transition{'s' if n != 1 else ''} learned" if n else "nothing learned yet")
        if self.controller is not None:
            self.controller.tick()
            self.controller.update_leds()

    def shutdown(self):
        if getattr(self, "_shut", False):
            return
        self._shut = True
        self._closing = True
        ts = getattr(self, "_timeshift", None)
        if ts is not None:
            ts.close()
        QApplication.instance().removeEventFilter(self.deck_keys)
        self.timer.stop()
        self.dev_timer.stop()
        self.engine.close_outputs()
        if self.stream is not None:
            self.stream.stop()
            self.stream = None
        self.automix.stop()
        if self.engine.recorder is not None:
            self.toggle_record()
        self.engine.stop()
        if self.controller:
            self.controller.close()
        self.engine.close()

    def closeEvent(self, ev):
        self.shutdown()
        super().closeEvent(ev)


# backwards-compatible name
DJWindow = DJPanel
