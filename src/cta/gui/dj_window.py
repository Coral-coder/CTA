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
    QListWidget,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..dj.automix import AutoMix, AutoMixSettings
from ..dj.engine import SR, Engine, TrackInfo, list_output_devices
from ..dj.fx import BEAT_DIVS, FX_TYPES
from ..dj.loader import load_track
from ..dj.midi import Controller, Learner, learn_steps
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
        self.setToolTip(tip + ("\n" if tip else "") + "Double-click to reset")

    def mouseDoubleClickEvent(self, _ev):
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

def _band_colors(b: np.ndarray) -> np.ndarray:
    """(n,3) band levels -> (n,3) uint8 RGB: red bass, green mids, blue highs."""
    tot = b.sum(axis=1, keepdims=True) + 1e-9
    mix = b / tot
    rgb = np.stack([0.25 + 0.75 * mix[:, 0] + 0.2 * mix[:, 1],
                    0.25 + 0.7 * mix[:, 1] + 0.15 * mix[:, 0],
                    0.3 + 0.9 * mix[:, 2]], axis=1)
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)


def _render(heights: np.ndarray, colors: np.ndarray, h: int, past: int | None = None) -> QImage:
    """Mirror-waveform image from per-column heights (0..1) and colours."""
    w = heights.size
    img = np.empty((h, w, 4), np.uint8)
    img[..., 3] = 255
    img[..., :3] = (18, 19, 22)
    half = h / 2
    ys = np.abs(np.arange(h)[:, None] + 0.5 - half)
    mask = ys <= heights[None, :] * (half - 1)
    col = colors.copy()
    if past is not None and past > 0:
        col[:past] = (col[:past] * 0.45).astype(np.uint8)
    img[..., :3] = np.where(mask[..., None], col[None, :, :], img[..., :3])
    return QImage(img.data, w, h, 4 * w, QImage.Format_RGBA8888).copy()


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
        self._over = None          # cached (width, heights, colours)
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._menu)
        self._drag_x = None
        if zoom:
            self.setCursor(Qt.OpenHandCursor)
            self.setToolTip("Drag to scratch · double-click to jump · wheel to zoom · right-click for markers")

    def set_track(self, peaks, bands, duration, grid):
        self.peaks, self.bands, self.duration, self.grid = peaks, bands, max(duration, 1e-3), grid
        self._over = None
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

    def wheelEvent(self, ev):
        if self.zoom:
            self.span = float(np.clip(self.span * (0.8 if ev.angleDelta().y() > 0 else 1.25), 2, 32))
            self.update()

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
            t0 = self.pos - self.span / 2
            idx = ((t0 + np.arange(w) / w * self.span) * SR / 256).astype(np.int64)
            valid = (idx >= 0) & (idx < self.bands.shape[0])
            b = np.zeros((w, 3), np.float32)
            b[valid] = self.bands[idx[valid]]
            p.drawImage(0, 0, _render(np.clip(b.max(axis=1), 0, 1), _band_colors(b), h))
        else:
            if self._over is None or self._over[0] != w:
                n = self.peaks.shape[0]
                b = self.peaks[np.linspace(0, n - 1, w).astype(int)]
                self._over = (w, np.clip(b.max(axis=1), 0, 1), _band_colors(b))
            past = int(self.pos / self.duration * w)
            p.drawImage(0, 0, _render(self._over[1], self._over[2], h, past))
        if self.loop:
            x0, x1 = self._x(self.loop[0] / SR, w), self._x(self.loop[1] / SR, w)
            p.fillRect(QRectF(x0, 0, max(x1 - x0, 2), h), QColor(80, 220, 120, 60))
        if self.zoom and self.grid is not None:
            t0 = self.pos - self.span / 2
            for bt in range(int(np.floor(self.grid.beat_at(t0))), int(np.ceil(self.grid.beat_at(t0 + self.span))) + 1):
                x = self._x(self.grid.time_of_beat(bt), w)
                bar = bt % 4 == 0
                p.setPen(QPen(QColor(255, 255, 255, 150 if bar else 45), 2 if bar else 1))
                p.drawLine(int(x), 0, int(x), h)
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
        # mix markers: green IN flag, red OUT flag + the blend region after it
        if self.mix_out is not None and self.grid is not None:
            x0 = self._x(self.mix_out, w)
            x1 = self._x(self.mix_out_end if self.mix_out_end is not None
                         else self.mix_out + self.mix_beats * self.grid.beat_len, w)
            p.fillRect(QRectF(x0, 0, max(x1 - x0, 2), h), QColor(229, 57, 53, 45))
        for t, label, colr in ((self.mix_in, "IN", QColor(67, 160, 71)), (self.mix_out, "OUT", QColor(229, 57, 53)),
                               (self.mix_out_end, "END", QColor(255, 152, 0))):
            if t is None:
                continue
            x = self._x(t, w)
            if -30 <= x <= w + 5:
                p.setPen(QPen(colr, 2))
                p.drawLine(int(x), 0, int(x), h)
                p.fillRect(QRectF(x, h - 14, 28, 14), colr)
                p.setPen(QColor(255, 255, 255))
                p.drawText(QRectF(x, h - 14, 28, 14), Qt.AlignCenter, label)
        cx = self._x(self.cue, w)
        p.setPen(QPen(QColor(255, 190, 0), 1, Qt.DashLine))
        p.drawLine(int(cx), 0, int(cx), h)
        px = w // 2 if self.zoom else int(self._x(self.pos, w))
        p.setPen(QPen(QColor(255, 60, 60) if self.zoom else QColor(255, 255, 255), 2))
        p.drawLine(px, 0, px, h)


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

    def paintEvent(self, _ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = self.rect().adjusted(4, 4, -4, -4)
        p.setPen(QPen(QColor(10, 11, 14), 2))
        p.setBrush(QColor(28, 30, 36))
        p.drawEllipse(r)
        # grooves
        p.setBrush(Qt.NoBrush)
        for k in range(3, 9):
            p.setPen(QPen(QColor(48, 50, 58), 1))
            inset = k * r.width() / 20
            p.drawEllipse(r.adjusted(inset, inset, -inset, -inset))
        # ring
        ring = QColor(self.color)
        ring.setAlpha(255 if self.held else 170)
        p.setPen(QPen(ring, 5))
        p.drawEllipse(r.adjusted(3, 3, -3, -3))
        # spinning marker
        c = r.center()
        rad = r.width() / 2 - 12
        a = np.radians(self.angle - 90)
        p.setPen(QPen(QColor(240, 240, 245), 5, Qt.SolidLine, Qt.RoundCap))
        p.drawLine(c, c + QPointF(np.cos(a) * rad, np.sin(a) * rad).toPoint())
        if self.art is not None and not self.art.isNull():
            # the record label: album art turning with the platter
            lr = r.width() * 0.22
            p.save()
            p.translate(QPointF(c))
            p.rotate(self.angle)
            clip = QPainterPath()
            clip.addEllipse(QPointF(0, 0), lr, lr)
            p.setClipPath(clip)
            p.drawPixmap(QRectF(-lr, -lr, 2 * lr, 2 * lr), self.art, QRectF(self.art.rect()))
            p.restore()
            p.setBrush(QColor(20, 20, 24))
            p.setPen(Qt.NoPen)
            p.drawEllipse(QPointF(c), 3, 3)
        else:
            p.setBrush(self.color)
            p.setPen(Qt.NoPen)
            p.drawEllipse(c, 14, 14)


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
        if not d.playing:
            if seconds:
                self.eng.seek(self.deck, d.time + seconds)      # paused: search
            return
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

    COLORS = {"in": QColor(67, 160, 71), "out": QColor(229, 57, 53), "out_end": QColor(255, 152, 0)}
    LABELS = {"in": "IN", "out": "OUT", "out_end": "OUT END"}

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

    def set_track(self, bands, duration, grid):
        self.bands, self.duration, self.grid = bands, max(duration, 1e-3), grid
        self._coarse = (None,)
        self.v0, self.v1 = 0.0, self.duration

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
        self.zoom(0.8 if ev.angleDelta().y() > 0 else 1.25, self._t(ev.position().x()))

    def mousePressEvent(self, ev):
        if ev.button() != Qt.LeftButton or self.bands is None:
            return
        x = ev.position().x()
        k = self._hit(x)
        self._drag = ("mark", k) if k else ("pan", x, self.v0, self.v1)
        self.setCursor(Qt.SizeHorCursor if k else Qt.ClosedHandCursor)

    def mouseMoveEvent(self, ev):
        x = ev.position().x()
        if self._drag is None:
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
        self.seek.emit(self._snap(self._t(ev.position().x())))

    # ---- paint ------------------------------------------------------------------------------
    def paintEvent(self, _ev):
        p = QPainter(self)
        w, h = self.width(), self.height()
        p.fillRect(0, 0, w, h, QColor(16, 17, 20))
        if self.bands is None or w < 4:
            return
        wave_h = h - 22
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
        p.drawImage(0, 0, _render(np.clip(b.max(axis=1), 0, 1), _band_colors(b), wave_h))
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
        marks = dict(self.marks)
        if self._preview:
            marks[self._preview[0]] = self._preview[1]
            if self._preview[0] == "out" and self.marks["out"] is not None and self.marks["out_end"] is not None:
                marks["out_end"] = self.marks["out_end"] + (self._preview[1] - self.marks["out"])
        # the outgoing blend: OUT .. OUT END (or the auto-mix length)
        if marks["out"] is not None:
            end = marks["out_end"]
            if end is None and self.grid is not None:
                end = marks["out"] + self.blend_beats * self.grid.beat_len
            if end is not None:
                x0, x1 = self._x(marks["out"]), self._x(end)
                p.fillRect(QRectF(x0, 0, max(x1 - x0, 2), wave_h), QColor(229, 57, 53, 40))
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
        lay.addWidget(self.wave, 1)
        hint = QLabel("Drag a flag to move it (it snaps to the start of the bar) · drag elsewhere to scroll · "
                      "wheel to zoom · double-click to play from there")
        hint.setStyleSheet("color: #9a9ba3;")
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
            b = _btn("At playhead", tip="Put it at the start of the bar that is playing")
            b.clicked.connect(lambda _=False, k=kind: self._at_playhead(k))
            g.addWidget(b, r, col)
            b = _btn("Show", tip="Zoom to it")
            b.clicked.connect(lambda _=False, k=kind: self._show(k))
            g.addWidget(b, r, col + 1)
            b = _btn("Clear")
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
        self.setStyleSheet(f"#deck {{ background: #17181c; border: 1px solid {col.darker(150).name()};"
                           f" border-top: 3px solid {col.name()}; border-radius: 8px; }}")
        self.scr = _Scratcher(self.eng, name)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 6, 8, 8)
        outer.setSpacing(5)

        # ---- header ----------------------------------------------------------
        head = QHBoxLayout()
        badge = QLabel(name)
        badge.setFixedSize(30, 30)
        badge.setAlignment(Qt.AlignCenter)
        badge.setStyleSheet(f"background: {col.name()}; color: #111; font-weight: 800; font-size: 16px;"
                            "border-radius: 6px;")
        head.addWidget(badge)
        self.art = QLabel()
        self.art.setFixedSize(46, 46)
        self.art.setScaledContents(True)
        self.art.setStyleSheet("background: #24262c; border-radius: 6px;")
        head.addWidget(self.art)
        titles = QVBoxLayout()
        titles.setSpacing(0)
        self.title = QLabel("—")
        self.title.setStyleSheet("font-size: 15px; font-weight: 700; color: #f2f2f5;")
        self.artist = QLabel("Drop a track here")
        self.artist.setStyleSheet("font-size: 12px; color: #9a9ba3;")
        titles.addWidget(self.title)
        titles.addWidget(self.artist)
        head.addLayout(titles, 1)
        self.key_lbl = QLabel("")
        self.key_lbl.setAlignment(Qt.AlignCenter)
        self.key_lbl.setMinimumWidth(78)
        head.addWidget(self.key_lbl)
        bpmbox = QVBoxLayout()
        bpmbox.setSpacing(0)
        self.bpm_lbl = QLabel("")
        self.bpm_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.bpm_lbl.setStyleSheet("font-size: 26px; font-weight: 800; color: #ffffff;")
        self.pct_lbl = QLabel("")
        self.pct_lbl.setAlignment(Qt.AlignRight)
        self.pct_lbl.setStyleSheet("font-size: 11px; color: #9a9ba3;")
        bpmbox.addWidget(self.bpm_lbl)
        bpmbox.addWidget(self.pct_lbl)
        head.addLayout(bpmbox)
        timebox = QVBoxLayout()
        timebox.setSpacing(0)
        self.remain_lbl = QLabel("")
        self.remain_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.remain_lbl.setMinimumWidth(92)
        self.remain_lbl.setStyleSheet("font-size: 20px; font-weight: 700; color: #f2f2f5;")
        self.time_lbl = QLabel("")
        self.time_lbl.setAlignment(Qt.AlignRight)
        self.time_lbl.setStyleSheet("font-size: 11px; color: #9a9ba3;")
        timebox.addWidget(self.remain_lbl)
        timebox.addWidget(self.time_lbl)
        head.addLayout(timebox)
        outer.addLayout(head)

        # ---- waveforms ---------------------------------------------------------
        self.zoom = Waveform(name, True)
        self.zoom.setMinimumHeight(110)
        self.over = Waveform(name, False)
        for wv in (self.zoom, self.over):
            wv.seek.connect(lambda t, n=name: self.eng.seek(n, t))
            wv.marker.connect(lambda kind, t, n=name: self.dj.set_marker(n, kind, t))
        self.zoom.scratch.connect(self.scr.move)
        self.zoom.scratch_end.connect(self.scr.end)
        outer.addWidget(self.zoom, 1)
        outer.addWidget(self.over)
        self.phase = PhaseMeter()
        outer.addWidget(self.phase)

        # ---- platter + pads + tempo -------------------------------------------------
        body = QHBoxLayout()
        body.setSpacing(8)
        self.platter = JogPlatter(col, 140)
        self.platter.scratch.connect(lambda revs: self.scr.move(revs * 1.8))
        self.platter.scratch_end.connect(self.scr.end)

        pads = QGridLayout()
        pads.setSpacing(4)
        self.hc = []
        for i in range(8):
            b = QPushButton(str(i + 1))
            b.setObjectName("pad")
            b.setMinimumHeight(30)
            b.setToolTip("Hot cue: click = set / jump · right-click = delete")
            b.clicked.connect(lambda _=False, i=i: self._hotcue(i))
            b.setContextMenuPolicy(Qt.CustomContextMenu)
            b.customContextMenuRequested.connect(lambda _p, i=i: self._hotcue_delete(i))
            self.hc.append(b)
            pads.addWidget(b, i // 4, i % 4)
        # loops
        lrow = QHBoxLayout()
        lrow.setSpacing(3)
        for txt, fn, tip in (("IN", lambda: self.eng.loop_in(name), "Loop in"),
                             ("OUT", lambda: self.eng.loop_out(name), "Loop out"),
                             ("½", lambda: self.eng.loop_resize(name, 0.5), "Halve the loop"),
                             ("×2", lambda: self.eng.loop_resize(name, 2.0), "Double the loop")):
            b = _btn(txt, tip=tip, h=26)
            b.setObjectName("small")
            b.clicked.connect(fn)
            lrow.addWidget(b)
        self.loop_size = QComboBox()
        self.loop_size.addItems(["1", "2", "4", "8", "16", "32"])
        self.loop_size.setCurrentText("4")
        self.loop_size.setToolTip("Auto-loop length (beats)")
        lrow.addWidget(self.loop_size)
        self.b_loop = _btn("LOOP", True, "Auto loop on/off", h=26)
        self.b_loop.setObjectName("loop")
        self.b_loop.clicked.connect(self._toggle_loop)
        lrow.addWidget(self.b_loop)
        # beat jump / key / grid
        jrow = QHBoxLayout()
        jrow.setSpacing(3)
        b_jb = _btn("◀◀", tip="Beat jump back", h=26)
        b_jf = _btn("▶▶", tip="Beat jump forward", h=26)
        self.jump = QComboBox()
        self.jump.addItems(["1", "2", "4", "8", "16", "32"])
        self.jump.setCurrentText("4")
        self.jump.setToolTip("Beat jump size (beats)")
        b_jb.clicked.connect(lambda: self.eng.beatjump(name, -int(self.jump.currentText())))
        b_jf.clicked.connect(lambda: self.eng.beatjump(name, int(self.jump.currentText())))
        self.b_flux = _btn("FLUX", True, "Flux: after a loop the track continues where it would have been", h=26)
        self.b_flux.clicked.connect(lambda on: self.eng.set(name, flux=on))
        for w in (b_jb, self.jump, b_jf, self.b_flux):
            jrow.addWidget(w)
        krow = QHBoxLayout()
        krow.setSpacing(3)
        self.key_shift = QSpinBox()
        self.key_shift.setRange(-12, 12)
        self.key_shift.setPrefix("Key ")
        self.key_shift.setSuffix(" st")
        self.key_shift.setToolTip("Transpose in semitones (tempo unaffected)")
        self.key_shift.valueChanged.connect(lambda v: self.eng.set(name, key_shift=float(v)))
        b_ks = _btn("KEY SYNC", tip="Shift to the nearest key that is harmonic with the master", h=26)
        b_ks.clicked.connect(lambda: self.eng.key_sync(name))
        krow.addWidget(self.key_shift)
        krow.addWidget(b_ks)
        for txt, ms in (("◂", -5), ("▸", 5)):
            b = _btn(txt, w=24, tip=f"Shift the beat grid {ms:+d} ms", h=26)
            b.clicked.connect(lambda _=False, ms=ms: self._grid_shift(ms / 1000))
            krow.addWidget(b)
        b1 = _btn("1↓", w=30, tip="Set beat 1 (downbeat) at the playhead", h=26)
        b1.clicked.connect(self._grid_downbeat)
        krow.addWidget(b1)
        b_mk = _btn("✎ MARKERS", tip="Zoom in and place the IN / OUT / OUT END mix markers precisely", h=26)
        b_mk.clicked.connect(lambda: self.dj.edit_markers(name))
        krow.addWidget(b_mk)
        mid = QVBoxLayout()
        mid.setSpacing(4)
        mid.addLayout(pads)
        mid.addLayout(lrow)
        mid.addLayout(jrow)
        mid.addLayout(krow)

        tb = QVBoxLayout()
        tb.setSpacing(2)
        lab = QLabel("TEMPO")
        lab.setStyleSheet("font-size: 9px; color: #9a9ba3;")
        tb.addWidget(lab, 0, Qt.AlignHCenter)
        self.tempo = QSlider(Qt.Vertical)
        self.tempo.setRange(-1000, 1000)
        self.tempo.setInvertedAppearance(True)       # Pioneer style: down = faster
        self.tempo.valueChanged.connect(lambda v: self.eng.set(name, tempo=v / 1000.0))
        tb.addWidget(self.tempo, 1, Qt.AlignHCenter)
        self.range = QComboBox()
        self.range.addItems(["±6%", "±8%", "±16%", "±50%"])
        self.range.setCurrentIndex(1)
        self.range.setFixedWidth(72)
        self.range.currentIndexChanged.connect(
            lambda i: self.eng.set(name, tempo_range=(0.06, 0.08, 0.16, 0.5)[i]))
        tb.addWidget(self.range, 0, Qt.AlignHCenter)
        b0 = _btn("0", tip="Reset tempo", w=40, h=22)
        b0.clicked.connect(lambda: self.tempo.setValue(0))
        tb.addWidget(b0, 0, Qt.AlignHCenter)

        parts = [self.platter, mid, tb]
        if mirrored:
            parts = [tb, mid, self.platter]
        for p_ in parts:
            if isinstance(p_, QWidget):
                body.addWidget(p_, 0, Qt.AlignVCenter)
            else:
                body.addLayout(p_, 1)
        outer.addLayout(body)

        # ---- transport ---------------------------------------------------------------
        row = QHBoxLayout()
        row.setSpacing(6)
        self.b_cue = QPushButton("CUE")
        self.b_cue.setObjectName("cue")
        self.b_cue.setToolTip("Cue (hold to preview from the cue point)")
        self.b_cue.pressed.connect(lambda: self.eng.cue_press(name))
        self.b_cue.released.connect(lambda: self.eng.cue_release(name))
        self.b_play = QPushButton("▶")
        self.b_play.setObjectName("play")
        self.b_play.setCheckable(True)
        self.b_play.setToolTip("Play / pause")
        self.b_play.clicked.connect(lambda: self.eng.toggle_play(name))
        self.b_sync = QPushButton("SYNC")
        self.b_sync.setObjectName("sync")
        self.b_sync.setCheckable(True)
        self.b_sync.setToolTip("Tempo + phase lock to the master deck")
        self.b_sync.clicked.connect(lambda on: self.eng.set(name, sync=on))
        self.b_master = QPushButton("MASTER")
        self.b_master.setObjectName("toggle")
        self.b_master.setCheckable(True)
        self.b_master.clicked.connect(lambda: self.eng.set_master(name))
        self.b_key = QPushButton("KEY LOCK")
        self.b_key.setObjectName("toggle")
        self.b_key.setCheckable(True)
        self.b_key.setChecked(True)
        self.b_key.setToolTip("Key lock: tempo changes keep the original key")
        self.b_key.clicked.connect(lambda on: self.eng.set(name, keylock=on))
        for b in (self.b_cue, self.b_play, self.b_sync, self.b_master, self.b_key):
            b.setMinimumHeight(38)
            row.addWidget(b)
        outer.addLayout(row)
        self._loaded_path = None

    def _toggle_loop(self, on):
        self.eng.loop_beats(self.name, int(self.loop_size.currentText()) if on else None)

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
            self.art.setPixmap(pm)
            self.platter.art = pm
            self.platter.update()

    def refresh(self):
        d = self.eng.decks[self.name]
        if d.info.path != self._loaded_path:
            self._loaded_path = d.info.path
            self.title.setText(d.info.title or "—")
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
                lb.setText("")
            self.key_lbl.setStyleSheet("")
            return
        pct = (d.base_rate - 1) * 100
        self.bpm_lbl.setText(f"{d.bpm:.2f}")
        self.pct_lbl.setText(f"{pct:+.2f}%  ·  orig {d.grid.bpm if d.grid else 0:.1f}")
        ek = d.effective_key
        if ek >= 0:
            kc = key_color(ek) or QColor(200, 200, 200)
            shifted = ek != d.info.key
            self.key_lbl.setText(f"{camelot_str(ek)}\n{key_name(ek)}")
            self.key_lbl.setStyleSheet(f"background: {kc.darker(260).name()}; color: {kc.name()};"
                                       f" border: 1px solid {kc.name()}; border-radius: 6px;"
                                       f" font-weight: 800; font-size: 14px; padding: 2px 6px;"
                                       + (" border-style: dashed;" if shifted else ""))
            self.key_lbl.setToolTip(f"original key {camelot_str(d.info.key)} {key_name(d.info.key)}"
                                    if shifted else "")
        remain = d.length / SR - d.time
        self.remain_lbl.setText(f"-{fmt_time(remain)}")
        self.time_lbl.setText(fmt_time(d.time))
        warn = d.playing and remain < 30
        self.remain_lbl.setStyleSheet("font-size: 20px; font-weight: 700; color: "
                                      + ("#ff5050;" if warn else "#f2f2f5;"))
        for wv in (self.zoom, self.over):
            wv.pos = d.time
            wv.cue = d.cue_point / SR
            wv.loop = d.loop
            wv.hotcues = d.hotcues
            wv.grid = d.grid
            wv.mix_in = d.mix_in / SR if d.mix_in is not None else None
            wv.mix_out = d.mix_out / SR if d.mix_out is not None else None
            wv.mix_out_end = d.mix_out_end / SR if d.mix_out_end is not None else None
            wv.mix_beats = 4.0 * self.dj.automix.s.mix_bars
            wv.update()
        if not self.platter.held:
            self.platter.angle = (d.time / 1.8) * 360.0 % 360.0
        self.platter.update()
        self.phase.err = self.eng.phase_error(self.name) if d.playing else None
        self.phase.update()
        for b, on in ((self.b_play, d.playing), (self.b_sync, d.sync), (self.b_key, d.keylock),
                      (self.b_master, self.eng.master_deck == self.name), (self.b_flux, d.flux),
                      (self.b_loop, d.loop is not None)):
            _sync_check(b, on)
        self.b_play.setText("❚❚" if d.playing else "▶")
        for i, b in enumerate(self.hc):
            st = (f"background: {CUE_COLORS[i].name()}; color: #111; font-weight: 800;"
                  if d.hotcues[i] is not None else "")
            if b.styleSheet() != st:
                b.setStyleSheet(st)
        _sync_value(self.tempo, int(d.tempo * 1000))
        _sync_value(self.key_shift, int(round(d.key_shift)), 0)


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
        for key, lab, tip in (("trim", "GAIN", "Channel gain"), ("high", "HI", "EQ high (left = kill)"),
                              ("mid", "MID", "EQ mid (left = kill)"), ("low", "LOW", "EQ low (left = kill)"),
                              ("filter", "FILTER", "Left = low-pass, right = high-pass")):
            k = Knob(50, 34, tip)
            k.valueChanged.connect(self._knobs_changed)
            self.knobs[key] = k
            lbl = QLabel(lab)
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet("font-size: 9px; color: #aaa;")
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
        self.fader = QSlider(Qt.Vertical)
        self.fader.setRange(0, 1000)
        self.fader.setValue(1000)
        self.fader.setMinimumHeight(110)
        self.fader.setToolTip(f"Channel {name} volume")
        self.fader.valueChanged.connect(self._fader)
        frow.addWidget(self.meter)
        frow.addWidget(self.fader)
        lay.addLayout(frow, 1)

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


class MixerPanel(QGroupBox):
    def __init__(self, dj: DJPanel):
        super().__init__("Mixer")
        self.dj = dj
        eng = dj.engine
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 10, 4, 4)
        strips = QHBoxLayout()
        self.strips = {n: ChannelStrip(n, dj) for n in "CABD"}
        for n in "CABD":
            strips.addWidget(self.strips[n])
        lay.addLayout(strips, 1)
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        lay.addWidget(line)
        g = QGridLayout()
        g.addWidget(QLabel("Master"), 0, 0, Qt.AlignHCenter)
        self.master = Knob(90, 34, "Master volume")
        self.master.valueChanged.connect(lambda v: setattr(eng, "master_volume", v / 100))
        g.addWidget(self.master, 1, 0, Qt.AlignHCenter)
        self.master_meter = LevelMeter(10)
        self.master_meter.setMinimumHeight(40)
        g.addWidget(self.master_meter, 0, 1, 2, 1)
        g.addWidget(QLabel("Cue mix"), 0, 2, Qt.AlignHCenter)
        self.cue_mix = Knob(0, 34, "Headphones: cue ↔ master")
        self.cue_mix.valueChanged.connect(lambda v: setattr(eng, "cue_mix", v / 100))
        g.addWidget(self.cue_mix, 1, 2, Qt.AlignHCenter)
        g.addWidget(QLabel("Phones"), 0, 3, Qt.AlignHCenter)
        self.phones = Knob(80, 34, "Headphone volume")
        self.phones.valueChanged.connect(lambda v: setattr(eng, "phones_volume", v / 100))
        g.addWidget(self.phones, 1, 3, Qt.AlignHCenter)
        lay.addLayout(g)
        lay.addWidget(QLabel("Crossfader"), 0, Qt.AlignHCenter)
        self.xf = QSlider(Qt.Horizontal)
        self.xf.setRange(-100, 100)
        self.xf.valueChanged.connect(self._xf)
        lay.addWidget(self.xf)
        row = QHBoxLayout()
        b = _btn("Centre", h=22)
        b.clicked.connect(lambda: self.xf.setValue(0))
        row.addWidget(b)
        self.rec = _btn("● REC", True, "Record the master output to a WAV file", h=22)
        self.rec.setStyleSheet("QPushButton:checked { background: #c62828; color: white; }")
        self.rec.clicked.connect(lambda _on: dj.toggle_record())
        row.addWidget(self.rec)
        lay.addLayout(row)
        self.rec_lbl = QLabel("")
        self.rec_lbl.setAlignment(Qt.AlignCenter)
        lay.addWidget(self.rec_lbl)

    def _xf(self, v):
        am = self.dj.automix
        if am.tr is not None and am.tr.started:
            am.user_took_over()
        self.dj.engine.crossfader = v / 100.0

    def refresh(self, show_cd: bool):
        eng = self.dj.engine
        for n, s in self.strips.items():
            s.setVisible(show_cd or n in "AB")
            if s.isVisible():
                s.refresh()
        self.master_meter.value = eng.peak_level
        self.master_meter.update()
        _sync_value(self.xf, int(eng.crossfader * 100), 1)
        r = eng.recorder
        _sync_check(self.rec, r is not None)
        self.rec_lbl.setText(f"REC {fmt_time(r.seconds)}" if r is not None else "")


class FXPanel(QGroupBox):
    def __init__(self, dj: DJPanel, unit: int):
        super().__init__(f"FX {unit + 1}")
        self.u = dj.engine.fx_units[unit]
        lay = QHBoxLayout(self)
        lay.setContentsMargins(6, 10, 6, 4)
        self.on = _btn("ON", True, w=40)
        self.on.toggled.connect(lambda v: setattr(self.u, "on", v))
        lay.addWidget(self.on)
        self.kind = QComboBox()
        self.kind.addItems(FX_TYPES)
        self.kind.setCurrentText(self.u.kind)
        self.kind.currentTextChanged.connect(lambda t: setattr(self.u, "kind", t))
        lay.addWidget(self.kind)
        lay.addWidget(QLabel("D/W"))
        self.wet = Knob(50, 32, "Dry / wet")
        self.wet.valueChanged.connect(lambda v: setattr(self.u, "wet", v / 100))
        lay.addWidget(self.wet)
        lay.addWidget(QLabel("Amt"))
        self.amt = Knob(50, 32, "Effect amount (feedback / size / depth)")
        self.amt.valueChanged.connect(lambda v: setattr(self.u, "amount", v / 100))
        lay.addWidget(self.amt)
        self.div = QComboBox()
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

class AudioSettings(QDialog):
    def __init__(self, settings: QSettings, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Audio output")
        self.s = settings
        form = QFormLayout(self)
        self.dev = QComboBox()
        self.dev.addItem("System default", None)
        for i, name, _ch in list_output_devices():
            self.dev.addItem(name, i)
        ix = self.dev.findText(settings.value("dj_device_name", ""))
        if ix >= 0:
            self.dev.setCurrentIndex(ix)
        form.addRow("Output device", self.dev)
        self.master = QComboBox()
        self.master.addItems(["1-2", "3-4"])
        self.master.setCurrentText(settings.value("dj_master", "1-2"))
        form.addRow("Master outputs", self.master)
        self.phones = QComboBox()
        self.phones.addItems(["off (master only)", "3-4", "1-2"])
        self.phones.setCurrentText(settings.value("dj_phones", "3-4"))
        form.addRow("Headphone outputs", self.phones)
        self.block = QComboBox()
        self.block.addItems(["256", "512", "1024", "2048"])
        self.block.setCurrentText(str(settings.value("dj_block", "512")))
        form.addRow("Buffer (samples)", self.block)
        self.rec_dir = QPushButton(settings.value("rec_dir", os.path.join(os.path.expanduser("~"), "Music",
                                                                          "CTA Recordings")))
        self.rec_dir.clicked.connect(self._pick_dir)
        form.addRow("Recordings folder", self.rec_dir)
        self.port = QSpinBox()
        self.port.setRange(1024, 65535)
        self.port.setValue(int(settings.value("stream_port", 8000)))
        form.addRow("Phone stream port", self.port)
        form.addRow(QLabel("<small>DDJ-T1: choose its sound card (WASAPI), master 1-2, headphones 3-4.<br>"
                           "Use a bigger buffer if you hear crackles.</small>"))
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        form.addRow(bb)

    def _pick_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Recordings folder", self.rec_dir.text())
        if d:
            self.rec_dir.setText(d)

    def accept(self):
        self.s.setValue("dj_device_name", self.dev.currentText() if self.dev.currentData() is not None else "")
        self.s.setValue("dj_master", self.master.currentText())
        self.s.setValue("dj_phones", self.phones.currentText())
        self.s.setValue("dj_block", self.block.currentText())
        self.s.setValue("rec_dir", self.rec_dir.text())
        self.s.setValue("stream_port", self.port.value())
        super().accept()


class LearnWizard(QDialog):
    got = Signal(object)

    def __init__(self, controller: Controller, decks=("A", "B"), parent=None):
        super().__init__(parent)
        self.setWindowTitle("Controller setup")
        self.ctl = controller
        self.decks = decks
        self.result = {(b.action, b.deck): b for b in controller.bindings}
        self.learner = Learner()
        lay = QVBoxLayout(self)
        self.adv = QCheckBox("Include hot cues, loops, beat jump, filter and FX controls")
        self.adv.setChecked(True)
        self.adv.toggled.connect(self._rebuild)
        lay.addWidget(self.adv)
        self.prog = QProgressBar()
        lay.addWidget(self.prog)
        self.prompt = QLabel()
        self.prompt.setStyleSheet("font-size: 16px; padding: 12px;")
        self.prompt.setWordWrap(True)
        lay.addWidget(self.prompt)
        self.heard = QLabel("")
        lay.addWidget(self.heard)
        row = QHBoxLayout()
        for txt, fn in (("Back", self._back), ("Skip", self._next)):
            b = QPushButton(txt)
            b.clicked.connect(fn)
            row.addWidget(b)
        row.addStretch()
        b = QPushButton("Finish && save")
        b.clicked.connect(self._finish)
        row.addWidget(b)
        lay.addLayout(row)
        self.got.connect(self._on_event)
        controller.learn_cb = lambda ev: self.got.emit(ev)
        self.resize(560, 240)
        self._rebuild()

    def _rebuild(self, *_):
        self.steps = learn_steps(self.decks, self.adv.isChecked())
        self.i = 0
        self.prog.setRange(0, len(self.steps))
        self._show()

    def _show(self):
        if self.i >= len(self.steps):
            self.prompt.setText("All done - click <b>Finish & save</b>.")
            self.prog.setValue(len(self.steps))
            return
        action, deck, text = self.steps[self.i]
        if action == "tempo":
            extra = "Move it slowly from the TOP to the BOTTOM."
        elif action in ("jog", "browse"):
            extra = "Turn it a little clockwise."
        elif action in ("volume", "eq_high", "eq_mid", "eq_low", "gain", "crossfader", "master_volume",
                        "cue_mix", "phones_volume", "filter", "fx1_wet", "fx1_amount", "fx2_wet", "fx2_amount"):
            extra = "Move it a little."
        else:
            extra = "Press it."
        have = self.result.get((action, deck))
        cur = f"<br><small>currently: {have.type} ch{have.ch + 1} #{have.num}</small>" if have else ""
        self.prompt.setText(f"<b>{text}</b><br><small>{extra}</small>{cur}")
        self.prog.setValue(self.i)
        self.learner = Learner()

    def _on_event(self, ev):
        self.heard.setText(f"heard: {ev.type} ch{ev.ch + 1} #{ev.num} = {ev.value}")
        if self.i >= len(self.steps):
            return
        action, deck, _ = self.steps[self.i]
        b = self.learner.offer(action, ev)
        if b is not None:
            b.deck = deck
            self.result[(action, deck)] = b
            QTimer.singleShot(350, self._next)

    def _next(self):
        self.i = min(self.i + 1, len(self.steps))
        self._show()

    def _back(self):
        self.i = max(0, self.i - 1)
        self._show()

    def _finish(self):
        self.ctl.set_bindings(list(self.result.values()), learned=True)
        self.accept()

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
        name = f"CTA history {time.strftime('%Y-%m-%d')}"
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

    def __init__(self, main_window):
        super().__init__()
        self.main = main_window
        self.settings = main_window.settings
        self.engine = Engine(block=int(self.settings.value("dj_block", 512)))
        self.automix = AutoMix(self.engine, self._automix_load, self._automix_settings())
        from ..dj.learn import MixLearner

        self.learner = MixLearner(main_window.db, self.automix)
        self.automix.learner = self.learner
        self.controller: Controller | None = None
        self.session = time.strftime("%Y-%m-%d ") + uuid.uuid4().hex[:6]
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
        self.quant = QCheckBox("Quantize")
        self.quant.setChecked(True)
        self.quant.setToolTip("Snap cues, loops and jumps to the beat grid")
        self.quant.toggled.connect(lambda on: setattr(self.engine, "quantize", on))
        self.show_cd = QCheckBox("Decks C/D")
        self.show_cd.toggled.connect(self._toggle_cd)
        b_hist = QPushButton("History…")
        b_hist.clicked.connect(lambda: HistoryDialog(self).exec())
        self.b_stream = QPushButton("📱 Phone remote")
        self.b_stream.setCheckable(True)
        self.b_stream.setToolTip("Mix server: control the mix and listen to it from your phone (same Wi-Fi)")
        self.b_stream.toggled.connect(self.toggle_stream)
        self.stream_lbl = QLabel("")
        self.stream_lbl.setTextFormat(Qt.RichText)
        self.stream_lbl.setTextInteractionFlags(Qt.TextBrowserInteraction)
        self.stream_lbl.linkActivated.connect(lambda _l: self.show_remote_info())
        self.stream = None
        for w in (self.audio_lbl, b_audio, QLabel("  MIDI:"), self.midi_combo, b_midi, b_learn):
            bar.addWidget(w)
        bar.addStretch()
        for w in (self.stream_lbl, self.b_stream, self.quant, self.show_cd, b_hist):
            bar.addWidget(w)
        outer.addLayout(bar)

        grid = QGridLayout()
        grid.setSpacing(4)
        self.decks = {n: DeckPanel(n, self, mirrored=n in "BD") for n in "ABCD"}
        self.mixer = MixerPanel(self)
        grid.addWidget(self.decks["C"], 0, 0)
        grid.addWidget(self.decks["D"], 0, 2)
        grid.addWidget(self.decks["A"], 1, 0)
        grid.addWidget(self.mixer, 0, 1, 2, 1)
        grid.addWidget(self.decks["B"], 1, 2)
        grid.setColumnStretch(0, 5)
        grid.setColumnStretch(2, 5)
        outer.addLayout(grid, 1)

        bottom = QHBoxLayout()
        self.fx = [FXPanel(self, 0), FXPanel(self, 1)]
        for f in self.fx:
            bottom.addWidget(f)
        bottom.addWidget(self._automix_panel(), 1)
        outer.addLayout(bottom)
        self._toggle_cd(False)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(33)
        self._refresh_midi_ports()
        QTimer.singleShot(200, self.start_audio)
        QTimer.singleShot(400, self._auto_connect_midi)

    def _automix_panel(self) -> QWidget:
        box = QGroupBox("Auto-mix / Assist")
        lay = QHBoxLayout(box)
        lay.setContentsMargins(6, 10, 6, 4)
        col = QVBoxLayout()
        self.b_auto = _btn("▶ Auto-mix", True, h=28)
        self.b_auto.setStyleSheet("font-weight: bold;")
        self.b_auto.clicked.connect(self.toggle_automix)
        col.addWidget(self.b_auto)
        b_tr = _btn("Auto transition now",
                    tip="Blend from the playing deck into the other loaded deck, from the next bar")
        b_tr.clicked.connect(lambda: self.automix.transition_now())
        col.addWidget(b_tr)
        b_q = _btn("Queue = setlist")
        b_q.clicked.connect(self.queue_from_setlist)
        col.addWidget(b_q)
        self.b_pause = _btn("⏸ Pause mix", True, tip="Fade the whole mix out and hold it (also from the phone)")
        self.b_pause.clicked.connect(lambda on: self.engine.set_paused(on))
        col.addWidget(self.b_pause)
        lay.addLayout(col)
        form = QFormLayout()
        self.auto_len = QCheckBox("Auto")
        self.auto_len.setToolTip("Auto: the blend length follows the music (up to the number of bars set) and "
                                 "never hangs on dead air.\nOff: always blend over exactly that many bars "
                                 "(the outgoing track is looped if it ends sooner).")
        self.auto_len.setChecked(self.settings.value("dj_auto_len", "true") == "true")
        self.auto_len.toggled.connect(self._update_am_settings)
        self.mix_bars = QSpinBox()
        self.mix_bars.setRange(1, 128)
        self.mix_bars.setToolTip("Blend length in bars (the longest blend when Auto is on)")
        self.mix_bars.setValue(int(self.settings.value("dj_mix_bars", 16)))
        self.mix_bars.valueChanged.connect(self._update_am_settings)
        brow = QHBoxLayout()
        brow.setSpacing(2)
        brow.addWidget(self.auto_len)
        brow.addWidget(self.mix_bars)
        for n in (4, 8, 16, 32, 64, 128):
            b = _btn(str(n), w=34, h=22, tip=f"Blend over {n} bars")
            b.clicked.connect(lambda _=False, n=n: self.mix_bars.setValue(n))
            brow.addWidget(b)
        form.addRow("Blend (bars)", brow)
        self.glide = QCheckBox("Glide tempo to each track's BPM")
        self.glide.setChecked(self.settings.value("dj_glide", "true") == "true")
        self.glide.toggled.connect(self._update_am_settings)
        form.addRow(self.glide)
        self.auto_suggest = QCheckBox("Auto-load best next track")
        self.auto_suggest.setChecked(self.settings.value("dj_auto_suggest", "true") == "true")
        form.addRow(self.auto_suggest)
        lrow = QHBoxLayout()
        self.learn_chk = QCheckBox("Learn from my mixes")
        self.learn_chk.setToolTip("Every transition you mix by hand teaches auto-mix where and how long you blend")
        self.learn_chk.setChecked(self.settings.value("dj_learn", "true") == "true")
        self.learn_chk.toggled.connect(self._learn_toggled)
        self.learn_lbl = QLabel("")
        self.learn_lbl.setStyleSheet("color: #9a9ba3;")
        b_forget = _btn("Forget", h=22, tip="Forget everything learned from your mixes")
        b_forget.clicked.connect(self._forget_learning)
        lrow.addWidget(self.learn_chk)
        lrow.addWidget(self.learn_lbl, 1)
        lrow.addWidget(b_forget)
        form.addRow(lrow)
        lay.addLayout(form)
        col = QVBoxLayout()
        self.am_status = QLabel("Auto-mix off")
        self.am_status.setStyleSheet("font-weight: bold;")
        col.addWidget(self.am_status)
        qh = QHBoxLayout()
        qh.addWidget(QLabel("Up next"))
        qh.addStretch()
        b_clear = _btn("Clear queue", h=22, tip="Remove every track from the auto-mix queue")
        b_clear.clicked.connect(self.clear_queue)
        qh.addWidget(b_clear)
        col.addLayout(qh)
        self.queue_list = QueueList()
        self.queue_list.setMaximumHeight(150)
        self.queue_list.removeRequested.connect(self.remove_from_queue)
        self.queue_list.reordered.connect(self._queue_reordered)
        col.addWidget(self.queue_list)
        lay.addLayout(col, 1)
        return box

    def _toggle_cd(self, on):
        self.decks["C"].setVisible(on)
        self.decks["D"].setVisible(on)

    # ---- audio ---------------------------------------------------------------
    def start_audio(self, quiet: bool = False):
        s = self.settings
        name = s.value("dj_device_name", "")
        dev = None
        try:
            devices = list_output_devices()
            for i, n, _ch in devices:
                if n == name:
                    dev = i
            if dev is None:
                for i, n, ch in devices:                 # prefer the DDJ-T1 sound card
                    if "DDJ-T1" in n.upper() and ch >= 4:
                        dev = i
                        break
            mc = (2, 3) if s.value("dj_master", "1-2") == "3-4" else (0, 1)
            ph = s.value("dj_phones", "3-4")
            pc = None if ph.startswith("off") else ((2, 3) if ph == "3-4" else (0, 1))
            if pc == mc:
                pc = None
            self.engine.start(dev, int(s.value("dj_block", 512)), mc, pc)
            phones = ("headphones " + ("3-4" if self.engine.phones_chans == (2, 3) else "1-2")
                      if self.engine.phones_chans else "no separate headphones")
            self.audio_lbl.setText(f"Audio: {self.engine.out_channels} ch, {self.engine.latency_ms:.0f} ms, {phones}")
        except Exception as e:  # noqa: BLE001
            log.exception("audio start failed")
            # keep the mix running for the phone stream / recording
            self.engine.start_virtual()
            self.audio_lbl.setText("Audio: no output device (phone stream only) - click Audio…")
            if self.main.isVisible() and not quiet:
                self._error.emit(f"Could not open the audio device:\n{e}\n\n"
                                 "The mix still runs for the phone remote and recordings.")

    def audio_settings(self):
        if AudioSettings(self.settings, self).exec():
            self.engine.stop()
            self.engine.block = int(self.settings.value("dj_block", 512))
            self.start_audio()

    def toggle_record(self):
        eng = self.engine
        if eng.recorder is not None:
            r = eng.recorder
            eng.recorder = None
            r.close()
            self.main.status_label.setText(f"Recording saved: {r.path}")
            return
        folder = self.settings.value("rec_dir", os.path.join(os.path.expanduser("~"), "Music", "CTA Recordings"))
        try:
            os.makedirs(folder, exist_ok=True)
            path = os.path.join(folder, f"CTA mix {time.strftime('%Y-%m-%d %H%M%S')}.wav")
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
                             "Another program (or another copy of CTA) may be using it - change the port under Audio….")
            return
        _sync_check(self.b_stream, True)
        self.stream_lbl.setText(f"Phone: <a href='#' style='color:#6fb3ff'>{self.stream.urls[0]}</a>")
        self.stream_lbl.setToolTip("Click for the QR code and PIN. Windows may ask to allow CTA through the "
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
        try:
            if not self.controller.open(port):
                raise RuntimeError("port not found")
        except Exception as e:  # noqa: BLE001
            self.controller = None
            self._error.emit(f"Could not open {port}:\n{e}")
            return
        self.main.status_label.setText(f"Controller connected: {port}")
        if not self.controller.learned:
            r = QMessageBox.question(
                self, "Controller setup",
                f"{port} is connected with a basic mapping (play, cue, jog, tempo, load, browse).\n\n"
                "Run the guided setup now so every fader, knob and button works exactly as on your unit?")
            if r == QMessageBox.Yes:
                self.learn()

    def learn(self):
        if self.controller is None:
            self.connect_midi()
            if self.controller is None:
                return
        decks = ("A", "B", "C", "D") if self.show_cd.isChecked() else ("A", "B")
        LearnWizard(self.controller, decks, self).exec()

    # controller callbacks (MIDI thread -> GUI thread via signals)
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
                lt = load_track(path, bpm, info, self.main.db)
                # never replaces a deck that started playing meanwhile
                self.engine.load_track(deck, lt, only_if_stopped=True)
            except Exception as e:  # noqa: BLE001
                log.exception("load failed")
                self._error.emit(f"Could not load {path}:\n{e}")
        threading.Thread(target=job, daemon=True, name=f"load-{deck}").start()

    def load_selected(self, deck: str):
        idx = self.main.current
        if idx is None:
            self.main.status_label.setText("Select a track in the library first")
            return
        self.load_index(deck, idx)

    def _automix_load(self, idx):
        path, bpm, info = self._track_info(idx)
        return load_track(path, bpm, info, self.main.db)

    # ---- auto-mix --------------------------------------------------------------------
    def _automix_settings(self) -> AutoMixSettings:
        return AutoMixSettings(mix_bars=int(self.settings.value("dj_mix_bars", 16)),
                               auto_length=self.settings.value("dj_auto_len", "true") == "true",
                               glide=self.settings.value("dj_glide", "true") == "true")

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
        if QMessageBox.question(self, "Forget learning", "Forget every transition learned from your mixes?") \
                == QMessageBox.Yes:
            self.learner.forget()

    def queue_from_setlist(self):
        sl = list(self.main.setlist)
        if not sl:
            QMessageBox.information(self, "Auto-mix", "Build a setlist in the Setlist Builder first "
                                    "(or just start auto-mix: it picks the best next tracks itself).")
            return
        self.automix.set_queue(sl)
        self.set_seed = sl[0]
        self._show_queue()

    def _show_queue(self):
        from PySide6.QtWidgets import QListWidgetItem

        self.queue_list.clear()
        lib = self.main.lib
        if lib is None:
            return
        for i in list(self.automix.queue):
            if not isinstance(i, (int, np.integer)) or not 0 <= int(i) < lib.n:
                continue
            t = lib.track(int(i))
            it = QListWidgetItem(f"{t.bpm:6.1f}  {camelot_str(t.key):>3}  {t.artist} - {t.title}")
            it.setData(Qt.UserRole, int(i))
            it.setToolTip("Drag to reorder · drag out of the list, click ✕ or press Delete to remove")
            self.queue_list.addItem(it)

    def clear_queue(self):
        with self.automix._lock:
            self.automix.queue.clear()
        self._show_queue()
        self.main.status_label.setText("Queue cleared (auto-mix picks the next track itself when it runs dry)"
                                        if self.auto_suggest.isChecked() else "Queue cleared")

    def remove_from_queue(self, pos: int):
        with self.automix._lock:
            q = list(self.automix.queue)
            if 0 <= pos < len(q):
                q.pop(pos)
                self.automix.queue.clear()
                self.automix.queue.extend(q)
        self._show_queue()

    def _queue_reordered(self):
        order = [self.queue_list.item(r).data(Qt.UserRole) for r in range(self.queue_list.count())]
        with self.automix._lock:
            self.automix.queue.clear()
            self.automix.queue.extend(int(i) for i in order if i is not None)

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

    def manual_touch(self, deck):
        am = self.automix
        if am.tr is not None and am.tr.started and deck in (am.tr.in_deck, am.tr.out_deck):
            am.user_took_over()

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
                except Exception:  # noqa: BLE001
                    log.exception("history failed")
                self._logged[n] = (d.info.path, -1.0)

    # ---- refresh -------------------------------------------------------------------------
    def _watchdog(self):
        """The sound card stopped asking for audio (device lost, driver
        hiccup, sleep): reopen it. The decks keep their place."""
        eng = self.engine
        now = time.monotonic()
        while eng.deck_errors:
            msg = eng.deck_errors.pop(0)
            self.main.status_label.setText(msg + " - see the log")
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

    def _tick(self):
        self._watchdog()
        show_cd = self.show_cd.isChecked()
        for p in self.decks.values():
            if p.isVisible():
                p.refresh()
        self.mixer.refresh(show_cd)
        for f in self.fx:
            f.refresh()
        am = self.automix
        self.am_status.setText(("⏸ PAUSED · " if self.engine.paused else "") + am.status)
        _sync_check(self.b_pause, self.engine.paused)
        _sync_check(self.b_auto, am.active)
        self.b_auto.setText("■ Stop auto-mix" if am.active else "▶ Auto-mix")
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
        self.learn_lbl.setText(f"{n} transition{'s' if n != 1 else ''} learned" if n else "nothing learned yet")
        if self.controller is not None:
            self.controller.tick()
            self.controller.update_leds()

    def shutdown(self):
        self.timer.stop()
        if self.stream is not None:
            self.stream.stop()
            self.stream = None
        self.automix.stop()
        if self.engine.recorder is not None:
            self.toggle_record()
        self.engine.stop()
        if self.controller:
            self.controller.close()

    def closeEvent(self, ev):
        self.shutdown()
        super().closeEvent(ev)


# backwards-compatible name
DJWindow = DJPanel
