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
from PySide6.QtCore import QRectF, QSettings, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen
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
from .models import TRACK_MIME

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
        self.mix_beats = 64.0      # blend length shown after the outro marker
        self._over = None          # cached (width, heights, colours)
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._menu)

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
        m.addAction(f"Set cue point here  [{stamp}]", lambda: self.marker.emit("cue", t))
        if self.mix_in is not None or self.mix_out is not None:
            m.addSeparator()
            m.addAction("Clear mix-in / mix-out markers", lambda: self.marker.emit("clear", 0.0))
        m.exec(self.mapToGlobal(pos))

    def mousePressEvent(self, ev):
        if self.peaks is None or ev.button() != Qt.LeftButton:
            return
        x = ev.position().x() / max(self.width(), 1)
        self.seek.emit(self.pos + (x - 0.5) * self.span if self.zoom else x * self.duration)

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
            x1 = self._x(self.mix_out + self.mix_beats * self.grid.beat_len, w)
            p.fillRect(QRectF(x0, 0, max(x1 - x0, 2), h), QColor(229, 57, 53, 45))
        for t, label, colr in ((self.mix_in, "IN", QColor(67, 160, 71)), (self.mix_out, "OUT", QColor(229, 57, 53))):
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

class DeckPanel(QGroupBox):
    def __init__(self, name: str, dj: DJPanel, mirrored: bool = False):
        super().__init__(f"Deck {name}")
        self.name = name
        self.dj = dj
        self.eng = dj.engine
        self.setAcceptDrops(True)
        col = DECK_COLORS[name]
        self.setStyleSheet(f"QGroupBox {{ border: 1px solid {col.name()}; border-radius: 6px; margin-top: 8px; }}"
                           f"QGroupBox::title {{ color: {col.name()}; left: 8px; }}")
        outer = QHBoxLayout(self)
        outer.setContentsMargins(6, 10, 6, 6)
        main = QVBoxLayout()
        main.setSpacing(4)

        top = QHBoxLayout()
        self.title = QLabel("—")
        f = QFont()
        f.setPointSize(11)
        f.setBold(True)
        self.title.setFont(f)
        top.addWidget(self.title, 1)
        self.bpm_lbl = QLabel("")
        self.bpm_lbl.setStyleSheet("font-size: 20px; font-weight: bold;")
        top.addWidget(self.bpm_lbl)
        main.addLayout(top)
        info = QHBoxLayout()
        self.key_lbl = QLabel("")
        self.key_lbl.setStyleSheet("font-weight: bold;")
        self.time_lbl = QLabel("")
        info.addWidget(self.key_lbl)
        info.addStretch()
        info.addWidget(self.time_lbl)
        main.addLayout(info)

        self.zoom = Waveform(name, True)
        self.over = Waveform(name, False)
        for wv in (self.zoom, self.over):
            wv.seek.connect(lambda t, n=name: self.eng.seek(n, t))
            wv.marker.connect(lambda kind, t, n=name: self.dj.set_marker(n, kind, t))
        main.addWidget(self.zoom, 1)
        main.addWidget(self.over)
        self.phase = PhaseMeter()
        main.addWidget(self.phase)

        row = QHBoxLayout()
        self.b_play = _btn("▶", True, "Play / pause", h=32)
        self.b_play.clicked.connect(lambda: self.eng.toggle_play(name))
        self.b_cue = _btn("CUE", tip="Cue (hold to preview from the cue point)", h=32)
        self.b_cue.pressed.connect(lambda: self.eng.cue_press(name))
        self.b_cue.released.connect(lambda: self.eng.cue_release(name))
        self.b_sync = _btn("SYNC", True, "Tempo + phase lock to the master deck", h=32)
        self.b_sync.clicked.connect(lambda on: self.eng.set(name, sync=on))
        self.b_master = _btn("MASTER", True, "Make this the tempo master", h=32)
        self.b_master.clicked.connect(lambda: self.eng.set_master(name))
        self.b_key = _btn("KEYLOCK", True, "Key lock: tempo changes keep the original key", h=32)
        self.b_key.setChecked(True)
        self.b_key.clicked.connect(lambda on: self.eng.set(name, keylock=on))
        for b in (self.b_play, self.b_cue, self.b_sync, self.b_master, self.b_key):
            row.addWidget(b)
        main.addLayout(row)

        row = QHBoxLayout()
        row.setSpacing(2)
        self.hc = []
        for i in range(8):
            b = _btn(str(i + 1), tip="Hot cue: click = set / jump, right-click = delete", h=24)
            b.clicked.connect(lambda _=False, i=i: self._hotcue(i))
            b.setContextMenuPolicy(Qt.CustomContextMenu)
            b.customContextMenuRequested.connect(lambda _p, i=i: self._hotcue_delete(i))
            self.hc.append(b)
            row.addWidget(b)
        main.addLayout(row)

        row = QHBoxLayout()
        row.setSpacing(2)
        b_in = _btn("IN", w=34)
        b_in.clicked.connect(lambda: self.eng.loop_in(name))
        b_out = _btn("OUT", w=40)
        b_out.clicked.connect(lambda: self.eng.loop_out(name))
        b_half = _btn("½", w=26, tip="Halve the loop")
        b_half.clicked.connect(lambda: self.eng.loop_resize(name, 0.5))
        b_dbl = _btn("×2", w=30, tip="Double the loop")
        b_dbl.clicked.connect(lambda: self.eng.loop_resize(name, 2.0))
        for b in (b_in, b_out, b_half, b_dbl):
            row.addWidget(b)
        for n in (1, 2, 4, 8, 16):
            b = _btn(str(n), w=28, tip=f"Loop {n} beats")
            b.clicked.connect(lambda _=False, n=n: self.eng.loop_beats(name, n))
            row.addWidget(b)
        self.b_exit = _btn("EXIT", w=40, tip="Leave the loop")
        self.b_exit.clicked.connect(lambda: self.eng.loop_beats(name, None))
        row.addWidget(self.b_exit)
        self.b_flux = _btn("FLUX", True, "Flux: after a loop the track continues where it would have been", w=44)
        self.b_flux.clicked.connect(lambda on: self.eng.set(name, flux=on))
        row.addWidget(self.b_flux)
        row.addStretch()
        main.addLayout(row)

        row = QHBoxLayout()
        row.setSpacing(2)
        b_jb = _btn("◀◀", w=34, tip="Beat jump back")
        b_jf = _btn("▶▶", w=34, tip="Beat jump forward")
        self.jump = QComboBox()
        self.jump.addItems(["1", "2", "4", "8", "16", "32"])
        self.jump.setCurrentText("4")
        self.jump.setToolTip("Beat jump size (beats)")
        b_jb.clicked.connect(lambda: self.eng.beatjump(name, -int(self.jump.currentText())))
        b_jf.clicked.connect(lambda: self.eng.beatjump(name, int(self.jump.currentText())))
        row.addWidget(b_jb)
        row.addWidget(self.jump)
        row.addWidget(b_jf)
        row.addSpacing(10)
        row.addWidget(QLabel("Key"))
        self.key_shift = QSpinBox()
        self.key_shift.setRange(-12, 12)
        self.key_shift.setSuffix(" st")
        self.key_shift.setToolTip("Transpose in semitones (tempo unaffected)")
        self.key_shift.valueChanged.connect(lambda v: self.eng.set(name, key_shift=float(v)))
        row.addWidget(self.key_shift)
        b_ks = _btn("KEY SYNC", tip="Shift to the nearest key that is harmonic with the master", w=70)
        b_ks.clicked.connect(lambda: self.eng.key_sync(name))
        row.addWidget(b_ks)
        row.addSpacing(10)
        row.addWidget(QLabel("Grid"))
        for txt, ms in (("◂", -5), ("▸", 5)):
            b = _btn(txt, w=22, tip=f"Shift the beat grid {ms:+d} ms")
            b.clicked.connect(lambda _=False, ms=ms: self._grid_shift(ms / 1000))
            row.addWidget(b)
        b1 = _btn("1↓", w=28, tip="Set beat 1 (downbeat) at the playhead")
        b1.clicked.connect(self._grid_downbeat)
        row.addWidget(b1)
        row.addStretch()
        main.addLayout(row)
        outer.addLayout(main, 1)

        tb = QVBoxLayout()
        tb.addWidget(QLabel("Tempo"), 0, Qt.AlignHCenter)
        self.tempo = QSlider(Qt.Vertical)
        self.tempo.setRange(-1000, 1000)
        self.tempo.setInvertedAppearance(True)       # Pioneer style: down = faster
        self.tempo.valueChanged.connect(lambda v: self.eng.set(name, tempo=v / 1000.0))
        tb.addWidget(self.tempo, 1, Qt.AlignHCenter)
        self.range = QComboBox()
        self.range.addItems(["±6%", "±8%", "±16%", "±50%"])
        self.range.setCurrentIndex(1)
        self.range.currentIndexChanged.connect(
            lambda i: self.eng.set(name, tempo_range=(0.06, 0.08, 0.16, 0.5)[i]))
        tb.addWidget(self.range)
        b0 = _btn("0", tip="Reset tempo", w=50)
        b0.clicked.connect(lambda: self.tempo.setValue(0))
        tb.addWidget(b0, 0, Qt.AlignHCenter)
        if mirrored:
            outer.insertLayout(0, tb)
        else:
            outer.addLayout(tb)
        self._loaded_path = None

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

    def refresh(self):
        d = self.eng.decks[self.name]
        if d.info.path != self._loaded_path:
            self._loaded_path = d.info.path
            self.title.setText(f"{d.info.artist} - {d.info.title}" if d.info.artist else (d.info.title or "—"))
            bands = getattr(d, "bands", None)
            for wv in (self.zoom, self.over):
                wv.set_track(d.peaks, bands, d.length / SR, d.grid)
            _sync_value(self.key_shift, int(d.key_shift), 0)
        if not d.loaded:
            for lb in (self.bpm_lbl, self.key_lbl, self.time_lbl):
                lb.setText("")
            return
        orig = d.grid.bpm if d.grid else 0
        pct = (d.base_rate - 1) * 100
        self.bpm_lbl.setText(f"{d.bpm:.2f}")
        self.bpm_lbl.setToolTip(f"original {orig:.2f} BPM, {pct:+.2f}%")
        ek = d.effective_key
        if ek >= 0:
            shifted = ek != d.info.key
            txt = f"{camelot_str(ek)} {key_name(ek)}" + (f"  (orig {camelot_str(d.info.key)})" if shifted else "")
            self.key_lbl.setText(f"{txt}   {pct:+.1f}%")
            self.key_lbl.setStyleSheet("font-weight: bold; color: #ffb74d;" if shifted else "font-weight: bold;")
        else:
            self.key_lbl.setText(f"{pct:+.1f}%")
        remain = d.length / SR - d.time
        self.time_lbl.setText(f"{fmt_time(d.time)}  /  -{fmt_time(remain)}")
        self.time_lbl.setStyleSheet("color: #ff5050; font-weight: bold;" if d.playing and remain < 30 else "")
        for wv in (self.zoom, self.over):
            wv.pos = d.time
            wv.cue = d.cue_point / SR
            wv.loop = d.loop
            wv.hotcues = d.hotcues
            wv.grid = d.grid
            wv.mix_in = d.mix_in / SR if d.mix_in is not None else None
            wv.mix_out = d.mix_out / SR if d.mix_out is not None else None
            wv.mix_beats = 4.0 * self.dj.automix.s.mix_bars
            wv.update()
        self.phase.err = self.eng.phase_error(self.name) if d.playing else None
        self.phase.update()
        for b, on in ((self.b_play, d.playing), (self.b_sync, d.sync), (self.b_key, d.keylock),
                      (self.b_master, self.eng.master_deck == self.name), (self.b_flux, d.flux)):
            _sync_check(b, on)
        self.b_play.setText("❚❚" if d.playing else "▶")
        self.b_exit.setStyleSheet("background: #2e7d32;" if d.loop else "")
        for i, b in enumerate(self.hc):
            st = (f"background: {CUE_COLORS[i].name()}; color: black; font-weight: bold;"
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
        self.fx1 = _btn("1", True, "Route this channel through FX unit 1", w=22, h=20)
        self.fx2 = _btn("2", True, "Route this channel through FX unit 2", w=22, h=20)
        for b in (self.fx1, self.fx2):
            b.toggled.connect(self._fx_changed)
            fxrow.addWidget(b)
        lay.addWidget(QLabel("FX"), 0, Qt.AlignHCenter)
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
        self.controller: Controller | None = None
        self.session = time.strftime("%Y-%m-%d ") + uuid.uuid4().hex[:6]
        self._logged: dict[str, tuple[str, float]] = {}
        self._played: set[int] = set()
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
        self.b_stream = QPushButton("📡 Stream to phone")
        self.b_stream.setCheckable(True)
        self.b_stream.setToolTip("Serve the live mix on your network: open the shown address on your phone")
        self.b_stream.toggled.connect(self.toggle_stream)
        self.stream_lbl = QLabel("")
        self.stream_lbl.setTextFormat(Qt.RichText)
        self.stream_lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
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
        lay.addLayout(col)
        form = QFormLayout()
        self.mix_bars = QSpinBox()
        self.mix_bars.setRange(1, 64)
        self.mix_bars.setToolTip("How many bars the auto-mix blend lasts (both tracks playing together)")
        self.mix_bars.setValue(int(self.settings.value("dj_mix_bars", 16)))
        self.mix_bars.valueChanged.connect(self._update_am_settings)
        brow = QHBoxLayout()
        brow.setSpacing(2)
        brow.addWidget(self.mix_bars)
        for n in (4, 8, 16, 32):
            b = _btn(str(n), w=30, h=22, tip=f"Blend over {n} bars")
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
        lay.addLayout(form)
        col = QVBoxLayout()
        self.am_status = QLabel("Auto-mix off")
        self.am_status.setStyleSheet("font-weight: bold;")
        col.addWidget(self.am_status)
        self.queue_list = QListWidget()
        self.queue_list.setMaximumHeight(70)
        col.addWidget(self.queue_list)
        lay.addLayout(col, 1)
        return box

    def _toggle_cd(self, on):
        self.decks["C"].setVisible(on)
        self.decks["D"].setVisible(on)

    # ---- audio ---------------------------------------------------------------
    def start_audio(self):
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
            self.audio_lbl.setText("Audio: failed - click Audio…")
            self._error.emit(f"Could not open the audio device:\n{e}")

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

    def toggle_stream(self, on: bool):
        from ..dj.stream import StreamServer

        if not on:
            if self.stream is not None:
                self.stream.stop()
                self.stream = None
            self.stream_lbl.setText("")
            return
        port = int(self.settings.value("stream_port", 8000))
        try:
            self.stream = StreamServer(self.engine, port=port, status_fn=self._stream_status)
            self.stream.start()
        except OSError as e:
            self.stream = None
            self.b_stream.blockSignals(True)
            self.b_stream.setChecked(False)
            self.b_stream.blockSignals(False)
            self._error.emit(f"Could not start the stream on port {port}:\n{e}\n\n"
                             "Another program may be using it - change the port under Audio….")
            return
        urls = self.stream.urls
        self.stream_lbl.setText("Phone: <b>" + "</b> or <b>".join(urls) + "</b>")
        self.stream_lbl.setToolTip("Open this address in the phone's browser (same Wi-Fi). "
                                   "Away from home use a VPN such as Tailscale, or forward the port on your router.\n"
                                   "Windows may ask to allow CTA through the firewall - allow it on private networks.")

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
        mi = d.mix_in / SR if d.mix_in is not None else None
        mo = d.mix_out / SR if d.mix_out is not None else None
        if kind == "in":
            mi = t
        elif kind == "out":
            mo = t
        elif kind == "clear":
            mi = mo = None
        if mi is not None and mo is not None and mo <= mi:
            self._error.emit("The mix-out (outro) marker must come after the mix-in (intro) marker.")
            return
        self.engine.set_markers(deck, mi, mo)
        self.automix.refresh_markers(deck)
        try:
            self.main.db.set_markers(d.info.path, mi, mo)
        except Exception:  # noqa: BLE001
            log.exception("saving markers failed")
        what = {"in": "Mix-in (intro)", "out": "Mix-out (outro)", "clear": "Markers"}[kind]
        self.main.status_label.setText(f"{what} {'cleared' if kind == 'clear' else 'set'} for {d.info.title} "
                                       "(saved with the track)")

    def grid_changed(self, deck: str):
        def save():
            d = self.engine.decks[deck]
            if d.grid is not None and d.info.path:
                self.main.db.set_grid(d.info.path, d.grid.first_beat, d.info.bpm)
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
                                                      genre=lib.parent_genres[idx] or lib.tag_genres[idx])

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
                               glide=self.settings.value("dj_glide", "true") == "true")

    def _update_am_settings(self, *_):
        self.settings.setValue("dj_mix_bars", self.mix_bars.value())
        self.settings.setValue("dj_glide", "true" if self.glide.isChecked() else "false")
        self.automix.s.mix_bars = self.mix_bars.value()
        self.automix.s.glide = self.glide.isChecked()

    def queue_from_setlist(self):
        sl = list(self.main.setlist)
        if not sl:
            QMessageBox.information(self, "Auto-mix", "Build a setlist in the Setlist Builder first "
                                    "(or just start auto-mix: it picks the best next tracks itself).")
            return
        self.automix.set_queue(sl)
        self._show_queue()

    def _show_queue(self):
        self.queue_list.clear()
        lib = self.main.lib
        for i in list(self.automix.queue):
            t = lib.track(i)
            self.queue_list.addItem(f"{t.bpm:6.1f}  {camelot_str(t.key):>3}  {t.artist} - {t.title}")

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
            self.automix.start()
        else:
            self.automix.stop()
        self._show_queue()

    def manual_touch(self, deck):
        am = self.automix
        if am.tr is not None and am.tr.started and deck in (am.tr.in_deck, am.tr.out_deck):
            am.user_took_over()

    def _keep_queue_filled(self):
        am = self.automix
        lib = self.main.lib
        if lib is None or am.queue or not self.auto_suggest.isChecked():
            return
        m = self.engine._master()
        if m is None or m.info.lib_index < 0:
            return
        if any(not d.playing and d.loaded and n in am.meta for n, d in self.engine.decks.items()):
            return
        self._played |= {d.info.lib_index for d in self.engine.decks.values() if d.info.lib_index >= 0}
        sug = suggest_next(lib, m.info.lib_index, self.main.mix_settings(), n=10, exclude=self._played)
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
                except Exception:  # noqa: BLE001
                    log.exception("history failed")
                self._logged[n] = (d.info.path, -1.0)

    # ---- refresh -------------------------------------------------------------------------
    def _tick(self):
        show_cd = self.show_cd.isChecked()
        for p in self.decks.values():
            if p.isVisible():
                p.refresh()
        self.mixer.refresh(show_cd)
        for f in self.fx:
            f.refresh()
        am = self.automix
        self.am_status.setText(am.status)
        _sync_check(self.b_auto, am.active)
        self.b_auto.setText("■ Stop auto-mix" if am.active else "▶ Auto-mix")
        if am.active:
            self._keep_queue_filled()
            if self.queue_list.count() != len(am.queue):
                self._show_queue()
        self._log_history()
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
