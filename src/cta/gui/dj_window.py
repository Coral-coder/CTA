"""DJ Mode window: decks, mixer, auto-mix and controller setup."""

from __future__ import annotations

import logging
import threading

import numpy as np
from PySide6.QtCore import QRectF, QSettings, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDial,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMainWindow,
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
from ..dj.loader import load_track
from ..dj.midi import Controller, Learner, learn_steps
from ..keys import camelot_str, key_name
from ..mixer import suggest_next

log = logging.getLogger(__name__)

DECK_COLORS = {"A": QColor(64, 156, 255), "B": QColor(255, 140, 60), "C": QColor(120, 200, 120),
               "D": QColor(200, 120, 220)}


def fmt_time(s: float) -> str:
    s = max(0.0, s)
    return f"{int(s // 60)}:{int(s % 60):02d}.{int((s * 10) % 10)}"


# --------------------------------------------------------------------------
# waveform
# --------------------------------------------------------------------------

class Waveform(QWidget):
    seek = Signal(float)          # seconds

    def __init__(self, deck: str, zoom: bool, parent=None):
        super().__init__(parent)
        self.deck = deck
        self.zoom = zoom
        self.color = DECK_COLORS[deck]
        self.setMinimumHeight(70 if zoom else 34)
        self.peaks = None          # overview (W, 2)
        self.fine = None           # (N,) peak per 256 samples for the zoom view
        self.duration = 1.0
        self.pos = 0.0
        self.grid = None
        self.cue = 0.0
        self.loop = None
        self.rate = 1.0

    def set_track(self, peaks, audio, grid):
        self.peaks = peaks
        self.grid = grid
        if audio is not None:
            mono = np.abs(audio).max(axis=0)
            n = mono.size // 256
            self.fine = mono[: n * 256].reshape(n, 256).max(axis=1) if n else None
            self.duration = audio.shape[1] / SR
        else:
            self.fine = None
        self.update()

    def mousePressEvent(self, ev):
        if self.peaks is None:
            return
        x = ev.position().x() / max(self.width(), 1)
        if self.zoom:
            span = 8.0
            self.seek.emit(self.pos + (x - 0.5) * span)
        else:
            self.seek.emit(x * self.duration)

    def paintEvent(self, _ev):
        p = QPainter(self)
        w, h = self.width(), self.height()
        p.fillRect(0, 0, w, h, QColor(22, 23, 26))
        if self.peaks is None:
            p.setPen(QColor(90, 90, 96))
            p.drawText(self.rect(), Qt.AlignCenter, "Load a track")
            return
        mid = h / 2
        if not self.zoom:
            n = self.peaks.shape[0]
            cols = np.linspace(0, n - 1, w).astype(int)
            pk = self.peaks[cols]
            past = int(self.pos / self.duration * w) if self.duration else 0
            for x in range(w):
                a = pk[x, 0] * (h / 2 - 1)
                c = self.color.darker(170) if x < past else self.color
                p.setPen(c)
                p.drawLine(x, int(mid - a), x, int(mid + a))
            if self.loop:
                x0 = self.loop[0] / SR / self.duration * w
                x1 = self.loop[1] / SR / self.duration * w
                p.fillRect(QRectF(x0, 0, max(x1 - x0, 2), h), QColor(80, 200, 120, 70))
            cx = int(self.cue / self.duration * w)
            p.setPen(QColor(255, 190, 0))
            p.drawLine(cx, 0, cx, h)
            p.setPen(QPen(QColor(255, 255, 255), 2))
            p.drawLine(past, 0, past, h)
            return
        # zoomed, scrolling view: 8 s window centred on the playhead
        span = 8.0
        t0 = self.pos - span / 2
        if self.fine is not None:
            bins_per_sec = SR / 256
            idx = (t0 + np.arange(w) / w * span) * bins_per_sec
            valid = (idx >= 0) & (idx < self.fine.size)
            vals = np.zeros(w)
            vals[valid] = self.fine[idx[valid].astype(int)]
            m = max(self.fine.max(), 1e-6)
            p.setPen(self.color)
            for x in range(w):
                a = vals[x] / m * (h / 2 - 2)
                if a > 0.5:
                    p.drawLine(x, int(mid - a), x, int(mid + a))
        if self.grid is not None:
            b0 = int(np.floor(self.grid.beat_at(t0)))
            b1 = int(np.ceil(self.grid.beat_at(t0 + span)))
            for b in range(b0, b1 + 1):
                t = self.grid.time_of_beat(b)
                x = (t - t0) / span * w
                bar = b % 4 == 0
                p.setPen(QPen(QColor(255, 255, 255, 170 if bar else 60), 2 if bar else 1))
                p.drawLine(int(x), 0, int(x), h)
        p.setPen(QPen(QColor(255, 60, 60), 2))
        p.drawLine(w // 2, 0, w // 2, h)


class PhaseMeter(QWidget):
    """Shows how far this deck's beats are from the master's (+-half a beat)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(10)
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


# --------------------------------------------------------------------------
# deck panel
# --------------------------------------------------------------------------

class DeckPanel(QGroupBox):
    def __init__(self, name: str, win: DJWindow):
        super().__init__(f"Deck {name}")
        self.name = name
        self.win = win
        self.eng = win.engine
        col = DECK_COLORS[name]
        self.setStyleSheet(f"QGroupBox {{ border: 1px solid {col.name()}; border-radius: 6px; margin-top: 8px; }}"
                           f"QGroupBox::title {{ color: {col.name()}; left: 8px; }}")
        lay = QVBoxLayout(self)
        top = QHBoxLayout()
        self.title = QLabel("—")
        f = QFont()
        f.setPointSize(11)
        f.setBold(True)
        self.title.setFont(f)
        self.title.setWordWrap(True)
        top.addWidget(self.title, 1)
        self.bpm_lbl = QLabel("")
        self.bpm_lbl.setStyleSheet("font-size: 18px; font-weight: bold;")
        self.bpm_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        top.addWidget(self.bpm_lbl)
        lay.addLayout(top)
        info = QHBoxLayout()
        self.key_lbl = QLabel("")
        self.time_lbl = QLabel("")
        self.time_lbl.setAlignment(Qt.AlignRight)
        info.addWidget(self.key_lbl)
        info.addStretch()
        info.addWidget(self.time_lbl)
        lay.addLayout(info)
        self.zoom = Waveform(name, True)
        self.over = Waveform(name, False)
        self.zoom.seek.connect(lambda t: self.eng.seek(name, t))
        self.over.seek.connect(lambda t: self.eng.seek(name, t))
        lay.addWidget(self.zoom)
        lay.addWidget(self.over)
        self.phase = PhaseMeter()
        lay.addWidget(self.phase)

        row = QHBoxLayout()
        self.b_play = QPushButton("▶ PLAY")
        self.b_play.setCheckable(True)
        self.b_play.clicked.connect(lambda: self.eng.toggle_play(name))
        self.b_cue = QPushButton("CUE")
        self.b_cue.pressed.connect(lambda: self.eng.cue_press(name))
        self.b_cue.released.connect(lambda: self.eng.cue_release(name))
        self.b_sync = QPushButton("SYNC")
        self.b_sync.setCheckable(True)
        self.b_sync.clicked.connect(lambda on: self.eng.set(name, sync=on))
        self.b_key = QPushButton("KEY LOCK")
        self.b_key.setCheckable(True)
        self.b_key.setChecked(True)
        self.b_key.clicked.connect(lambda on: self.eng.set(name, keylock=on))
        self.b_master = QPushButton("MASTER")
        self.b_master.setCheckable(True)
        self.b_master.clicked.connect(lambda: self.eng.set_master(name))
        self.b_phones = QPushButton("🎧")
        self.b_phones.setCheckable(True)
        self.b_phones.setToolTip("Pre-listen in headphones")
        self.b_phones.clicked.connect(lambda on: self.eng.set(name, cue_phones=on))
        for b in (self.b_play, self.b_cue, self.b_sync, self.b_key, self.b_master, self.b_phones):
            b.setMinimumHeight(30)
            row.addWidget(b)
        lay.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Loop"))
        for n in (1, 2, 4, 8, 16, 32):
            b = QPushButton(str(n))
            b.setMaximumWidth(36)
            b.clicked.connect(lambda _=False, n=n: self._loop(n))
            row.addWidget(b)
        self.b_exit_loop = QPushButton("exit")
        self.b_exit_loop.clicked.connect(lambda: self.eng.loop_beats(name, None))
        row.addWidget(self.b_exit_loop)
        row.addStretch()
        b_load = QPushButton(f"Load selected → {name}")
        b_load.clicked.connect(lambda: win.load_selected(name))
        row.addWidget(b_load)
        lay.addLayout(row)

        ctl = QHBoxLayout()
        ctl.setSpacing(10)
        eqbox = QGridLayout()
        eqbox.setVerticalSpacing(2)
        self.dials = {}
        # channel strip: knobs stacked vertically like on the hardware
        for i, (key, lab) in enumerate((("trim", "TRIM"), ("high", "HI"), ("mid", "MID"), ("low", "LOW"))):
            d = QDial()
            d.setRange(0, 100)
            d.setValue(50)
            d.setNotchesVisible(True)
            d.setFixedSize(42, 42)
            d.valueChanged.connect(self._eq_changed)
            d.setToolTip(f"{lab} (centre = flat, fully left = kill)")
            self.dials[key] = d
            eqbox.addWidget(QLabel(lab), i, 0, Qt.AlignRight | Qt.AlignVCenter)
            eqbox.addWidget(d, i, 1)
        b_flat = QPushButton("flat")
        b_flat.setToolTip("Reset trim and EQ")
        b_flat.clicked.connect(lambda: [d.setValue(50) for d in self.dials.values()])
        eqbox.addWidget(b_flat, 4, 0, 1, 2)
        ctl.addLayout(eqbox)
        ctl.addStretch(1)
        tb = QVBoxLayout()
        tb.addWidget(QLabel("Tempo"))
        self.tempo = QSlider(Qt.Vertical)
        self.tempo.setRange(-1000, 1000)
        self.tempo.setInvertedAppearance(True)       # Pioneer style: down = faster
        self.tempo.valueChanged.connect(lambda v: self.eng.set(name, tempo=v / 1000.0))
        self.tempo.setMinimumHeight(140)
        self.tempo.setMaximumHeight(260)
        tb.addWidget(self.tempo, 1, Qt.AlignHCenter)
        self.range = QComboBox()
        self.range.addItems(["±6%", "±8%", "±16%", "±50%"])
        self.range.setCurrentIndex(1)
        self.range.currentIndexChanged.connect(
            lambda i: self.eng.set(name, tempo_range=(0.06, 0.08, 0.16, 0.5)[i]))
        tb.addWidget(self.range)
        b_reset = QPushButton("0")
        b_reset.setToolTip("Reset tempo")
        b_reset.clicked.connect(lambda: self.tempo.setValue(0))
        tb.addWidget(b_reset)
        ctl.addLayout(tb)
        vb = QVBoxLayout()
        vb.addWidget(QLabel("Vol"))
        self.vol = QSlider(Qt.Vertical)
        self.vol.setRange(0, 1000)
        self.vol.setValue(1000)
        self.vol.valueChanged.connect(self._vol_changed)
        self.vol.setMaximumHeight(260)
        vb.addWidget(self.vol, 1, Qt.AlignHCenter)
        ctl.addLayout(vb)
        lay.addLayout(ctl)
        self._loaded_path = None

    def _loop(self, n):
        self.eng.loop_beats(self.name, n)

    def _vol_changed(self, v):
        self.win.manual_touch(self.name)
        self.eng.set(self.name, volume=v / 1000.0)

    @staticmethod
    def _eq_gain(v: int) -> float:
        x = v / 100.0
        if x <= 0.5:
            return (x / 0.5) ** 2 if x > 0.01 else 0.0
        return 1.0 + (x - 0.5) * 2.0

    def _eq_changed(self, *_):
        self.win.manual_touch(self.name)
        g = self._eq_gain
        self.eng.set(self.name, eq=(g(self.dials["low"].value()), g(self.dials["mid"].value()),
                                    g(self.dials["high"].value())),
                     trim=float(10 ** ((self.dials["trim"].value() / 100.0 - 0.5) * 24 / 20)))

    def refresh(self):
        d = self.eng.decks[self.name]
        if d.info.path != self._loaded_path:
            self._loaded_path = d.info.path
            self.title.setText(f"{d.info.artist} - {d.info.title}" if d.info.artist else (d.info.title or "—"))
            self.zoom.set_track(d.peaks, d.audio, d.grid)
            self.over.set_track(d.peaks, d.audio, d.grid)
        if not d.loaded:
            self.bpm_lbl.setText("")
            self.key_lbl.setText("")
            self.time_lbl.setText("")
            return
        orig = d.grid.bpm if d.grid else 0
        pct = (d.base_rate - 1) * 100
        self.bpm_lbl.setText(f"{d.bpm:.2f}")
        self.bpm_lbl.setToolTip(f"original {orig:.2f} BPM, {pct:+.2f}%")
        k = d.info.key
        if k >= 0:
            if d.keylock or abs(d.base_rate - 1) < 1e-4:
                self.key_lbl.setText(f"{camelot_str(k)} {key_name(k)}   {pct:+.1f}%")
            else:
                shift = 12 * np.log2(d.base_rate)
                self.key_lbl.setText(f"{camelot_str(k)} {key_name(k)} ({shift:+.1f} st)   {pct:+.1f}%")
        else:
            self.key_lbl.setText(f"{pct:+.1f}%")
        remain = d.length / SR - d.time
        self.time_lbl.setText(f"{fmt_time(d.time)}  /  -{fmt_time(remain)}")
        self.time_lbl.setStyleSheet("color: #ff5050;" if d.playing and remain < 30 else "")
        for wv in (self.zoom, self.over):
            wv.pos = d.time
            wv.cue = d.cue_point / SR
            wv.loop = d.loop
            wv.update()
        self.phase.err = self.eng.phase_error(self.name) if d.playing else None
        self.phase.update()
        for b, on in ((self.b_play, d.playing), (self.b_sync, d.sync), (self.b_key, d.keylock),
                      (self.b_master, self.eng.master_deck == self.name), (self.b_phones, d.cue_phones)):
            if b.isChecked() != on:
                b.blockSignals(True)
                b.setChecked(on)
                b.blockSignals(False)
        self.b_play.setText("❚❚ PAUSE" if d.playing else "▶ PLAY")
        # reflect controller / auto-mix moves on the on-screen controls
        for w, val in ((self.vol, int(d.volume * 1000)), (self.tempo, int(d.tempo * 1000))):
            if abs(w.value() - val) > 2 and not w.isSliderDown():
                w.blockSignals(True)
                w.setValue(val)
                w.blockSignals(False)


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
        self.devices = list_output_devices()
        self.dev.addItem("System default", None)
        for i, name, _ch in self.devices:
            self.dev.addItem(name, i)
        cur = settings.value("dj_device_name", "")
        ix = self.dev.findText(cur)
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
        form.addRow(QLabel("<small>DDJ-T1: choose its sound card, master 1-2, headphones 3-4.<br>"
                           "Use a bigger buffer if you hear crackles.</small>"))
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        form.addRow(bb)

    def accept(self):
        self.s.setValue("dj_device_name", self.dev.currentText() if self.dev.currentData() is not None else "")
        self.s.setValue("dj_master", self.master.currentText())
        self.s.setValue("dj_phones", self.phones.currentText())
        self.s.setValue("dj_block", self.block.currentText())
        super().accept()


class LearnWizard(QDialog):
    """Guided MIDI learn: asks for each control in turn."""

    got = Signal(object)

    def __init__(self, controller: Controller, decks=("A", "B"), parent=None):
        super().__init__(parent)
        self.setWindowTitle("Controller setup")
        self.ctl = controller
        self.steps = learn_steps(decks)
        self.i = 0
        self.result = {(b.action, b.deck): b for b in controller.bindings}
        self.learner = Learner()
        lay = QVBoxLayout(self)
        self.prog = QProgressBar()
        self.prog.setRange(0, len(self.steps))
        lay.addWidget(self.prog)
        self.prompt = QLabel()
        self.prompt.setStyleSheet("font-size: 16px; padding: 12px;")
        self.prompt.setWordWrap(True)
        lay.addWidget(self.prompt)
        self.heard = QLabel("")
        lay.addWidget(self.heard)
        row = QHBoxLayout()
        b_skip = QPushButton("Skip")
        b_skip.clicked.connect(self._next)
        b_back = QPushButton("Back")
        b_back.clicked.connect(self._back)
        b_done = QPushButton("Finish && save")
        b_done.clicked.connect(self._finish)
        row.addWidget(b_back)
        row.addWidget(b_skip)
        row.addStretch()
        row.addWidget(b_done)
        lay.addLayout(row)
        self.got.connect(self._on_event)
        controller.learn_cb = lambda ev: self.got.emit(ev)
        self.resize(520, 220)
        self._show()

    def _show(self):
        if self.i >= len(self.steps):
            self.prompt.setText("All done - click <b>Finish & save</b>.")
            return
        action, deck, text = self.steps[self.i]
        extra = ""
        if action == "tempo":
            extra = "<br><small>Move it slowly from the TOP to the BOTTOM.</small>"
        elif action in ("jog", "browse"):
            extra = "<br><small>Turn it a little clockwise.</small>"
        elif action in ("volume", "eq_high", "eq_mid", "eq_low", "gain", "crossfader", "master_volume",
                        "cue_mix", "phones_volume"):
            extra = "<br><small>Move it a little.</small>"
        else:
            extra = "<br><small>Press it.</small>"
        have = self.result.get((action, deck))
        cur = f"<br><small>currently: {have.type} ch{have.ch + 1} #{have.num}</small>" if have else ""
        self.prompt.setText(f"<b>{text}</b>{extra}{cur}")
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


# --------------------------------------------------------------------------
# main DJ window
# --------------------------------------------------------------------------

class DJWindow(QMainWindow):
    # worker/MIDI threads must reach the GUI through queued signals
    _error = Signal(str)
    _browse_sig = Signal(int)
    _load_sig = Signal(str)

    def __init__(self, main_window):
        super().__init__()
        self._error.connect(lambda msg: QMessageBox.warning(self, "DJ Mode", msg))
        self._browse_sig.connect(lambda d: self.main.browse_library(d))
        self._load_sig.connect(lambda deck: self.load_selected(deck))
        self.main = main_window
        self.settings = main_window.settings
        self.setWindowTitle("CTA - DJ Mode")
        self.resize(1500, 860)
        self.engine = Engine(block=int(self.settings.value("dj_block", 512)))
        self.automix = AutoMix(self.engine, self._automix_load, self._automix_settings())
        self.controller: Controller | None = None
        self._touch_lock = threading.Lock()

        central = QWidget()
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)
        bar = QHBoxLayout()
        self.audio_lbl = QLabel("Audio: off")
        b_audio = QPushButton("Audio…")
        b_audio.clicked.connect(self.audio_settings)
        self.midi_combo = QComboBox()
        self.midi_combo.setMinimumWidth(220)
        b_midi = QPushButton("Connect controller")
        b_midi.clicked.connect(self.connect_midi)
        b_learn = QPushButton("Controller setup…")
        b_learn.clicked.connect(self.learn)
        self.show_cd = QCheckBox("Decks C/D")
        self.show_cd.toggled.connect(self._toggle_cd)
        for w in (self.audio_lbl, b_audio, QLabel("   MIDI:"), self.midi_combo, b_midi, b_learn):
            bar.addWidget(w)
        bar.addStretch()
        bar.addWidget(self.show_cd)
        outer.addLayout(bar)

        grid = QGridLayout()
        self.panels = {n: DeckPanel(n, self) for n in "ABCD"}
        grid.addWidget(self.panels["A"], 0, 0)
        grid.addWidget(self._mixer_panel(), 0, 1)
        grid.addWidget(self.panels["B"], 0, 2)
        grid.addWidget(self.panels["C"], 1, 0)
        grid.addWidget(self.panels["D"], 1, 2)
        grid.setColumnStretch(0, 5)
        grid.setColumnStretch(2, 5)
        grid.setColumnStretch(1, 2)
        outer.addLayout(grid, 1)
        outer.addWidget(self._automix_panel())
        self._toggle_cd(False)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(33)
        self._refresh_midi_ports()
        QTimer.singleShot(200, self.start_audio)
        QTimer.singleShot(400, self._auto_connect_midi)

    # ---- layout pieces -----------------------------------------------------
    def _mixer_panel(self) -> QWidget:
        box = QGroupBox("Mixer")
        lay = QVBoxLayout(box)
        self.level = QProgressBar()
        self.level.setRange(0, 100)
        self.level.setTextVisible(False)
        self.level.setFixedHeight(10)
        lay.addWidget(QLabel("Master level"))
        lay.addWidget(self.level)
        self.master_vol = QSlider(Qt.Horizontal)
        self.master_vol.setRange(0, 100)
        self.master_vol.setValue(90)
        self.master_vol.valueChanged.connect(lambda v: setattr(self.engine, "master_volume", v / 100))
        lay.addWidget(QLabel("Master volume"))
        lay.addWidget(self.master_vol)
        self.cue_mix = QSlider(Qt.Horizontal)
        self.cue_mix.setRange(0, 100)
        self.cue_mix.valueChanged.connect(lambda v: setattr(self.engine, "cue_mix", v / 100))
        lay.addWidget(QLabel("Headphones: cue ↔ master"))
        lay.addWidget(self.cue_mix)
        self.phones_vol = QSlider(Qt.Horizontal)
        self.phones_vol.setRange(0, 100)
        self.phones_vol.setValue(80)
        self.phones_vol.valueChanged.connect(lambda v: setattr(self.engine, "phones_volume", v / 100))
        lay.addWidget(QLabel("Headphones volume"))
        lay.addWidget(self.phones_vol)
        lay.addStretch()
        lay.addWidget(QLabel("Crossfader"))
        self.xf = QSlider(Qt.Horizontal)
        self.xf.setRange(-100, 100)
        self.xf.setValue(0)
        self.xf.valueChanged.connect(self._xf_changed)
        lay.addWidget(self.xf)
        b = QPushButton("Centre")
        b.clicked.connect(lambda: self.xf.setValue(0))
        lay.addWidget(b)
        return box

    def _automix_panel(self) -> QWidget:
        box = QGroupBox("Auto-mix / Assist")
        lay = QHBoxLayout(box)
        col = QVBoxLayout()
        self.b_auto = QPushButton("▶ Start auto-mix")
        self.b_auto.setCheckable(True)
        self.b_auto.setStyleSheet("font-weight: bold; padding: 8px;")
        self.b_auto.clicked.connect(self.toggle_automix)
        col.addWidget(self.b_auto)
        b_tr = QPushButton("Auto transition now")
        b_tr.setToolTip("Blend from the playing deck into the other loaded deck, starting at the next bar")
        b_tr.clicked.connect(lambda: self.automix.transition_now())
        col.addWidget(b_tr)
        b_q = QPushButton("Queue = current setlist")
        b_q.clicked.connect(self.queue_from_setlist)
        col.addWidget(b_q)
        lay.addLayout(col)
        form = QFormLayout()
        self.mix_bars = QSpinBox()
        self.mix_bars.setRange(2, 64)
        self.mix_bars.setValue(int(self.settings.value("dj_mix_bars", 16)))
        self.mix_bars.valueChanged.connect(self._update_am_settings)
        form.addRow("Blend length (bars)", self.mix_bars)
        self.glide = QCheckBox("Glide tempo to each track's own BPM (key stays locked)")
        self.glide.setChecked(self.settings.value("dj_glide", "true") == "true")
        self.glide.toggled.connect(self._update_am_settings)
        form.addRow(self.glide)
        self.bass_swap = QCheckBox("Bass swap")
        self.bass_swap.setChecked(True)
        self.bass_swap.toggled.connect(self._update_am_settings)
        form.addRow(self.bass_swap)
        self.auto_suggest = QCheckBox("Assist: auto-load the best next track when a deck is empty")
        self.auto_suggest.setChecked(self.settings.value("dj_auto_suggest", "true") == "true")
        form.addRow(self.auto_suggest)
        lay.addLayout(form)
        col = QVBoxLayout()
        self.am_status = QLabel("Auto-mix off")
        self.am_status.setStyleSheet("font-size: 14px; font-weight: bold;")
        col.addWidget(self.am_status)
        self.queue_list = QListWidget()
        self.queue_list.setMaximumHeight(110)
        col.addWidget(self.queue_list)
        lay.addLayout(col, 1)
        return box

    def _toggle_cd(self, on):
        self.panels["C"].setVisible(on)
        self.panels["D"].setVisible(on)

    # ---- audio --------------------------------------------------------------
    def start_audio(self):
        s = self.settings
        name = s.value("dj_device_name", "")
        dev = None
        try:
            for i, n, _ch in list_output_devices():
                if n == name:
                    dev = i
            if dev is None:
                # prefer the DDJ-T1 sound card when it is connected
                for i, n, ch in list_output_devices():
                    if "DDJ-T1" in n.upper() and ch >= 4:
                        dev = i
            mc = (2, 3) if s.value("dj_master", "1-2") == "3-4" else (0, 1)
            ph = s.value("dj_phones", "3-4")
            pc = None if ph.startswith("off") else ((2, 3) if ph == "3-4" else (0, 1))
            if pc == mc:
                pc = None
            self.engine.start(dev, int(s.value("dj_block", 512)), mc, pc)
            phones = "headphones on " + ("3-4" if self.engine.phones_chans == (2, 3) else "1-2") \
                if self.engine.phones_chans else "no separate headphones"
            self.audio_lbl.setText(f"Audio: {self.engine.out_channels} ch, {self.engine.latency_ms:.0f} ms, {phones}")
        except Exception as e:  # noqa: BLE001
            log.exception("audio start failed")
            self.audio_lbl.setText("Audio: failed - click Audio…")
            QMessageBox.warning(self, "Audio", f"Could not open the audio device:\n{e}")

    def audio_settings(self):
        if AudioSettings(self.settings, self).exec():
            self.engine.stop()
            self.engine.block = int(self.settings.value("dj_block", 512))
            self.start_audio()

    # ---- MIDI ----------------------------------------------------------------
    def _refresh_midi_ports(self):
        self.midi_combo.clear()
        ports = Controller.list_inputs()
        self.midi_combo.addItems(ports or ["(no MIDI devices)"])
        for i, p in enumerate(ports):
            if "DDJ" in p.upper():
                self.midi_combo.setCurrentIndex(i)

    def _auto_connect_midi(self):
        port = self.midi_combo.currentText()
        if "DDJ" in port.upper():
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
            QMessageBox.warning(self, "Controller", f"Could not open {port}:\n{e}")
            self.controller = None
            return
        self.statusBar().showMessage(f"Controller connected: {port}", 5000)
        if not self.controller.learned:
            r = QMessageBox.question(
                self, "Controller setup",
                f"{port} is connected with a basic default mapping (play, cue, jog, tempo, load, browse).\n\n"
                "Run the 2-minute guided setup now so every fader, knob and button works exactly "
                "as on your unit?")
            if r == QMessageBox.Yes:
                self.learn()

    def learn(self):
        if self.controller is None:
            self.connect_midi()
            if self.controller is None:
                return
        decks = ("A", "B", "C", "D") if self.show_cd.isChecked() else ("A", "B")
        LearnWizard(self.controller, decks, self).exec()

    # controller UI callbacks (called from the MIDI thread -> marshal to GUI)
    def browse(self, delta: int):
        self._browse_sig.emit(int(delta))

    def load(self, deck: str):
        self._load_sig.emit(deck)

    # ---- loading ---------------------------------------------------------------
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
                                                      bpm=float(lib.bpm[idx]), loudness=loud, lib_index=idx)

    def load_index(self, deck: str, idx: int):
        d = self.engine.decks[deck]
        if d.playing:
            QMessageBox.information(self, "Load", f"Deck {deck} is playing - stop it first.")
            return
        path, bpm, info = self._track_info(idx)
        self.panels[deck].title.setText(f"Loading {info.title}…")

        def job():
            try:
                lt = load_track(path, bpm, info, self.main.db)
                self.engine.load(deck, lt.audio, lt.grid, lt.info, lt.peaks)
            except Exception as e:  # noqa: BLE001
                log.exception("load failed")
                self._error.emit(f"Could not load {path}:\n{e}")
        threading.Thread(target=job, daemon=True, name=f"load-{deck}").start()

    def load_selected(self, deck: str):
        idx = self.main.current
        if idx is None:
            self.statusBar().showMessage("Select a track in the library first", 4000)
            return
        self.load_index(deck, idx)

    def _automix_load(self, idx):
        path, bpm, info = self._track_info(idx)
        return load_track(path, bpm, info, self.main.db)

    # ---- auto-mix -----------------------------------------------------------------
    def _automix_settings(self) -> AutoMixSettings:
        return AutoMixSettings(mix_bars=int(self.settings.value("dj_mix_bars", 16)),
                               glide=self.settings.value("dj_glide", "true") == "true")

    def _update_am_settings(self, *_):
        self.settings.setValue("dj_mix_bars", self.mix_bars.value())
        self.settings.setValue("dj_glide", "true" if self.glide.isChecked() else "false")
        self.automix.s.mix_bars = self.mix_bars.value()
        self.automix.s.glide = self.glide.isChecked()
        self.automix.s.bass_swap = self.bass_swap.isChecked()

    def queue_from_setlist(self):
        sl = list(self.main.setlist)
        if not sl:
            QMessageBox.information(self, "Auto-mix", "Build a setlist in the main window first "
                                    "(or just start auto-mix: it will pick the best next tracks itself).")
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
            self.b_auto.setText("■ Stop auto-mix")
        else:
            self.automix.stop()
            self.b_auto.setText("▶ Start auto-mix")
        self._show_queue()

    def manual_touch(self, deck):
        am = self.automix
        if am.tr is not None and am.tr.started and deck in (am.tr.in_deck, am.tr.out_deck):
            am.user_took_over()

    def _xf_changed(self, v):
        if self.automix.tr is not None and self.automix.tr.started:
            self.automix.user_took_over()
        self.engine.crossfader = v / 100.0

    def _keep_queue_filled(self):
        """Assist / endless auto-mix: when the queue runs dry, add the best
        next track for whatever is playing."""
        am = self.automix
        lib = self.main.lib
        if lib is None or am.queue or not self.auto_suggest.isChecked():
            return
        m = self.engine._master()
        if m is None or m.info.lib_index < 0:
            return
        if any(not d.playing and d.loaded and n in am.meta for n, d in self.engine.decks.items()):
            return
        played = {d.info.lib_index for d in self.engine.decks.values()}
        played |= set(getattr(self, "_played", set()))
        self._played = played
        sug = suggest_next(lib, m.info.lib_index, self.main.mix_settings(), n=10, exclude=played)
        if sug:
            am.set_queue([sug[0].idx])
            self._show_queue()

    # ---- refresh --------------------------------------------------------------------
    def _tick(self):
        for p in self.panels.values():
            if p.isVisible():
                p.refresh()
        self.level.setValue(int(min(1.0, self.engine.peak_level) * 100))
        if abs(self.xf.value() - int(self.engine.crossfader * 100)) > 1 and not self.xf.isSliderDown():
            self.xf.blockSignals(True)
            self.xf.setValue(int(self.engine.crossfader * 100))
            self.xf.blockSignals(False)
        am = self.automix
        self.am_status.setText(am.status)
        if self.b_auto.isChecked() != am.active:
            self.b_auto.setChecked(am.active)
            self.b_auto.setText("■ Stop auto-mix" if am.active else "▶ Start auto-mix")
        if am.active:
            self._keep_queue_filled()
            if self.queue_list.count() != len(am.queue):
                self._show_queue()
        if self.controller is not None:
            self.controller.tick()
            self.controller.update_leds()
        if self.engine.underruns:
            self.statusBar().showMessage(f"Audio dropouts: {self.engine.underruns} - "
                                         "increase the buffer in Audio…", 2000)

    def closeEvent(self, ev):
        self.automix.stop()
        self.engine.stop()
        if self.controller:
            self.controller.close()
        super().closeEvent(ev)
