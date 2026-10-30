"""Time-Shift: mix the next transition now, in your headphones - auto-mix
plays it to the room later, on the same beat with the same moves.

The window has its own preview player (two decks that are not the live
ones), zoomable full waveforms to place the intro / outro markers, a small
mixer the controller can drive, and a recorder for the moves.
"""

from __future__ import annotations

import threading

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..dj.engine import SR, Engine
from ..dj.rehearsal import LANES

LANE_STYLE = {  # lane: (colour, label) - the ones drawn in the lane view
    "xf": (QColor(235, 235, 240), "Crossfader"),
    "out_vol": (QColor(66, 133, 244), "Out fader"),
    "in_vol": (QColor(255, 152, 0), "In fader"),
    "out_low": (QColor(120, 160, 255), "Out bass"),
    "in_low": (QColor(255, 200, 90), "In bass"),
}


class LanesView(QWidget):
    """The recorded moves, beat by beat (one line per control)."""

    def __init__(self):
        super().__init__()
        self.setMinimumHeight(90)
        self.r = None
        self.x = None

    def paintEvent(self, _ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        p.fillRect(0, 0, w, h, QColor(16, 17, 20))
        r = self.r
        if r is None or r.beats <= 0:
            p.setPen(QColor(110, 112, 120))
            p.drawText(self.rect(), Qt.AlignCenter,
                       "Mix the two decks (or press Rehearse): your moves appear here")
            return
        span = max(r.beats, 1.0)
        for bar in range(0, int(span) + 1, 4):
            x = bar / span * (w - 1)
            p.setPen(QPen(QColor(255, 255, 255, 60 if bar % 16 == 0 else 22), 1))
            p.drawLine(int(x), 0, int(x), h)
        for k, (col, _lab) in LANE_STYLE.items():
            pts = r.lanes.get(k) or []
            if not pts:
                continue
            p.setPen(QPen(col, 2 if k.endswith("vol") or k == "xf" else 1,
                          Qt.SolidLine if not k.endswith("low") else Qt.DashLine))
            lo, hi = (-1.0, 1.0) if k == "xf" else (0.0, 1.5 if k.endswith("low") else 1.0)
            prev = None
            for x, v in pts:
                q = QPointF(x / span * (w - 1), h - 4 - (v - lo) / (hi - lo) * (h - 8))
                if prev is not None:
                    p.drawLine(prev, q)
                prev = q
        if self.x is not None:
            px = float(np.clip(self.x / span, 0, 1)) * (w - 1)
            p.setPen(QPen(QColor(255, 255, 255), 1))
            p.drawLine(int(px), 0, int(px), h)
        p.setPen(QColor(180, 182, 190))
        x0 = 6
        for _k, (col, lab) in LANE_STYLE.items():
            p.fillRect(QRectF(x0, 6, 10, 3), col)
            p.drawText(x0 + 14, 12, lab)
            x0 += 14 + p.fontMetrics().horizontalAdvance(lab) + 12


class TimeShiftWindow(QDialog):
    _loaded = Signal(str, object)          # preview deck, LoadedTrack (from a loader thread)

    def __init__(self, dj, out_idx: int | None = None, in_idx: int | None = None):
        super().__init__(dj)
        from .dj_window import DECK_COLORS, Knob, MarkerWave, ResetSlider, _btn

        self.dj = dj
        self.live: Engine = dj.engine
        self.setWindowTitle("⏱ Time-Shift - mix ahead, play it live later")
        self.resize(1100, 760)
        self.pv = self.live.create_preview()   # its own decks, rendered inside the live engine
        self.session = self.pv.session
        self.store = dj.rehearsals
        self.idx = {"A": None, "B": None}
        self.paths = {"A": "", "B": ""}
        self._saved_result = None
        self._loaded.connect(self._install)
        # headphones when there are any; the main output only while the room hears nothing
        live_on = any(d.playing for d in self.live.decks.values())
        self.live.preview_route = "phones" if self._has_phones() or live_on else "master"

        lay = QVBoxLayout(self)
        intro = QLabel("Mix the next transition now, on your headphones, while the room keeps listening to the "
                       "live decks. Your fader, crossfader, EQ and filter moves are recorded on the beat; "
                       "when the live set reaches these two tracks, auto-mix plays your mix exactly.")
        intro.setWordWrap(True)
        intro.setStyleSheet("color: #a9abb3;")
        lay.addWidget(intro)

        # ---- the two tracks: zoomable full waveforms with their markers --------------------
        self.waves, self.titles = {}, {}
        for deck, role in (("A", "OUTGOING"), ("B", "INCOMING")):
            box = QGroupBox(f"{role}  (preview deck {deck})")
            bl = QVBoxLayout(box)
            row = QHBoxLayout()
            t = QLabel("—")
            t.setStyleSheet(f"font-weight: 800; color: {DECK_COLORS[deck].name()};")
            self.titles[deck] = t
            row.addWidget(t, 1)
            for text, tip, fn in (("Use selected", "Use the track selected in the library",
                                   lambda _=False, d=deck: self.use_selected(d)),
                                  ("−", "Zoom out", lambda _=False, d=deck: self.waves[d].zoom(1.5)),
                                  ("+", "Zoom in", lambda _=False, d=deck: self.waves[d].zoom(1 / 1.5)),
                                  ("Fit", "Show the whole track", lambda _=False, d=deck: self.waves[d].fit())):
                b = QPushButton(text)
                b.setToolTip(tip)
                b.clicked.connect(fn)
                row.addWidget(b)
            play = _btn("▶", True, "Play / pause this preview deck (headphones)", h=24, w=34)
            play.clicked.connect(lambda _on, d=deck: self.pv.toggle_play(d))
            setattr(self, f"play_{deck}", play)
            row.addWidget(play)
            bl.addLayout(row)
            wv = MarkerWave()
            wv.setMinimumHeight(120)
            wv.moved.connect(lambda kind, secs, d=deck: self._marker_moved(d, kind, secs))
            wv.seek.connect(lambda secs, d=deck: self.pv.seek(d, secs))
            self.waves[deck] = wv
            bl.addWidget(wv, 1)
            lay.addWidget(box, 1)
        swap = QPushButton("⇅ Swap outgoing / incoming")
        swap.clicked.connect(self.swap)

        # ---- preview mixer -----------------------------------------------------------------
        mix = QGroupBox("PREVIEW MIXER  (only you hear this)")
        ml = QHBoxLayout(mix)
        self.faders, self.knobs = {}, {}
        for deck in "AB":
            g = QGridLayout()
            g.setSpacing(2)
            head = QLabel(deck)
            head.setStyleSheet(f"font-weight: 900; color: {DECK_COLORS[deck].name()};")
            g.addWidget(head, 0, 0, 1, 2, Qt.AlignHCenter)
            for r, (key, cap, default) in enumerate((("high", "HI", 50), ("mid", "MID", 50), ("low", "LOW", 50),
                                                      ("filter", "FILTER", 50)), start=1):
                k = Knob(default, 26, cap)
                k.valueChanged.connect(lambda _v, d=deck: self._knobs_changed(d))
                self.knobs[(deck, key)] = k
                lab = QLabel(cap)
                lab.setStyleSheet("font-size: 8px; color: #8d8f97; font-weight: 800;")
                g.addWidget(lab, r, 0, Qt.AlignRight)
                g.addWidget(k, r, 1)
            f = ResetSlider(Qt.Vertical, 1000, f"Preview {deck} fader")
            f.setRange(0, 1000)
            f.setValue(1000)
            f.setMinimumHeight(110)
            f.valueChanged.connect(lambda v, d=deck: self.pv.set(d, volume=v / 1000.0))
            self.faders[deck] = f
            g.addWidget(f, 1, 2, 4, 1)
            ml.addLayout(g)
        xcol = QVBoxLayout()
        xcol.addStretch(1)
        self.xf = ResetSlider(Qt.Horizontal, 0, "Preview crossfader")
        self.xf.setRange(-100, 100)
        self.xf.setMinimumWidth(180)
        self.xf.valueChanged.connect(lambda v: setattr(self.pv, "crossfader", v / 100.0))
        xcol.addWidget(self.xf)
        xcol.addWidget(QLabel("A  ·  crossfader  ·  B"), 0, Qt.AlignHCenter)
        ml.insertLayout(1, xcol)
        side = QVBoxLayout()
        self.route = QComboBox()
        self.route.addItems(["Headphones", "Main output (the room hears it!)"])
        self.route.setCurrentIndex(0 if self.live.preview_route == "phones" else 1)
        self.route.currentIndexChanged.connect(
            lambda i: setattr(self.live, "preview_route", "phones" if i == 0 else "master"))
        side.addWidget(QLabel("Listen on"))
        side.addWidget(self.route)
        self.ctl = QCheckBox("My controller mixes the preview")
        self.ctl.setToolTip("While this window is open your MIDI controller's faders, crossfader, EQ and "
                            "transport drive the preview decks instead of the live ones")
        self.ctl.setEnabled(dj.controller is not None)
        self.ctl.toggled.connect(self._controller_to_preview)
        side.addWidget(self.ctl)
        side.addWidget(swap)
        side.addStretch(1)
        ml.addLayout(side)
        lay.addWidget(mix)

        # ---- rehearsal ---------------------------------------------------------------------
        rec = QGroupBox("RECORD THE TRANSITION")
        rl = QVBoxLayout(rec)
        trow = QHBoxLayout()
        st = dj.settings
        self.auto_rec = QCheckBox("Record whenever I mix the two decks")
        self.auto_rec.setToolTip("On: starting the incoming while the outgoing plays records the transition, "
                                 "saved for auto-mix.\nOff: just mix and listen - nothing is recorded "
                                 "(● Rehearse still records).")
        self.auto_rec.setChecked(st.value("ts_auto_record", "true") == "true")
        self.auto_rec.toggled.connect(self._auto_rec_toggled)
        self.fill = QCheckBox("Controls I didn't touch follow auto-mix")
        self.fill.setToolTip("On: only what you moved is replayed; auto-mix does the rest (e.g. its bass swap).\n"
                             "Off: the transition plays exactly as recorded.")
        self.fill.setChecked(st.value("ts_fill", "true") == "true")
        self.fill.toggled.connect(self._fill_toggled)
        trow.addWidget(self.auto_rec)
        trow.addWidget(self.fill)
        trow.addStretch(1)
        rl.addLayout(trow)
        self.session.auto_record = self.auto_rec.isChecked()
        self.session.fill = self.fill.isChecked()
        row = QHBoxLayout()
        self.b_rec = QPushButton("● Rehearse")
        self.b_rec.setToolTip("The outgoing plays from 8 bars before its OUT marker; the incoming drops in on the "
                              "OUT marker (or press Drop now). Mix it - every move is recorded.")
        self.b_rec.setStyleSheet("QPushButton { font-weight: 800; color: #ff6b6b; }")
        self.b_rec.clicked.connect(self.rehearse)
        self.b_drop = QPushButton("⤓ Drop incoming now")
        self.b_drop.setToolTip("Bring the incoming in on the next bar line instead of at the OUT marker")
        self.b_drop.clicked.connect(self.session.drop_now)
        self.b_stop = QPushButton("■ Stop")
        self.b_stop.clicked.connect(self.stop)
        self.b_play = QPushButton("▶ Play back")
        self.b_play.setToolTip("Hear the rehearsed transition the way auto-mix will play it")
        self.b_play.clicked.connect(self.play_back)
        self.use = QCheckBox("Auto-mix plays this transition live")
        self.use.setChecked(True)
        self.use.toggled.connect(self._use_toggled)
        self.b_del = QPushButton("Delete")
        self.b_del.clicked.connect(self.delete)
        for w in (self.b_rec, self.b_drop, self.b_stop, self.b_play):
            row.addWidget(w)
        row.addStretch(1)
        row.addWidget(self.use)
        row.addWidget(self.b_del)
        rl.addLayout(row)
        self.lanes = LanesView()
        rl.addWidget(self.lanes)
        self.status = QLabel("")
        self.status.setStyleSheet("font-weight: 700;")
        rl.addWidget(self.status)
        lay.addWidget(rec)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(33)
        self._pick_defaults(out_idx, in_idx)

    # ---- tracks ------------------------------------------------------------------------
    def _has_phones(self) -> bool:
        return self.live._pstream is not None or bool(getattr(self.live, "phones_chans", None))

    def _pick_defaults(self, out_idx, in_idx):
        e = self.live
        m = e._master()
        if out_idx is None and m is not None and m.info.lib_index >= 0:
            out_idx = m.info.lib_index
        if in_idx is None:
            for d in e.decks.values():
                if d.loaded and not d.playing and d.info.lib_index >= 0 and d.info.lib_index != out_idx:
                    in_idx = d.info.lib_index
                    break
        if in_idx is None:
            q = [i for i in list(self.dj.automix.queue) if isinstance(i, (int, np.integer))]
            in_idx = int(q[0]) if q else None
        for deck, idx in (("A", out_idx), ("B", in_idx)):
            if idx is not None:
                self.load(deck, int(idx))
        self._refresh_status()

    def use_selected(self, deck: str):
        idx = self.dj.main.current
        if idx is None:
            QMessageBox.information(self, "Time-Shift", "Select a track in the library first.")
            return
        self.load(deck, idx)

    def swap(self):
        a, b = self.idx["A"], self.idx["B"]
        for deck, idx in (("A", b), ("B", a)):
            if idx is not None:
                self.load(deck, idx)

    def load(self, deck: str, idx: int):
        self.idx[deck] = idx
        lib = self.dj.main.lib
        self.titles[deck].setText(f"{lib.artists[idx]} - {lib.titles[idx]}  (loading…)")

        def job():
            try:
                lt = self.dj._automix_load(idx)
            except Exception as ex:  # noqa: BLE001
                lt = ex
            self._loaded.emit(deck, lt)
        threading.Thread(target=job, daemon=True, name=f"timeshift-{deck}").start()

    def _install(self, deck: str, lt):
        if isinstance(lt, Exception):
            self.titles[deck].setText(f"Could not load: {lt}")
            return
        self.pv.load_track(deck, lt, only_if_stopped=False)
        self.pv.set(deck, volume=1.0)
        if deck == "B":                          # the incoming follows the outgoing, beat for beat
            self.pv.set("B", sync=True, keylock=True)
        self.paths[deck] = lt.info.path
        self.titles[deck].setText(f"{lt.info.artist + ' - ' if lt.info.artist else ''}{lt.info.title}"
                                  f"   ·  {lt.grid.bpm:.1f} BPM")
        wv = self.waves[deck]
        wv.set_track(lt.bands, lt.audio.shape[1] / SR, lt.grid)
        wv.marks = {"in": lt.mix_in, "out": lt.mix_out, "out_end": lt.mix_out_end}
        focus = lt.mix_out if deck == "A" else lt.mix_in
        if focus is not None:
            wv.show_around(focus)
        else:
            wv.fit()
        self._refresh_status()

    def _marker_moved(self, deck: str, kind: str, secs: float):
        """Markers placed here are the track's markers everywhere (library, live decks, auto-mix)."""
        wv = self.waves[deck]
        wv.marks[kind] = secs
        mi, mo, me = wv.marks["in"], wv.marks["out"], wv.marks["out_end"]
        path = self.paths[deck]
        try:
            self.dj.main.db.set_markers(path, mi, mo, me)
        except Exception:  # noqa: BLE001
            pass
        self.pv.set_markers(deck, mi, mo, me)
        for name, d in self.live.decks.items():
            if d.loaded and d.info.path == path:
                try:
                    self.dj.set_markers(name, mi, mo, me)
                except ValueError:
                    pass
        wv.update()
        self._refresh_status()

    # ---- preview mixer ----------------------------------------------------------------------
    def _knobs_changed(self, deck: str):
        from .dj_window import knob_eq

        k = self.knobs
        self.pv.set(deck, eq=(knob_eq(k[(deck, "low")].value()), knob_eq(k[(deck, "mid")].value()),
                              knob_eq(k[(deck, "high")].value())),
                    filter=float((k[(deck, "filter")].value() - 50) / 50.0))

    def _controller_to_preview(self, on: bool):
        c = self.dj.controller
        if c is None:
            return
        if on:
            c.eng, c.automix = self.pv, None
        else:
            c.eng, c.automix = self.live, self.dj.automix

    # ---- rehearsal ------------------------------------------------------------------------
    def _beats(self) -> tuple[float, float] | None:
        a, b = self.pv.decks["A"], self.pv.decks["B"]
        if not (a.loaded and b.loaded and a.grid and b.grid):
            return None
        mo = self.waves["A"].marks["out"]
        out_beat = a.grid.beat_at(mo) if mo is not None else max(a.grid.beat_at(a.length / SR) - 96.0, 32.0)
        out_beat = float(np.round(out_beat / 4.0) * 4.0)
        mi = self.waves["B"].marks["in"]
        in_beat = float(np.round(b.grid.beat_at(mi) / 4.0) * 4.0) if mi is not None else 0.0
        return out_beat, in_beat

    def rehearse(self):
        beats = self._beats()
        if beats is None:
            QMessageBox.information(self, "Time-Shift", "Load an outgoing and an incoming track first.")
            return
        self._saved_result = None
        # start from the incoming at zero - you bring it in
        self.faders["B"].setValue(0)
        self.faders["A"].setValue(1000)
        self.xf.setValue(0)
        for (_d, _k), kn in self.knobs.items():
            kn.setValue(kn.default)
        self.session.rehearse(*beats)

    def stop(self):
        self.session.stop()

    def play_back(self):
        r = self._current()
        if r is None:
            QMessageBox.information(self, "Time-Shift", "Nothing rehearsed for these two tracks yet.")
            return
        self.session.replay(r)

    def _current(self):
        if not (self.paths["A"] and self.paths["B"]):
            return None
        return self.store.find(self.paths["A"], self.paths["B"])

    def _auto_rec_toggled(self, on: bool):
        self.session.auto_record = bool(on)
        self.dj.settings.setValue("ts_auto_record", "true" if on else "false")

    def _fill_toggled(self, on: bool):
        self.session.fill = bool(on)
        self.dj.automix.rehearsal_fill = bool(on)
        self.dj.settings.setValue("ts_fill", "true" if on else "false")

    def _use_toggled(self, on: bool):
        r = self._current()
        if r is not None:
            r.enabled = bool(on)
            self.store.put(r)
            self._refresh_status()

    def delete(self):
        if self._current() is None:
            return
        self.store.delete(self.paths["A"], self.paths["B"])
        self._refresh_status()

    def _refresh_status(self):
        r = self._current()
        self.lanes.r = r
        self.lanes.update()
        self.use.blockSignals(True)
        self.use.setChecked(r is None or r.enabled)
        self.use.blockSignals(False)
        self.b_play.setEnabled(r is not None)
        self.b_del.setEnabled(r is not None)
        if r is not None:
            self.status.setText(f"Rehearsed: {r.out_title} → {r.in_title}, {r.beats / 4:.0f} bars"
                                + ("  ·  auto-mix will play it live" if r.enabled else "  ·  not used by auto-mix"))

    def _tick(self):
        from .dj_window import _sync_check, _sync_value, eq_knob

        s = self.session
        for deck in "AB":
            d = self.pv.decks[deck]
            self.waves[deck].pos = d.time
            self.waves[deck].update()
            _sync_check(getattr(self, f"play_{deck}"), d.playing)
            # the controller (or a replay) moves the preview: show it
            _sync_value(self.faders[deck], int(d.volume * 1000))
            for key, g in (("low", d.eq[0]), ("mid", d.eq[1]), ("high", d.eq[2])):
                _sync_value(self.knobs[(deck, key)], eq_knob(g))
            _sync_value(self.knobs[(deck, "filter")], int(50 + d.filter * 50))
        _sync_value(self.xf, int(self.pv.crossfader * 100), 1)
        if s.mode == "lead":
            a = self.pv.decks["A"]
            left = (s.drop_beat - a.grid.beat_at(a.time)) / 4.0 if a.grid and s.drop_beat is not None else 0
            self.status.setText(f"Lead-in: the incoming drops in {max(0, int(np.ceil(left)))} bars - "
                                + ("listen and get ready" if s.replaying is None else "playback"))
        elif s.mode in ("recording", "replay"):
            self.status.setText(("● REC  " if s.mode == "recording" else "▶ PLAYBACK  ")
                                + f"bar {s.x / 4.0 + 1:.0f}" + ("  ·  take the outgoing out to finish (or ■ Stop)"
                                                                if s.mode == "recording" else ""))
            self.lanes.x = s.x
            if s.mode == "recording":
                from ..dj.rehearsal import Rehearsal

                self.lanes.r = Rehearsal("", "", 0, 0, max(s.x, 4.0), s.lanes)
            self.lanes.update()
        elif s.mode == "done" and s.result is not None and s.result is not self._saved_result:
            self._saved_result = s.result
            self.store.put(s.result)
            self.lanes.x = None
            self._refresh_status()
        self.b_drop.setEnabled(s.mode == "lead" and s.replaying is None)
        self.b_stop.setEnabled(s.mode in ("lead", "recording", "replay"))

    def _cleanup(self):
        if getattr(self, "_closed", False):
            return
        self._closed = True
        self.timer.stop()
        self._controller_to_preview(False)
        self.session.stop()
        self.live.close_preview()

    def closeEvent(self, ev):
        self._cleanup()
        super().closeEvent(ev)

    def done(self, r):
        self._cleanup()
        super().done(r)

__all__ = ["TimeShiftWindow", "LANES"]
