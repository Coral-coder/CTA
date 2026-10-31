"""Track details: everything about one song on a page - the numbers, how it
feels, how it sits against what is playing, a map of its sections with the
energy curve, and the full waveform. Opened by double-clicking a track."""
from __future__ import annotations

import logging
import os

import numpy as np
from PySide6.QtCore import QObject, QRectF, Qt, QThread, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QDialog, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from ..keys import camelot_str, compatibility, key_name
from ..styles import NEIGHBOURS, blend_score

log = logging.getLogger(__name__)

ENERGY_WORDS = ((3.0, "laid back"), (5.0, "steady"), (7.0, "driving"), (8.6, "high-energy"), (99, "peak-time"))
SECTION_COLORS = {"intro": QColor(67, 160, 71), "groove": QColor(90, 120, 200), "build": QColor(255, 193, 7),
                  "drop": QColor(229, 57, 53), "breakdown": QColor(120, 90, 200), "outro": QColor(255, 152, 0)}


def _word(scale, v):
    return next(w for lim, w in scale if v < lim)


def feel_text(lib, idx: int, row: dict | None = None, medians: dict | None = None) -> str:
    """One paragraph on how the track feels: energy, tempo, brightness,
    busyness, style family."""
    t = lib.track(idx)
    e = float(lib.energy[idx])
    parts = [f"<b>{_word(ENERGY_WORDS, e).capitalize()}</b> at {t.bpm:.0f} BPM (energy {e:.1f}/10)"]
    if row and medians:
        if row.get("centroid") and medians.get("centroid"):
            r = row["centroid"] / medians["centroid"]
            parts.append("bright, open top end" if r > 1.25 else "dark, warm tone" if r < 0.8 else "balanced tone")
        if row.get("onset_rate") and medians.get("onset_rate"):
            r = row["onset_rate"] / medians["onset_rate"]
            parts.append("busy, lots going on" if r > 1.25 else "sparse and spacious" if r < 0.8 else "even groove")
        if row.get("loudness") is not None and medians.get("loudness") is not None:
            d = row["loudness"] - medians["loudness"]
            if abs(d) >= 2:
                parts.append(f"{'louder' if d > 0 else 'quieter'} than most of the library ({d:+.0f} dB)")
    fam = lib.families[idx] if idx < len(lib.families) else ""
    style = lib.genres[idx]
    if style:
        parts.append(f"{style}" + (f" ({fam} family)" if fam and fam != style else ""))
    return " · ".join(parts) + "."


def against_text(lib, idx: int, other: int | None, label: str = "what's playing") -> str:
    """How the track would feel after `other` (the playing track): energy
    step, tempo change, key, style fit."""
    if other is None or other == idx or not (0 <= other < lib.n):
        return "Nothing is playing right now - load something on a deck and this will say how the two sit together."
    o = lib.track(other)
    t = lib.track(idx)
    de = float(lib.energy[idx] - lib.energy[other])
    if abs(de) < 0.7:
        energy = "keeps the energy where it is"
    elif de > 0:
        energy = f"a {'big ' if de > 2 else ''}lift (+{de:.1f} energy)"
    else:
        energy = f"a {'big ' if de < -2 else ''}drop ({de:.1f} energy)"
    tempo = ""
    if o.bpm > 0 and t.bpm > 0:
        r = t.bpm / o.bpm
        r = min((r, r * 2, r / 2), key=lambda x: abs(np.log(x)))
        pct = (r - 1) * 100
        tempo = "same tempo" if abs(pct) < 0.5 else f"{pct:+.1f} % tempo"
    harm = compatibility(o.key, t.key) if o.key >= 0 and t.key >= 0 else None
    key = ("" if harm is None else "in key" if harm >= 0.9 else "harmonic" if harm >= 0.5
           else "a key clash - mix short, or echo out")
    fit = blend_score(lib.genres[idx], lib.genres[other], lib.parent_genres[idx], lib.parent_genres[other])
    style = ("same style" if fit >= 0.99 else "same family" if fit >= 0.8 else "a family that blends" if fit >= 0.5
             else "a different style")
    bits = [b for b in (energy, tempo, key, style) if b]
    verdict = ("an easy next step" if (harm is None or harm >= 0.5) and fit >= 0.5 and abs(de) <= 2
               else "a bold move" if fit >= 0.5 else "a change of direction")
    return (f"After <b>{o.artist} - {o.title}</b> ({o.bpm:.0f} BPM, {camelot_str(o.key)}, energy "
            f"{lib.energy[other]:.1f}) this is {', '.join(bits)} - {verdict}.")


class _Loader(QObject):
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, path, bpm, info, db):
        super().__init__()
        self.path, self.bpm, self.info, self.db = path, bpm, info, db

    def run(self):
        try:
            from ..dj.loader import load_track
            from ..dj.structure import analyse

            lt = load_track(self.path, self.bpm, self.info, self.db)
            st = analyse(lt.bands, lt.grid, phrase_origin=lt.phrase_origin)
            self.done.emit((lt, st))
        except Exception as e:  # noqa: BLE001
            log.exception("details: loading failed")
            self.failed.emit(str(e))


class SongMap(QWidget):
    """The song's sections as coloured blocks, the energy per bar as a curve,
    drops marked, mix markers as flags; the full waveform underneath."""

    def __init__(self):
        super().__init__()
        self.setMinimumHeight(190)
        self.lt = None
        self.st = None
        self.peaks_img = None

    def set_track(self, lt, st):
        self.lt, self.st = lt, st
        self.peaks_img = None
        self.update()

    def paintEvent(self, _ev):
        p = QPainter(self)
        w, h = self.width(), self.height()
        p.fillRect(0, 0, w, h, QColor(16, 17, 20))
        if self.lt is None or self.st is None or w < 8:
            p.setPen(QColor(120, 120, 128))
            p.drawText(self.rect(), Qt.AlignCenter, "Reading the track…")
            return
        lt, st = self.lt, self.st
        dur = max(lt.audio.shape[1] / 44100.0, 1e-3)
        g = lt.grid
        map_h = 20
        curve_h = h - map_h - 80
        wave_y = map_h + curve_h + 4
        wave_h = h - wave_y

        def x_of(t):
            return t / dur * w

        # sections
        p.setFont(self.font())
        for s in st.sections:
            x0, x1 = x_of(g.time_of_beat(s.start_bar * 4)), x_of(g.time_of_beat(s.end_bar * 4))
            c = SECTION_COLORS.get(s.kind, QColor(120, 120, 120))
            p.fillRect(QRectF(x0, 0, max(x1 - x0 - 1, 1), map_h - 2), c)
            if x1 - x0 > 54:
                p.setPen(QColor(255, 255, 255))
                p.drawText(QRectF(x0 + 4, 0, x1 - x0 - 8, map_h - 2), Qt.AlignLeft | Qt.AlignVCenter,
                           f"{s.kind.upper()} {s.bars} bars")
        # energy curve per bar (+ bass as a darker line)
        if st.bars:
            xs = [x_of(g.time_of_beat(b * 4)) for b in range(st.bars + 1)]
            base = map_h + curve_h
            p.setPen(QPen(QColor(255, 255, 255, 25), 1))
            for frac in (0.25, 0.5, 0.75):
                y = base - frac * curve_h
                p.drawLine(0, int(y), w, int(y))
            for b in range(st.bars):
                e = float(np.clip(st.bar_energy[b], 0, 1))
                bass = float(np.clip(st.bar_bass[b], 0, 1))
                x0, x1 = xs[b], xs[b + 1]
                col = SECTION_COLORS.get((st.section_at(b) or st.sections[0]).kind if st.sections else "groove",
                                         QColor(120, 120, 120))
                p.fillRect(QRectF(x0, base - e * curve_h, max(x1 - x0 - 1, 1), e * curve_h),
                           QColor(col.red(), col.green(), col.blue(), 140))
                p.fillRect(QRectF(x0, base - bass * curve_h * 0.6, max(x1 - x0 - 1, 1), bass * curve_h * 0.6),
                           QColor(255, 255, 255, 40))
            if st.first_drop_bar is not None:
                x = xs[min(st.first_drop_bar, st.bars)]
                p.setPen(QPen(QColor(229, 57, 53), 2))
                p.drawLine(int(x), map_h, int(x), base)
                p.setPen(QColor(229, 57, 53))
                p.drawText(QRectF(x + 3, map_h + 2, 120, 16), Qt.AlignLeft, "first drop")
            p.setPen(QColor(170, 172, 180))
            p.drawText(QRectF(4, base - 16, 200, 14), Qt.AlignLeft, "energy per bar (white: bass)")
        # the waveform
        if self.peaks_img is None or self.peaks_img.width() != w:
            from .dj_window import _wave_image, colour_range

            n = lt.peaks.shape[0]
            cols = lt.peaks[np.linspace(0, n - 1, max(w, 1)).astype(int)]
            self.peaks_img = _wave_image(cols, max(wave_h, 8), rng=colour_range(lt.bands))
        p.drawImage(0, wave_y, self.peaks_img)
        for t, label, colr in ((lt.mix_in, "IN", QColor(67, 160, 71)), (lt.mix_out, "OUT", QColor(229, 57, 53)),
                               (lt.mix_out_end, "END", QColor(255, 152, 0))):
            if t is None:
                continue
            x = x_of(t)
            p.setPen(QPen(colr, 2))
            p.drawLine(int(x), wave_y, int(x), h)
            fx = x - 30 if label == "END" else x
            p.fillRect(QRectF(fx, wave_y, 30, 14), colr)
            p.setPen(QColor(255, 255, 255))
            p.drawText(QRectF(fx, wave_y, 30, 14), Qt.AlignCenter, label)
        # time ruler
        p.setPen(QColor(150, 152, 160))
        step = 30 if dur > 240 else 15
        for t in range(0, int(dur), step):
            x = x_of(t)
            p.drawLine(int(x), h - 4, int(x), h)
            p.drawText(QRectF(x + 2, h - 16, 50, 14), Qt.AlignLeft, f"{t // 60}:{t % 60:02d}")


class TrackDetails(QDialog):
    """The details page for one library track (non-modal)."""

    def __init__(self, main, idx: int):
        super().__init__(main)
        self.main = main
        self.lib = main.lib
        self.idx = idx
        t = self.lib.track(idx)
        self.setWindowTitle(f"{t.artist} - {t.title}" if t.artist else t.title)
        self.resize(1100, 760)
        lay = QVBoxLayout(self)
        head = QLabel(self._head_html(t))
        head.setWordWrap(True)
        head.setTextInteractionFlags(Qt.TextSelectableByMouse)
        lay.addWidget(head)
        self.feel = QLabel("")
        self.feel.setWordWrap(True)
        lay.addWidget(self.feel)
        self.against = QLabel("")
        self.against.setWordWrap(True)
        self.against.setStyleSheet("padding: 6px; background: rgba(255,255,255,0.06); border-radius: 4px;")
        lay.addWidget(self.against)
        self.map = SongMap()
        lay.addWidget(self.map, 1)
        self.structure = QLabel("")
        self.structure.setWordWrap(True)
        self.structure.setStyleSheet("color: #9a9ba3;")
        lay.addWidget(self.structure)
        row = QHBoxLayout()
        for deck in ("A", "B"):
            b = QPushButton(f"Load to Deck {deck}")
            b.clicked.connect(lambda _=False, d=deck: self.main.load_to_deck(d, self.idx))
            row.addWidget(b)
        b = QPushButton("Add to setlist")
        b.clicked.connect(lambda: self.main._append(self.idx))
        row.addWidget(b)
        b = QPushButton("Show what mixes after this")
        b.clicked.connect(lambda: (self.main.focus_track(self.idx), self.main.tabs.setCurrentIndex(0)))
        row.addWidget(b)
        row.addStretch()
        b = QPushButton("Show in folder")
        b.clicked.connect(lambda: self.main.reveal(self.idx))
        row.addWidget(b)
        b = QPushButton("Open in external player")
        b.clicked.connect(lambda: self.main.play(self.idx))
        row.addWidget(b)
        lay.addLayout(row)
        self.refresh_text()
        self._start_load(t)

    def _head_html(self, t) -> str:
        lib = self.lib
        styles = ", ".join(f"{lab.split('---')[-1]} {p:.0%}" for lab, p in lib.styles[self.idx][:5])
        fam = lib.families[self.idx]
        fam_txt = (f"<br><b>{fam}</b> family - blends with {', '.join(NEIGHBOURS[fam])}"
                   if fam in NEIGHBOURS else "")
        mins, secs = divmod(int(t.duration), 60)
        album = f" · {t.album}" if t.album else ""
        return (f"<span style='font-size:20px'><b>{t.artist}</b> - {t.title}</span>{album}<br>"
                f"<span style='font-size:26px'>{t.bpm:.2f} BPM &nbsp; {camelot_str(t.key)} {key_name(t.key)} "
                f"&nbsp; Energy {t.energy:.1f}</span> &nbsp; {mins}:{secs:02d}<br>"
                f"<small>{styles or t.tag_genre or 'no AI style yet'}{fam_txt}<br>"
                f"{os.path.basename(t.path)}</small>")

    def _medians(self):
        try:
            rows = self.lib.db.conn().execute("SELECT loudness, centroid, onset_rate FROM tracks "
                                              "WHERE status='ok'").fetchall()
            arr = np.array([[r[0] or -20, r[1] or 0, r[2] or 0] for r in rows], np.float64)
            med = np.median(arr, axis=0) if arr.size else np.zeros(3)
            me = self.lib.db.conn().execute("SELECT loudness, centroid, onset_rate FROM tracks WHERE id=?",
                                            (int(self.lib.ids[self.idx]),)).fetchone()
            row = {"loudness": me[0], "centroid": me[1], "onset_rate": me[2]} if me else None
            return row, {"loudness": float(med[0]), "centroid": float(med[1]), "onset_rate": float(med[2])}
        except Exception:  # noqa: BLE001
            return None, None

    def playing_index(self) -> int | None:
        """The library track on the master deck, if DJ mode is up and playing."""
        dj = getattr(self.main, "dj", None)
        if dj is None:
            return None
        try:
            m = dj.engine._master()
            if m is not None and m.info.lib_index >= 0:
                return int(m.info.lib_index)
            for d in dj.engine.decks.values():
                if d.loaded and d.info.lib_index >= 0:
                    return int(d.info.lib_index)
        except Exception:  # noqa: BLE001
            return None
        return None

    def refresh_text(self):
        row, med = self._medians()
        self.feel.setText(feel_text(self.lib, self.idx, row, med))
        other = self.playing_index()
        if other is None and self.main.current is not None and self.main.current != self.idx:
            self.against.setText(against_text(self.lib, self.idx, self.main.current)
                                 .replace("After", "After the focused track", 1))
        else:
            self.against.setText(against_text(self.lib, self.idx, other))

    def _start_load(self, t):
        from ..dj.engine import TrackInfo

        info = TrackInfo(path=t.path, title=t.title, artist=t.artist, key=t.key, bpm=t.bpm, lib_index=self.idx)
        self._thread = QThread(self)
        self._loader = _Loader(t.path, t.bpm, info, self.lib.db)
        self._loader.moveToThread(self._thread)
        self._thread.started.connect(self._loader.run)
        self._loader.done.connect(self._loaded)
        self._loader.failed.connect(lambda e: self.structure.setText(f"Could not read the audio: {e}"))
        self._loader.done.connect(self._thread.quit)
        self._loader.failed.connect(self._thread.quit)
        self._thread.start()

    def _loaded(self, payload):
        lt, st = payload
        self.map.set_track(lt, st)
        secs = " · ".join(f"{s.kind} {s.bars} bars" for s in st.sections)
        drop = f"first drop at bar {st.first_drop_bar + 1}" if st.first_drop_bar is not None else "no clear drop"
        self.structure.setText(f"Sections: {secs}. {drop.capitalize()}; "
                               f"{st.bars} bars at {lt.grid.bpm:.2f} BPM.")

    def closeEvent(self, ev):
        th = getattr(self, "_thread", None)
        if th is not None and th.isRunning():
            th.quit()
            th.wait(5000)
        super().closeEvent(ev)
