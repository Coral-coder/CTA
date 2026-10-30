"""Main window: library browser, next-track suggestions and setlist builder."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import time

import numpy as np
from PySide6.QtCore import QObject, QSettings, Qt, QThread, QTimer, QUrl, Signal
from PySide6.QtGui import QAction, QDesktopServices, QKeySequence
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMenu,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSlider,
    QSpinBox,
    QSplitter,
    QTableView,
    QTabWidget,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from .. import __version__, paths
from ..db import Database
from ..export import export
from ..keys import camelot_str, key_name, parse_key
from ..library import Library
from ..mixer import ENERGY_CURVES, MixSettings, SetlistOptions, build_setlist, suggest_next, transition_report
from ..scanner import Scanner, ScanSettings, default_workers
from .models import LibraryModel, TrackListModel
from .update_ui import UpdateController

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# background workers
# --------------------------------------------------------------------------

class ScanWorker(QObject):
    progress = Signal(object)
    finished = Signal(object)

    def __init__(self, db: Database, folders: list[str], settings: ScanSettings):
        super().__init__()
        self.scanner = Scanner(db, settings, on_progress=self._prog)
        self.folders = folders
        self._last = 0.0

    def _prog(self, p):
        now = time.time()
        if now - self._last > 0.1 or p.phase not in ("analyzing",):
            self._last = now
            self.progress.emit(p)

    def run(self):
        res = None
        try:
            res = self.scanner.run(self.folders)
        except Exception:  # noqa: BLE001
            log.exception("scan crashed")
        self.finished.emit(res)


class LoadWorker(QObject):
    loaded = Signal(object)

    def __init__(self, db: Database):
        super().__init__()
        self.db = db

    def run(self):
        try:
            lib = Library(self.db)
        except Exception:  # noqa: BLE001
            log.exception("library load failed")
            lib = None
        self.loaded.emit(lib)


def _table(model) -> QTableView:
    t = QTableView()
    t.setModel(model)
    t.setSelectionBehavior(QAbstractItemView.SelectRows)
    t.setSelectionMode(QAbstractItemView.SingleSelection)
    t.setAlternatingRowColors(True)
    t.setShowGrid(False)
    t.setWordWrap(False)
    t.verticalHeader().setDefaultSectionSize(22)
    t.horizontalHeader().setHighlightSections(False)
    t.horizontalHeader().setStretchLastSection(True)
    t.setEditTriggers(QAbstractItemView.NoEditTriggers)
    return t


# --------------------------------------------------------------------------
# settings dialog
# --------------------------------------------------------------------------

class SettingsDialog(QDialog):
    def __init__(self, settings: QSettings, db: Database, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.s = settings
        self.db = db
        lay = QVBoxLayout(self)

        lay.addWidget(QLabel("<b>Music folders</b>"))
        self.folders = QListWidget()
        self.folders.addItems(db.folders())
        lay.addWidget(self.folders)
        row = QHBoxLayout()
        b_add = QPushButton("Add folder…")
        b_rem = QPushButton("Remove selected")
        b_add.clicked.connect(self._add)
        b_rem.clicked.connect(self._remove)
        row.addWidget(b_add)
        row.addWidget(b_rem)
        row.addStretch()
        lay.addLayout(row)

        form = QFormLayout()
        self.workers = QSpinBox()
        self.workers.setRange(0, 61)
        self.workers.setSpecialValueText(f"Auto ({default_workers()})")
        self.workers.setValue(int(settings.value("workers", 0)))
        form.addRow("Analysis processes", self.workers)
        self.min_bpm = QDoubleSpinBox()
        self.min_bpm.setRange(40, 150)
        self.min_bpm.setValue(float(settings.value("min_bpm", 70)))
        self.max_bpm = QDoubleSpinBox()
        self.max_bpm.setRange(90, 250)
        self.max_bpm.setValue(float(settings.value("max_bpm", 180)))
        form.addRow("BPM range: min", self.min_bpm)
        form.addRow("BPM range: max", self.max_bpm)
        self.use_ai = QCheckBox("Use AI genre / sound analysis")
        self.use_ai.setChecked(settings.value("use_ai", "true") == "true")
        form.addRow(self.use_ai)
        self.device = QComboBox()
        self.device.addItems(["auto", "cpu"])
        self.device.setCurrentText(settings.value("ai_device", "auto"))
        form.addRow("AI device", self.device)
        self.ui_scale = QComboBox()
        for pct in (70, 80, 90, 100, 110, 125, 150):
            self.ui_scale.addItem(f"{pct} %", pct)
        cur = int(settings.value("ui_scale", 100))
        self.ui_scale.setCurrentIndex(max(0, self.ui_scale.findData(cur)))
        self.ui_scale.setToolTip("Make everything smaller to fit a laptop screen (or bigger). Applies after a restart.")
        form.addRow("Interface size (restart)", self.ui_scale)
        self.auto_update = QCheckBox("Check for updates automatically")
        self.auto_update.setChecked(settings.value("auto_update_check", "true") == "true")
        form.addRow(self.auto_update)
        lay.addLayout(form)

        lay.addWidget(QLabel("<b>Mix server (phone remote)</b>"))
        from .. import autostart

        self.server_mode = QCheckBox("Run the mix server whenever CTA is open (keeps running in the tray)")
        self.server_mode.setChecked(settings.value("server_mode", "false") == "true")
        self.login = QCheckBox("Start CTA in the background when I log in")
        self.login.setToolTip("CTA starts hidden in the system tray with the phone remote running")
        self.login.setChecked(autostart.is_enabled())
        self.login.toggled.connect(lambda on: on and self.server_mode.setChecked(True))
        lay.addWidget(self.server_mode)
        lay.addWidget(self.login)
        lay.addWidget(QLabel(f"<small>Data folder: {paths.data_dir()}</small>"))

        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        self.resize(560, 460)

    def _add(self):
        d = QFileDialog.getExistingDirectory(self, "Add music folder")
        if d:
            d = os.path.normpath(d)
            self.db.add_folder(d)
            self.folders.clear()
            self.folders.addItems(self.db.folders())

    def _remove(self):
        for it in self.folders.selectedItems():
            self.db.remove_folder(it.text())
        self.folders.clear()
        self.folders.addItems(self.db.folders())

    def accept(self):
        self.s.setValue("workers", self.workers.value())
        self.s.setValue("min_bpm", self.min_bpm.value())
        self.s.setValue("max_bpm", self.max_bpm.value())
        self.s.setValue("use_ai", "true" if self.use_ai.isChecked() else "false")
        self.s.setValue("ai_device", self.device.currentText())
        self.s.setValue("auto_update_check", "true" if self.auto_update.isChecked() else "false")
        self.s.setValue("server_mode", "true" if self.server_mode.isChecked() else "false")
        if int(self.s.value("ui_scale", 100)) != self.ui_scale.currentData():
            self.s.setValue("ui_scale", self.ui_scale.currentData())
            QMessageBox.information(self, "Interface size", "The new interface size applies when CTA restarts.")
        from .. import autostart

        if self.login.isChecked() != autostart.is_enabled():
            try:
                autostart.set_enabled(self.login.isChecked())
            except OSError as e:
                QMessageBox.warning(self, "Settings", f"Could not change the start-up setting:\n{e}")
        super().accept()


# --------------------------------------------------------------------------
# main window
# --------------------------------------------------------------------------

class MainWindow(QMainWindow):
    def __init__(self, db: Database, settings: QSettings | None = None):
        super().__init__()
        self.db = db
        self.settings = settings if settings is not None else QSettings("CTA", "CTA")
        self.lib: Library | None = None
        self.current: int | None = None          # library index of the focused track
        self.setlist: list[int] = []
        self._scan_thread: QThread | None = None
        self._scan_worker: ScanWorker | None = None
        self._load_thread: QThread | None = None
        self.tray = None                         # system tray (set by the app when available)
        self._quitting = False
        self._want_server = False

        self.setWindowTitle(f"CTA {__version__} - DJ Library & Setlist Builder")
        self.resize(1500, 900)
        self.updates = UpdateController(self, self.settings)
        self._build_ui()
        geo = self.settings.value("geometry")
        if geo is not None:
            self.restoreGeometry(geo)
        self.reload_library()
        QTimer.singleShot(0, self.updates.startup)

    def is_scanning(self) -> bool:
        return self._scan_thread is not None

    # ---------------------------------------------------------------- UI --
    def _build_ui(self):
        tb = QToolBar("Main")
        tb.setMovable(False)
        self.addToolBar(tb)
        self.act_add = QAction("Add Folder…", self)
        self.act_add.triggered.connect(self.add_folder)
        self.act_scan = QAction("Scan Library", self)
        self.act_scan.setShortcut(QKeySequence("Ctrl+R"))
        self.act_scan.triggered.connect(lambda: self.start_scan())
        self.act_stop = QAction("Stop", self)
        self.act_stop.setEnabled(False)
        self.act_stop.triggered.connect(self.stop_scan)
        self.act_settings = QAction("Settings…", self)
        self.act_settings.triggered.connect(self.open_settings)
        self.act_dj = QAction("🎧 DJ Mode", self)
        self.act_dj.setShortcut(QKeySequence("Ctrl+D"))
        self.act_dj.triggered.connect(self.toggle_dj)
        for keyseq, deck in (("Ctrl+Left", "A"), ("Ctrl+Right", "B"), ("Ctrl+Shift+Left", "C"),
                             ("Ctrl+Shift+Right", "D")):
            a = QAction(self)
            a.setShortcut(QKeySequence(keyseq))
            a.triggered.connect(lambda _=False, deck=deck: self.current is not None
                                and self.load_to_deck(deck, self.current))
            self.addAction(a)
        self.act_update = QAction("Check for Updates", self)
        self.act_update.triggered.connect(lambda: self.updates.check(manual=True))
        for a in (self.act_add, self.act_scan, self.act_stop, self.act_dj):
            tb.addAction(a)
        tb.addSeparator()

        self.search = QLineEdit()
        self.search.setPlaceholderText("Search artist, title, album, genre…  (Ctrl+F)")
        self.search.setClearButtonEnabled(True)
        self.search.setMinimumWidth(320)
        self._search_timer = QTimer(self, singleShot=True, interval=180)
        self._search_timer.timeout.connect(self.apply_filter)
        self.search.textChanged.connect(lambda _: self._search_timer.start())
        tb.addWidget(self.search)
        f_find = QAction(self)
        f_find.setShortcut(QKeySequence("Ctrl+F"))
        f_find.triggered.connect(lambda: (self.search.setFocus(), self.search.selectAll()))
        self.addAction(f_find)

        tb.addWidget(QLabel("  BPM "))
        self.f_bpm_min = QDoubleSpinBox()
        self.f_bpm_min.setRange(0, 300)
        self.f_bpm_min.setSpecialValueText("min")
        self.f_bpm_max = QDoubleSpinBox()
        self.f_bpm_max.setRange(0, 300)
        self.f_bpm_max.setSpecialValueText("max")
        for w in (self.f_bpm_min, self.f_bpm_max):
            w.setDecimals(0)
            w.valueChanged.connect(lambda _: self._search_timer.start())
            tb.addWidget(w)
        tb.addWidget(QLabel("  Genre "))
        self.f_genre = QComboBox()
        self.f_genre.setMinimumWidth(140)
        self.f_genre.currentIndexChanged.connect(lambda _: self.apply_filter())
        tb.addWidget(self.f_genre)
        self.f_compat = QCheckBox("Only keys compatible with focused track")
        self.f_compat.toggled.connect(lambda _: self.apply_filter())
        tb.addWidget(self.f_compat)
        tb.addSeparator()
        tb.addAction(self.act_settings)
        tb.addAction(self.act_update)

        # library table
        self.lib_model = LibraryModel(self)
        self.lib_view = _table(self.lib_model)
        self.lib_view.setSortingEnabled(True)
        self.lib_view.sortByColumn(-1, Qt.AscendingOrder)
        self.lib_view.selectionModel().currentRowChanged.connect(self._on_lib_row)
        self.lib_view.doubleClicked.connect(lambda ix: self.play(self.lib_model.lib_index(ix.row())))
        self.lib_view.setContextMenuPolicy(Qt.CustomContextMenu)
        self.lib_view.customContextMenuRequested.connect(self._lib_menu)
        self.lib_view.setDragEnabled(True)
        self.lib_view.setDragDropMode(QAbstractItemView.DragOnly)
        h = self.lib_view.horizontalHeader()
        for col, w in enumerate([62, 170, 260, 70, 80, 60, 140, 110, 55, 160]):
            h.resizeSection(col, w)

        # right side tabs
        self.tabs = QTabWidget()
        def scrolled(w):
            # tab pages scroll instead of forcing a minimum height (small screens)
            from PySide6.QtWidgets import QScrollArea

            sa = QScrollArea()
            sa.setWidget(w)
            sa.setWidgetResizable(True)
            sa.setFrameShape(QFrame.NoFrame)
            return sa
        self.tabs.addTab(scrolled(self._build_next_tab()), "Next Track")
        self.tabs.addTab(scrolled(self._build_setlist_tab()), "Setlist Builder")

        split = QSplitter(Qt.Horizontal)
        split.addWidget(self.lib_view)
        split.addWidget(self.tabs)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        # decks above, library below, with a handle to give either more room
        self.vsplit = QSplitter(Qt.Vertical)
        self.vsplit.setChildrenCollapsible(False)
        self.vsplit.setHandleWidth(7)
        self.vsplit.addWidget(split)
        self.setCentralWidget(self.vsplit)

        # status bar
        self.status_label = QLabel("Ready")
        self.device_label = QLabel("")
        self.progress = QProgressBar()
        self.progress.setMaximumWidth(320)
        self.progress.setVisible(False)
        sb = self.statusBar()
        sb.addWidget(self.status_label, 1)
        sb.addPermanentWidget(self.device_label)
        sb.addPermanentWidget(self.progress)

    def _build_next_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        self.now_label = QLabel("Select a track in the library.")
        self.now_label.setWordWrap(True)
        self.now_label.setStyleSheet("font-size: 14px; padding: 4px;")
        lay.addWidget(self.now_label)

        row = QHBoxLayout()
        row.addWidget(QLabel("Pitch range ±%"))
        self.pitch = QDoubleSpinBox()
        self.pitch.setRange(0.5, 50)
        self.pitch.setValue(float(self.settings.value("pitch", 6.0)))
        self.pitch.valueChanged.connect(self._mix_changed)
        row.addWidget(self.pitch)
        self.strict_key = QCheckBox("Harmonic only")
        self.strict_key.setChecked(self.settings.value("strict_key", "false") == "true")
        self.strict_key.toggled.connect(self._mix_changed)
        row.addWidget(self.strict_key)
        self.half_double = QCheckBox("Half/double time")
        self.half_double.setChecked(self.settings.value("half_double", "true") == "true")
        self.half_double.toggled.connect(self._mix_changed)
        row.addWidget(self.half_double)
        self.genre_lock = QCheckBox("Same genre family")
        self.genre_lock.setChecked(self.settings.value("genre_lock", "false") == "true")
        self.genre_lock.toggled.connect(self._mix_changed)
        row.addWidget(self.genre_lock)
        row.addStretch()
        lay.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Energy"))
        self.energy_dir = QComboBox()
        self.energy_dir.addItems(["Keep level", "Lift (+1)", "Big lift (+2)", "Drop (-1)", "Big drop (-2)"])
        self.energy_dir.setCurrentIndex(int(self.settings.value("energy_dir", 0)))
        self.energy_dir.currentIndexChanged.connect(self._mix_changed)
        row.addWidget(self.energy_dir)
        row.addWidget(QLabel("  Weights: tempo"))
        self.w_sliders = {}
        for name, default in (("tempo", 30), ("key", 25), ("sound", 25), ("genre", 10)):
            if name != "tempo":
                row.addWidget(QLabel(name))
            sl = QSlider(Qt.Horizontal)
            sl.setRange(0, 100)
            sl.setValue(int(self.settings.value(f"w_{name}", default)))
            sl.setMaximumWidth(70)
            sl.valueChanged.connect(self._mix_changed)
            self.w_sliders[name] = sl
            row.addWidget(sl)
        row.addStretch()
        lay.addLayout(row)

        self.next_model = TrackListModel(["Match", "Pitch"], self)
        self.next_model.header_tips = {"Match": "Overall mix compatibility (0-100)",
                                       "Pitch": "Tempo change to apply to this track to match the focused one"}
        self.next_view = _table(self.next_model)
        self.next_view.doubleClicked.connect(self._next_activated)
        self.next_view.setContextMenuPolicy(Qt.CustomContextMenu)
        self.next_view.setDragEnabled(True)
        self.next_view.setDragDropMode(QAbstractItemView.DragOnly)
        self.next_view.customContextMenuRequested.connect(self._next_menu)
        h = self.next_view.horizontalHeader()
        for col, wd in enumerate([55, 60, 60, 75, 50, 110, 140, 200, 50]):
            h.resizeSection(col, wd)
        lay.addWidget(self.next_view, 1)

        row = QHBoxLayout()
        b_follow = QPushButton("Follow →  (mix into selected)")
        b_follow.setToolTip("Make the selected suggestion the focused track and show what comes after it")
        b_follow.clicked.connect(lambda: self._next_activated(self.next_view.currentIndex()))
        b_add = QPushButton("Add to setlist")
        b_add.clicked.connect(self._add_suggestion_to_setlist)
        row.addWidget(b_follow)
        row.addWidget(b_add)
        row.addStretch()
        lay.addLayout(row)
        lay.addWidget(QLabel("<small>Double-click a suggestion to follow it. Double-click a library "
                             "track to play it in your default player.</small>"))
        return w

    def _build_setlist_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        form = QFormLayout()
        self.seed_label = QLabel("Seed: (focused track)")
        form.addRow(self.seed_label)
        self.sl_len = QSpinBox()
        self.sl_len.setRange(2, 500)
        self.sl_len.setValue(int(self.settings.value("sl_len", 20)))
        form.addRow("Tracks", self.sl_len)
        self.sl_end_bpm = QDoubleSpinBox()
        self.sl_end_bpm.setRange(0, 250)
        self.sl_end_bpm.setDecimals(1)
        self.sl_end_bpm.setSpecialValueText("off (stay near seed tempo)")
        self.sl_end_bpm.setValue(float(self.settings.value("sl_end_bpm", 0)))
        form.addRow("Ramp tempo to BPM", self.sl_end_bpm)
        self.sl_curve = QComboBox()
        self.sl_curve.addItems(ENERGY_CURVES)
        self.sl_curve.setCurrentText(self.settings.value("sl_curve", "steady"))
        form.addRow("Energy curve", self.sl_curve)
        self.sl_variety = QSlider(Qt.Horizontal)
        self.sl_variety.setRange(0, 100)
        self.sl_variety.setValue(int(self.settings.value("sl_variety", 15)))
        form.addRow("Variety", self.sl_variety)
        lay.addLayout(form)

        row = QHBoxLayout()
        b_gen = QPushButton("Generate Setlist")
        b_gen.setDefault(True)
        b_gen.setStyleSheet("font-weight: bold; padding: 6px 14px;")
        b_gen.clicked.connect(self.generate_setlist)
        b_cont = QPushButton("Extend from last")
        b_cont.setToolTip("Append more tracks continuing from the last track")
        b_cont.clicked.connect(self.extend_setlist)
        row.addWidget(b_gen)
        row.addWidget(b_cont)
        row.addStretch()
        lay.addLayout(row)

        self.sl_model = TrackListModel(["Mix", "Pitch"], self)
        self.sl_view = _table(self.sl_model)
        self.sl_view.doubleClicked.connect(lambda ix: self.play(self.setlist[ix.row()]))
        self.sl_view.setContextMenuPolicy(Qt.CustomContextMenu)
        self.sl_view.setDragEnabled(True)
        self.sl_view.setDragDropMode(QAbstractItemView.DragOnly)
        self.sl_view.customContextMenuRequested.connect(self._setlist_menu)
        h = self.sl_view.horizontalHeader()
        for col, wd in enumerate([45, 60, 60, 75, 50, 110, 140, 200, 50]):
            h.resizeSection(col, wd)
        lay.addWidget(self.sl_view, 1)

        row = QHBoxLayout()
        for text, fn in (("▲", lambda: self._move(-1)), ("▼", lambda: self._move(1)),
                         ("Remove", self._remove_from_setlist), ("Clear", self._clear_setlist)):
            b = QPushButton(text)
            b.clicked.connect(fn)
            row.addWidget(b)
        row.addStretch()
        self.sl_info = QLabel("")
        row.addWidget(self.sl_info)
        b_exp = QPushButton("Export…")
        b_exp.setStyleSheet("font-weight: bold;")
        b_exp.clicked.connect(self.export_setlist)
        row.addWidget(b_exp)
        lay.addLayout(row)
        return w

    # ----------------------------------------------------------- library --
    def reload_library(self):
        if self._load_thread is not None:
            return
        self.status_label.setText("Loading library…")
        th = QThread(self)
        wk = LoadWorker(self.db)
        wk.moveToThread(th)
        th.started.connect(wk.run)
        wk.loaded.connect(self._on_loaded)
        wk.loaded.connect(th.quit)
        th.finished.connect(lambda: setattr(self, "_load_thread", None))
        self._load_thread = th
        self._load_worker = wk
        th.start()

    def _on_loaded(self, lib):
        if lib is None:
            self.status_label.setText("Failed to load library - see log")
            return
        prev_path = self.lib.paths[self.current] if (self.lib is not None and self.current is not None) else None
        self.lib = lib
        self.lib_model.set_library(lib)
        g = self.f_genre.currentText()
        self.f_genre.blockSignals(True)
        self.f_genre.clear()
        self.f_genre.addItem("")
        self.f_genre.addItems([x for x in lib.parent_names if x])
        self.f_genre.setCurrentText(g)
        self.f_genre.blockSignals(False)
        # keep the setlist only if all tracks still exist
        self.current = None
        if prev_path and prev_path in lib.paths:
            self.focus_track(lib.paths.index(prev_path))
        self.setlist = []
        self._refresh_setlist()
        ai = int(lib.has_ai.sum())
        msg = f"{lib.n} tracks analysed"
        if lib.n:
            msg += f" · {ai} with AI genre/sound data"
        if lib.errors:
            msg += f" · {len(lib.errors)} could not be read"
        self.status_label.setText(msg)
        if (lib.n == 0 and not self.db.folders() and self._scan_thread is None
                and not os.environ.get("CTA_SMOKE_EXIT_MS") and self.isVisible()):
            QTimer.singleShot(300, self._first_run)
        self.apply_filter()
        if self._want_server:
            self._want_server = False
            QTimer.singleShot(0, self.start_server)

    def _first_run(self):
        r = QMessageBox.question(
            self, "Welcome to CTA",
            "Your library is empty.\n\nAdd your music folder now? CTA will analyse BPM, key, energy "
            "and genre for every track. Re-scans only process new or changed files.")
        if r == QMessageBox.Yes:
            self.add_folder()

    def apply_filter(self):
        if self.lib is None:
            return
        keys = None
        if self.f_compat.isChecked() and self.current is not None:
            from ..keys import compatibility
            k = int(self.lib.key[self.current])
            keys = {j for j in range(24) if compatibility(k, j) >= 0.55}
        self.lib_model.set_filter(text=self.search.text().strip(), bpm_min=self.f_bpm_min.value(),
                                  bpm_max=self.f_bpm_max.value(), genre=self.f_genre.currentText(), keys=keys)
        if self.current is not None:
            r = self.lib_model.row_of(self.current)
            if r >= 0:
                self.lib_view.selectionModel().blockSignals(True)
                self.lib_view.selectRow(r)
                self.lib_view.selectionModel().blockSignals(False)

    def _on_lib_row(self, cur, _prev):
        if cur.isValid():
            self.focus_track(self.lib_model.lib_index(cur.row()), select=False)

    def focus_track(self, idx: int, select: bool = True):
        if self.lib is None or idx is None or idx >= self.lib.n:
            return
        self.current = idx
        t = self.lib.track(idx)
        styles = ", ".join(f"{lab.split('---')[-1]} {p:.0%}" for lab, p in self.lib.styles[idx][:3])
        self.now_label.setText(
            f"<b>{t.artist} - {t.title}</b><br>"
            f"<span style='font-size:20px'>{t.bpm:.2f} BPM &nbsp; {camelot_str(t.key)} {key_name(t.key)} "
            f"&nbsp; Energy {t.energy:.1f}</span><br><small>{styles or t.tag_genre}</small>")
        self.seed_label.setText(f"Seed: <b>{t.artist} - {t.title}</b> ({t.bpm:.1f} BPM, {camelot_str(t.key)})")
        if select:
            r = self.lib_model.row_of(idx)
            if r < 0:   # filtered out: clear filters so it can be shown
                self.search.blockSignals(True)
                self.search.clear()
                self.search.blockSignals(False)
                self.f_bpm_min.setValue(0)
                self.f_bpm_max.setValue(0)
                self.f_genre.setCurrentIndex(0)
                self.apply_filter()
                r = self.lib_model.row_of(idx)
            if r >= 0:
                self.lib_view.selectionModel().blockSignals(True)
                self.lib_view.selectRow(r)
                self.lib_view.scrollTo(self.lib_model.index(r, 0), QAbstractItemView.PositionAtCenter)
                self.lib_view.selectionModel().blockSignals(False)
        if self.f_compat.isChecked():
            self.apply_filter()
        self.refresh_suggestions()

    # ------------------------------------------------------------ mixing --
    def mix_settings(self) -> MixSettings:
        steps = [0.0, 1.0, 2.0, -1.0, -2.0]
        w = {k: s.value() / 100.0 for k, s in self.w_sliders.items()}
        return MixSettings(max_pitch_pct=self.pitch.value(), allow_half_double=self.half_double.isChecked(),
                           strict_key=self.strict_key.isChecked(), genre_lock=self.genre_lock.isChecked(),
                           energy_step=steps[self.energy_dir.currentIndex()], w_tempo=w["tempo"],
                           w_key=w["key"], w_sound=w["sound"], w_genre=w["genre"], w_energy=0.10)

    def _mix_changed(self, *_):
        s = self.settings
        s.setValue("pitch", self.pitch.value())
        s.setValue("strict_key", "true" if self.strict_key.isChecked() else "false")
        s.setValue("half_double", "true" if self.half_double.isChecked() else "false")
        s.setValue("genre_lock", "true" if self.genre_lock.isChecked() else "false")
        s.setValue("energy_dir", self.energy_dir.currentIndex())
        for k, sl in self.w_sliders.items():
            s.setValue(f"w_{k}", sl.value())
        self.refresh_suggestions()

    def refresh_suggestions(self):
        if self.lib is None or self.current is None:
            self.next_model.set_items(self.lib, [], [])
            return
        t0 = time.perf_counter()
        sug = suggest_next(self.lib, self.current, self.mix_settings(), n=100, exclude=set(self.setlist))
        dt = (time.perf_counter() - t0) * 1000
        self.next_model.set_items(self.lib, [s.idx for s in sug],
                                  [[f"{s.score * 100:.0f}", f"{s.pitch_pct:+.1f}%"] for s in sug])
        self.tabs.setTabText(0, f"Next Track ({len(sug)})")
        log.debug("suggestions in %.1f ms", dt)

    def _next_activated(self, ix):
        if not ix.isValid():
            return
        self.focus_track(self.next_model.items[ix.row()])

    def _add_suggestion_to_setlist(self):
        ix = self.next_view.currentIndex()
        if not ix.isValid():
            return
        i = self.next_model.items[ix.row()]
        if not self.setlist and self.current is not None:
            self.setlist.append(self.current)
        if i not in self.setlist:
            self.setlist.append(i)
        self._refresh_setlist()
        self.focus_track(i)

    # ----------------------------------------------------------- setlist --
    def _setlist_options(self, length: int) -> SetlistOptions:
        s = self.settings
        s.setValue("sl_len", self.sl_len.value())
        s.setValue("sl_end_bpm", self.sl_end_bpm.value())
        s.setValue("sl_curve", self.sl_curve.currentText())
        s.setValue("sl_variety", self.sl_variety.value())
        return SetlistOptions(length=length, end_bpm=self.sl_end_bpm.value() or None,
                              energy_curve=self.sl_curve.currentText(),
                              variety=self.sl_variety.value() / 100.0, mix=self.mix_settings())

    def generate_setlist(self):
        if self.lib is None or self.current is None:
            QMessageBox.information(self, "Setlist", "Select a seed track in the library first.")
            return
        t0 = time.perf_counter()
        self.setlist = build_setlist(self.lib, self.current, self._setlist_options(self.sl_len.value()),
                                     anchor_w=0.35)
        dt = (time.perf_counter() - t0) * 1000
        self._refresh_setlist()
        self.tabs.setCurrentIndex(1)
        self.status_label.setText(f"Setlist of {len(self.setlist)} tracks built in {dt:.0f} ms")
        if len(self.setlist) < min(self.sl_len.value(), self.lib.n):
            QMessageBox.information(
                self, "Setlist",
                f"Only {len(self.setlist)} tracks could be chained within ±{self.pitch.value():g}% tempo"
                + (" with harmonic-only mixing" if self.strict_key.isChecked() else "")
                + ".\n\nWiden the pitch range, allow half/double time or relax the filters for a longer set.")

    def extend_setlist(self):
        if self.lib is None:
            return
        if not self.setlist:
            return self.generate_setlist()
        opt = self._setlist_options(self.sl_len.value() + 1)
        more = build_setlist(self.lib, self.setlist[-1], opt, exclude=set(self.setlist), anchor_w=0.35,
                             anchor_idx=self.setlist[0])
        for i in more[1:]:
            if i not in self.setlist:
                self.setlist.append(i)
        self._refresh_setlist()

    def _refresh_setlist(self):
        if self.lib is None:
            return
        rep = transition_report(self.lib, self.setlist, self.mix_settings()) if len(self.setlist) > 1 else []
        extras = [["seed", ""]] if self.setlist else []
        extras += [[f"{r['score'] * 100:.0f}", f"{r['pitch_pct']:+.1f}%"] for r in rep]
        self.sl_model.set_items(self.lib, self.setlist, extras)
        total = sum(self.lib.duration[i] for i in self.setlist)
        self.sl_info.setText(f"{len(self.setlist)} tracks · {int(total // 3600)}h {int(total % 3600 // 60):02d}m  ")
        self.tabs.setTabText(1, f"Setlist Builder ({len(self.setlist)})")

    def _move(self, d):
        r = self.sl_view.currentIndex().row()
        j = r + d
        if r < 0 or j < 0 or j >= len(self.setlist):
            return
        self.setlist[r], self.setlist[j] = self.setlist[j], self.setlist[r]
        self._refresh_setlist()
        self.sl_view.selectRow(j)

    def _remove_from_setlist(self):
        r = self.sl_view.currentIndex().row()
        if 0 <= r < len(self.setlist):
            del self.setlist[r]
            self._refresh_setlist()
            self.sl_view.selectRow(min(r, len(self.setlist) - 1))

    def _clear_setlist(self):
        self.setlist = []
        self._refresh_setlist()

    def export_setlist(self):
        if not self.setlist or self.lib is None:
            QMessageBox.information(self, "Export", "The setlist is empty.")
            return
        last = self.settings.value("export_dir", os.path.expanduser("~"))
        seed = self.lib.track(self.setlist[0])
        name = f"CTA {time.strftime('%Y-%m-%d %H%M')} - {seed.title}"[:80]
        safe = "".join(c for c in name if c not in '<>:"/\\|?*')
        path, flt = QFileDialog.getSaveFileName(
            self, "Export setlist", os.path.join(last, safe + ".m3u8"),
            "M3U8 playlist (*.m3u8);;Traktor NML (*.nml);;rekordbox XML (*.xml);;CSV spreadsheet (*.csv)")
        if not path:
            return
        if not os.path.splitext(path)[1]:
            path += {"M3U8": ".m3u8", "Trakt": ".nml", "rekor": ".xml", "CSV s": ".csv"}.get(flt[:5], ".m3u8")
        try:
            export(self.lib, self.setlist, path, name)
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "Export failed", str(e))
            return
        self.settings.setValue("export_dir", os.path.dirname(path))
        self.status_label.setText(f"Exported {len(self.setlist)} tracks to {path}")

    # ------------------------------------------------------ context menus --
    def _add_deck_actions(self, m: QMenu, i: int):
        """'Load to Deck X' entries; a playing deck is shown but disabled."""
        dj = getattr(self, "dj", None)
        decks = "ABCD" if dj is not None and dj.show_cd.isChecked() else "AB"
        keys = {"A": "Ctrl+Left", "B": "Ctrl+Right", "C": "Ctrl+Shift+Left", "D": "Ctrl+Shift+Right"}
        for deck in decks:
            playing = dj is not None and dj.deck_playing(deck)
            act = m.addAction(f"Load to Deck {deck}" + ("   (playing)" if playing else f"\t{keys[deck]}"),
                              lambda deck=deck: self.load_to_deck(deck, i))
            act.setEnabled(not playing)

    def _lib_menu(self, pos):
        ix = self.lib_view.indexAt(pos)
        if not ix.isValid() or self.lib is None:
            return
        i = self.lib_model.lib_index(ix.row())
        m = QMenu(self)
        m.addAction("Build setlist from this track", lambda: (self.focus_track(i), self.generate_setlist()))
        m.addAction("Add to end of setlist", lambda: self._append(i))
        m.addSeparator()
        self._add_deck_actions(m, i)
        m.addSeparator()
        m.addAction("Preview in default player", lambda: self.play(i))
        m.addAction("Show in folder", lambda: self.reveal(i))
        m.addSeparator()
        b = self.lib.bpm[i]
        m.addAction(f"Double BPM  ({b:.1f} → {b * 2:.1f})", lambda: self._set_bpm(i, round(b * 2, 2)))
        m.addAction(f"Halve BPM  ({b:.1f} → {b / 2:.1f})", lambda: self._set_bpm(i, round(b / 2, 2)))
        m.addAction("Set BPM…", lambda: self._ask_bpm(i))
        m.addAction("Set key…", lambda: self._ask_key(i))
        if self.lib.bpm_overridden[i] or self.lib.key_overridden[i]:
            m.addAction("Reset to detected BPM/key", lambda: self._reset_override(i))
        m.exec(self.lib_view.viewport().mapToGlobal(pos))

    def _next_menu(self, pos):
        ix = self.next_view.indexAt(pos)
        if not ix.isValid():
            return
        i = self.next_model.items[ix.row()]
        m = QMenu(self)
        self._add_deck_actions(m, i)
        m.addSeparator()
        m.addAction("Follow (show what mixes after this)", lambda: self.focus_track(i))
        m.addAction("Add to setlist", self._add_suggestion_to_setlist)
        m.addAction("Preview in default player", lambda: self.play(i))
        m.exec(self.next_view.viewport().mapToGlobal(pos))

    def _setlist_menu(self, pos):
        ix = self.sl_view.indexAt(pos)
        if not ix.isValid():
            return
        r = ix.row()
        i = self.setlist[r]
        m = QMenu(self)
        self._add_deck_actions(m, i)
        m.addSeparator()
        m.addAction("Preview in default player", lambda: self.play(i))
        m.addAction("Show next-track options for this", lambda: (self.focus_track(i), self.tabs.setCurrentIndex(0)))
        m.addAction("Regenerate everything after this", lambda: self._regen_after(r))
        m.addAction("Remove", self._remove_from_setlist)
        m.exec(self.sl_view.viewport().mapToGlobal(pos))

    def _regen_after(self, r):
        keep = self.setlist[: r + 1]
        opt = self._setlist_options(max(2, self.sl_len.value() - r))
        more = build_setlist(self.lib, keep[-1], opt, exclude=set(keep), anchor_w=0.35, anchor_idx=keep[0])
        self.setlist = keep + [i for i in more[1:] if i not in keep]
        self._refresh_setlist()

    def _append(self, i):
        if i not in self.setlist:
            self.setlist.append(i)
            self._refresh_setlist()

    def _set_bpm(self, i, bpm):
        self.db.set_override(int(self.lib.ids[i]), bpm=float(bpm))
        self._patch_track(i)

    def _ask_bpm(self, i):
        v, ok = QInputDialog.getDouble(self, "Set BPM", "BPM:", float(self.lib.bpm[i]), 20, 400, 2)
        if ok:
            self._set_bpm(i, v)

    def _ask_key(self, i):
        cur = f"{camelot_str(int(self.lib.key[i]))}"
        v, ok = QInputDialog.getText(self, "Set key", "Key (e.g. 8A, Am, F#m, Eb):", text=cur)
        if ok:
            k = parse_key(v)
            if k < 0:
                QMessageBox.warning(self, "Set key", f"Could not understand key '{v}'.")
                return
            self.db.set_override(int(self.lib.ids[i]), key=k)
            self._patch_track(i)

    def _reset_override(self, i):
        self.db.set_override(int(self.lib.ids[i]), clear_bpm=True, clear_key=True)
        self._patch_track(i)

    def _patch_track(self, i):
        """Apply an override to the in-memory library without a full reload."""
        r = self.db.conn().execute("SELECT bpm, bpm_override, key, key_override FROM tracks WHERE id=?",
                                   (int(self.lib.ids[i]),)).fetchone()
        lib = self.lib
        lib.bpm_overridden[i] = r["bpm_override"] is not None
        lib.bpm[i] = r["bpm_override"] if r["bpm_override"] is not None else (r["bpm"] or 0)
        lib.key_overridden[i] = r["key_override"] is not None
        lib.key[i] = r["key_override"] if r["key_override"] is not None else r["key"]
        self.lib_model.layoutChanged.emit()
        if self.current is not None:
            self.focus_track(self.current, select=False)
        self._refresh_setlist()

    # -------------------------------------------------------------- files --
    # ---------------------------------------------------------------- DJ --
    def open_dj(self, show: bool = True):
        """Show the DJ decks docked above the library (Traktor layout)."""
        if self.lib is None:
            return
        if getattr(self, "dj", None) is None:
            try:
                from .dj_window import DJPanel
            except Exception as e:  # noqa: BLE001
                QMessageBox.critical(self, "DJ Mode", f"DJ Mode is unavailable:\n{e}")
                return
            self.dj = DJPanel(self)
            self.dj_dock = self.dj                # the decks live in the top half of the splitter
            self.vsplit.insertWidget(0, self.dj)
            st = self.settings.value("dj_split")
            if st is None or not self.vsplit.restoreState(st):
                h = max(self.vsplit.height(), 600)
                self.vsplit.setSizes([int(h * 0.62), int(h * 0.38)])
            if not show:
                self.dj.hide()
        if show:
            self.dj.show()

    def toggle_dj(self):
        if getattr(self, "dj", None) is not None and self.dj_dock.isVisible():
            self.dj_dock.hide()
        else:
            self.open_dj()

    def load_to_deck(self, deck: str, i: int):
        self.open_dj()
        if getattr(self, "dj", None) is not None:
            self.dj.load_index(deck, i)

    def session_played(self) -> set[int]:
        if self.lib is None:
            return set()
        return set(np.nonzero(self.lib.session_played)[0].tolist())

    def track_played(self, idx: int):
        """Called by DJ Mode when a track has been on air for 30 s."""
        lib = self.lib
        if lib is None or not (0 <= idx < lib.n):
            return
        lib.session_played[idx] = True
        lib.play_count[idx] += 1
        lib.last_played[idx] = time.time()
        self.lib_model.layoutChanged.emit()

    def browse_library(self, delta: int):
        """Move the library selection (controller browse knob)."""
        n = self.lib_model.rowCount()
        if n == 0:
            return
        r = self.lib_view.currentIndex().row()
        r = max(0, min(n - 1, (r if r >= 0 else 0) + int(delta)))
        self.lib_view.selectRow(r)
        self.lib_view.scrollTo(self.lib_model.index(r, 0))

    def play(self, i):
        if self.lib is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(self.lib.paths[i]))

    def reveal(self, i):
        p = self.lib.paths[i]
        try:
            if sys.platform == "win32":
                subprocess.Popen(["explorer", "/select,", os.path.normpath(p)])
            elif sys.platform == "darwin":
                subprocess.Popen(["open", "-R", p])
            else:
                QDesktopServices.openUrl(QUrl.fromLocalFile(os.path.dirname(p)))
        except OSError:
            QDesktopServices.openUrl(QUrl.fromLocalFile(os.path.dirname(p)))

    # --------------------------------------------------------------- scan --
    def add_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Add music folder")
        if not d:
            return
        self.db.add_folder(os.path.normpath(d))
        self.start_scan()

    def open_settings(self):
        dlg = SettingsDialog(self.settings, self.db, self)
        if dlg.exec() and self.settings.value("server_mode", "false") == "true" and not self.server_running():
            self.start_server()

    def scan_settings(self, force=False) -> ScanSettings:
        s = self.settings
        return ScanSettings(workers=int(s.value("workers", 0)), min_bpm=float(s.value("min_bpm", 70)),
                            max_bpm=float(s.value("max_bpm", 180)), use_ai=s.value("use_ai", "true") == "true",
                            ai_device=s.value("ai_device", "auto"), force=force)

    def start_scan(self, force=False):
        if self._scan_thread is not None:
            return
        folders = [f for f in self.db.folders() if os.path.isdir(f)]
        if not folders:
            missing = self.db.folders()
            if missing:
                QMessageBox.warning(self, "Scan", "None of the library folders are reachable:\n" + "\n".join(missing))
            else:
                self.add_folder()
            return
        self.act_scan.setEnabled(False)
        self.act_stop.setEnabled(True)
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)
        th = QThread(self)
        wk = ScanWorker(self.db, folders, self.scan_settings(force))
        wk.moveToThread(th)
        th.started.connect(wk.run)
        wk.progress.connect(self._on_scan_progress)
        wk.finished.connect(self._on_scan_finished)
        wk.finished.connect(th.quit)
        self._scan_thread = th
        self._scan_worker = wk
        th.start()

    def stop_scan(self):
        if self._scan_worker is not None:
            self._scan_worker.scanner.stop()
            self.status_label.setText("Stopping…")

    def _on_scan_progress(self, p):
        if p.device:
            self.device_label.setText(f"AI: {p.device}  ")
        if p.phase == "analyzing" and p.total:
            self.progress.setRange(0, p.total)
            self.progress.setValue(p.done)
            eta = int(p.eta)
            self.status_label.setText(
                f"Analysing {p.done}/{p.total} · {p.rate:.1f} files/s · ETA {eta // 60}m{eta % 60:02d}s"
                + (f" · {p.failed} failed" if p.failed else "") + f" · {p.current}")
        elif p.message:
            self.status_label.setText(p.message)

    def _on_scan_finished(self, p):
        self._scan_thread = None
        self._scan_worker = None
        self.act_scan.setEnabled(True)
        self.act_stop.setEnabled(False)
        self.progress.setVisible(False)
        self.reload_library()
        if p is not None and p.message:
            QTimer.singleShot(1500, lambda: self.status_label.setText(p.message))

    # ---------------------------------------------------------- mix server --
    def server_running(self) -> bool:
        dj = getattr(self, "dj", None)
        return dj is not None and dj.stream is not None

    def start_server(self):
        """Run the mix server (phone remote + live stream); waits for the library."""
        if self.lib is None:
            self._want_server = True
            return
        self.open_dj(show=self.isVisible())
        dj = getattr(self, "dj", None)
        if dj is None:
            return
        dj.toggle_stream(True, show_info=False)
        if self.tray is not None:
            self.tray.refresh()

    def show_from_tray(self):
        if self.isMinimized():
            self.showNormal()
        self.show()
        self.raise_()
        self.activateWindow()
        if self.lib is not None and self.lib.n == 0 and not self.db.folders():
            QTimer.singleShot(300, self._first_run)

    def quit_app(self):
        self._quitting = True
        self.close()

    def closeEvent(self, ev):
        if getattr(self, "dj", None) is not None:
            self.settings.setValue("dj_split", self.vsplit.saveState())
        if self.tray is not None and not self._quitting and self.server_running():
            # keep serving the phone remote from the tray
            ev.ignore()
            self.settings.setValue("geometry", self.saveGeometry())
            self.hide()
            self.tray.hidden_notice()
            return
        self.settings.setValue("geometry", self.saveGeometry())
        if getattr(self, "dj", None) is not None:
            self.dj.shutdown()
        if self._scan_worker is not None:
            self._scan_worker.scanner.stop()
        if self._scan_thread is not None:
            self._scan_thread.quit()
            self._scan_thread.wait(5000)
        if self.tray is not None:
            self.tray.hide()
            QTimer.singleShot(0, QApplication.quit)
        super().closeEvent(ev)
