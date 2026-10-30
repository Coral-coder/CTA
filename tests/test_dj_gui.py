"""Headless smoke test of DJ Mode: load two decks, sync, auto transition."""
import os

import pytest
from conftest import synth_track, write_audio

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")
pytest.importorskip("pylibrb")


def test_dj_window(tmp_data_dir, monkeypatch):
    from PySide6.QtCore import QEventLoop, QSettings, QTimer

    import cta.gui.dj_window as DW
    from cta.db import Database
    from cta.dj.engine import Engine
    from cta.gui.main_window import MainWindow
    from cta.scanner import Scanner, ScanSettings

    monkeypatch.setattr(Engine, "start", lambda self, *a, **k: None)     # no sound card in CI
    monkeypatch.setattr(DW, "list_output_devices", lambda: [])
    monkeypatch.setattr(DW.Controller, "list_inputs", staticmethod(lambda: []))

    music = tmp_data_dir / "music"
    music.mkdir()
    for n, bpm in enumerate((124.0, 126.0)):
        write_audio(str(music / f"t{n}.flac"), synth_track(bpm, 40, chord=(57, 60, 64)))
    db = Database(str(tmp_data_dir / "lib.sqlite3"))
    Scanner(db, ScanSettings(workers=1, use_ai=False)).run([str(music)])

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def wait(ms):
        loop = QEventLoop()
        QTimer.singleShot(ms, loop.quit)
        loop.exec()

    w = MainWindow(db, QSettings(str(tmp_data_dir / "s.ini"), QSettings.IniFormat))
    w.show()
    for _ in range(100):
        if w.lib is not None:
            break
        wait(50)
    w.open_dj()
    dj = w.dj
    dj.load_index("A", 0)
    dj.load_index("B", 1)
    for _ in range(200):
        if dj.engine.decks["A"].loaded and dj.engine.decks["B"].loaded:
            break
        wait(50)
    eng = dj.engine
    assert eng.decks["A"].loaded and eng.decks["B"].loaded
    eng.toggle_play("A")
    eng.set("B", sync=True)
    for _ in range(200):
        eng.render(512)
    app.processEvents()
    assert abs(eng.decks["B"].bpm - eng.decks["A"].bpm) < 0.01
    dj.automix.transition_now(2)
    for _ in range(int(44100 * 6 / 512)):
        eng.render(512)
    assert eng.master_deck == "B"
    dj._tick()
    # never the same song twice: decks, queue and loading items are all excluded
    used = dj.played_indices()
    assert {0, 1} <= used
    dj.automix._loading.add("C")
    dj.automix.queue.clear()
    dj._keep_queue_filled()                   # must not queue anything while a load is running
    assert len(dj.automix.queue) == 0
    dj.automix._loading.clear()
    # history marks tracks as played (library column)
    w.track_played(0)
    assert w.lib.session_played[0] and w.lib.play_count[0] >= 1
    assert 0 in w.session_played()
    dj.shutdown()
    w.close()


def test_phone_remote_drives_the_app(tmp_data_dir, monkeypatch):
    """The phone remote end to end: HTTP -> GUI thread -> auto-mix/engine."""
    import json
    import socket
    import threading
    import urllib.request

    from PySide6.QtCore import QEventLoop, QSettings, QTimer

    import cta.gui.dj_window as DW
    from cta.db import Database
    from cta.dj.engine import SR, Engine
    from cta.gui.main_window import MainWindow
    from cta.scanner import Scanner, ScanSettings

    monkeypatch.setattr(Engine, "start", lambda self, *a, **k: None)
    monkeypatch.setattr(DW, "list_output_devices", lambda: [])
    monkeypatch.setattr(DW.Controller, "list_inputs", staticmethod(lambda: []))
    music = tmp_data_dir / "music"
    music.mkdir()
    for n, bpm in enumerate((124.0, 125.0, 126.0)):
        write_audio(str(music / f"r{n}.flac"), synth_track(bpm, 40, chord=(57, 60, 64), seed=n))
    db = Database(str(tmp_data_dir / "lib.sqlite3"))
    Scanner(db, ScanSettings(workers=1, use_ai=False)).run([str(music)])
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def wait(ms):
        loop = QEventLoop()
        QTimer.singleShot(ms, loop.quit)
        loop.exec()

    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    settings = QSettings(str(tmp_data_dir / "s.ini"), QSettings.IniFormat)
    settings.setValue("stream_port", port)
    settings.setValue("remote_pin", "4242")
    w = MainWindow(db, settings)
    w.start_server()                         # before the library is loaded: starts when it is
    for _ in range(100):
        if w.server_running():
            break
        wait(50)
    assert w.server_running()
    dj = w.dj
    eng = dj.engine
    base = f"http://127.0.0.1:{port}"

    def call(fn):
        """HTTP from a worker thread while the GUI keeps running (like a phone)."""
        box = {}
        t = threading.Thread(target=lambda: box.setdefault("r", fn()), daemon=True)
        t.start()
        while t.is_alive():
            wait(20)
        return box["r"]

    def get(path):
        req = urllib.request.Request(base + path, headers={"X-CTA-Key": "4242"})
        return json.loads(urllib.request.urlopen(req, timeout=10).read())

    def cmd(**body):
        req = urllib.request.Request(base + "/api/cmd", data=json.dumps(body).encode(), method="POST",
                                     headers={"X-CTA-Key": "4242", "Content-Type": "application/json"})
        try:
            return json.loads(urllib.request.urlopen(req, timeout=10).read())
        except urllib.error.HTTPError as e:
            return json.loads(e.read())

    import urllib.error

    lib = w.lib
    seed = next(i for i in range(lib.n) if lib.titles[i] == "r1")
    res = call(lambda: get("/api/search?q=r1"))
    assert [r["idx"] for r in res["results"]] == [seed]
    assert "error" in call(lambda: cmd(action="skip"))            # nothing to play yet
    res = call(lambda: cmd(action="seed", idx=seed))
    assert res["ok"] and "r1" in res["message"]
    assert w.setlist[0] == seed                                  # the set shows up on the PC too
    for _ in range(200):                                         # decks load in threads
        eng.render(1024)
        if eng._master() is not None and eng._master().info.lib_index == seed:
            break
        wait(10)
    m = eng._master()
    assert m is not None and m.info.lib_index == seed
    st = call(lambda: get("/api/state"))
    assert st["automix"]["active"] and any(d["master"] and d["idx"] == seed for d in st["decks"])
    # fade markers from the phone: snapped to the start of a bar, saved with the track
    bar = 4 * m.grid.beat_len
    res = call(lambda: cmd(action="markers", deck=m.name, **{"in": m.grid.first_beat + 1.1 * bar,
                                                               "out": m.grid.first_beat + 8.6 * bar}))
    assert res["ok"]
    for _ in range(3):
        eng.render(512)
    assert abs(eng.decks[m.name].mix_out / SR - (m.grid.first_beat + 8 * bar)) < 0.01
    assert abs(db.get_markers(m.info.path)[0] - (m.grid.first_beat + bar)) < 0.01
    wv = call(lambda: get("/api/wave?deck=" + m.name))
    assert len(wv["low"]) == 4000 and wv["bpm"] > 0
    # bump / outro end / outro length from the phone
    out0 = eng.decks[m.name].mix_out / SR
    res = call(lambda: cmd(action="bump", deck=m.name, which="out", beats=4))
    assert res["ok"]
    for _ in range(3):
        eng.render(512)
    assert abs(eng.decks[m.name].mix_out / SR - (out0 + bar)) < 0.01
    res = call(lambda: cmd(action="out_length", deck=m.name, bars=64))
    for _ in range(3):
        eng.render(512)
    assert "64 bars" in res["message"]
    assert abs((eng.decks[m.name].mix_out_end - eng.decks[m.name].mix_out) / SR - 64 * bar) < 0.02
    assert abs(db.get_out_end(m.info.path) - eng.decks[m.name].mix_out_end / SR) < 0.01
    res = call(lambda: cmd(action="bump", deck=m.name, which="in", beats=-4))      # moves IN one bar back
    assert res["ok"]
    res = call(lambda: cmd(action="markers", deck=m.name, out_end=None))           # clear just the end
    for _ in range(3):
        eng.render(512)
    assert eng.decks[m.name].mix_out_end is None and eng.decks[m.name].mix_out is not None
    call(lambda: cmd(action="blend", auto=False, bars=128))
    assert not dj.automix.s.auto_length and dj.automix.s.mix_bars == 128
    call(lambda: cmd(action="blend", auto=True))
    assert dj.automix.s.auto_length
    # queue editing and pause
    other = next(i for i in range(lib.n) if i != seed)
    call(lambda: cmd(action="queue_add", idx=other))
    assert other in list(dj.automix.queue)
    q = list(dj.automix.queue)
    call(lambda: cmd(action="queue_remove", pos=q.index(other)))
    assert list(dj.automix.queue).count(other) == q.count(other) - 1
    call(lambda: cmd(action="queue_add", idx=other))
    call(lambda: cmd(action="queue_add", idx=seed))
    call(lambda: cmd(action="queue_clear"))
    assert len(dj.automix.queue) == 0 and dj.queue_list.count() == 0
    call(lambda: cmd(action="toggle_pause"))
    for _ in range(30):
        eng.render(1024)
    assert eng.paused and not any(d.playing for d in eng.decks.values())
    call(lambda: cmd(action="toggle_pause"))
    for _ in range(5):
        eng.render(1024)
    assert eng._master() is not None
    call(lambda: cmd(action="blend", bars=8))
    assert dj.mix_bars.value() == 8 and dj.automix.s.mix_bars == 8
    # audio watchdog: the sound card stops calling back -> reopen it
    calls = []
    monkeypatch.setattr(dj, "start_audio", lambda quiet=False: calls.append(quiet))
    monkeypatch.setattr(type(eng), "running", property(lambda self: True))
    import time as _time

    dj._wd = (eng.frames_rendered, _time.monotonic() - 5)
    dj._watchdog()
    assert calls == [True]
    dj._wd = (eng.frames_rendered - 1, _time.monotonic() - 5)    # frames moving: all fine
    dj._watchdog()
    assert calls == [True]
    monkeypatch.undo()
    # PC queue list: ✕ / Delete / drag-out remove, drag to reorder
    with dj.automix._lock:
        dj.automix.queue.clear()
        dj.automix.queue.extend([0, 1, 2])
    dj._show_queue()
    assert dj.queue_list.count() == 3
    dj.queue_list.removeRequested.emit(1)
    assert list(dj.automix.queue) == [0, 2]
    dj.queue_list.model().moveRow(dj.queue_list.model().index(1, 0).parent(), 1,
                                  dj.queue_list.model().index(0, 0).parent(), 0)
    assert list(dj.automix.queue) == [2, 0]
    dj.clear_queue()
    assert not dj.automix.queue
    # PC marker editor: opens, paints zoomed in, bumps
    dj.edit_markers(m.name)
    ed = dj._editors[m.name]
    app.processEvents()
    ed.wave.zoom(0.1)
    ed.grab()
    o1 = eng.decks[m.name].mix_out / SR
    dj.bump_marker(m.name, "out", -4)
    for _ in range(3):
        eng.render(512)
    assert abs(eng.decks[m.name].mix_out / SR - (o1 - bar)) < 0.01
    ed.refresh()
    ed.close()
        # PC side: QR/PIN dialog and tray
    dlg = DW.RemoteInfoDialog(dj)
    dlg.show()
    app.processEvents()
    dlg.grab()                                   # paints the QR code
    dlg._new_pin()
    assert dj.stream.pin != "4242" and len(dj.stream.pin) == 6
    dlg.close()
    from cta.gui.tray import Tray

    t = Tray(w)
    t.refresh()
    assert "mix server" in t.icon.toolTip()
    t.hide()
    app.processEvents()
    w.quit_app()


def test_decks_scale_to_a_laptop_screen(tmp_data_dir, monkeypatch):
    """13-inch MacBook: the window fits, the decks switch view as they get
    shorter (Full -> Essential -> Micro) and nothing is drawn on top of anything."""
    from PySide6.QtCore import QEventLoop, QSettings, QTimer

    import cta.gui.dj_window as DW
    from cta.db import Database
    from cta.dj.engine import Engine
    from cta.gui.main_window import MainWindow
    from cta.scanner import Scanner, ScanSettings

    monkeypatch.setattr(Engine, "start", lambda self, *a, **k: None)
    monkeypatch.setattr(DW, "list_output_devices", lambda: [])
    monkeypatch.setattr(DW.Controller, "list_inputs", staticmethod(lambda: []))
    music = tmp_data_dir / "music"
    music.mkdir()
    write_audio(str(music / "a.flac"), synth_track(124.0, 20))
    db = Database(str(tmp_data_dir / "lib.sqlite3"))
    Scanner(db, ScanSettings(workers=1, use_ai=False)).run([str(music)])
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def wait(ms):
        loop = QEventLoop()
        QTimer.singleShot(ms, loop.quit)
        loop.exec()

    w = MainWindow(db, QSettings(str(tmp_data_dir / "s.ini"), QSettings.IniFormat))
    w.resize(1280, 760)
    w.show()
    for _ in range(100):
        if w.lib is not None:
            break
        wait(50)
    w.open_dj()
    wait(200)
    mw = w.minimumSizeHint().width()
    assert mw <= 1280 and w.width() <= 1280, (mw, {
        "dj": w.dj.minimumSizeHint().width(), "deck": w.dj.decks["A"].minimumSizeHint().width(),
        "mixer": w.dj.mixer.minimumSizeHint().width(),
        "bar": w.dj.layout().itemAt(0).widget().minimumSizeHint().width(),
        "lib": w.vsplit.widget(1).minimumSizeHint().width()})
    dj = w.dj
    deck = dj.decks["A"]
    seen = set()
    for frac in (0.95, 0.7, 0.45, 0.3):
        hh = w.vsplit.height()
        w.vsplit.setSizes([int(hh * frac), hh - int(hh * frac)])
        wait(150)
        seen.add(deck.view)
        need = deck._needs()
        # the richest view that fits the deck's height
        fits = (v for v in deck.VIEWS if deck.height() >= need[v] and deck.width() >= need[v + "_w"])
        assert deck.view == next(fits, "micro"), (deck.height(), deck.width(), need)
        # nothing inside the deck or the mixer spills past its frame
        for panel in (deck, dj.mixer):
            for child in panel.findChildren(QtWidgets.QWidget):
                if child.isVisible() and child.parentWidget() is panel:
                    assert child.geometry().bottom() <= panel.height() + 1, (panel, child, deck.view)
    assert len(seen) >= 2 and "micro" in seen        # it really changes view as it shrinks
    # Windows asks for size hints synchronously while rows are shown/hidden:
    # measuring must not re-enter itself (it recursed forever on Windows CI)
    orig_show = type(deck)._show

    def eager_show(self, v):
        orig_show(self, v)
        self.minimumSizeHint()
        dj.mixer.minimumSizeHint()
    monkeypatch.setattr(type(deck), "_show", eager_show)
    deck._need_key = None
    dj.mixer._need = None
    assert deck._needs()["micro"] > 0 and dj.mixer._needs()[2] > 0
    monkeypatch.setattr(type(deck), "_show", orig_show)
    dj.view_combo.setCurrentText("Full")
    assert deck.view == "full"
    dj.view_combo.setCurrentText("Auto")
    dj.shutdown()
    w.close()


def test_keyboard_cue_button_and_playlists(tmp_data_dir, monkeypatch):
    from PySide6.QtCore import QEvent, QEventLoop, QPoint, QPointF, QSettings, Qt, QTimer
    from PySide6.QtGui import QKeyEvent, QMouseEvent

    import cta.gui.dj_window as DW
    from cta.db import Database
    from cta.dj.engine import Engine
    from cta.gui.main_window import MainWindow
    from cta.scanner import Scanner, ScanSettings

    monkeypatch.setattr(Engine, "start", lambda self, *a, **k: None)
    monkeypatch.setattr(DW, "list_output_devices", lambda: [])
    monkeypatch.setattr(DW.Controller, "list_inputs", staticmethod(lambda: []))
    music = tmp_data_dir / "music"
    music.mkdir()
    for n, bpm in enumerate((124.0, 125.0, 126.0)):
        write_audio(str(music / f"t{n}.flac"), synth_track(bpm, 30, chord=(57, 60, 64)))
    db = Database(str(tmp_data_dir / "lib.sqlite3"))
    Scanner(db, ScanSettings(workers=1, use_ai=False)).run([str(music)])
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def wait(ms):
        loop = QEventLoop()
        QTimer.singleShot(ms, loop.quit)
        loop.exec()

    w = MainWindow(db, QSettings(str(tmp_data_dir / "s.ini"), QSettings.IniFormat))
    w.show()
    for _ in range(100):
        if w.lib is not None:
            break
        wait(50)
    w.open_dj()
    dj = w.dj
    eng = dj.engine
    dj.load_index("A", 0)
    for _ in range(200):
        if eng.decks["A"].loaded:
            break
        wait(50)
    w.lib_view.setFocus()
    monkeypatch.setattr(QtWidgets.QApplication, "activeWindow", staticmethod(lambda: w))
    d = eng.decks["A"]

    def key(k, release=False, mods=Qt.NoModifier):
        ev = QKeyEvent(QEvent.KeyRelease if release else QEvent.KeyPress, k, mods)
        app.sendEvent(w.lib_view, ev)

    # Traktor keys: W plays deck A, W again pauses, Q holds the cue preview
    key(Qt.Key_W)
    assert d.playing
    key(Qt.Key_W)
    assert not d.playing
    key(Qt.Key_Q)
    assert d.playing and d.cue_preview
    key(Qt.Key_Q, release=True)
    assert not d.playing
    key(Qt.Key_U)
    assert d.sync
    key(Qt.Key_1)
    assert d.hotcues[0] is not None
    key(Qt.Key_1, mods=Qt.ControlModifier)
    assert d.hotcues[0] is None
    # typing in the search box is not deck control
    w.search.setFocus()
    app.sendEvent(w.search, QKeyEvent(QEvent.KeyPress, Qt.Key_W, Qt.NoModifier, "w"))
    assert not d.playing and w.search.text() == "w"
    w.search.clear()

    # CUE button: dragging off it before letting go keeps the track playing
    b = dj.decks["A"].b_cue
    c = QPointF(b.rect().center())
    app.sendEvent(b, QMouseEvent(QEvent.MouseButtonPress, c, c, Qt.LeftButton, Qt.LeftButton, Qt.NoModifier))
    assert d.playing and d.cue_preview
    off = QPointF(QPoint(b.width() + 40, 5))
    app.sendEvent(b, QMouseEvent(QEvent.MouseMove, off, off, Qt.NoButton, Qt.LeftButton, Qt.NoModifier))
    app.sendEvent(b, QMouseEvent(QEvent.MouseButtonRelease, off, off, Qt.LeftButton, Qt.NoButton, Qt.NoModifier))
    assert d.playing and not d.cue_preview
    eng.set_playing("A", False)

    # playlists: two songs get a dimmed bridge between them; accepting it adds it
    pl = w.playlists
    k = pl.new_list(name="Friday")
    pl.add_tracks([0, 2], playlist=k)
    assert pl.items() == [0, 2]
    kinds = pl.model.kinds
    assert kinds == ["track", "bridge", "track"]
    bridge = pl.model.items[1]
    assert bridge == 1
    from cta.gui.playlists import DIM

    assert pl.model.data(pl.model.index(1, 6), Qt.ForegroundRole) == DIM
    pl.accept_bridge(1)
    assert pl.items() == [0, 1, 2]
    # saved with the library
    pl2 = type(pl)(w)
    assert pl2.names() == ["Friday"] and pl2.items() == [0, 1, 2]
    pl.to_automix()
    assert list(dj.automix.queue) == [0, 1, 2]
    dj.shutdown()
    w.close()


def test_zoomed_waveform_scrolls_without_flicker():
    """Moving the playhead only shifts the columns: none of them changes
    (each column is the peak of a fixed slice of the track)."""
    import numpy as np

    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from cta.gui.dj_window import SR, Waveform

    w = Waveform("A", True)
    w.bands = np.random.default_rng(0).random((SR * 60 // 256, 3)).astype(np.float32)
    for span in (2.0, 8.0, 32.0):
        w.span = span
        col = span / 600
        w.pos = 20.0
        a, _ = w._zoom_columns(600)
        w.pos = 20.0 + col * 3.4
        b, frac = w._zoom_columns(600)
        assert abs(frac - 0.4) < 1e-6
        assert np.allclose(a[3:], b[:-3])
