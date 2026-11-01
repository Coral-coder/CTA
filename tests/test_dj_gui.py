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

    monkeypatch.setattr(Engine, "start", lambda self, *a, **k: None)
    monkeypatch.setattr("cta.dj.engine.list_output_devices", lambda: [])   # the audio server scans too
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
    dj._tick()
    app.processEvents()
    # the track loaded last (either deck: the loads run in threads) is the one the
    # library shows; Next Track lists what mixes from it
    assert w.current in {eng.decks["A"].info.lib_index, eng.decks["B"].info.lib_index}
    # the controller's browse knob and LOAD work on the right panel's open tab
    w.tabs.setCurrentIndex(0)                                   # Next Track
    assert w.next_model.rowCount() >= 1
    w.browse_library(1)
    w.browse_library(-5)
    assert w.next_view.currentIndex().row() == 0
    assert w.selected_for_load() == w.next_model.items[0]
    w.tabs.setCurrentIndex(2)                                   # Playlists (empty): falls back to the library
    w.browse_library(1)
    assert w.selected_for_load() == w.lib_model.lib_index(w.lib_view.currentIndex().row())
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
    monkeypatch.setattr("cta.dj.engine.list_output_devices", lambda: [])   # the audio server scans too
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
    import time as _time

    t_end = _time.monotonic() + 20                               # decks load in threads (slow on busy CI)
    while _time.monotonic() < t_end:
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
    # queue editing and pause (auto-suggest off: it would refill an emptied queue on the next tick)
    dj.auto_suggest.setChecked(False)
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
    dj.automix.set_queue([0, 1, 2])
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
    # add a marker by arming a button and clicking; clicks never move the deck
    from PySide6.QtCore import QEvent, QPointF, Qt
    from PySide6.QtGui import QMouseEvent

    ed.wave.fit()
    ed.wave.resize(1000, 240)
    t_play = eng.decks[m.name].time
    x_in = ed.wave._x(2 * bar)                                   # before OUT (IN must come first)
    ed.place_btns["in"].setChecked(True)
    assert ed.wave.place == "in"
    ev = QMouseEvent(QEvent.MouseButtonPress, QPointF(x_in, 100), QPointF(x_in, 100), Qt.LeftButton, Qt.LeftButton,
                     Qt.NoModifier)
    ed.wave.mousePressEvent(ev)
    for _ in range(3):
        eng.render(512)
    assert ed.wave.place is None and not ed.place_btns["in"].isChecked()
    assert abs(eng.decks[m.name].mix_in / SR - ed.wave._snap(2 * bar)) < 0.01
    ev = QMouseEvent(QEvent.MouseButtonDblClick, QPointF(300, 100), QPointF(300, 100), Qt.LeftButton, Qt.LeftButton,
                     Qt.NoModifier)
    ed.wave.mouseDoubleClickEvent(ev)
    for _ in range(3):
        eng.render(512)
    assert abs(eng.decks[m.name].time - t_play) < 0.2            # a double-click did not jump the deck
    assert ed.wave.sections                                       # the section strip is there
    ed._clear("in")
    for _ in range(3):
        eng.render(512)
    assert eng.decks[m.name].mix_in is None
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
    monkeypatch.setattr("cta.dj.engine.list_output_devices", lambda: [])   # the audio server scans too
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
    monkeypatch.setattr("cta.dj.engine.list_output_devices", lambda: [])   # the audio server scans too
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

    # double-clicking the deck title shows what mixes next from that track
    dj.decks["A"].title.doubleClicked.emit()
    assert w.current == 0 and w.tabs.currentIndex() == 0
    # Serato-style layout: every deck's waveform rides in the stack across the top, playheads in line
    dj.set_wave_layout("stacked")
    assert dj.stack_w.isVisibleTo(dj) and dj.decks["A"].zoom.parent() is dj.stack_rows["A"]
    assert dj.decks["A"].platter.parent() is dj.stack_rows["A"]          # the platter rides left of its track
    assert dj.stack_rows["A"].isVisibleTo(dj) and not dj.stack_rows["C"].isVisibleTo(dj)
    dj.show_cd.setChecked(True)
    assert dj.stack_rows["C"].isVisibleTo(dj) and dj.automix.decks == ("A", "B", "C", "D")
    dj.show_cd.setChecked(False)
    assert dj.automix.decks == ("A", "B")
    app.processEvents()
    dj.stack_w.grab()
    dj.set_wave_layout("decks")
    assert not dj.stack_w.isVisibleTo(dj) and dj.decks["A"].zoom.parent() is dj.decks["A"].w_body
    assert dj.decks["A"].platter.parent() is dj.decks["A"].w_body
    assert dj.settings.value("dj_wave_layout") == "decks"
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

    # every set becomes a playlist named after when it started; a track is in it once
    pl = w.playlists
    t0 = 1_800_000_000.0
    pl.log_played(0, t0)
    pl.log_played(1, t0)
    pl.log_played(0, t0)
    import time as _time

    name = _time.strftime("Set %Y-%m-%d %H:%M", _time.localtime(t0))
    assert [p["name"] for p in pl.lists].count(name) == 1
    assert pl.items(next(p for p in pl.lists if p["name"] == name)) == [0, 1]
    pl.lists = [p for p in pl.lists if p["name"] != name]          # out of the way of the checks below
    pl._save()
    pl._fill_combo()
    # playlists: two songs get a dimmed bridge between them; accepting it adds it
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
    # Sort for flow reorders (never drops) the songs; Undo sort brings the old order back
    pl.lists[k]["paths"].append("/gone/missing.flac")              # a file no longer in the library stays
    pl.sort_for_flow()
    assert sorted(pl.items()) == [0, 1, 2] and pl.lists[k]["paths"][-1] == "/gone/missing.flac"
    pl.lists[k]["paths"] = [pl.lists[k]["paths"][i] for i in (2, 0, 1, 3)]
    pl.undo_sort()
    assert pl.items() == [0, 1, 2]
    pl.lists[k]["paths"].pop()
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
        a, _ = w._zoom_image(600, 60)
        w.pos = 20.0 + col * 3.4
        b, frac = w._zoom_image(600, 60)
        assert abs(frac - 0.4) < 1e-6
        assert np.array_equal(a._buf[:, 3:], b._buf[:, :-3])


def test_waveform_themes_render_saturated_colours():
    import numpy as np

    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from cta.gui import dj_window as DW

    bands = np.array([[1.0, 0.05, 0.02], [0.05, 1.0, 0.05], [0.02, 0.05, 1.0]], np.float32)
    try:
        for name in DW.WAVE_THEMES:
            DW.set_wave_theme(name)
            img = DW._wave_image(bands, 40)
            assert img.width() == 3 and img.height() == 40
        DW.set_wave_theme("Spectrum")
        rgb = DW._band_colors(bands).astype(int)
        # bass red, mids green-ish, highs blue: each a strong, pure colour
        assert rgb[0, 0] > 200 and rgb[0, 2] < 40
        assert rgb[1, 1] > 200 and rgb[1, 0] < 120
        assert rgb[2, 2] > 200 and rgb[2, 1] < 60
        assert all(c.max() - c.min() > 180 for c in rgb)          # saturated, not washed out
    finally:
        DW.set_wave_theme("Spectrum")


def test_timeshift_window_rehearses_on_the_preview_only(tmp_data_dir, monkeypatch):
    from PySide6.QtCore import QEventLoop, QSettings, QTimer

    import cta.gui.dj_window as DW
    from cta.db import Database
    from cta.dj.engine import Engine
    from cta.gui.main_window import MainWindow
    from cta.scanner import Scanner, ScanSettings

    monkeypatch.setattr(Engine, "start", lambda self, *a, **k: None)
    monkeypatch.setattr("cta.dj.engine.list_output_devices", lambda: [])   # the audio server scans too
    monkeypatch.setattr(DW, "list_output_devices", lambda: [])
    monkeypatch.setattr(DW.Controller, "list_inputs", staticmethod(lambda: []))
    music = tmp_data_dir / "music"
    music.mkdir()
    for n, bpm in enumerate((124.0, 124.0)):
        write_audio(str(music / f"t{n}.flac"), synth_track(bpm, 60, chord=(57, 60, 64)))
    db = Database(str(tmp_data_dir / "lib.sqlite3"))
    Scanner(db, ScanSettings(workers=1, use_ai=False)).run([str(music)])
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

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
    live = dj.engine
    ts = dj.open_timeshift(0, 1)
    assert live.preview is ts.pv
    for _ in range(200):
        if ts.pv.decks["A"].loaded and ts.pv.decks["B"].loaded:
            break
        live.render(512)                       # the preview runs inside the live engine
        wait(20)
    assert ts.pv.decks["A"].loaded and ts.pv.decks["B"].loaded
    # place an OUT marker on the outgoing in the zoomable waveform: saved with the track
    g = ts.pv.decks["A"].grid
    ts._marker_moved("A", "out", g.time_of_beat(48))
    assert abs(db.get_markers(ts.paths["A"])[1] - g.time_of_beat(48)) < 1e-6
    ts.rehearse()
    ts.session.drop_now()
    for _ in range(400):
        live.render(512)
        if ts.session.mode == "recording":
            break
    assert ts.session.mode == "recording"
    # the DJ mixes on the preview mixer: incoming up, outgoing down over 4 bars
    ts.faders["B"].setValue(1000)
    for _ in range(int(8 * 44100 / 512)):
        live.render(512)
    ts.faders["A"].setValue(0)
    for _ in range(int(6 * 44100 / 512)):
        live.render(512)
        if ts.session.mode == "done":
            break
    ts._tick()
    r = dj.rehearsals.get(ts.paths["A"], ts.paths["B"])
    assert r is not None and r.beats >= 4
    assert r.value("in_vol", r.beats) > 0.9 and r.value("out_vol", r.beats) < 0.05
    # the live decks never moved
    assert not any(d.playing for d in live.decks.values())
    ts.close()
    assert live.preview is None
    dj.shutdown()
    w.close()


def test_scopes_draw_the_master_and_the_stereo_image():
    import numpy as np
    from PySide6.QtGui import QColor

    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from cta.gui.dj_window import Oscilloscope, xy_scope_image

    t = np.arange(2048) / 44100
    mono = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    img = xy_scope_image(np.stack([mono, mono]), 41, QColor(255, 0, 0))
    lit = (img._buf >> 24) > 0
    cols = np.nonzero(lit.any(axis=0))[0]
    assert cols.min() >= 19 and cols.max() <= 21          # mono: one vertical line
    wide = xy_scope_image(np.stack([mono, -mono]), 41, QColor(255, 0, 0))
    assert ((wide._buf >> 24) > 0).any(axis=1).sum() <= 3  # out of phase: a horizontal line
    osc = Oscilloscope(110, 46)
    osc.data = np.stack([mono, mono])
    osc.resize(110, 46)
    pm = osc.grab()                                         # paints without errors
    assert pm.width() == 110


def test_knobs_turn_with_vertical_drag_and_never_jump():
    from PySide6.QtCore import QEvent, QPointF, Qt
    from PySide6.QtGui import QMouseEvent

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from cta.gui.dj_window import Knob

    k = Knob(50, 38)
    k.resize(38, 38)

    def ev(kind, x, y, mods=Qt.NoModifier):
        btn = Qt.LeftButton if kind != QEvent.MouseMove else Qt.NoButton
        held = Qt.NoButton if kind == QEvent.MouseButtonRelease else Qt.LeftButton
        p = QPointF(x, y)
        app.sendEvent(k, QMouseEvent(kind, p, p, btn, held, mods))
    ev(QEvent.MouseButtonPress, 2, 36)            # click at the bottom-left edge: no jump
    assert k.value() == 50
    ev(QEvent.MouseMove, 2, 16)                   # 20 px up: +10 (200 px = the whole range)
    assert k.value() == 60
    ev(QEvent.MouseMove, 2, 56)                   # back down 40 px
    assert k.value() == 40
    ev(QEvent.MouseMove, 2, 56, Qt.ShiftModifier)  # Shift: fine, carries on from here
    ev(QEvent.MouseMove, 2, 6, Qt.ShiftModifier)   # 50 px up, fine: +5
    assert k.value() == 45
    ev(QEvent.MouseButtonRelease, 2, 6)
    ev(QEvent.MouseMove, 2, -500)                 # not held: nothing
    assert k.value() == 45
    p = QPointF(2, 6)
    app.sendEvent(k, QMouseEvent(QEvent.MouseButtonPress, p, p, Qt.RightButton, Qt.RightButton, Qt.NoModifier))
    assert k.value() == 50                        # right-click: back to the default


def test_faders_and_crossfader_drag_from_where_they_are():
    from PySide6.QtCore import QEvent, QPointF, Qt
    from PySide6.QtGui import QMouseEvent

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from cta.gui.dj_window import ResetSlider

    def drag(sl, pts, mods=Qt.NoModifier):
        for i, (x, y) in enumerate(pts):
            kind = QEvent.MouseButtonPress if i == 0 else (
                QEvent.MouseButtonRelease if i == len(pts) - 1 else QEvent.MouseMove)
            btn = Qt.LeftButton if kind != QEvent.MouseMove else Qt.NoButton
            held = Qt.NoButton if kind == QEvent.MouseButtonRelease else Qt.LeftButton
            p = QPointF(x, y)
            app.sendEvent(sl, QMouseEvent(kind, p, p, btn, held, mods))

    fader = ResetSlider(Qt.Vertical, 1000)
    fader.setRange(0, 1000)
    fader.resize(20, 216)                          # 200 px of travel = the whole range
    fader.setValue(500)
    drag(fader, [(10, 200), (10, 200)])            # click near the bottom: no jump
    assert fader.value() == 500
    drag(fader, [(10, 100), (10, 60), (10, 60)])   # 40 px up: +200
    assert fader.value() == 700
    p = QPointF(10, 60)
    app.sendEvent(fader, QMouseEvent(QEvent.MouseButtonPress, p, p, Qt.RightButton, Qt.RightButton, Qt.NoModifier))
    assert fader.value() == 1000                   # right-click: back to the default
    xf = ResetSlider(Qt.Horizontal, 0)
    xf.setRange(-100, 100)
    xf.resize(216, 20)
    drag(xf, [(5, 10), (55, 10), (55, 10)])        # grabbed at the far left, 50 px right: +50
    assert xf.value() == 50
    tempo = ResetSlider(Qt.Vertical, 0)
    tempo.setRange(-1000, 1000)
    tempo.setInvertedAppearance(True)              # down = faster, like a turntable
    tempo.resize(20, 216)
    drag(tempo, [(10, 100), (10, 110), (10, 110)])
    assert tempo.value() == 100


def test_unload_decks_and_reset_auto_mix_for_the_next_set(tmp_data_dir, monkeypatch):
    from PySide6.QtCore import QEventLoop, QSettings, QTimer

    import cta.gui.dj_window as DW
    from cta.db import Database
    from cta.dj.engine import Engine
    from cta.gui.main_window import MainWindow
    from cta.scanner import Scanner, ScanSettings

    monkeypatch.setattr(Engine, "start", lambda self, *a, **k: None)
    monkeypatch.setattr("cta.dj.engine.list_output_devices", lambda: [])   # the audio server scans too
    monkeypatch.setattr(DW, "list_output_devices", lambda: [])
    monkeypatch.setattr(DW.Controller, "list_inputs", staticmethod(lambda: []))
    music = tmp_data_dir / "music"
    music.mkdir()
    for n, bpm in enumerate((124.0, 125.0, 126.0)):
        write_audio(str(music / f"t{n}.flac"), synth_track(bpm, 30, chord=(57, 60, 64)))
    db = Database(str(tmp_data_dir / "lib.sqlite3"))
    Scanner(db, ScanSettings(workers=1, use_ai=False)).run([str(music)])
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

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
    dj, eng = w.dj, w.dj.engine
    dj.load_index("A", 0)
    dj.load_index("B", 1)
    for _ in range(200):
        if eng.decks["A"].loaded and eng.decks["B"].loaded:
            break
        wait(50)
    am = dj.automix
    am.history[:] = ["t0"]
    am.used_items[:] = [0, 1]
    am.set_queue([2])
    dj.set_seed = 0
    eng.set_playing("A", True)
    eng.render(512)
    dj._tick()
    assert not dj.unload_btn.isVisibleTo(w)            # the queue still holds a track
    dj.clear_queue()
    dj._tick()
    assert dj.unload_btn.isVisibleTo(w)                # auto-mix off, queue empty: offered
    dj.unload_decks(False)
    eng.render(512)
    assert eng.decks["A"].loaded and eng.decks["A"].playing     # the track playing is never cut
    assert not eng.decks["B"].loaded
    assert not am.history and not am.used_items and dj.set_seed is None and not am.queue
    dj._tick()                                         # an emptied deck draws without errors
    app = QtWidgets.QApplication.instance()
    app.processEvents()
    monkeypatch.setattr(DW.QMessageBox, "question", staticmethod(lambda *a, **k: DW.QMessageBox.Yes))
    dj.unload_decks(True)                              # everything, after asking
    eng.render(512)
    assert not any(d.loaded for d in eng.decks.values())
    dj._tick()
    app.processEvents()
    dj.shutdown()
    w.close()


def test_dragging_the_overview_scrubs():
    import numpy as np
    from PySide6.QtCore import QEvent, QPointF, Qt
    from PySide6.QtGui import QMouseEvent

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from cta.gui.dj_window import Waveform

    w = Waveform("A", False)
    w.resize(200, 30)
    w.set_track(np.ones((200, 3), np.float32) * 0.5, None, 100.0, None)
    seeks = []
    w.seek.connect(seeks.append)
    for kind, x, held in ((QEvent.MouseButtonPress, 20, Qt.LeftButton), (QEvent.MouseMove, 60, Qt.LeftButton),
                          (QEvent.MouseMove, 150, Qt.LeftButton), (QEvent.MouseButtonRelease, 150, Qt.NoButton)):
        p = QPointF(x, 10)
        btn = Qt.LeftButton if kind != QEvent.MouseMove else Qt.NoButton
        app.sendEvent(w, QMouseEvent(kind, p, p, btn, held, Qt.NoModifier))
    assert [round(s_) for s_ in seeks] == [10, 30, 75]


def test_learned_mixes_dialog_edits_templates_and_round_trips_a_file(tmp_data_dir, monkeypatch):
    from PySide6.QtCore import QSettings, Qt

    import cta.gui.dj_window as DW
    from cta.db import Database
    from cta.dj.engine import Engine
    from cta.dj.learn import TransitionExample
    from cta.gui.main_window import MainWindow

    monkeypatch.setattr(Engine, "start", lambda self, *a, **k: None)
    monkeypatch.setattr("cta.dj.engine.list_output_devices", lambda: [])
    monkeypatch.setattr(DW, "list_output_devices", lambda: [])
    monkeypatch.setattr(DW.Controller, "list_inputs", staticmethod(lambda: []))
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    db = Database(str(tmp_data_dir / "lib.sqlite3"))
    w = MainWindow(db, QSettings(str(tmp_data_dir / "s.ini"), QSettings.IniFormat))
    w.show()
    for _ in range(100):
        if w.lib is not None:
            break
        app.processEvents()
    w.open_dj()
    dj = w.dj
    dj.learner.examples.append(TransitionExample(1.0, "House", "a", "b", 16.0, 0.0, 8.0, "outro", "intro"))
    dj.learner._save()
    dlg = dj.open_learned()
    assert dlg.ex.rowCount() == 1
    dlg.ex.item(0, 3).setText("24")                      # perfect it: 24 bars
    assert dj.learner.examples[0].bars == 24.0
    dlg.ex.selectRow(0)
    dlg._template_from_example()
    assert dj.learner.preferences("House")["bars"] == 24 and dlg.tpl.rowCount() == 1
    out = tmp_data_dir / "mixes.json"
    monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (str(out), "")))
    dlg.export()
    assert out.exists()
    dj.learner.forget()
    dlg.refresh()
    assert dlg.ex.rowCount() == 0 and dlg.tpl.rowCount() == 0
    dlg.import_(str(out))
    assert dlg.ex.rowCount() == 1 and dj.learner.preferences("House")["template"]
    dlg.tpl.selectRow(0)
    dlg._remove_template()
    assert "House" not in dj.learner.templates and dlg.tpl.rowCount() == 0 and Qt is not None
    dlg.close()
    dj.shutdown()
    w.close()


def test_dj_mode_is_built_once_even_when_two_openers_overlap(tmp_data_dir, monkeypatch):
    """At start-up DJ mode opens by itself and a live update opens it too; if one
    runs while the other is still building the decks, there is still one set."""
    from PySide6.QtCore import QEventLoop, QSettings, QTimer

    import cta.gui.dj_window as DW
    from cta.db import Database
    from cta.dj.engine import Engine
    from cta.gui.main_window import MainWindow
    from cta.scanner import Scanner, ScanSettings

    monkeypatch.setattr(Engine, "start", lambda self, *a, **k: None)
    monkeypatch.setattr("cta.dj.engine.list_output_devices", lambda: [])
    monkeypatch.setattr(DW, "list_output_devices", lambda: [])
    monkeypatch.setattr(DW.Controller, "list_inputs", staticmethod(lambda: []))
    music = tmp_data_dir / "music"
    music.mkdir()
    write_audio(str(music / "a.flac"), synth_track(124.0, 20))
    db = Database(str(tmp_data_dir / "lib.sqlite3"))
    Scanner(db, ScanSettings(workers=1, use_ai=False)).run([str(music)])
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    w = MainWindow(db, QSettings(str(tmp_data_dir / "s.ini"), QSettings.IniFormat))
    w.show()
    for _ in range(100):
        if w.lib is not None:
            break
        loop = QEventLoop()
        QTimer.singleShot(50, loop.quit)
        loop.exec()
    real_init = DW.DJPanel.__init__

    def init(self, main):
        real_init(self, main)
        main.open_dj()                     # the other opener, while the first is still building
    monkeypatch.setattr(DW.DJPanel, "__init__", init)
    w.open_dj()
    assert len(w.findChildren(DW.DJPanel)) == 1
    assert [w.tabs.tabText(i) for i in range(w.tabs.count())].count("Auto-mix queue") == 1
    w.dj.shutdown()
    w.close()
