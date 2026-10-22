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
    assert len(wv["low"]) == 480 and wv["bpm"] > 0
    # queue editing and pause
    other = next(i for i in range(lib.n) if i != seed)
    call(lambda: cmd(action="queue_add", idx=other))
    assert other in list(dj.automix.queue)
    q = list(dj.automix.queue)
    call(lambda: cmd(action="queue_remove", pos=q.index(other)))
    assert list(dj.automix.queue).count(other) == q.count(other) - 1
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
