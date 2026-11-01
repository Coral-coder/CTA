"""Live update: the running version hands its mix to the new one without a stop."""
import os
import time

import numpy as np
import pytest
from conftest import synth_track, write_audio

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")
pytest.importorskip("pylibrb")


def test_engines_line_up_and_crossfade():
    """The new engine lands where the old one is, and the two fades add up to one."""
    from cta.dj.beatgrid import BeatGrid
    from cta.dj.engine import SR, Engine, TrackInfo

    t = np.arange(SR * 20) / SR
    y = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    a, b = Engine(), Engine()
    for e in (a, b):
        e.load("A", np.stack([y, y]), BeatGrid(120.0, 0.0, 20.0), TrackInfo(loudness=-10))
    a.set_playing("A", True)
    for _ in range(40):
        a.render(512)
    b.handover_fade(float("inf"), 0.15, True)            # the new one starts silent
    b.set_playing("A", True)
    m, _ = b.render(512)
    assert np.abs(m).max() == 0.0
    clock = a.handover_clock()
    b.handover_align(clock)
    b.render(512)
    a.render(512)
    # same place (to within a block's timing) on both
    assert abs(b.decks["A"].play_pos - a.decks["A"].play_pos) < 1500
    t0 = time.perf_counter()
    a.handover_fade(t0, 0.15, False)
    b.handover_fade(t0, 0.15, True)
    ga = a._ho_gain(t0 + 0.075, 4)
    gb = b._ho_gain(t0 + 0.075, 4)
    assert np.allclose(ga ** 2 + gb ** 2, 1.0, atol=1e-3)  # equal power: no dip, no bump
    assert a._ho_gain(t0 + 1.0, 4).max() == 0.0 and b._ho_gain(t0 + 1.0, 4).min() == 1.0


def test_the_new_version_takes_over_the_running_mix(tmp_data_dir, monkeypatch):
    """Old and new version side by side (one process, real socket): the new one
    loads the same tracks, sets them up the same way, plays muted in step and
    switches in as the old one fades out."""
    from PySide6.QtCore import QEventLoop, QSettings, QTimer

    import cta.dj.engine as E
    import cta.gui.dj_window as DW
    import cta.gui.live_update as LU
    from cta.db import Database
    from cta.gui.main_window import MainWindow
    from cta.scanner import Scanner, ScanSettings

    monkeypatch.setattr(E.Engine, "start", lambda self, *a, **k: None)
    monkeypatch.setattr("cta.dj.engine.list_output_devices", lambda: [])
    monkeypatch.setattr(DW, "list_output_devices", lambda: [])
    monkeypatch.setattr(DW.Controller, "list_inputs", staticmethod(lambda: []))
    monkeypatch.setattr(LU, "_on_card", lambda eng: True)          # no sound card on the test machine
    music = tmp_data_dir / "music"
    music.mkdir()
    for n, bpm in enumerate((124.0, 126.0)):
        write_audio(str(music / f"t{n}.flac"), synth_track(bpm, 40, chord=(57, 60, 64)))
    db = Database(str(tmp_data_dir / "lib.sqlite3"))
    Scanner(db, ScanSettings(workers=1, use_ai=False)).run([str(music)])
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def wait(ms):
        loop = QEventLoop()
        QTimer.singleShot(ms, loop.quit)
        loop.exec()

    def window(name):
        st = QSettings(str(tmp_data_dir / f"{name}.ini"), QSettings.IniFormat)
        st.setValue("auto_update_check", "false")
        w = MainWindow(db, st)
        w.show()
        for _ in range(100):
            if w.lib is not None:
                break
            wait(50)
        return w

    # ---- the running version, mid-set ----
    old = window("old")
    old.open_dj()
    od = old.dj
    od.load_index("A", 0)
    if os.environ.get("LU_FOUR"):
        od.show_cd.setChecked(True)
        od.load_index("C", 1)
    for _ in range(200):
        if od.engine.decks["A"].loaded and (not os.environ.get("LU_FOUR") or od.engine.decks["C"].loaded):
            break
        wait(50)
    od.engine.set("A", volume=0.8, eq=(1.0, 0.5, 1.0), keylock=True)
    od.engine.set_playing("A", True)

    rendering = {"on": True}

    due = {}

    def pump():                                        # stands in for both sound cards: real-time blocks
        if rendering["on"]:
            now = time.perf_counter()
            for w_ in (old, getattr(pump, "new", None)):
                if w_ is not None and getattr(w_, "dj", None) is not None:
                    t = due.setdefault(id(w_), now)
                    while t <= now:
                        w_.dj.engine.render(512)
                        t += 512 / 44100
                    due[id(w_)] = t
            QTimer.singleShot(3, pump)
    pump()
    wait(300)

    host = LU.LiveUpdateHost(old, "/nonexistent", on_fail=lambda why: pytest.fail(why))
    monkeypatch.setattr(host, "_copy_aside", lambda: "/nonexistent")
    launched = {}
    monkeypatch.setattr(LU, "_launch", lambda root, env: launched.update(env=env))
    finished = {}
    monkeypatch.setattr(host, "_quit_into_new", lambda: finished.setdefault("t", time.perf_counter()))
    host._finish.disconnect()
    host._finish.connect(lambda: finished.setdefault("t", time.perf_counter()))
    host.start()
    for _ in range(100):
        if "env" in launched:
            break
        wait(20)

    # ---- the new version ----
    monkeypatch.setenv("TEMPEST_HANDOVER", launched["env"])
    monkeypatch.delenv("CTA_NO_AUTO_DJ", raising=False)  # as in real life: DJ mode opens by itself too
    st = QSettings(str(tmp_data_dir / "new.ini"), QSettings.IniFormat)
    st.setValue("auto_update_check", "false")
    new = MainWindow(db, st)                           # as in real life: the hand-over starts at once,
    new.show()                                         # before the library has loaded
    pump.new = new
    guest = LU.LiveUpdateGuest(new, launched["env"])
    gave_up = {}
    monkeypatch.setattr(guest, "_give_up", lambda why, tell=True: gave_up.setdefault("why", why))
    guest.start()
    for _ in range(600):
        if "t" in finished or gave_up:
            break
        wait(50)
    assert not gave_up, gave_up
    assert "t" in finished                             # the old version faded out and let go
    if os.environ.get("LU_SHOT"):
        new.resize(1500, 950)
        wait(300)
        new.grab().save(os.environ["LU_SHOT"])
    print("DJ panels in the new window:", len(new.findChildren(DW.DJPanel)),
          "deck panels:", len(new.findChildren(DW.DeckPanel)))
    assert len(new.findChildren(DW.DJPanel)) == 1      # one set of decks, never two
    nd = new.dj.engine.decks["A"]
    assert nd.loaded and nd.info.path == od.engine.decks["A"].info.path
    assert nd.playing and nd.keylock and abs(nd.volume - 0.8) < 1e-6 and abs(nd.eq[1] - 0.5) < 1e-6
    # in step with the old one (both rendered by the same loop here)
    assert abs(nd.play_pos - od.engine.decks["A"].play_pos) < 0.05 * 44100
    # after the switch: old silent, new at full level
    wait(int((LU.LEAD_S + LU.FADE_S) * 1000) + 300)
    m_old, _ = od.engine.render(512)
    m_new, _ = new.dj.engine.render(512)
    assert np.abs(m_old).max() == 0.0 and np.abs(m_new).max() > 1e-3
    rendering["on"] = False
    for w_ in (old, new):
        w_.dj.shutdown()
        w_.close()


def test_the_running_version_carries_on_when_the_new_one_backs_out(monkeypatch):
    import json
    import socket
    from types import SimpleNamespace

    from PySide6.QtCore import QEventLoop, QObject, QTimer

    import cta.gui.live_update as LU

    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def wait(ms):
        loop = QEventLoop()
        QTimer.singleShot(ms, loop.quit)
        loop.exec()

    main = QObject()
    main.status_label = SimpleNamespace(setText=lambda t: None)
    main.dj = SimpleNamespace(engine=SimpleNamespace(handover_clock=lambda: None))
    monkeypatch.setattr(LU, "capture_state", lambda m: {"decks": {}})
    failed = []
    host = LU.LiveUpdateHost(main, "/nonexistent", on_fail=failed.append)
    monkeypatch.setattr(host, "_copy_aside", lambda: "/nonexistent")
    launched = {}
    monkeypatch.setattr(LU, "_launch", lambda root, env: launched.update(env=env))
    host.start()
    for _ in range(100):
        if "env" in launched:
            break
        wait(20)
    h, port, token = launched["env"].split(":", 2)
    s = socket.create_connection((h, int(port)))
    s.sendall((json.dumps({"op": "hello", "token": token}) + "\n").encode())
    assert json.loads(s.makefile().readline())["op"] == "state"
    s.sendall((json.dumps({"op": "abort", "why": "could not load"}) + "\n").encode())
    for _ in range(100):
        if failed:
            break
        wait(20)
    assert failed and "could not load" in failed[0]           # the update waits for the decks to stop
    assert not host.switched                                  # nothing was handed over, nothing stopped
    s.close()
