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
