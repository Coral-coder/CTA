"""Headless GUI smoke test: build the window on a real library and drive it."""
import os

import pytest
from conftest import synth_track, write_audio

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")


def test_main_window(tmp_data_dir):
    from PySide6.QtCore import QEventLoop, QTimer

    from cta.db import Database
    from cta.gui.main_window import MainWindow
    from cta.scanner import Scanner, ScanSettings

    music = tmp_data_dir / "music"
    music.mkdir()
    for n, bpm in enumerate((124.0, 125.0, 126.0, 128.0)):
        write_audio(str(music / f"t{n}.flac"), synth_track(bpm, 15, chord=(57, 60, 64)))
    db = Database(str(tmp_data_dir / "lib.sqlite3"))
    db.add_folder(str(music))
    Scanner(db, ScanSettings(workers=1, use_ai=False)).run([str(music)])

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from PySide6.QtCore import QSettings

    w = MainWindow(db, QSettings(str(tmp_data_dir / "settings.ini"), QSettings.IniFormat))
    w.show()
    for _ in range(100):
        if w.lib is not None:
            break
        loop = QEventLoop()
        QTimer.singleShot(50, loop.quit)
        loop.exec()
    assert w.lib is not None and w.lib.n == 4
    w.lib_view.selectRow(0)
    app.processEvents()
    assert w.current is not None
    assert w.next_model.rowCount() == 3
    w.sl_len.setValue(4)
    w.generate_setlist()
    assert len(w.setlist) == 4
    w.search.setText("t1")
    w.apply_filter()
    assert w.lib_model.rowCount() == 1
    w.close()


def test_drag_out_carries_audio_files_for_other_apps(tmp_path):
    """Dragging rows out of CTA gives file URLs (what Traktor, rekordbox and
    Finder / Explorer accept) as well as CTA's own deck format."""
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    from cta.gui.models import TRACK_MIME, TrackListModel

    QApplication.instance() or QApplication([])

    class Lib:
        n = 3
        paths = [str(tmp_path / f"t{i}.mp3") for i in range(3)]

    m = TrackListModel([])
    m.lib = Lib()
    m.items = [2, 0, 1]
    data = m.mimeData([m.index(0, 0), m.index(0, 3), m.index(1, 0)])
    assert "text/uri-list" in m.mimeTypes()
    assert [u.toLocalFile() for u in data.urls()] == [Lib.paths[2], Lib.paths[0]]
    assert bytes(data.data(TRACK_MIME)).decode() == "2"
