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
    # (Qt gives forward slashes on Windows: compare normalised paths)
    assert [os.path.normpath(u.toLocalFile()) for u in data.urls()] == \
        [os.path.normpath(Lib.paths[2]), os.path.normpath(Lib.paths[0])]
    assert bytes(data.data(TRACK_MIME)).decode() == "2"


def test_double_click_opens_the_details_page_not_a_player(tmp_data_dir, monkeypatch):
    """Double-clicking a track shows its details (feel, song map, waveform)
    instead of handing the file to an external player."""
    from PySide6.QtCore import QEventLoop, QSettings, QTimer

    from cta.db import Database
    from cta.gui.details import against_text, feel_text
    from cta.gui.main_window import MainWindow
    from cta.scanner import Scanner, ScanSettings

    music = tmp_data_dir / "music"
    music.mkdir()
    for n, bpm in enumerate((124.0, 128.0)):
        write_audio(str(music / f"t{n}.flac"), synth_track(bpm, 20, chord=(57, 60, 64)))
    db = Database(str(tmp_data_dir / "lib.sqlite3"))
    Scanner(db, ScanSettings(workers=1, use_ai=False)).run([str(music)])
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    opened = []
    monkeypatch.setattr("PySide6.QtGui.QDesktopServices.openUrl", staticmethod(lambda u: opened.append(u)))
    w = MainWindow(db, QSettings(str(tmp_data_dir / "settings.ini"), QSettings.IniFormat))
    w.show()

    def wait(ms):
        loop = QEventLoop()
        QTimer.singleShot(ms, loop.quit)
        loop.exec()

    for _ in range(100):
        if w.lib is not None:
            break
        wait(50)
    assert w.lib is not None and w.lib.n == 2
    w.focus_track(1)
    w.lib_view.doubleClicked.emit(w.lib_model.index(w.lib_model.row_of(0), 0))
    dlg = w._details
    assert dlg.isVisible() and not opened
    for _ in range(200):                                   # the audio is read in a thread
        if dlg.map.st is not None:
            break
        wait(50)
    assert dlg.map.st is not None and dlg.map.st.bars >= 10
    assert "BPM" in dlg.feel.text() and "After the focused track" in dlg.against.text()
    dlg.map.resize(800, 240)
    dlg.map.grab()                                          # paints the map + waveform
    assert "Sections:" in dlg.structure.text()
    assert "energy" in feel_text(w.lib, 0) and "tempo" in against_text(w.lib, 0, 1)
    assert "Nothing is playing" in against_text(w.lib, 0, None)
    dlg.close()
    w.close()
