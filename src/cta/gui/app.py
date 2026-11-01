"""Desktop application entry point."""

from __future__ import annotations

import logging
import os
import sys
import traceback

log = logging.getLogger(__name__)


def _dark_palette(app):
    from PySide6.QtGui import QColor, QPalette

    p = QPalette()
    base = QColor(30, 31, 34)
    alt = QColor(38, 39, 43)
    text = QColor(225, 226, 230)
    hl = QColor(52, 120, 220)
    p.setColor(QPalette.Window, QColor(36, 37, 40))
    p.setColor(QPalette.WindowText, text)
    p.setColor(QPalette.Base, base)
    p.setColor(QPalette.AlternateBase, alt)
    p.setColor(QPalette.ToolTipBase, QColor(50, 51, 56))
    p.setColor(QPalette.ToolTipText, text)
    p.setColor(QPalette.Text, text)
    p.setColor(QPalette.Button, QColor(46, 47, 52))
    p.setColor(QPalette.ButtonText, text)
    p.setColor(QPalette.BrightText, QColor(255, 80, 80))
    p.setColor(QPalette.Highlight, hl)
    p.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
    p.setColor(QPalette.PlaceholderText, QColor(130, 132, 138))
    for role in (QPalette.Text, QPalette.ButtonText, QPalette.WindowText):
        p.setColor(QPalette.Disabled, role, QColor(110, 111, 116))
    app.setPalette(p)


def app_settings():
    """The app's QSettings; the first time, everything saved under the old
    Tempest name is carried over."""
    from PySide6.QtCore import QSettings

    from .. import APP_NAME, LEGACY_NAME

    s = QSettings(APP_NAME, APP_NAME)
    if not s.allKeys():
        old = QSettings(LEGACY_NAME, LEGACY_NAME)
        for k in old.allKeys():
            s.setValue(k, old.value(k))
        s.sync()
    return s


def _instance_name() -> str:
    import getpass
    import hashlib

    from .. import paths

    try:
        user = getpass.getuser()
    except Exception:  # noqa: BLE001
        user = "user"
    return "cta-" + hashlib.sha1(f"{user}|{paths.data_dir()}".encode("utf-8", "replace")).hexdigest()[:16]


def _signal_running_instance(name: str, msg: str) -> bool:
    """True when another Tempest is running (it was told to show itself)."""
    from PySide6.QtNetwork import QLocalSocket

    s = QLocalSocket()
    s.connectToServer(name)
    if not s.waitForConnected(400):
        return False
    s.write(msg.encode())
    s.flush()
    s.waitForBytesWritten(400)
    s.disconnectFromServer()
    return True


def _listen_for_instances(name: str, on_message):
    from PySide6.QtNetwork import QLocalServer

    srv = QLocalServer()
    QLocalServer.removeServer(name)            # stale socket from a crash
    if not srv.listen(name):
        log.warning("single-instance server: %s", srv.errorString())
        return srv

    def conn():
        while srv.hasPendingConnections():
            c = srv.nextPendingConnection()

            def read(c=c):
                on_message(bytes(c.readAll().data()).decode("utf-8", "ignore").strip())
            c.readyRead.connect(read)
            c.disconnected.connect(c.deleteLater)
    srv.newConnection.connect(conn)
    return srv


CRASH_MARK = "--- clean exit"


def _crash_log():
    """Native crashes (audio driver, Qt) kill the process without a Python
    traceback: faulthandler writes one to crash.log. Returns (file, the
    previous session crashed)."""
    import faulthandler

    from .. import __version__, paths

    fn = os.path.join(paths.data_dir(), "crash.log")
    crashed = False
    try:
        if os.path.exists(fn):
            with open(fn, "rb") as fh:
                fh.seek(max(0, os.path.getsize(fn) - 4096))
                tail = fh.read().decode("utf-8", "replace").strip().splitlines()
            crashed = bool(tail) and not tail[-1].startswith(CRASH_MARK)
            if os.path.getsize(fn) > 2_000_000:
                os.replace(fn, fn + ".old")
        f = open(fn, "a", encoding="utf-8")  # noqa: SIM115 - kept open for faulthandler
        import time

        f.write(f"\n--- Tempest {__version__} started {time.strftime('%Y-%m-%d %H:%M:%S')} pid {os.getpid()}\n")
        f.flush()
        faulthandler.enable(file=f, all_threads=True)
        return f, crashed
    except OSError:
        return None, crashed


def run_gui(db_path: str | None = None, background: bool = False) -> int:
    from PySide6.QtWidgets import QApplication, QMessageBox, QSystemTrayIcon

    from .. import paths
    from ..db import Database
    from .main_window import MainWindow

    if QApplication.instance() is None and "QT_SCALE_FACTOR" not in os.environ:
        # Settings -> Interface size (small laptop screens); must be set before Qt starts
        try:
            pct = int(app_settings().value("ui_scale", 100))
        except (TypeError, ValueError):
            pct = 100
        if pct != 100 and 50 <= pct <= 200:
            os.environ["QT_SCALE_FACTOR"] = f"{pct / 100:.2f}"
    # the audio callback needs Python's interpreter lock within a few ms of every
    # block: make busy threads (drawing, analysis) hand it over 5x more often than
    # the default 5 ms, so the music never stutters while the screen is busy
    sys.setswitchinterval(0.001)
    # fewer, smaller memory clean-ups while the music plays (see main_window.quiet_memory)
    import gc

    gc.set_threshold(20000, 20, 1000)
    app = QApplication.instance() or QApplication(sys.argv)
    from .. import APP_NAME

    app.setApplicationName(APP_NAME)
    app.setOrganizationName(APP_NAME)
    app.setStyle("Fusion")
    _dark_palette(app)
    from .theme import QSS

    app.setStyleSheet(QSS)
    try:
        from importlib import resources

        from PySide6.QtGui import QIcon, QPixmap

        pm = QPixmap()
        pm.loadFromData(resources.files("cta").joinpath("icon.png").read_bytes())
        app.setWindowIcon(QIcon(pm))
    except Exception:  # noqa: BLE001 - cosmetic only
        pass

    def excepthook(etype, value, tb):
        msg = "".join(traceback.format_exception(etype, value, tb))
        log.error("unhandled exception:\n%s", msg)
        try:
            QMessageBox.critical(None, "Tempest - unexpected error",
                                 f"{value}\n\nDetails were written to {paths.log_path()}")
        except Exception:  # noqa: BLE001
            pass

    sys.excepthook = excepthook

    def thread_hook(a):
        # worker threads (loaders, phone remote): windowed builds have no stderr
        log.error("unhandled exception in thread %s", getattr(a.thread, "name", "?"),
                  exc_info=(a.exc_type, a.exc_value, a.exc_traceback))

    import threading

    threading.excepthook = thread_hook

    name = _instance_name()
    handover = os.environ.get("TEMPEST_HANDOVER")         # a live update: the old version is still running
    if not os.environ.get("CTA_SMOKE_EXIT_MS") and not handover and \
            _signal_running_instance(name, "background" if background else "show"):
        log.info("Tempest is already running - asked it to show itself")
        return 0

    crash_file, crashed = _crash_log()
    if crashed:
        log.warning("the previous Tempest session ended without closing cleanly - details (if any) in %s",
                    os.path.join(paths.data_dir(), "crash.log"))

    db = Database(db_path or paths.db_path())
    w = MainWindow(db)
    if QSystemTrayIcon.isSystemTrayAvailable():
        from .tray import Tray

        w.tray = Tray(w)
        app.setQuitOnLastWindowClosed(False)
    w._instance_name = name
    if handover:
        from PySide6.QtCore import QTimer

        from .live_update import LiveUpdateGuest

        w._guest = LiveUpdateGuest(w, handover)
        QTimer.singleShot(0, w._guest.start)
    else:
        w.claim_single_instance()
        try:
            from .live_update import cleanup_old_copies

            cleanup_old_copies()
        except Exception:  # noqa: BLE001
            pass
    if background and w.tray is not None:
        log.info("started in the background (system tray)")
    elif background:
        w.showMinimized()
    else:
        w.show()
    if background or w.settings.value("server_mode", "false") == "true":
        w.start_server()
    try:
        from .. import autostart

        autostart.refresh()
    except Exception:  # noqa: BLE001
        pass
    from PySide6.QtCore import QTimer

    smoke = os.environ.get("CTA_SMOKE_EXIT_MS")
    if smoke:
        # CI smoke test of the packaged GUI: start, render, quit cleanly
        QTimer.singleShot(int(smoke), lambda: (print("GUI SMOKE OK", flush=True), w.quit_app(), app.quit()))
    if crashed:
        QTimer.singleShot(1500, lambda: w.status_label.setText(
            "Tempest did not close cleanly last time - crash details were saved to crash.log in the data folder"))
    rc = app.exec()
    if crash_file is not None:
        try:
            import faulthandler
            import time

            faulthandler.disable()
            crash_file.write(f"{CRASH_MARK} {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            crash_file.close()
        except OSError:
            pass
    return rc
