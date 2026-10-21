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
    """True when another CTA is running (it was told to show itself)."""
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


def run_gui(db_path: str | None = None, background: bool = False) -> int:
    from PySide6.QtWidgets import QApplication, QMessageBox, QSystemTrayIcon

    from .. import paths
    from ..db import Database
    from .main_window import MainWindow

    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("CTA")
    app.setOrganizationName("CTA")
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
            QMessageBox.critical(None, "CTA - unexpected error",
                                 f"{value}\n\nDetails were written to {paths.log_path()}")
        except Exception:  # noqa: BLE001
            pass

    sys.excepthook = excepthook

    name = _instance_name()
    if not os.environ.get("CTA_SMOKE_EXIT_MS") and \
            _signal_running_instance(name, "background" if background else "show"):
        log.info("CTA is already running - asked it to show itself")
        return 0

    db = Database(db_path or paths.db_path())
    w = MainWindow(db)
    if QSystemTrayIcon.isSystemTrayAvailable():
        from .tray import Tray

        w.tray = Tray(w)
        app.setQuitOnLastWindowClosed(False)
    single = _listen_for_instances(name, lambda msg: msg == "show" and w.show_from_tray())  # noqa: F841
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
    smoke = os.environ.get("CTA_SMOKE_EXIT_MS")
    if smoke:
        # CI smoke test of the packaged GUI: start, render, quit cleanly
        from PySide6.QtCore import QTimer

        QTimer.singleShot(int(smoke), lambda: (print("GUI SMOKE OK", flush=True), w.quit_app(), app.quit()))
    return app.exec()
