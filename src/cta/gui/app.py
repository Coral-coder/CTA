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


def run_gui(db_path: str | None = None) -> int:
    from PySide6.QtWidgets import QApplication, QMessageBox

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

    db = Database(db_path or paths.db_path())
    w = MainWindow(db)
    w.show()
    smoke = os.environ.get("CTA_SMOKE_EXIT_MS")
    if smoke:
        # CI smoke test of the packaged GUI: start, render, quit cleanly
        from PySide6.QtCore import QTimer

        QTimer.singleShot(int(smoke), lambda: (print("GUI SMOKE OK", flush=True), app.quit()))
    return app.exec()
