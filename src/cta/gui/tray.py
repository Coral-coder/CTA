"""System tray icon: CTA keeps serving the phone remote with the window closed."""

from __future__ import annotations

from PySide6.QtCore import QObject, QTimer
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QApplication, QMenu, QSystemTrayIcon


class Tray(QObject):
    def __init__(self, win):
        super().__init__(win)
        self.win = win
        self._noticed = False
        self.icon = QSystemTrayIcon(QApplication.windowIcon(), win)
        self.menu = QMenu()
        a_open = QAction("Open CTA", self.menu)
        a_open.triggered.connect(win.show_from_tray)
        self.a_remote = QAction("Phone remote (QR code && PIN)…", self.menu)
        self.a_remote.triggered.connect(self._remote)
        self.a_pause = QAction("Pause mix", self.menu)
        self.a_pause.triggered.connect(self._pause)
        self.a_login = QAction("Start at login", self.menu)
        self.a_login.setCheckable(True)
        self.a_login.toggled.connect(self._login)
        a_quit = QAction("Quit CTA", self.menu)
        a_quit.triggered.connect(win.quit_app)
        for a in (a_open, self.a_remote, self.a_pause):
            self.menu.addAction(a)
        self.menu.addSeparator()
        self.menu.addAction(self.a_login)
        self.menu.addAction(a_quit)
        self.menu.aboutToShow.connect(self.refresh)
        self.icon.setContextMenu(self.menu)
        self.icon.activated.connect(self._activated)
        self.icon.show()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(5000)
        self.refresh()

    def _activated(self, reason):
        if reason in (QSystemTrayIcon.Trigger, QSystemTrayIcon.DoubleClick):
            self.win.show_from_tray()

    def _dj(self):
        return getattr(self.win, "dj", None)

    def _remote(self):
        dj = self._dj()
        if dj is None or dj.stream is None:
            self.win.start_server()
            dj = self._dj()
        if dj is not None and dj.stream is not None:
            dj.show_remote_info()

    def _pause(self):
        dj = self._dj()
        if dj is not None:
            dj.engine.set_paused(not dj.engine.paused)

    def _login(self, on: bool):
        from .. import autostart

        if on == autostart.is_enabled():
            return
        try:
            autostart.set_enabled(on)
            if on:
                self.win.settings.setValue("server_mode", "true")
        except OSError as e:
            self.icon.showMessage("CTA", f"Could not change the start-up setting: {e}")

    def refresh(self):
        from .. import autostart

        dj = self._dj()
        self.a_login.blockSignals(True)
        self.a_login.setChecked(autostart.is_enabled())
        self.a_login.blockSignals(False)
        self.a_pause.setEnabled(dj is not None)
        if dj is not None:
            self.a_pause.setText("Resume mix" if dj.engine.paused else "Pause mix")
        tip = "CTA"
        if dj is not None and dj.stream is not None:
            tip += " mix server - " + dj.stream.urls[0]
            m = dj.engine._master()
            if m is not None and m.info.title:
                tip += f"\nNow: {m.info.artist + ' - ' if m.info.artist else ''}{m.info.title}"[:120]
        self.icon.setToolTip(tip)

    def hidden_notice(self):
        if not self._noticed:
            self._noticed = True
            self.icon.showMessage("CTA is still running",
                                  "The mix server keeps running in the tray. Right-click the icon to quit.",
                                  QSystemTrayIcon.Information, 4000)

    def hide(self):
        self.timer.stop()
        self.icon.hide()
