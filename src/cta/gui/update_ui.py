"""Update checking / installing UI for the main window."""

from __future__ import annotations

import logging
import os
import threading
import time

from PySide6.QtCore import QObject, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QApplication, QMessageBox, QProgressDialog

from .. import __version__, updater

log = logging.getLogger(__name__)

CHECK_INTERVAL = 12 * 3600


class UpdateController(QObject):
    _checked = Signal(object, object, bool)       # release | None, error | None, manual
    _progress = Signal(int, int)
    _prepared = Signal(object, object)            # new_root | None, error | None

    def __init__(self, window, settings):
        super().__init__(window)
        self.win = window
        self.settings = settings
        self._busy = False
        self._cancel = False
        self._dlg: QProgressDialog | None = None
        self._checked.connect(self._on_checked)
        self._progress.connect(self._on_progress)
        self._prepared.connect(self._on_prepared)

    # ---- start-up --------------------------------------------------------
    def startup(self):
        try:
            result = updater.cleanup_after_update()
        except Exception:  # noqa: BLE001
            log.exception("update cleanup failed")
            result = None
        if result == "ok":
            self.win.status_label.setText(f"Updated to Tempest {__version__}")
        elif result and result.startswith("failed"):
            QMessageBox.warning(self.win, "Update",
                                f"The last update could not be installed ({result[7:]}).\n"
                                f"Tempest {__version__} is still installed and working.")
        if os.environ.get("CTA_SMOKE_EXIT_MS") or os.environ.get("TEMPEST_HANDOVER"):
            return
        if self.auto_enabled() and time.time() - float(self.settings.value("update_last_check", 0)) > CHECK_INTERVAL:
            QTimer.singleShot(4000, lambda: self.check(manual=False))

    def auto_enabled(self) -> bool:
        return self.settings.value("auto_update_check", "true") == "true"

    # ---- check -----------------------------------------------------------
    def check(self, manual: bool = True):
        if self._busy:
            return
        self._busy = True
        if manual:
            self.win.status_label.setText("Checking for updates…")

        def run():
            try:
                rel = updater.check_latest()
                self._checked.emit(rel, None, manual)
            except Exception as e:  # noqa: BLE001
                self._checked.emit(None, e, manual)

        threading.Thread(target=run, daemon=True, name="update-check").start()

    def _on_checked(self, rel, err, manual):
        self._busy = False
        self.settings.setValue("update_last_check", time.time())
        if err is not None:
            log.info("update check failed: %s", err)
            if manual:
                QMessageBox.warning(self.win, "Check for updates", f"Could not check for updates:\n{err}")
                self.win.status_label.setText("Update check failed")
            return
        if not updater.is_newer(rel.version):
            if manual:
                QMessageBox.information(self.win, "Check for updates",
                                        f"You have the latest version (Tempest {__version__}).")
                self.win.status_label.setText("Tempest is up to date")
            return
        if not manual and self.settings.value("update_skip_version", "") == rel.version:
            return
        self._offer(rel)

    def _offer(self, rel):
        notes = rel.notes.strip()
        if len(notes) > 1500:
            notes = notes[:1500] + "…"
        box = QMessageBox(self.win)
        box.setWindowTitle("Update available")
        box.setIcon(QMessageBox.Information)
        box.setText(f"<b>Tempest {rel.version}</b> is available (you have {__version__}).")
        box.setInformativeText("Your library, settings and setlists are kept.\n\n"
                               "Update now? Tempest restarts when it's done.")
        if notes:
            box.setDetailedText(notes)
        b_now = box.addButton("Update now", QMessageBox.AcceptRole)
        box.addButton("Later", QMessageBox.RejectRole)
        b_skip = box.addButton("Skip this version", QMessageBox.DestructiveRole)
        box.setDefaultButton(b_now)
        box.exec()
        clicked = box.clickedButton()
        if clicked is b_skip:
            self.settings.setValue("update_skip_version", rel.version)
        elif clicked is b_now:
            self.install(rel)

    # ---- install ---------------------------------------------------------
    def install(self, rel):
        ok, why = updater.can_self_update()
        if not ok or not rel.installable:
            msg = why if not ok else "This release has no build for your platform."
            r = QMessageBox.question(self.win, "Update",
                                     f"Tempest can't update itself here:\n{msg}\n\nOpen the download page instead?")
            if r == QMessageBox.Yes:
                QDesktopServices.openUrl(QUrl(rel.url))
            return
        if self.win.is_scanning():
            r = QMessageBox.question(self.win, "Update",
                                     "A library scan is running. Stop it and update? "
                                     "The scan resumes where it left off next time.")
            if r != QMessageBox.Yes:
                return
            self.win.stop_scan()
        self._cancel = False
        self._busy = True
        dlg = QProgressDialog(f"Downloading Tempest {rel.version}…", "Cancel", 0, 100, self.win)
        dlg.setWindowTitle("Updating Tempest")
        dlg.setWindowModality(Qt.WindowModal)
        dlg.setMinimumDuration(0)
        dlg.setAutoClose(False)
        dlg.setAutoReset(False)
        dlg.canceled.connect(self._on_cancel)
        dlg.show()
        self._dlg = dlg

        def run():
            try:
                new_root = updater.prepare_update(rel, lambda d, t: self._progress.emit(d, t),
                                                  lambda: self._cancel)
                self._prepared.emit(new_root, None)
            except Exception as e:  # noqa: BLE001
                log.exception("update failed")
                self._prepared.emit(None, e)

        threading.Thread(target=run, daemon=True, name="update-download").start()

    def _on_cancel(self):
        self._cancel = True

    def _on_progress(self, done, total):
        if self._dlg is None:
            return
        if total:
            self._dlg.setValue(int(done * 100 / total))
            self._dlg.setLabelText(f"Downloading… {done / 1e6:.0f} / {total / 1e6:.0f} MB")
        if total and done >= total:
            self._dlg.setLabelText("Verifying and testing the new version…")

    def _on_prepared(self, new_root, err):
        self._busy = False
        if self._dlg is not None:
            self._dlg.close()
            self._dlg = None
        if err is not None:
            if self._cancel:
                self.win.status_label.setText("Update cancelled")
                return
            QMessageBox.warning(self.win, "Update failed",
                                f"The update could not be prepared:\n{err}\n\n"
                                "Nothing was changed; Tempest keeps running as before.")
            return
        from . import live_update

        dj = getattr(self.win, "dj", None)
        if live_update.supported() and live_update.playing(dj):
            self._live(new_root)                 # music playing: hand the mix over, no stop
            return
        self._install_now(new_root)              # decks stopped (or Linux): the restart way

    def _install_now(self, new_root):
        try:
            updater.launch_swap(new_root, relaunch=True)
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self.win, "Update failed", f"Could not start the installer:\n{e}")
            return
        self.win.status_label.setText("Installing update - Tempest will restart…")
        self.win.quit_app()
        QApplication.instance().quit()

    # ---- live update (Windows, macOS) ------------------------------------------------
    def _live(self, new_root):
        from . import live_update

        dj = self.win.dj
        if dj.automix.tr is not None and dj.automix.tr.started:
            # never in the middle of an auto-mix blend: right after it
            self.win.status_label.setText("Update ready - it goes live after this blend")
            QTimer.singleShot(2000, lambda: self._live(new_root))
            return
        if not live_update.playing(dj):
            self._install_now(new_root)
            return
        self._host = live_update.LiveUpdateHost(self.win, new_root,
                                                on_fail=lambda why: self._when_stopped(new_root, why))
        self._host.start()

    def _when_stopped(self, new_root, why: str):
        """The live hand-over did not work: install the restart way once the decks stop."""
        from . import live_update

        self.win.status_label.setText(f"Live update did not go through ({why}) - "
                                      "it installs when the decks are stopped")
        self._stopped_since = None

        def poll():
            if live_update.playing(getattr(self.win, "dj", None)):
                self._stopped_since = None
            elif self._stopped_since is None:
                self._stopped_since = time.monotonic()
            elif time.monotonic() - self._stopped_since > 3.0:
                self._install_now(new_root)
                return
            QTimer.singleShot(1000, poll)
        QTimer.singleShot(1000, poll)
