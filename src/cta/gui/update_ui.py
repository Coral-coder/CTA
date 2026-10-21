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
            self.win.status_label.setText(f"Updated to CTA {__version__}")
        elif result and result.startswith("failed"):
            QMessageBox.warning(self.win, "Update",
                                f"The last update could not be installed ({result[7:]}).\n"
                                f"CTA {__version__} is still installed and working.")
        if os.environ.get("CTA_SMOKE_EXIT_MS"):
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
                                        f"You have the latest version (CTA {__version__}).")
                self.win.status_label.setText("CTA is up to date")
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
        box.setText(f"<b>CTA {rel.version}</b> is available (you have {__version__}).")
        box.setInformativeText("Your library, settings and setlists are kept.\n\n"
                               "Update now? CTA restarts when it's done.")
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
                                     f"CTA can't update itself here:\n{msg}\n\nOpen the download page instead?")
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
        dlg = QProgressDialog(f"Downloading CTA {rel.version}…", "Cancel", 0, 100, self.win)
        dlg.setWindowTitle("Updating CTA")
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
                                "Nothing was changed; CTA keeps running as before.")
            return
        try:
            updater.launch_swap(new_root, relaunch=True)
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self.win, "Update failed", f"Could not start the installer:\n{e}")
            return
        self.win.status_label.setText("Installing update - CTA will restart…")
        self.win.close()
        QApplication.instance().quit()
