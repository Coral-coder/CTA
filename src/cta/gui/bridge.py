"""The library bridge in the app: sharing this library (home side) and the
home libraries mirrored here (away side), with the downloader."""
from __future__ import annotations

import json
import logging
import os
import threading

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .. import paths
from ..share import DEFAULT_PORT, Downloader, ShareClient, ShareServer, forget_library, mirror_catalog, parse_code

log = logging.getLogger(__name__)


class Bridge(QObject):
    """Owned by the main window: the share server (when enabled), the remote
    libraries (settings), their clients and the downloader."""

    synced = Signal(str, dict)          # library id, {"added", "updated", "removed"}
    sync_failed = Signal(str, str)      # library name, error
    downloaded = Signal(str)            # local path

    def __init__(self, main):
        super().__init__(main)
        self.main = main
        self.settings = main.settings
        self.db = main.db
        self.server: ShareServer | None = None
        self._clients: dict[str, ShareClient] = {}
        self.downloader = Downloader(self.db, self.client_of, paths.data_dir())
        self.downloader.on_change = lambda p: self.downloaded.emit(p)
        self._syncing: set[str] = set()

    # ---- home side: sharing ---------------------------------------------------------------
    def share_key(self) -> str:
        import secrets

        k = str(self.settings.value("share_key", "") or "")
        if not k:
            k = secrets.token_urlsafe(32)
            self.settings.setValue("share_key", k)
        return k

    def start_share(self) -> bool:
        if self.server is not None and self.server.running:
            return True
        port = int(self.settings.value("share_port", DEFAULT_PORT))
        try:
            self.server = ShareServer(self.db, paths.data_dir(), port=port, key=self.share_key(),
                                      name=str(self.settings.value("share_name", "") or ""))
            self.server.start()
            return True
        except OSError as e:
            log.exception("library share failed")
            self.server = None
            self.main.status_label.setText(f"Could not share the library on port {port}: {e}")
            return False

    def stop_share(self):
        if self.server is not None:
            self.server.stop()
            self.server = None

    def apply_share_setting(self):
        if self.settings.value("share_enabled", "false") == "true":
            self.start_share()
        else:
            self.stop_share()

    def pairing_code(self) -> str:
        """The code to paste into the CTA you take along (host editable there)."""
        from ..dj.stream import lan_addresses

        host = str(self.settings.value("share_host", "") or "") or (lan_addresses() or ["your-public-address"])[0]
        port = int(self.settings.value("share_port", DEFAULT_PORT))
        if self.server is not None:
            return self.server.code(host)
        from ..share import ensure_cert, make_code

        tls = ensure_cert(paths.data_dir())
        return make_code(host, port, self.share_key(), tls[2] if tls else None)

    # ---- away side: remote libraries ----------------------------------------------------------
    def remotes(self) -> list[dict]:
        try:
            v = json.loads(self.settings.value("remote_libraries", "") or "[]")
            return v if isinstance(v, list) else []
        except ValueError:
            return []

    def _save_remotes(self, items: list[dict]):
        self.settings.setValue("remote_libraries", json.dumps(items))
        self._clients.clear()

    def client_of(self, library_id: str) -> ShareClient | None:
        c = self._clients.get(library_id)
        if c is None:
            r = next((x for x in self.remotes() if x.get("lib") == library_id), None)
            if r is None:
                return None
            c = self._clients[library_id] = ShareClient(r["host"], r["port"], r["key"], r.get("fingerprint", ""))
        return c

    def add_remote(self, code: str) -> dict:
        """Pair with a home library from its code: contacts it, remembers it, syncs."""
        info = parse_code(code)
        client = ShareClient(info["host"], info["port"], info["key"], info["fingerprint"])
        meta = client.info()                                     # raises when unreachable / wrong key
        items = [x for x in self.remotes() if x.get("lib") != meta["library"]]
        entry = {**info, "lib": meta["library"], "name": meta.get("name") or info["host"]}
        items.append(entry)
        self._save_remotes(items)
        self.sync(meta["library"])
        return entry

    def remove_remote(self, library_id: str, delete_files: bool = False) -> int:
        items = [x for x in self.remotes() if x.get("lib") != library_id]
        self._save_remotes(items)
        n = forget_library(self.db, paths.data_dir(), library_id, delete_files)
        self.main.reload_library()
        return n

    def sync(self, library_id: str | None = None):
        """Fetch the catalogue(s) in the background and reload the library."""
        for r in self.remotes():
            lib = r.get("lib")
            if lib is None or (library_id is not None and lib != library_id) or lib in self._syncing:
                continue
            self._syncing.add(lib)
            threading.Thread(target=self._sync_one, args=(r,), daemon=True, name="library-sync").start()

    def _sync_one(self, r: dict):
        lib = r["lib"]
        try:
            client = ShareClient(r["host"], r["port"], r["key"], r.get("fingerprint", ""))
            stats = mirror_catalog(self.db, paths.data_dir(), lib, client.catalog())
            self.synced.emit(lib, stats)
        except Exception as e:  # noqa: BLE001
            log.warning("sync of %s failed: %s", r.get("name"), e)
            self.sync_failed.emit(str(r.get("name") or r.get("host")), str(e))
        finally:
            self._syncing.discard(lib)

    # ---- downloads --------------------------------------------------------------------------
    def ensure_local(self, path: str, status=None) -> str:
        """Called from a loader thread: the file is here when this returns."""
        if os.path.isfile(path):
            return path
        if status is not None:
            status("Downloading…")
        return self.downloader.ensure(path)

    def download(self, idx: int, first: bool = False):
        lib = self.main.lib
        if lib is not None and 0 <= idx < lib.n and not lib.local[idx]:
            self.downloader.request(lib.paths[idx], first=first)

    def download_many(self, idxs):
        for i in idxs:
            self.download(i)

    def progress_text(self, path: str) -> str | None:
        p = self.downloader.progress.get(path)
        if p is None:
            return None
        done, total = p
        return f"Downloading {done / total * 100:.0f} %" if total else f"Downloading {done // 1024} kB"

    def close(self):
        self.stop_share()
        self.downloader.close()


# --------------------------------------------------------------------------
# settings section
# --------------------------------------------------------------------------

class BridgeSettings(QWidget):
    """The 'Library bridge' part of the Settings dialog."""

    def __init__(self, settings, bridge: Bridge | None, parent=None):
        super().__init__(parent)
        self.s = settings
        self.bridge = bridge
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(QLabel("<b>Library bridge</b> - play your home collection from anywhere"))
        # ---- share (home side)
        row = QHBoxLayout()
        self.share = QCheckBox("Share this library")
        self.share.setChecked(settings.value("share_enabled", "false") == "true")
        self.share.setToolTip("Serve the catalogue and the files to a CTA holding the pairing code (keeps "
                              "running in the tray with the mix server)")
        row.addWidget(self.share)
        row.addWidget(QLabel("port"))
        self.port = QSpinBox()
        self.port.setRange(1024, 65535)
        self.port.setValue(int(settings.value("share_port", DEFAULT_PORT)))
        row.addWidget(self.port)
        row.addWidget(QLabel("public address"))
        self.host = QLineEdit(str(settings.value("share_host", "") or ""))
        self.host.setPlaceholderText("your-home.example.org or 203.0.113.7 (empty: this LAN address)")
        self.host.setToolTip("How the other machine reaches this one over the internet: forward the port on your "
                             "router, or use a tunnel such as Tailscale and put its address here")
        row.addWidget(self.host, 1)
        lay.addLayout(row)
        row = QHBoxLayout()
        self.code = QLineEdit()
        self.code.setReadOnly(True)
        self.code.setToolTip("Paste this into Settings → Library bridge on the CTA you take along")
        row.addWidget(self.code, 1)
        b = QPushButton("Copy")
        b.clicked.connect(lambda: QApplication.clipboard().setText(self.code.text()))
        row.addWidget(b)
        b = QPushButton("New key")
        b.setToolTip("Replace the key: every paired machine has to pair again")
        b.clicked.connect(self._new_key)
        row.addWidget(b)
        lay.addLayout(row)
        self.share_info = QLabel("")
        self.share_info.setStyleSheet("color: #9a9ba3;")
        self.share_info.setWordWrap(True)
        lay.addWidget(self.share_info)
        self.port.valueChanged.connect(self._refresh_code)
        self.host.textChanged.connect(self._refresh_code)
        self._refresh_code()
        # ---- remote libraries (away side)
        lay.addWidget(QLabel("<b>Home libraries on this machine</b> - their tracks appear in your library (⬇ until "
                             "downloaded); a deck or the auto-mix downloads what it needs, or right-click → "
                             "Download now"))
        row = QHBoxLayout()
        self.paste = QLineEdit()
        self.paste.setPlaceholderText("Paste a pairing code (CTA-SHARE|host:port|key|fingerprint)")
        row.addWidget(self.paste, 1)
        b = QPushButton("Add")
        b.clicked.connect(self._add)
        row.addWidget(b)
        lay.addLayout(row)
        self.list = QListWidget()
        self.list.setMaximumHeight(90)
        lay.addWidget(self.list)
        row = QHBoxLayout()
        b = QPushButton("Sync now")
        b.clicked.connect(self._sync)
        row.addWidget(b)
        b = QPushButton("Remove")
        b.clicked.connect(lambda: self._remove(False))
        row.addWidget(b)
        b = QPushButton("Remove + delete downloads")
        b.clicked.connect(lambda: self._remove(True))
        row.addWidget(b)
        row.addStretch()
        lay.addLayout(row)
        self.remote_info = QLabel(f"<small>Downloads go to {os.path.join(paths.data_dir(), 'remote')}</small>")
        self.remote_info.setStyleSheet("color: #9a9ba3;")
        lay.addWidget(self.remote_info)
        self._fill()

    def _refresh_code(self):
        if self.bridge is None:
            return
        self.s.setValue("share_port", self.port.value())
        self.s.setValue("share_host", self.host.text().strip())
        try:
            self.code.setText(self.bridge.pairing_code())
        except Exception as e:  # noqa: BLE001
            self.code.setText(f"(no code yet: {e})")
        on = self.bridge.server is not None and self.bridge.server.running
        tls = self.bridge.server.tls if on else None
        self.share_info.setText(("Sharing now" if on else "Not sharing yet (tick the box and press OK)")
                                + (" · encrypted (the code carries the certificate fingerprint)" if tls or not on
                                   else " · plain HTTP: install the 'cryptography' package for encryption"))

    def _new_key(self):
        import secrets

        if QMessageBox.question(self, "New key", "Make a new key? Every machine paired with this one must pair "
                                "again with the new code.") != QMessageBox.Yes:
            return
        self.s.setValue("share_key", secrets.token_urlsafe(32))
        if self.bridge is not None and self.bridge.server is not None:
            self.bridge.stop_share()
            self.bridge.start_share()
        self._refresh_code()

    def _fill(self):
        self.list.clear()
        if self.bridge is None:
            return
        for r in self.bridge.remotes():
            self.list.addItem(f"{r.get('name', '')}  ·  {r['host']}:{r['port']}  ·  {r.get('lib', '')}")

    def _selected(self) -> dict | None:
        r = self.list.currentRow()
        items = self.bridge.remotes() if self.bridge is not None else []
        return items[r] if 0 <= r < len(items) else None

    def _add(self):
        if self.bridge is None:
            return
        code = self.paste.text().strip()
        if not code:
            return
        try:
            entry = self.bridge.add_remote(code)
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, "Library bridge", f"Could not reach that library:\n{e}\n\nIs it sharing, "
                                "is the port reachable from here, and is the code complete?")
            return
        self.paste.clear()
        self._fill()
        self.remote_info.setText(f"Paired with {entry['name']} - fetching its catalogue…")

    def _sync(self):
        r = self._selected()
        if self.bridge is not None:
            self.bridge.sync(r["lib"] if r else None)
            self.remote_info.setText("Syncing…")

    def _remove(self, delete_files: bool):
        r = self._selected()
        if r is None or self.bridge is None:
            return
        n = self.bridge.remove_remote(r["lib"], delete_files)
        self._fill()
        self.remote_info.setText(f"Removed {r.get('name', '')} ({n} tracks)")

    def save(self):
        self.s.setValue("share_enabled", "true" if self.share.isChecked() else "false")
        self.s.setValue("share_port", self.port.value())
        self.s.setValue("share_host", self.host.text().strip())
        if self.bridge is not None:
            self.bridge.apply_share_setting()
