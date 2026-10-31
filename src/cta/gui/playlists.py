"""Your own playlists, with bridge suggestions: between every two songs the
list shows (dimmed) the library track that best carries the flow from one
to the next. Double-click a suggestion, or right-click it, to keep it.
"""

from __future__ import annotations

import os
import time

from PySide6.QtCore import QModelIndex, Qt
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMenu,
    QMessageBox,
    QPushButton,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from ..mixer import score_from, suggest_bridge
from .models import TRACK_MIME, TrackListModel

META_KEY = "playlists"
DIM = QColor(112, 114, 122)


class PlaylistModel(TrackListModel):
    """Rows are playlist tracks and, between them, dimmed bridge suggestions."""

    def __init__(self, parent=None):
        super().__init__(["#", "Flow"], parent)
        self.kinds: list[str] = []         # "track" | "bridge" per row
        self.pos: list[int] = []           # playlist position (bridge: the track it follows)
        self.header_tips = {"Flow": "How well this mixes out of the track before it",
                            "#": "Dimmed rows are suggestions that bridge the gap: double-click to add"}

    def set_rows(self, lib, rows: list[tuple[str, int, int, list]]):
        self.kinds = [r[0] for r in rows]
        self.pos = [r[2] for r in rows]
        self.set_items(lib, [r[1] for r in rows], [r[3] for r in rows])

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        if self.kinds and self.kinds[index.row()] == "bridge":
            if role == Qt.ForegroundRole:
                return DIM
            if role == Qt.FontRole:
                f = QFont()
                f.setItalic(True)
                return f
            if role == Qt.ToolTipRole:
                return "Suggested bridge - double-click (or right-click) to add it to the playlist"
        return super().data(index, role)


class PlaylistView(QTableView):
    """Takes tracks dragged from the library (or files), reorders by drag."""

    def __init__(self, panel):
        super().__init__()
        self.panel = panel
        self.setAcceptDrops(True)
        self.setDragEnabled(True)
        self.setDragDropMode(QAbstractItemView.DragDrop)
        self.setDropIndicatorShown(True)

    def _accepts(self, md) -> bool:
        return md.hasFormat(TRACK_MIME) or md.hasUrls()

    def dragEnterEvent(self, ev):
        if self._accepts(ev.mimeData()):
            ev.acceptProposedAction()
        else:
            super().dragEnterEvent(ev)

    def dragMoveEvent(self, ev):
        if self._accepts(ev.mimeData()):
            ev.acceptProposedAction()
        else:
            super().dragMoveEvent(ev)

    def dropEvent(self, ev):
        md = ev.mimeData()
        row = self.indexAt(ev.position().toPoint()).row()
        at = self.panel.insert_pos_for_row(row)
        if ev.source() is self:
            self.panel.move_rows(sorted({ix.row() for ix in self.selectionModel().selectedRows()}), at)
        else:
            self.panel.add_mime(md, at)
        ev.acceptProposedAction()

    def keyPressEvent(self, ev):
        if ev.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            self.panel.remove_selected()
            return
        super().keyPressEvent(ev)


class PlaylistPanel(QWidget):
    def __init__(self, main):
        super().__init__()
        self.main = main
        self.lists: list[dict] = []          # [{"name", "paths"}]
        self._bridge_skip: dict[tuple[int, int], int] = {}
        self._path_idx: dict[str, int] = {}
        self._path_lib = None
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)

        row = QHBoxLayout()
        row.addWidget(QLabel("Playlist"))
        self.combo = QComboBox()
        self.combo.setMinimumWidth(160)
        self.combo.currentIndexChanged.connect(lambda _: self.refresh())
        row.addWidget(self.combo, 1)
        for text, fn, tip in (("New…", self.new_list, "Create a playlist"),
                              ("Rename…", self.rename_list, "Rename this playlist"),
                              ("Delete", self.delete_list, "Delete this playlist")):
            b = QPushButton(text)
            b.setToolTip(tip)
            b.clicked.connect(fn)
            row.addWidget(b)
        lay.addLayout(row)

        self.model = PlaylistModel(self)
        self.view = PlaylistView(self)
        self.view.setModel(self.model)
        self.view.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.view.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.view.setAlternatingRowColors(True)
        self.view.verticalHeader().setVisible(False)
        self.view.verticalHeader().setDefaultSectionSize(22)
        self.view.horizontalHeader().setStretchLastSection(True)
        for col, wd in enumerate([58, 46, 55, 75, 45, 100, 130, 190, 45]):
            self.view.horizontalHeader().resizeSection(col, wd)
        self.view.doubleClicked.connect(self._activated)
        self.view.setContextMenuPolicy(Qt.CustomContextMenu)
        self.view.customContextMenuRequested.connect(self._menu)
        lay.addWidget(self.view, 1)

        row = QHBoxLayout()
        self.show_bridges = QCheckBox("Suggest bridges")
        self.show_bridges.setToolTip("Show (dimmed) a track between each pair of songs that keeps the flow going")
        self.show_bridges.setChecked(str(main.settings.value("pl_bridges", "true")).lower() != "false")
        self.show_bridges.toggled.connect(self._bridges_toggled)
        row.addWidget(self.show_bridges)
        for text, fn, tip in (("▲", lambda: self.move_selected(-1), "Move up"),
                              ("▼", lambda: self.move_selected(1), "Move down"),
                              ("Remove", self.remove_selected, "Remove the selected songs (Delete)"),
                              ("Sort for flow", self.sort_for_flow,
                               "Reorder this playlist so every track mixes into the next as well as possible "
                               "(tempo, key, sound, style, energy - the Next Track rules). Nothing is removed; "
                               "the previous order comes back with Undo sort.")):
            b = QPushButton(text)
            b.setToolTip(tip)
            b.clicked.connect(fn)
            row.addWidget(b)
        row.addStretch()
        self.info = QLabel("")
        row.addWidget(self.info)
        for text, fn, tip in (("Queue in auto-mix", self.to_automix, "Play this playlist with auto-mix"),
                              ("To setlist", self.to_setlist, "Copy into the Setlist Builder"),
                              ("Export…", self.export_list, "Save as M3U8, Traktor NML, rekordbox XML or CSV")):
            b = QPushButton(text)
            b.setToolTip(tip)
            b.clicked.connect(fn)
            row.addWidget(b)
        lay.addLayout(row)
        self.hint = QLabel("Drag tracks here from the library, or right-click a track → Add to playlist.")
        self.hint.setStyleSheet("color: #8d8f97;")
        lay.addWidget(self.hint)
        self._load()

    # ---- storage -----------------------------------------------------------------
    def _load(self):
        try:
            rows = self.main.db.get_meta(META_KEY, []) or []
            self.lists = [{"name": str(r["name"]), "paths": [str(p) for p in r.get("paths", [])]} for r in rows]
        except Exception:  # noqa: BLE001
            self.lists = []
        self._fill_combo()

    def _save(self):
        try:
            self.main.db.set_meta(META_KEY, self.lists)
        except Exception:  # noqa: BLE001
            pass

    def _fill_combo(self, select: int | None = None):
        cur = self.combo.currentIndex() if select is None else select
        self.combo.blockSignals(True)
        self.combo.clear()
        for pl in self.lists:
            self.combo.addItem(pl["name"])
        if self.lists:
            self.combo.setCurrentIndex(min(max(cur, 0), len(self.lists) - 1))
        self.combo.blockSignals(False)
        self.refresh()

    @property
    def current(self) -> dict | None:
        i = self.combo.currentIndex()
        return self.lists[i] if 0 <= i < len(self.lists) else None

    def names(self) -> list[str]:
        return [pl["name"] for pl in self.lists]

    # ---- library mapping -------------------------------------------------------------
    @property
    def lib(self):
        return self.main.lib

    def _index_of(self, path: str) -> int:
        lib = self.lib
        if lib is None:
            return -1
        if self._path_lib is not lib:
            self._path_idx = {os.path.normcase(os.path.normpath(p)): i for i, p in enumerate(lib.paths)}
            self._path_lib = lib
        return self._path_idx.get(os.path.normcase(os.path.normpath(path)), -1)

    def items(self, pl: dict | None = None) -> list[int]:
        """Library indices of a playlist (files no longer in the library are skipped)."""
        pl = pl or self.current
        if pl is None:
            return []
        return [i for i in (self._index_of(p) for p in pl["paths"]) if i >= 0]

    # ---- the set as it happens ------------------------------------------------------------
    def log_played(self, idx: int, started: float):
        """Every track that went on air lands in a playlist named after the
        moment the set started ("Set 2026-10-31 21:04"), made on its first
        track; a track is added once per set."""
        lib = self.lib
        if lib is None or not (0 <= idx < lib.n):
            return
        name = time.strftime("Set %Y-%m-%d %H:%M", time.localtime(started))
        k = next((i for i, pl in enumerate(self.lists) if pl["name"] == name), -1)
        if k < 0:
            self.lists.append({"name": name, "paths": []})
            k = len(self.lists) - 1
            self._fill_combo(self.combo.currentIndex() if self.combo.currentIndex() >= 0 else k)
        path = lib.paths[idx]
        if path in self.lists[k]["paths"]:
            return
        self.lists[k]["paths"].append(path)
        self._save()
        if k == self.combo.currentIndex():
            self.refresh()

    # ---- editing ---------------------------------------------------------------------
    def new_list(self, _=False, name: str | None = None) -> int:
        if name is None:
            name, ok = QInputDialog.getText(self, "New playlist", "Name:", text=f"Playlist {len(self.lists) + 1}")
            if not ok or not name.strip():
                return -1
        self.lists.append({"name": name.strip(), "paths": []})
        self._save()
        self._fill_combo(len(self.lists) - 1)
        return len(self.lists) - 1

    def rename_list(self):
        pl = self.current
        if pl is None:
            return
        name, ok = QInputDialog.getText(self, "Rename playlist", "Name:", text=pl["name"])
        if ok and name.strip():
            pl["name"] = name.strip()
            self._save()
            self._fill_combo()

    def delete_list(self):
        pl = self.current
        if pl is None:
            return
        if QMessageBox.question(self, "Delete playlist", f"Delete the playlist “{pl['name']}”?\n"
                                "(The songs stay in your library.)") != QMessageBox.Yes:
            return
        self.lists.remove(pl)
        if getattr(self, "_undo_sort", None) and self._undo_sort[0] is pl:
            self._undo_sort = None
        self._save()
        self._fill_combo()

    def add_tracks(self, idx: list[int], playlist: int | None = None, at: int | None = None):
        """Append (or insert at playlist position `at`) library tracks."""
        lib = self.lib
        if lib is None or not idx:
            return
        if playlist is None:
            playlist = self.combo.currentIndex()
            if playlist < 0:
                playlist = self.new_list(name="My playlist")
        pl = self.lists[playlist]
        paths = [lib.paths[i] for i in idx if 0 <= i < lib.n]
        if at is None or at < 0 or at > len(pl["paths"]):
            pl["paths"].extend(paths)
        else:
            pl["paths"][at:at] = paths
        self._save()
        if playlist != self.combo.currentIndex():
            self.combo.setCurrentIndex(playlist)
        else:
            self.refresh()
        self.main.status_label.setText(f"Added {len(paths)} track(s) to “{pl['name']}”")

    def add_mime(self, md, at: int | None):
        lib = self.lib
        if lib is None:
            return
        idx = []
        if md.hasUrls():
            idx = [self._index_of(u.toLocalFile()) for u in md.urls() if u.isLocalFile()]
            idx = [i for i in idx if i >= 0]
        if not idx and md.hasFormat(TRACK_MIME):
            try:
                idx = [int(bytes(md.data(TRACK_MIME)).decode())]
            except ValueError:
                idx = []
        self.add_tracks(idx, at=at)

    def insert_pos_for_row(self, row: int) -> int:
        """Playlist position a drop on a table row goes to."""
        if row < 0 or row >= len(self.model.kinds):
            return -1
        p = self.model.pos[row]
        return p + 1 if self.model.kinds[row] == "bridge" else p

    def _selected_positions(self, rows=None) -> list[int]:
        rows = rows if rows is not None else sorted({ix.row() for ix in self.view.selectionModel().selectedRows()})
        return sorted({self.model.pos[r] for r in rows if 0 <= r < len(self.model.kinds)
                       and self.model.kinds[r] == "track"})

    def move_rows(self, rows: list[int], at: int):
        pl = self.current
        sel = self._selected_positions(rows)
        if pl is None or not sel:
            return
        paths = pl["paths"]
        at = len(paths) if at < 0 else at
        moving = [paths[p] for p in sel]
        before = sum(1 for p in sel if p < at)
        rest = [p for k, p in enumerate(paths) if k not in set(sel)]
        at -= before
        pl["paths"] = rest[:at] + moving + rest[at:]
        self._save()
        self.refresh()

    def move_selected(self, d: int):
        pl = self.current
        sel = self._selected_positions()
        if pl is None or len(sel) != 1:
            return
        p, j = sel[0], sel[0] + d
        if not 0 <= j < len(pl["paths"]):
            return
        pl["paths"][p], pl["paths"][j] = pl["paths"][j], pl["paths"][p]
        self._save()
        self.refresh()
        self._select_pos(j)

    def sort_for_flow(self):
        """Reorder the current playlist for the best run of transitions."""
        from ..mixer import order_for_flow

        pl, lib = self.current, self.lib
        if pl is None or lib is None:
            return
        items = self.items(pl)
        if len(items) < 3:
            self.main.status_label.setText("Sort for flow needs at least three songs")
            return
        s = self.main.mix_settings()
        before = sum(score_from(lib, a, s).total[b] for a, b in zip(items, items[1:]))
        order = order_for_flow(lib, items, s)
        after = sum(score_from(lib, a, s).total[b] for a, b in zip(order, order[1:]))
        known = {lib.paths[i] for i in items}
        missing = [p for p in pl["paths"] if p not in known]          # files no longer in the library stay (at the end)
        self._undo_sort = (pl, list(pl["paths"]), None)
        pl["paths"] = [lib.paths[i] for i in order] + missing
        self._save()
        self.refresh()
        n = len(items) - 1
        self.main.status_label.setText(f"“{pl['name']}” sorted for flow: average transition {before / n * 100:.0f} "
                                       f"→ {after / n * 100:.0f} (right-click the list → Undo sort to go back)")

    def undo_sort(self):
        u = getattr(self, "_undo_sort", None)
        self._undo_sort = None
        if not u:
            return
        pl, before, _after = u
        k = next((i for i, x in enumerate(self.lists) if x is pl), -1)
        if k < 0:
            return                                     # that playlist is gone: nothing to undo
        added = [p for p in pl["paths"] if p not in set(before)]    # e.g. tracks the live set logged since
        pl["paths"] = list(before) + added
        self._save()
        if k != self.combo.currentIndex():
            self.combo.setCurrentIndex(k)
        else:
            self.refresh()

    def remove_selected(self):
        pl = self.current
        sel = self._selected_positions()
        if pl is None or not sel:
            return
        pl["paths"] = [p for k, p in enumerate(pl["paths"]) if k not in set(sel)]
        self._save()
        self.refresh()

    def accept_bridge(self, row: int):
        """Make the suggestion on `row` part of the playlist."""
        pl = self.current
        if pl is None or self.model.kinds[row] != "bridge":
            return
        self.add_tracks([self.model.items[row]], at=self.model.pos[row] + 1)

    def other_bridge(self, row: int):
        a = self.model.pos[row]
        items = self.items()
        key = (items[a], items[a + 1])
        self._bridge_skip[key] = self._bridge_skip.get(key, 0) + 1
        self.refresh()

    def _select_pos(self, p: int):
        for r, (k, q) in enumerate(zip(self.model.kinds, self.model.pos)):
            if k == "track" and q == p:
                self.view.selectRow(r)
                return

    def _bridges_toggled(self, on):
        self.main.settings.setValue("pl_bridges", bool(on))
        self.refresh()

    # ---- display -----------------------------------------------------------------------
    def refresh(self):
        lib = self.lib
        items = self.items()
        self.hint.setVisible(not items)
        if lib is None:
            self.model.set_rows(None, [])
            return
        s = self.main.mix_settings()
        rows = []
        inlist = set(items)
        cache: dict[int, object] = {}

        def sc(i):
            if i not in cache:
                cache[i] = score_from(lib, i, s)
            return cache[i]
        for k, i in enumerate(items):
            flow = f"{sc(items[k - 1]).total[i] * 100:.0f}" if k else ""
            rows.append(("track", i, k, [str(k + 1), flow]))
            if self.show_bridges.isChecked() and k + 1 < len(items):
                c = items[k + 1]
                opts = suggest_bridge(lib, i, c, s, exclude=inlist)
                if opts:
                    b = opts[self._bridge_skip.get((i, c), 0) % len(opts)]
                    rows.append(("bridge", b, k, ["  + bridge", f"{min(sc(i).total[b], sc(b).total[c]) * 100:.0f}"]))
        self.model.set_rows(lib, rows)
        total = sum(lib.duration[i] for i in items)
        self.info.setText(f"{len(items)} songs · {int(total // 3600)}h {int(total % 3600 // 60):02d}m ")

    def _activated(self, ix: QModelIndex):
        r = ix.row()
        if self.model.kinds[r] == "bridge":
            self.accept_bridge(r)
        else:
            self.main.show_details(self.model.items[r])     # never an external player

    def _menu(self, pos):
        ix = self.view.indexAt(pos)
        if not ix.isValid():
            return
        r = ix.row()
        i = self.model.items[r]
        m = QMenu(self)
        if self.model.kinds[r] == "bridge":
            m.addAction("Add this bridge to the playlist", lambda: self.accept_bridge(r))
            m.addAction("Suggest another bridge", lambda: self.other_bridge(r))
            m.addSeparator()
        self.main._add_deck_actions(m, i)
        m.addSeparator()
        m.addAction("Preview in default player", lambda: self.main.play(i))
        m.addAction("Show next-track options for this", lambda: (self.main.focus_track(i),
                                                                 self.main.tabs.setCurrentIndex(0)))
        if self.model.kinds[r] == "track":
            m.addSeparator()
            m.addAction("Remove from playlist", self.remove_selected)
        m.addSeparator()
        m.addAction("Sort for flow", self.sort_for_flow)
        if getattr(self, "_undo_sort", None):
            m.addAction("Undo sort", self.undo_sort)
        m.exec(self.view.viewport().mapToGlobal(pos))

    # ---- using a playlist ---------------------------------------------------------------
    def to_setlist(self):
        items = self.items()
        if not items:
            return
        self.main.setlist = list(items)
        self.main._refresh_setlist()
        self.main.tabs.setCurrentIndex(1)

    def to_automix(self):
        items = self.items()
        if not items:
            return
        self.main.open_dj()
        dj = getattr(self.main, "dj", None)
        if dj is not None:
            dj.queue_items(items)

    def export_list(self):
        pl = self.current
        items = self.items()
        if pl is None or not items:
            QMessageBox.information(self, "Export", "The playlist is empty.")
            return
        from ..export import export

        last = self.main.settings.value("export_dir", os.path.expanduser("~"))
        safe = "".join(c for c in pl["name"] if c not in '<>:"/\\|?*') or "playlist"
        path, flt = QFileDialog.getSaveFileName(
            self, "Export playlist", os.path.join(last, safe + ".m3u8"),
            "M3U8 playlist (*.m3u8);;Traktor NML (*.nml);;rekordbox XML (*.xml);;CSV spreadsheet (*.csv)")
        if not path:
            return
        if not os.path.splitext(path)[1]:
            path += {"M3U8": ".m3u8", "Trakt": ".nml", "rekor": ".xml", "CSV s": ".csv"}.get(flt[:5], ".m3u8")
        try:
            export(self.lib, items, path, pl["name"])
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "Export failed", str(e))
            return
        self.main.settings.setValue("export_dir", os.path.dirname(path))
        self.main.status_label.setText(f"Exported {len(items)} tracks to {path}")

    def add_menu(self, m: QMenu, idx: list[int]):
        """'Add to playlist ▸' submenu for a context menu."""
        sub = m.addMenu("Add to playlist")
        for k, name in enumerate(self.names()):
            sub.addAction(name, lambda k=k: self.add_tracks(idx, playlist=k))
        if self.lists:
            sub.addSeparator()

        def new():
            k = self.new_list()
            if k >= 0:
                self.add_tracks(idx, playlist=k)
        sub.addAction("New playlist…", new)
