"""Qt table models backed directly by the numpy arrays of :class:`Library`.

Sorting and filtering happen in numpy (not QSortFilterProxyModel) so a
50k-track library sorts instantly.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QAbstractTableModel, QMimeData, QModelIndex, Qt
from PySide6.QtGui import QColor

from ..keys import camelot_str, key_name
from ..library import Library

TRACK_MIME = "application/x-cta-track"


def _track_mime(lib_index: int) -> QMimeData:
    m = QMimeData()
    m.setData(TRACK_MIME, str(int(lib_index)).encode())
    return m


class _DragMixin:
    """Rows can be dragged onto a DJ deck."""

    def flags(self, index):
        f = super().flags(index)
        return f | Qt.ItemIsDragEnabled if index.isValid() else f

    def mimeTypes(self):
        return [TRACK_MIME]

    def mimeData(self, indexes):
        if not indexes:
            return None
        return _track_mime(self._lib_index_for_row(indexes[0].row()))

    def supportedDragActions(self):
        return Qt.CopyAction


def fmt_duration(sec: float) -> str:
    sec = int(round(sec or 0))
    return f"{sec // 60}:{sec % 60:02d}"


# Camelot wheel colours (hue per number) to make compatible keys easy to spot
def key_color(key: int) -> QColor | None:
    if key is None or key < 0:
        return None
    num = int(camelot_str(key)[:-1])
    c = QColor.fromHsv(int((num - 1) * 30) % 360, 120, 230)
    return c


class LibraryModel(_DragMixin, QAbstractTableModel):
    COLUMNS = ["Played", "Artist", "Title", "BPM", "Key", "Energy", "AI Genre", "Tag Genre", "Time", "Album"]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.lib: Library | None = None
        self.rows = np.zeros(0, np.int64)
        self._sort = (-1, Qt.AscendingOrder)
        self._filter = {"text": "", "bpm_min": 0.0, "bpm_max": 0.0, "genre": "", "keys": None}

    # ---- data source ------------------------------------------------------
    def set_library(self, lib: Library):
        self.beginResetModel()
        self.lib = lib
        self._apply()
        self.endResetModel()

    def set_filter(self, **kw):
        self._filter.update(kw)
        self.beginResetModel()
        self._apply()
        self.endResetModel()

    def _apply(self):
        lib = self.lib
        if lib is None or lib.n == 0:
            self.rows = np.zeros(0, np.int64)
            return
        f = self._filter
        mask = np.zeros(lib.n, bool)
        mask[lib.search(f["text"])] = True
        if f["bpm_min"] > 0:
            mask &= lib.bpm >= f["bpm_min"]
        if f["bpm_max"] > 0:
            mask &= lib.bpm <= f["bpm_max"]
        if f["genre"]:
            g = f["genre"]
            mask &= np.array([pg == g for pg in lib.parent_genres])
        if f["keys"] is not None:
            mask &= np.isin(lib.key, list(f["keys"]))
        rows = np.nonzero(mask)[0]
        self.rows = self._sorted(rows)

    def _sorted(self, rows: np.ndarray) -> np.ndarray:
        col, order = self._sort
        lib = self.lib
        if col < 0 or rows.size == 0:
            return rows
        name = self.COLUMNS[col]
        if name == "BPM":
            keys = lib.bpm[rows]
        elif name == "Key":
            keys = np.array([self._camelot_sort(k) for k in lib.key[rows]])
        elif name == "Energy":
            keys = lib.energy[rows]
        elif name == "Time":
            keys = lib.duration[rows]
        elif name == "Played":
            keys = lib.session_played[rows] * 1e12 + lib.last_played[rows]
        else:
            src = {"Artist": lib.artists, "Title": lib.titles, "AI Genre": lib.genres,
                   "Tag Genre": lib.tag_genres, "Album": lib.albums}[name]
            keys = np.array([src[i].lower() for i in rows], dtype=object)
        o = np.argsort(keys, kind="stable")
        if order == Qt.DescendingOrder:
            o = o[::-1]
        return rows[o]

    @staticmethod
    def _camelot_sort(k: int) -> int:
        c = camelot_str(int(k))
        if not c:
            return 999
        return int(c[:-1]) * 2 + (1 if c[-1] == "B" else 0)

    # ---- Qt API -----------------------------------------------------------
    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else int(self.rows.size)

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.COLUMNS)

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return self.COLUMNS[section]
        return None

    def sort(self, column, order=Qt.AscendingOrder):
        self._sort = (column, order)
        self.layoutAboutToBeChanged.emit()
        self.rows = self._sorted(self.rows)
        self.layoutChanged.emit()

    def lib_index(self, row: int) -> int:
        return int(self.rows[row])

    def _lib_index_for_row(self, row: int) -> int:
        return self.lib_index(row)

    def row_of(self, lib_idx: int) -> int:
        hits = np.nonzero(self.rows == lib_idx)[0]
        return int(hits[0]) if hits.size else -1

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid() or self.lib is None:
            return None
        i = int(self.rows[index.row()])
        lib = self.lib
        name = self.COLUMNS[index.column()]
        if role == Qt.DisplayRole:
            if name == "Played":
                if lib.session_played[i]:
                    return "● now" if lib.play_count[i] <= 1 else f"● {lib.play_count[i]}×"
                if lib.play_count[i]:
                    return f"✓ {lib.play_count[i]}×"
                return ""
            if name == "Artist":
                return lib.artists[i]
            if name == "Title":
                return lib.titles[i]
            if name == "BPM":
                b = lib.bpm[i]
                return (f"{b:.2f}".rstrip("0").rstrip(".") if b else "") + (" *" if lib.bpm_overridden[i] else "")
            if name == "Key":
                k = int(lib.key[i])
                return f"{camelot_str(k):>3}  {key_name(k)}" + (" *" if lib.key_overridden[i] else "")
            if name == "Energy":
                return f"{lib.energy[i]:.1f}"
            if name == "AI Genre":
                return lib.genres[i]
            if name == "Tag Genre":
                return lib.tag_genres[i]
            if name == "Time":
                return fmt_duration(lib.duration[i])
            if name == "Album":
                return lib.albums[i]
        elif role == Qt.ForegroundRole and name == "Key":
            return key_color(int(lib.key[i]))
        elif role == Qt.ForegroundRole and name == "Played":
            return QColor(255, 120, 90) if lib.session_played[i] else QColor(120, 200, 130)
        elif role == Qt.ForegroundRole and lib.session_played[i]:
            return QColor(120, 122, 130)          # already played this session: dimmed
        elif role == Qt.TextAlignmentRole and name in ("BPM", "Energy", "Time"):
            return int(Qt.AlignRight | Qt.AlignVCenter)
        elif role == Qt.ToolTipRole:
            if name == "AI Genre" and lib.styles[i]:
                return "\n".join(f"{lab.replace('---', ' / ')}: {p:.0%}" for lab, p in lib.styles[i])
            if name in ("Title", "Artist"):
                return lib.paths[i]
            if name == "Played" and lib.play_count[i]:
                import time as _t
                when = _t.strftime("%Y-%m-%d %H:%M", _t.localtime(lib.last_played[i]))
                return (f"Played {lib.play_count[i]} time(s), last {when}"
                        + ("\nAlready played in this session" if lib.session_played[i] else ""))
            if name == "BPM":
                return f"detected {lib.bpm_detected[i]:.2f} (confidence {lib.bpm_conf[i]:.0%})" + (
                    f", tag {lib.tag_bpm[i]:.2f}" if lib.tag_bpm[i] else "")
            if name == "Key":
                tk = int(lib.tag_key[i])
                return f"confidence {lib.key_conf[i]:.0%}" + (f", tag {key_name(tk)}" if tk >= 0 else "")
        return None


class TrackListModel(_DragMixin, QAbstractTableModel):
    """Generic list of library indices with optional per-row extra columns
    (used for suggestions and the setlist)."""

    def __init__(self, extra_columns: list[str], parent=None):
        super().__init__(parent)
        self.extra_columns = extra_columns
        self.columns = extra_columns + ["BPM", "Key", "Energy", "Genre", "Artist", "Title", "Time"]
        self.lib: Library | None = None
        self.items: list[int] = []
        self.extras: list[list] = []
        self.header_tips: dict[str, str] = {}

    def set_items(self, lib: Library, items: list[int], extras: list[list] | None = None):
        self.beginResetModel()
        self.lib = lib
        self.items = list(items)
        self.extras = extras or [[""] * len(self.extra_columns) for _ in items]
        self.endResetModel()

    def _lib_index_for_row(self, row: int) -> int:
        return int(self.items[row])

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.items)

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.columns)

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                return self.columns[section]
            return str(section + 1)
        if role == Qt.ToolTipRole and orientation == Qt.Horizontal:
            return self.header_tips.get(self.columns[section])
        return None

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid() or self.lib is None:
            return None
        r, c = index.row(), index.column()
        i = self.items[r]
        lib = self.lib
        ne = len(self.extra_columns)
        if c < ne:
            v = self.extras[r][c]
            if role == Qt.DisplayRole:
                return v
            if role == Qt.TextAlignmentRole:
                return int(Qt.AlignRight | Qt.AlignVCenter)
            return None
        name = self.columns[c]
        if role == Qt.DisplayRole:
            return {
                "BPM": f"{lib.bpm[i]:.2f}".rstrip("0").rstrip("."),
                "Key": f"{camelot_str(int(lib.key[i])):>3}  {key_name(int(lib.key[i]))}",
                "Energy": f"{lib.energy[i]:.1f}",
                "Genre": lib.genres[i] or lib.tag_genres[i],
                "Artist": lib.artists[i],
                "Title": lib.titles[i],
                "Time": fmt_duration(lib.duration[i]),
            }[name]
        if role == Qt.ForegroundRole and name == "Key":
            return key_color(int(lib.key[i]))
        if role == Qt.ToolTipRole:
            return lib.paths[i]
        return None
