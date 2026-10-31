"""Learned mixes & templates: see, perfect, export and reuse what auto-mix
learned from the DJ (transition examples) and the Mix Ahead recordings.

A *template* is one set of numbers (or one recording) that every transition
of a style group is mixed with from then on - no averaging, exactly that.
"""
from __future__ import annotations

import json
import logging
import time

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from ..dj.learn import describe
from ..styles import FAMILIES

log = logging.getLogger(__name__)

GROUPS = list(FAMILIES) + ["long", "club", "short"]
GROUP_TIPS = {"long": "trance / progressive (no AI style)", "club": "other club music (no AI style)",
              "short": "hip hop / pop / rock (no AI style)"}


def _item(text, editable=False, data=None):
    it = QTableWidgetItem(str(text))
    if not editable:
        it.setFlags(it.flags() & ~Qt.ItemIsEditable)
    if data is not None:
        it.setData(Qt.UserRole, data)
    return it


class LearnedMixesDialog(QDialog):
    """Two tables: what was learned by watching the DJ, and the Mix Ahead
    recordings; each row can be edited, deleted, or made the template for a
    style group. Export / import carries both."""

    def __init__(self, dj):
        super().__init__(dj)
        self.dj = dj
        self.learner = dj.learner
        self.store = dj.rehearsals
        self.setWindowTitle("Learned mixes & templates")
        self.resize(1100, 720)
        lay = QVBoxLayout(self)

        lay.addWidget(QLabel("<b>Learned from your mixes</b> - where in each track you blended, and for how long. "
                             "Edit the numbers in place (bars = how long both played together; out offset = bars "
                             "after the outgoing's last drop; in lead = bars before the incoming's drop). "
                             "<i>Make template</i> fixes a row's numbers for its whole style."))
        self.ex = QTableWidget(0, 7)
        self.ex.setHorizontalHeaderLabels(["When", "Style", "Out → In", "Bars", "Out offset", "In lead", "Sections"])
        self.ex.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.ex.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.ex.itemChanged.connect(self._example_edited)
        lay.addWidget(self.ex, 3)
        row = QHBoxLayout()
        b = QPushButton("Make template for its style")
        b.setToolTip("Every transition of this row's style is mixed with exactly these numbers from now on")
        b.clicked.connect(self._template_from_example)
        row.addWidget(b)
        b = QPushButton("Delete row")
        b.clicked.connect(self._delete_example)
        row.addWidget(b)
        row.addStretch()
        lay.addLayout(row)

        lay.addWidget(QLabel("<b>Templates</b> - one per style; a recording template replays your moves, a "
                             "numbers template only the timing."))
        self.tpl = QTableWidget(0, 3)
        self.tpl.setHorizontalHeaderLabels(["Style", "Template", "Kind"])
        self.tpl.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.tpl.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tpl.setEditTriggers(QAbstractItemView.NoEditTriggers)
        lay.addWidget(self.tpl, 1)
        row = QHBoxLayout()
        b = QPushButton("Remove template")
        b.clicked.connect(self._remove_template)
        row.addWidget(b)
        row.addStretch()
        lay.addLayout(row)

        lay.addWidget(QLabel("<b>Mix Ahead recordings</b> - transitions you mixed on headphones ahead of time "
                             "(replayed exactly for that pair of tracks). <i>Use as template</i> replays one for "
                             "every transition of a style."))
        self.rec = QTableWidget(0, 5)
        self.rec.setHorizontalHeaderLabels(["When", "Out → In", "Bars", "On", "Template for"])
        self.rec.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.rec.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.rec.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.rec.itemChanged.connect(self._rec_toggled)
        lay.addWidget(self.rec, 2)
        row = QHBoxLayout()
        row.addWidget(QLabel("Use as template for"))
        self.group_box = QComboBox()
        for g in GROUPS:
            self.group_box.addItem(g if g not in GROUP_TIPS else f"{g}  ({GROUP_TIPS[g]})", g)
        row.addWidget(self.group_box)
        b = QPushButton("Use selected recording")
        b.clicked.connect(self._template_from_recording)
        row.addWidget(b)
        b = QPushButton("Delete recording")
        b.clicked.connect(self._delete_recording)
        row.addWidget(b)
        row.addStretch()
        b = QPushButton("Export…")
        b.setToolTip("Save everything here (learned mixes, templates, recordings) to a JSON file")
        b.clicked.connect(self.export)
        row.addWidget(b)
        b = QPushButton("Import…")
        b.setToolTip("Merge a file exported here (or on another machine)")
        b.clicked.connect(self.import_)
        row.addWidget(b)
        lay.addLayout(row)
        self.status = QLabel("")
        lay.addWidget(self.status)
        self.refresh()

    # ---- tables ---------------------------------------------------------------------
    def refresh(self):
        self.ex.blockSignals(True)
        self.ex.setRowCount(0)
        for i, x in enumerate(self.learner.examples):
            r = self.ex.rowCount()
            self.ex.insertRow(r)
            when = time.strftime("%Y-%m-%d %H:%M", time.localtime(x.ts)) if x.ts else ""
            self.ex.setItem(r, 0, _item(when, data=i))
            self.ex.setItem(r, 1, _item(x.style_group, editable=True))
            self.ex.setItem(r, 2, _item(f"{x.out_title} → {x.in_title}"))
            self.ex.setItem(r, 3, _item(f"{x.bars:g}", editable=True))
            self.ex.setItem(r, 4, _item(f"{x.out_offset:g}", editable=True))
            self.ex.setItem(r, 5, _item(f"{x.in_lead:g}", editable=True))
            self.ex.setItem(r, 6, _item(f"{x.out_section} → {x.in_section}"))
        self.ex.blockSignals(False)

        self.tpl.setRowCount(0)
        for g, t in sorted(self.learner.templates.items()):
            r = self.tpl.rowCount()
            self.tpl.insertRow(r)
            self.tpl.setItem(r, 0, _item(g, data=("learned", g)))
            self.tpl.setItem(r, 1, _item(describe({**t, "n": 0, "template": True})))
            self.tpl.setItem(r, 2, _item("numbers"))
        for g in sorted(self.store.templates()):
            rec = self.store.template(g)
            if rec is None:
                continue
            r = self.tpl.rowCount()
            self.tpl.insertRow(r)
            self.tpl.setItem(r, 0, _item(g, data=("recording", g)))
            self.tpl.setItem(r, 1, _item(f"{rec.out_title} → {rec.in_title}  ({rec.beats / 4:.0f} bars)"))
            self.tpl.setItem(r, 2, _item("recording (your moves)"))

        self.rec.blockSignals(True)
        self.rec.setRowCount(0)
        for rec in self.store.items():
            r = self.rec.rowCount()
            self.rec.insertRow(r)
            when = time.strftime("%Y-%m-%d %H:%M", time.localtime(rec.created))
            self.rec.setItem(r, 0, _item(when, data=(rec.out_path, rec.in_path)))
            self.rec.setItem(r, 1, _item(f"{rec.out_title or rec.out_path} → {rec.in_title or rec.in_path}"))
            self.rec.setItem(r, 2, _item(f"{rec.beats / 4:.0f}"))
            on = QTableWidgetItem("")
            on.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            on.setCheckState(Qt.Checked if rec.enabled else Qt.Unchecked)
            self.rec.setItem(r, 3, on)
            self.rec.setItem(r, 4, _item(self.store.template_group(rec) or "—"))
        self.rec.blockSignals(False)

    # ---- learned examples -------------------------------------------------------------
    def _example_edited(self, it: QTableWidgetItem):
        i = self.ex.item(it.row(), 0).data(Qt.UserRole)
        col = it.column()
        try:
            if col == 1:
                self.learner.update(i, style_group=it.text().strip() or "club")
            elif col in (3, 4, 5):
                v = float(it.text())
                if col == 3 and v <= 0:
                    raise ValueError("bars must be positive")
                self.learner.update(i, **{{3: "bars", 4: "out_offset", 5: "in_lead"}[col]: v})
            self.status.setText("Saved")
        except ValueError as e:
            self.status.setText(f"⚠ {e}")
        self.refresh()

    def _selected_example(self) -> int | None:
        rows = self.ex.selectionModel().selectedRows()
        return self.ex.item(rows[0].row(), 0).data(Qt.UserRole) if rows else None

    def _template_from_example(self):
        i = self._selected_example()
        if i is None:
            self.status.setText("Select a learned mix first")
            return
        x = self.learner.examples[i]
        self.learner.set_template(x.style_group, x.bars, x.out_offset, x.in_lead)
        self.store.set_template(x.style_group, None)       # a numbers template replaces a recording one
        self.status.setText(f"Every {x.style_group} transition is now mixed like this one")
        self.refresh()

    def _delete_example(self):
        i = self._selected_example()
        if i is None:
            return
        self.learner.remove(i)
        self.refresh()

    # ---- templates ------------------------------------------------------------------------
    def _remove_template(self):
        rows = self.tpl.selectionModel().selectedRows()
        if not rows:
            return
        kind, g = self.tpl.item(rows[0].row(), 0).data(Qt.UserRole)
        if kind == "learned":
            self.learner.clear_template(g)
        else:
            self.store.set_template(g, None)
        self.status.setText(f"{g}: back to what was learned")
        self.refresh()

    # ---- recordings ------------------------------------------------------------------------
    def _selected_recording(self):
        rows = self.rec.selectionModel().selectedRows()
        if not rows:
            return None
        out_path, in_path = self.rec.item(rows[0].row(), 0).data(Qt.UserRole)
        return self.store.find(out_path, in_path)

    def _rec_toggled(self, it: QTableWidgetItem):
        if it.column() != 3:
            return
        out_path, in_path = self.rec.item(it.row(), 0).data(Qt.UserRole)
        r = self.store.find(out_path, in_path)
        if r is not None:
            r.enabled = it.checkState() == Qt.Checked
            self.store.put(r)

    def _template_from_recording(self):
        r = self._selected_recording()
        if r is None:
            self.status.setText("Select a recording first")
            return
        g = self.group_box.currentData()
        self.store.set_template(g, r)
        self.learner.clear_template(g)                      # a recording template replaces a numbers one
        self.status.setText(f"Every {g} transition now replays {r.out_title} → {r.in_title}")
        self.refresh()

    def _delete_recording(self):
        r = self._selected_recording()
        if r is None:
            return
        if QMessageBox.question(self, "Delete recording", f"Delete the recorded mix {r.out_title} → {r.in_title}?"
                                ) != QMessageBox.Yes:
            return
        self.store.delete(r.out_path, r.in_path)
        self.refresh()

    # ---- files -------------------------------------------------------------------------------
    def export(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export learned mixes", "cta-learned-mixes.json",
                                              "JSON (*.json)")
        if not path:
            return
        data = {"version": 1, "learned": json.loads(self.learner.export_json()),
                "rehearsals": json.loads(self.store.export_json())}
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=1)
            self.status.setText(f"Exported to {path}")
        except OSError as e:
            self.status.setText(f"⚠ {e}")

    def import_(self, path: str | None = None):
        if not path:
            path, _ = QFileDialog.getOpenFileName(self, "Import learned mixes", "", "JSON (*.json)")
        if not path:
            return
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            n = self.learner.import_json(json.dumps(data.get("learned", data)))
            m = self.store.import_json(json.dumps(data.get("rehearsals", {}))) if "rehearsals" in data else 0
            self.status.setText(f"Imported {n} learned mix{'es' if n != 1 else ''} and {m} recording"
                                f"{'s' if m != 1 else ''}")
        except (OSError, ValueError, KeyError) as e:
            self.status.setText(f"⚠ {e}")
        self.refresh()


def ask_group(parent) -> str | None:
    g, ok = QInputDialog.getItem(parent, "Template style", "Use it for every transition of:", GROUPS, 0, False)
    return g if ok else None
