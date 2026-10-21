"""Application style sheet: a modern dark DJ-software look."""

QSS = """
QWidget { font-size: 12px; }
QMainWindow, QDialog { background: #121316; }
QToolBar { background: #17181c; border: 0; border-bottom: 1px solid #26272d; spacing: 4px; padding: 4px; }
QToolBar QToolButton { background: transparent; color: #d8d9df; padding: 5px 10px; border-radius: 6px; }
QToolBar QToolButton:hover { background: #26282f; }
QToolBar QToolButton:checked { background: #2b3550; color: #8fb4ff; }

QPushButton {
    background: #24262c; color: #e4e5ea; border: 1px solid #33353d; border-radius: 6px;
    padding: 4px 10px;
}
QPushButton:hover { background: #2d3038; border-color: #454852; }
QPushButton:pressed { background: #1c1d22; }
QPushButton:checked { background: #2b3550; border-color: #4d6bb3; color: #ffffff; }
QPushButton:disabled { color: #5c5e66; }

QPushButton#mini { padding: 0px 2px; font-size: 11px; }
QPushButton#play { font-size: 18px; font-weight: 800; color: #7dff9b; border: 2px solid #2f7a45; }
QPushButton#play:checked { background: #1e5a33; color: #ffffff; border-color: #48d17a; }
QPushButton#cue { font-size: 14px; font-weight: 800; color: #ffc04d; border: 2px solid #8a6420; }
QPushButton#cue:pressed { background: #8a6420; color: #fff; }
QPushButton#sync { font-weight: 800; color: #8fb4ff; border: 2px solid #34508f; }
QPushButton#sync:checked { background: #34508f; color: #fff; border-color: #6d95ff; }
QPushButton#toggle { font-weight: 700; color: #b9bac2; }
QPushButton#toggle:checked { background: #3a2f55; border-color: #8a6cff; color: #fff; }
QPushButton#pad { background: #1f2026; border: 1px solid #2f3138; font-weight: 700; color: #8d8f98; }
QPushButton#pad:hover { border-color: #5a5d68; }
QPushButton#loop:checked { background: #1f5a35; border-color: #45c46f; color: #fff; }

QGroupBox {
    background: #17181c; border: 1px solid #2a2c33; border-radius: 8px; margin-top: 10px; padding-top: 6px;
}
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; color: #9a9ba3; font-weight: 700; }

QSlider::groove:vertical { background: #0e0f12; width: 6px; border-radius: 3px; }
QSlider::handle:vertical {
    background: #dfe1e8; border: 1px solid #8a8d96; height: 14px; width: 30px; margin: 0 -12px; border-radius: 3px;
}
QSlider::groove:horizontal { background: #0e0f12; height: 6px; border-radius: 3px; }
QSlider::handle:horizontal {
    background: #dfe1e8; border: 1px solid #8a8d96; width: 16px; height: 26px; margin: -10px 0; border-radius: 3px;
}
QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {
    background: #1d1e23; border: 1px solid #33353d; border-radius: 5px; padding: 3px 6px; color: #e4e5ea;
}
QComboBox:hover, QSpinBox:hover, QLineEdit:hover { border-color: #4a4d57; }
QLineEdit:focus { border-color: #4d6bb3; }
QTableView {
    background: #141518; alternate-background-color: #18191d; gridline-color: #22232a;
    selection-background-color: #2b3f6b; selection-color: #ffffff; border: 1px solid #24262c;
}
QHeaderView::section {
    background: #1b1c21; color: #9a9ba3; border: 0; border-right: 1px solid #26272d;
    padding: 4px 6px; font-weight: 700;
}
QTabWidget::pane { border: 1px solid #26272d; border-radius: 6px; top: -1px; }
QTabBar::tab {
    background: #1b1c21; color: #9a9ba3; padding: 6px 14px; border-top-left-radius: 6px;
    border-top-right-radius: 6px; margin-right: 2px;
}
QTabBar::tab:selected { background: #24262c; color: #ffffff; }
QDockWidget::title { background: #17181c; padding: 4px 8px; color: #9a9ba3; }
QToolTip { background: #24262c; color: #e4e5ea; border: 1px solid #3a3c44; padding: 4px; }
QProgressBar { background: #1d1e23; border: 1px solid #33353d; border-radius: 4px; text-align: center; }
QProgressBar::chunk { background: #4d6bb3; border-radius: 3px; }
QListWidget { background: #141518; border: 1px solid #24262c; border-radius: 6px; }
"""
