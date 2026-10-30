"""Application style sheet: flat graphite, compact controls - the look of
Traktor-style DJ software (dark panels, square-edged buttons, coloured
transport states, small all-caps labels)."""

QSS = """
QWidget { font-size: 12px; color: #d9dade; }
QMainWindow, QDialog { background: #141414; }
QSplitter::handle { background: #0b0b0b; }
QSplitter::handle:hover { background: #3a5a9a; }
QToolBar { background: #1b1b1c; border: 0; border-bottom: 1px solid #000; spacing: 2px; padding: 2px 4px; }
QToolBar QToolButton { background: transparent; color: #cfd0d4; padding: 4px 9px; border-radius: 2px; }
QToolBar QToolButton:hover { background: #2a2a2c; }
QToolBar QToolButton:checked { background: #2b3550; color: #9cc0ff; }

QPushButton {
    background: #2c2c2e; color: #dcdde1; border: 1px solid #0d0d0e; border-radius: 2px;
    padding: 3px 8px;
}
QPushButton:hover { background: #363639; }
QPushButton:pressed { background: #1e1e20; }
QPushButton:checked { background: #34507f; border-color: #0d0d0e; color: #ffffff; }
QPushButton:disabled { color: #5c5e66; }

QPushButton#mini { padding: 0px 2px; font-size: 11px; }
QPushButton#small { padding: 2px 4px; font-size: 11px; font-weight: 700; }
QPushButton#play { font-size: 17px; font-weight: 800; color: #86e39c; background: #262f28; }
QPushButton#play:checked { background: #2f8a4a; color: #ffffff; }
QPushButton#cue { font-size: 13px; font-weight: 800; color: #ffb547; background: #2f2a22; }
QPushButton#cue:pressed { background: #c98516; color: #111; }
QPushButton#sync { font-weight: 800; color: #9cc0ff; background: #232a36; }
QPushButton#sync:checked { background: #3d6bc4; color: #fff; }
QPushButton#toggle { font-weight: 700; color: #b0b1b8; font-size: 11px; }
QPushButton#toggle:checked { background: #c9701a; color: #111; }
QPushButton#pad { background: #232325; border: 1px solid #0d0d0e; font-weight: 700; color: #7f8189; }
QPushButton#pad:hover { background: #2e2e31; }
QPushButton#loop:checked { background: #2f8a4a; color: #fff; }

QFrame#deck { background: #1c1c1e; }
QGroupBox {
    background: #1c1c1e; border: 1px solid #0b0b0b; border-radius: 3px; margin-top: 12px; padding-top: 4px;
}
QGroupBox::title {
    subcontrol-origin: margin; left: 8px; padding: 0 4px; color: #8d8f97; font-weight: 800;
    font-size: 10px; text-transform: uppercase;
}

QSlider::groove:vertical { background: #070708; width: 5px; border-radius: 2px; }
QSlider::handle:vertical {
    background: #d6d8de; border: 1px solid #6d7079; height: 12px; width: 26px; margin: 0 -11px; border-radius: 2px;
}
QSlider::groove:horizontal { background: #070708; height: 5px; border-radius: 2px; }
QSlider::handle:horizontal {
    background: #d6d8de; border: 1px solid #6d7079; width: 14px; height: 22px; margin: -9px 0; border-radius: 2px;
}
QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {
    background: #232325; border: 1px solid #0d0d0e; border-radius: 2px; padding: 2px 5px; color: #dcdde1;
}
QComboBox:hover, QSpinBox:hover, QLineEdit:hover { border-color: #4a4d57; }
QLineEdit:focus { border-color: #4d6bb3; }
QCheckBox { spacing: 5px; }
QTableView {
    background: #161617; alternate-background-color: #1b1b1d; gridline-color: #232325;
    selection-background-color: #34507f; selection-color: #ffffff; border: 1px solid #0b0b0b;
}
QHeaderView::section {
    background: #202022; color: #8d8f97; border: 0; border-right: 1px solid #0b0b0b;
    padding: 3px 6px; font-weight: 800; font-size: 11px;
}
QTabWidget::pane { border: 1px solid #0b0b0b; top: -1px; }
QTabBar::tab {
    background: #1d1d1f; color: #8d8f97; padding: 5px 12px; margin-right: 1px; font-weight: 700;
}
QTabBar::tab:selected { background: #2c2c2e; color: #ffffff; }
QToolTip { background: #2c2c2e; color: #e4e5ea; border: 1px solid #0b0b0b; padding: 4px; }
QProgressBar { background: #1d1d1f; border: 1px solid #0b0b0b; border-radius: 2px; text-align: center; }
QProgressBar::chunk { background: #3d6bc4; }
QListWidget { background: #161617; border: 1px solid #0b0b0b; }
QScrollBar:vertical { background: #161617; width: 10px; }
QScrollBar::handle:vertical { background: #3a3a3d; min-height: 24px; border-radius: 3px; }
QScrollBar:horizontal { background: #161617; height: 10px; }
QScrollBar::handle:horizontal { background: #3a3a3d; min-width: 24px; border-radius: 3px; }
QScrollBar::add-line, QScrollBar::sub-line { width: 0; height: 0; }
"""
