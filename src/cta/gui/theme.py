"""Application themes: one style sheet, several colour sets. Every theme is
dark (club lighting); "Contrast" is for daylight and bright rooms. Switched
live from DJ Options -> App theme, remembered in the settings."""

from __future__ import annotations

GRAPHITE = dict(
    font="12px", text="#d9dade", text2="#cfd0d4", dim="#8d8f97", disabled="#5c5e66",
    bg="#141414", line="#0b0b0b", toolbar="#1b1b1c", hover="#2a2a2c", pressed="#1e1e20",
    button="#2c2c2e", btn_text="#dcdde1", btn_hover="#363639", border="#0d0d0e",
    field="#232325", field_hover="#4a4d57", panel="#1c1c1e", table="#161617", table_alt="#1b1b1d",
    grid="#232325", header="#202022", tab="#1d1d1f", scroll="#3a3a3d",
    groove="#070708", handle="#d6d8de", handle_edge="#6d7079",
    accent="#3d6bc4", accent_soft="#34507f", accent_dim="#2b3550", accent_text="#9cc0ff",
    focus="#4d6bb3", split_hover="#3a5a9a", sync_bg="#232a36",
    play_fg="#86e39c", play_bg="#262f28", play_on="#2f8a4a",
    cue_fg="#ffb547", cue_bg="#2f2a22", cue_on="#c98516", warm_on="#c9701a", pad_fg="#7f8189",
)

THEMES: dict[str, dict] = {
    "Graphite": GRAPHITE,
    "Midnight": {**GRAPHITE,
                 "text": "#c9d1d9", "text2": "#c2cad4", "dim": "#7d8896", "bg": "#0b1018", "line": "#05080d",
                 "toolbar": "#111823", "hover": "#1b2534", "pressed": "#0e141d", "button": "#18212e",
                 "btn_text": "#d3dbe5", "btn_hover": "#223042", "border": "#060a10", "field": "#141c27",
                 "field_hover": "#35506f", "panel": "#111925", "table": "#0d131c", "table_alt": "#111823",
                 "grid": "#1a2431", "header": "#131b27", "tab": "#111823", "scroll": "#2a3a50",
                 "accent": "#2f81f7", "accent_soft": "#1f4b8f", "accent_dim": "#16263f",
                 "accent_text": "#79c0ff", "focus": "#388bfd", "split_hover": "#2f81f7", "sync_bg": "#14223a"},
    "Neon": {**GRAPHITE,
             "text": "#e6e1ef", "text2": "#d9d2e6", "dim": "#8e84a3", "bg": "#08070c", "line": "#000000",
             "toolbar": "#100e17", "hover": "#1d1828", "pressed": "#0c0a12", "button": "#17131f",
             "btn_text": "#ece6f7", "btn_hover": "#241d31", "border": "#000000", "field": "#120f19",
             "field_hover": "#6b2a75", "panel": "#100d17", "table": "#0b0910", "table_alt": "#100d16",
             "grid": "#1a1524", "header": "#120f1a", "tab": "#100d17", "scroll": "#3b2c4c",
             "accent": "#ff2bd6", "accent_soft": "#7a1a6c", "accent_dim": "#2c0f2a", "accent_text": "#ff8af0",
             "focus": "#ff4fe0", "split_hover": "#ff2bd6", "sync_bg": "#26102a",
             "play_fg": "#3dffd0", "play_bg": "#0f2420", "play_on": "#00b894",
             "cue_fg": "#ffd23d", "cue_bg": "#2a2410", "cue_on": "#e0a800", "warm_on": "#ff2bd6"},
    "Ember": {**GRAPHITE,
              "text": "#e8ddd4", "text2": "#dccfc4", "dim": "#9a8a7e", "bg": "#120e0b", "line": "#070504",
              "toolbar": "#1a1410", "hover": "#2a201a", "pressed": "#150f0c", "button": "#261d17",
              "btn_text": "#efe3d8", "btn_hover": "#33271f", "border": "#080605", "field": "#1d1611",
              "field_hover": "#7a4a2a", "panel": "#1a1410", "table": "#140f0c", "table_alt": "#1a1410",
              "grid": "#271e18", "header": "#1c1611", "tab": "#1a1410", "scroll": "#4a3527",
              "accent": "#e8742a", "accent_soft": "#8a3f14", "accent_dim": "#3a2416", "accent_text": "#ffb37a",
              "focus": "#f08a45", "split_hover": "#e8742a", "sync_bg": "#30200f"},
    "Contrast": {**GRAPHITE,
                 "text": "#ffffff", "text2": "#ffffff", "dim": "#c8c8c8", "disabled": "#8a8a8a",
                 "bg": "#000000", "line": "#5a5a5a", "toolbar": "#0a0a0a", "hover": "#2a2a2a",
                 "pressed": "#000000", "button": "#1a1a1a", "btn_text": "#ffffff", "btn_hover": "#333333",
                 "border": "#8a8a8a", "field": "#111111", "field_hover": "#ffd400", "panel": "#0d0d0d",
                 "table": "#000000", "table_alt": "#141414", "grid": "#3a3a3a", "header": "#1a1a1a",
                 "tab": "#141414", "scroll": "#9a9a9a", "groove": "#3a3a3a", "handle": "#ffffff",
                 "handle_edge": "#ffd400", "accent": "#ffd400", "accent_soft": "#806a00", "accent_dim": "#3d3300",
                 "accent_text": "#ffe866", "focus": "#ffd400", "split_hover": "#ffd400", "sync_bg": "#2a2600",
                 "play_fg": "#5dff7a", "play_on": "#1fa83a", "cue_fg": "#ffb000", "pad_fg": "#d0d0d0"},
}

DEFAULT = "Graphite"

_QSS = """
QWidget { font-size: {font}; color: {text}; }
QMainWindow, QDialog { background: {bg}; }
QSplitter::handle { background: {line}; }
QSplitter::handle:hover { background: {split_hover}; }
QToolBar { background: {toolbar}; border: 0; border-bottom: 1px solid #000; spacing: 2px; padding: 2px 4px; }
QToolBar QToolButton { background: transparent; color: {text2}; padding: 4px 9px; border-radius: 2px; }
QToolBar QToolButton:hover { background: {hover}; }
QToolBar QToolButton:checked { background: {accent_dim}; color: {accent_text}; }

QPushButton {
    background: {button}; color: {btn_text}; border: 1px solid {border}; border-radius: 2px;
    padding: 3px 8px;
}
QPushButton:hover { background: {btn_hover}; }
QPushButton:pressed { background: {pressed}; }
QPushButton:checked { background: {accent_soft}; border-color: {border}; color: #ffffff; }
QPushButton:disabled { color: {disabled}; }

QPushButton#mini { padding: 0px 2px; font-size: 11px; }
QWidget#deckrow QPushButton { padding: 0px 2px; font-size: 11px; }
QWidget#deckrow QComboBox, QWidget#deckrow QSpinBox { padding: 0px 3px; font-size: 11px; }
QPushButton#small { padding: 2px 4px; font-size: 11px; font-weight: 700; }
QPushButton#play { font-size: 17px; font-weight: 800; color: {play_fg}; background: {play_bg}; }
QPushButton#play:checked { background: {play_on}; color: #ffffff; }
QPushButton#cue { font-size: 13px; font-weight: 800; color: {cue_fg}; background: {cue_bg}; }
QPushButton#cue:pressed { background: {cue_on}; color: #111; }
QPushButton#sync { font-weight: 800; color: {accent_text}; background: {sync_bg}; }
QPushButton#sync:checked { background: {accent}; color: #fff; }
QPushButton#toggle { font-weight: 700; color: {dim}; font-size: 11px; }
QPushButton#toggle:checked { background: {warm_on}; color: #111; }
QPushButton#pad { background: {field}; border: 1px solid {border}; font-weight: 700; color: {pad_fg}; }
QPushButton#pad:hover { background: {btn_hover}; }
QPushButton#loop:checked { background: {play_on}; color: #fff; }

QFrame#deck { background: {panel}; }
QGroupBox {
    background: {panel}; border: 1px solid {line}; border-radius: 3px; margin-top: 12px; padding-top: 4px;
}
QGroupBox::title {
    subcontrol-origin: margin; left: 8px; padding: 0 4px; color: {dim}; font-weight: 800;
    font-size: 10px; text-transform: uppercase;
}

QSlider::groove:vertical { background: {groove}; width: 5px; border-radius: 2px; }
QSlider::handle:vertical {
    background: {handle}; border: 1px solid {handle_edge}; height: 12px; width: 26px; margin: 0 -11px;
    border-radius: 2px;
}
QSlider::groove:horizontal { background: {groove}; height: 5px; border-radius: 2px; }
QSlider::handle:horizontal {
    background: {handle}; border: 1px solid {handle_edge}; width: 14px; height: 22px; margin: -9px 0;
    border-radius: 2px;
}
QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {
    background: {field}; border: 1px solid {border}; border-radius: 2px; padding: 2px 5px; color: {btn_text};
}
QComboBox:hover, QSpinBox:hover, QLineEdit:hover { border-color: {field_hover}; }
QLineEdit:focus { border-color: {focus}; }
QCheckBox { spacing: 5px; }
QTableView {
    background: {table}; alternate-background-color: {table_alt}; gridline-color: {grid};
    selection-background-color: {accent_soft}; selection-color: #ffffff; border: 1px solid {line};
}
QHeaderView::section {
    background: {header}; color: {dim}; border: 0; border-right: 1px solid {line};
    padding: 3px 6px; font-weight: 800; font-size: 11px;
}
QTabWidget::pane { border: 1px solid {line}; top: -1px; }
QTabBar::tab {
    background: {tab}; color: {dim}; padding: 5px 12px; margin-right: 1px; font-weight: 700;
}
QTabBar::tab:selected { background: {button}; color: #ffffff; }
QToolTip { background: {button}; color: {text}; border: 1px solid {line}; padding: 4px; }
QProgressBar { background: {tab}; border: 1px solid {line}; border-radius: 2px; text-align: center; }
QProgressBar::chunk { background: {accent}; }
QListWidget { background: {table}; border: 1px solid {line}; }
QScrollBar:vertical { background: {table}; width: 10px; }
QScrollBar::handle:vertical { background: {scroll}; min-height: 24px; border-radius: 3px; }
QScrollBar:horizontal { background: {table}; height: 10px; }
QScrollBar::handle:horizontal { background: {scroll}; min-width: 24px; border-radius: 3px; }
QScrollBar::add-line, QScrollBar::sub-line { width: 0; height: 0; }
"""


def qss(name: str = DEFAULT) -> str:
    t = THEMES.get(name, GRAPHITE)
    out = _QSS
    for k, v in t.items():
        out = out.replace("{" + k + "}", v)
    return out


QSS = qss(DEFAULT)          # (kept for anything that imports the plain style sheet)


def palette(name: str = DEFAULT):
    """The Qt palette that matches a theme (dialogs and native bits use it)."""
    from PySide6.QtGui import QColor, QPalette

    t = THEMES.get(name, GRAPHITE)
    p = QPalette()
    text = QColor(t["text"])
    p.setColor(QPalette.Window, QColor(t["panel"]))
    p.setColor(QPalette.WindowText, text)
    p.setColor(QPalette.Base, QColor(t["table"]))
    p.setColor(QPalette.AlternateBase, QColor(t["table_alt"]))
    p.setColor(QPalette.ToolTipBase, QColor(t["button"]))
    p.setColor(QPalette.ToolTipText, text)
    p.setColor(QPalette.Text, text)
    p.setColor(QPalette.Button, QColor(t["button"]))
    p.setColor(QPalette.ButtonText, QColor(t["btn_text"]))
    p.setColor(QPalette.BrightText, QColor(255, 80, 80))
    p.setColor(QPalette.Highlight, QColor(t["accent"]))
    p.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
    p.setColor(QPalette.PlaceholderText, QColor(t["dim"]))
    for role in (QPalette.Text, QPalette.ButtonText, QPalette.WindowText):
        p.setColor(QPalette.Disabled, role, QColor(t["disabled"]))
    return p


def apply(app, name: str) -> str:
    """Switch the whole app to a theme (live). Returns the name actually used."""
    name = name if name in THEMES else DEFAULT
    app.setPalette(palette(name))
    app.setStyleSheet(qss(name))
    return name
