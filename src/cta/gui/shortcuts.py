"""Keyboard control of the decks, laid out like Traktor Pro's default
keyboard mapping: deck A on the top letter row, deck B on the home row.

            Deck A   Deck B
  Cue        Q        A        (hold to preview, like the button)
  Play       W        S
  Beatjump   E        D        (Shift: back)
  Bend       T        G        (hold; Shift: slow down)
  Sync       U        J        (Shift: tempo master)
  Tap        I        K        (Shift: snap the phase to the master)
  Loop in    O        L        (Shift: halve loop size)
  Loop out   P        ;        (Shift: double loop size)
  Loop       [        '        (loop the chosen size on / off)
  Key lock   Shift+Y  Shift+H
  Hot cues   1-5      6-0      (Ctrl: delete; Shift: loop 1/2/4/8/16 beats)
  Load       Shift+←  Shift+→  (the selected library track)

Keys typed into a text box, spin box or open dialog are left alone.
"""

from __future__ import annotations

from PySide6.QtCore import QEvent, QObject, Qt
from PySide6.QtWidgets import QAbstractSpinBox, QApplication, QComboBox, QLineEdit, QPlainTextEdit, QTextEdit

HELP = __doc__.split("\n\n", 1)[1].rsplit("\n\n", 1)[0]

# Shift turns the digit row and some punctuation into other key codes
_SHIFTED = {Qt.Key_Exclam: Qt.Key_1, Qt.Key_At: Qt.Key_2, Qt.Key_NumberSign: Qt.Key_3, Qt.Key_Dollar: Qt.Key_4,
            Qt.Key_Percent: Qt.Key_5, Qt.Key_AsciiCircum: Qt.Key_6, Qt.Key_Ampersand: Qt.Key_7,
            Qt.Key_Asterisk: Qt.Key_8, Qt.Key_ParenLeft: Qt.Key_9, Qt.Key_ParenRight: Qt.Key_0,
            Qt.Key_Colon: Qt.Key_Semicolon, Qt.Key_QuoteDbl: Qt.Key_Apostrophe, Qt.Key_BraceLeft: Qt.Key_BracketLeft}

ROWS = {  # action: (deck A key, deck B key)
    "cue": (Qt.Key_Q, Qt.Key_A), "play": (Qt.Key_W, Qt.Key_S), "jump": (Qt.Key_E, Qt.Key_D),
    "bend": (Qt.Key_T, Qt.Key_G), "sync": (Qt.Key_U, Qt.Key_J), "tap": (Qt.Key_I, Qt.Key_K),
    "loop_in": (Qt.Key_O, Qt.Key_L), "loop_out": (Qt.Key_P, Qt.Key_Semicolon),
    "loop": (Qt.Key_BracketLeft, Qt.Key_Apostrophe), "keylock": (Qt.Key_Y, Qt.Key_H),
    "load": (Qt.Key_Left, Qt.Key_Right),
}
KEYMAP = {k: (act, "AB"[side]) for act, keys in ROWS.items() for side, k in enumerate(keys)}
DIGITS = [Qt.Key_1, Qt.Key_2, Qt.Key_3, Qt.Key_4, Qt.Key_5, Qt.Key_6, Qt.Key_7, Qt.Key_8, Qt.Key_9, Qt.Key_0]
LOOP_SIZES = (1, 2, 4, 8, 16)


class DeckKeys(QObject):
    """Application-wide key filter driving a DJPanel."""

    def __init__(self, dj):
        super().__init__(dj)
        self.dj = dj
        self.enabled = True
        self._cue_held: set[str] = set()

    def _typing(self) -> bool:
        app = QApplication.instance()
        if app.activeModalWidget() is not None or app.activePopupWidget() is not None:
            return True
        w = app.focusWidget()
        if isinstance(w, (QLineEdit, QAbstractSpinBox, QTextEdit, QPlainTextEdit)):
            return True
        return isinstance(w, QComboBox) and w.isEditable()

    def eventFilter(self, obj, ev):
        t = ev.type()
        if t not in (QEvent.KeyPress, QEvent.KeyRelease, QEvent.ShortcutOverride):
            return False
        if not self.enabled or not self.dj.isVisible() or self._typing():
            return False
        if self.dj.window() is not QApplication.activeWindow():
            return False
        key = _SHIFTED.get(ev.key(), ev.key())
        mods = ev.modifiers() & (Qt.ShiftModifier | Qt.ControlModifier | Qt.AltModifier | Qt.MetaModifier)
        shift = bool(mods & Qt.ShiftModifier)
        ctrl = bool(mods & (Qt.ControlModifier | Qt.MetaModifier))
        if mods & Qt.AltModifier:
            return False
        if t == QEvent.ShortcutOverride:
            # claim our keys before any widget shortcut (table type-ahead etc.)
            if self._action(key, shift, ctrl) is not None:
                ev.accept()
                return True
            return False
        act = self._action(key, shift, ctrl)
        if act is None:
            return False
        if t == QEvent.KeyRelease:
            if not ev.isAutoRepeat() and act[0] == "cue" and act[1] in self._cue_held:
                self._cue_held.discard(act[1])
                self.dj.engine.cue_release(act[1])
            return True
        if ev.isAutoRepeat() and act[0] != "bend":
            return True
        self.run(*act, shift=shift, ctrl=ctrl)
        return True

    def _action(self, key, shift: bool, ctrl: bool):
        if key in DIGITS:
            n = DIGITS.index(key)
            return ("hotcue", "AB"[n // 5], n % 5)
        hit = KEYMAP.get(key)
        if hit is None or ctrl:
            return None
        act, deck = hit
        if act in ("keylock", "load") and not shift:
            return None
        return (act, deck, None)

    def run(self, act: str, deck: str, arg=None, shift: bool = False, ctrl: bool = False):
        dj = self.dj
        eng = dj.engine
        panel = dj.decks[deck]
        if act != "load":
            dj.manual_touch(deck, act)          # before it happens: see AutoMix.manual
        if act == "hotcue":
            if ctrl:
                panel._hotcue_delete(arg)
            elif shift:
                panel._loop_size(LOOP_SIZES[arg])
            else:
                panel._hotcue(arg)
        elif act == "cue":
            self._cue_held.add(deck)
            eng.cue_press(deck)
        elif act == "play":
            eng.toggle_play(deck)
        elif act == "jump":
            n = int(panel.jump.currentText())
            eng.beatjump(deck, -n if shift else n)
        elif act == "bend":
            eng.jog(deck, -6.0 if shift else 6.0, False)
        elif act == "sync":
            if shift:
                eng.set_master(deck)
            else:
                eng.set(deck, sync=not eng.decks[deck].sync)
        elif act == "tap":
            if shift:
                eng.phase_snap(deck)
            else:
                panel._tap()
        elif act == "loop_in":
            panel._loop_step(0.5) if shift else eng.loop_in(deck)
        elif act == "loop_out":
            panel._loop_step(2.0) if shift else eng.loop_out(deck)
        elif act == "loop":
            panel._toggle_loop(eng.decks[deck].loop is None)
        elif act == "keylock":
            eng.set(deck, keylock=not eng.decks[deck].keylock)
        elif act == "load":
            dj.load_selected(deck)
