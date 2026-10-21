"""MIDI controller support (Pioneer DDJ-T1 default map + MIDI learn).

A mapping is a list of bindings ``{type, ch, num, action, deck, ...}``:

* type: "note" (buttons), "cc" (7-bit), "cc14" (14-bit MSB on num, LSB on num+32)
* relative: True for endless encoders / jog wheels
* invert: flip an absolute control (e.g. tempo fader direction)

Mappings are saved as JSON in the user data folder, so the guided "learn"
setup only has to be done once.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import asdict, dataclass

from .. import paths

log = logging.getLogger(__name__)

DECK_ACTIONS = {
    # name: (kind, description)
    "play": ("button", "PLAY/PAUSE button"),
    "cue": ("button", "CUE button"),
    "sync": ("button", "SYNC button"),
    "keylock": ("button", "KEY LOCK button"),
    "tempo": ("fader", "TEMPO slider"),
    "jog": ("relative", "JOG wheel (turn the outer ring)"),
    "jog_touch": ("button", "JOG wheel top (touch it)"),
    "volume": ("fader", "channel VOLUME fader"),
    "eq_high": ("fader", "EQ HI knob"),
    "eq_mid": ("fader", "EQ MID knob"),
    "eq_low": ("fader", "EQ LOW knob"),
    "gain": ("fader", "TRIM / GAIN knob"),
    "phones_cue": ("button", "headphone CUE button"),
    "load": ("button", "LOAD button"),
    "loop": ("button", "AUTO LOOP (press)"),
    "filter": ("fader", "FILTER knob"),
    "fx1_assign": ("button", "FX 1 assign button"),
    "fx2_assign": ("button", "FX 2 assign button"),
    "loop_in": ("button", "LOOP IN button"),
    "loop_out": ("button", "LOOP OUT button"),
    "loop_halve": ("button", "loop size ½ (or turn the loop knob left)"),
    "loop_double": ("button", "loop size ×2 (or turn the loop knob right)"),
    "beatjump_back": ("button", "BEAT JUMP back / BWD"),
    "beatjump_fwd": ("button", "BEAT JUMP forward / FWD"),
    "key_sync": ("button", "a button for KEY SYNC"),
    **{f"hotcue{i}": ("button", f"HOT CUE {i}") for i in range(1, 9)},
}
ADVANCED = {"filter", "fx1_assign", "fx2_assign", "loop_in", "loop_out", "loop_halve", "loop_double",
            "beatjump_back", "beatjump_fwd", "key_sync"} | {f"hotcue{i}" for i in range(1, 9)}
GLOBAL_ACTIONS = {
    "crossfader": ("fader", "CROSSFADER"),
    "master_volume": ("fader", "MASTER volume"),
    "cue_mix": ("fader", "HEADPHONES MIX knob"),
    "phones_volume": ("fader", "HEADPHONES volume"),
    "browse": ("relative", "BROWSE rotary selector (turn)"),
    "automix": ("button", "a button to toggle AUTO-MIX (e.g. FAVORITES)"),
    "transition": ("button", "a button for AUTO TRANSITION (e.g. browse press)"),
    "fx1_on": ("button", "FX 1 on/off"),
    "fx1_wet": ("fader", "FX 1 DRY/WET knob"),
    "fx1_amount": ("fader", "FX 1 parameter knob"),
    "fx2_on": ("button", "FX 2 on/off"),
    "fx2_wet": ("fader", "FX 2 DRY/WET knob"),
    "fx2_amount": ("fader", "FX 2 parameter knob"),
    "rec": ("button", "a button to start/stop RECORDING"),
}
ADVANCED |= {"fx1_on", "fx1_wet", "fx1_amount", "fx2_on", "fx2_wet", "fx2_amount", "rec"}
JOG_COUNTS_PER_REV = {"DDJ-T1": 2430.0}
REV_SECONDS = 1.8          # one platter revolution = 1.8 s of audio (33 1/3 rpm)


@dataclass
class Binding:
    type: str
    ch: int
    num: int
    action: str
    deck: str | None = None
    relative: bool = False
    invert: bool = False


def ddj_t1_default() -> list[Binding]:
    """Bindings confirmed from Pioneer's DDJ-T1 MIDI message list. Controls
    not listed there are added with the guided setup (MIDI learn)."""
    out = []
    for ch, deck in enumerate("ABCD"):
        out += [
            Binding("note", ch, 11, "play", deck),
            Binding("note", ch, 12, "cue", deck),
            Binding("cc14", ch, 1, "jog", deck, relative=True),
            Binding("cc14", ch, 0, "tempo", deck),
        ]
    out += [
        Binding("cc", 6, 64, "browse", relative=True),
        Binding("note", 6, 65, "transition"),
        Binding("note", 6, 67, "automix"),
        Binding("note", 6, 70, "load", "A"),
        Binding("note", 6, 71, "load", "B"),
        Binding("note", 6, 72, "load", "C"),
        Binding("note", 6, 73, "load", "D"),
    ]
    return out


def mapping_path(device: str) -> str:
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in device)[:60] or "controller"
    d = os.path.join(paths.data_dir(), "controllers")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{safe}.json")


def load_mapping(device: str) -> tuple[list[Binding], bool]:
    """(bindings, learned) - falls back to the built-in default for known devices."""
    p = mapping_path(device)
    if os.path.exists(p):
        try:
            data = json.load(open(p, encoding="utf-8"))
            return [Binding(**b) for b in data["bindings"]], bool(data.get("learned"))
        except Exception:  # noqa: BLE001
            log.exception("bad mapping file %s", p)
    if "DDJ-T1" in device.upper():
        return ddj_t1_default(), False
    return [], False


def save_mapping(device: str, bindings: list[Binding], learned: bool = True):
    with open(mapping_path(device), "w", encoding="utf-8") as f:
        json.dump({"device": device, "learned": learned, "bindings": [asdict(b) for b in bindings]}, f, indent=1)


# --------------------------------------------------------------------------
# message decoding
# --------------------------------------------------------------------------

@dataclass
class Event:
    type: str          # note / cc / cc14
    ch: int
    num: int
    value: int         # note: 127 press / 0 release; cc: 0..127; cc14: 0..16383
    t: float


class Decoder:
    """Turns raw MIDI bytes into note/cc/cc14 events."""

    def __init__(self, cc14: set[tuple[int, int]] | None = None):
        self.cc14 = set(cc14 or ())      # (ch, msb_num) pairs that are 14-bit
        self._msb: dict[tuple[int, int], int] = {}

    def feed(self, msg: list[int], t: float | None = None) -> Event | None:
        if not msg:
            return None
        t = time.perf_counter() if t is None else t
        st = msg[0] & 0xF0
        ch = msg[0] & 0x0F
        if st in (0x90, 0x80) and len(msg) >= 3:
            v = msg[2] if st == 0x90 else 0
            return Event("note", ch, msg[1], 127 if v > 0 else 0, t)
        if st == 0xB0 and len(msg) >= 3:
            num, v = msg[1], msg[2]
            if (ch, num) in self.cc14:
                self._msb[(ch, num)] = v
                return None                       # wait for the LSB
            if num >= 32 and (ch, num - 32) in self.cc14:
                msb = self._msb.get((ch, num - 32), 0)
                return Event("cc14", ch, num - 32, (msb << 7) | v, t)
            return Event("cc", ch, num, v, t)
        return None


def rel7(v: int) -> int:
    return v if v < 64 else v - 128


def rel14(v: int) -> int:
    return v if v < 8192 else v - 16384


def eq_curve(x: float) -> float:
    """Knob 0..1 -> gain: full cut at 0, unity at the centre, +6 dB at max."""
    if x <= 0.5:
        return (x / 0.5) ** 2 if x > 0.01 else 0.0
    return 1.0 + (x - 0.5) * 2.0


# --------------------------------------------------------------------------
# controller
# --------------------------------------------------------------------------

class Controller:
    """Routes MIDI events to the engine / auto-mix / UI."""

    def __init__(self, engine, automix=None, ui=None, device: str = "DDJ-T1"):
        self.eng = engine
        self.automix = automix
        self.ui = ui                       # object with browse(delta) and load(deck)
        self.device = device
        self.bindings, self.learned = load_mapping(device)
        self.counts_per_rev = JOG_COUNTS_PER_REV.get(device.split()[0] if device else "", 2430.0)
        for k, v in JOG_COUNTS_PER_REV.items():
            if k in device.upper():
                self.counts_per_rev = v
        self._index()
        self.decoder = Decoder({(b.ch, b.num) for b in self.bindings if b.type == "cc14"})
        self._touch: dict[str, bool] = {}
        self._jog_last: dict[str, float] = {}
        self.learn_cb = None               # set by the learn wizard: receives every Event
        self.monitor_cb = None
        self._in = None
        self._out = None
        self._eq: dict[str, list[float]] = {d: [1.0, 1.0, 1.0] for d in "ABCD"}
        self._led_state: dict[tuple[int, int], int] = {}
        self._lock = threading.Lock()
        self.jump_beats = 4

    def _index(self):
        self._map = {(b.type, b.ch, b.num): b for b in self.bindings}

    def set_bindings(self, bindings: list[Binding], learned: bool = True):
        self.bindings = bindings
        self.learned = learned
        self._index()
        self.decoder.cc14 = {(b.ch, b.num) for b in bindings if b.type == "cc14"}
        save_mapping(self.device, bindings, learned)

    # ---- ports --------------------------------------------------------------------
    @staticmethod
    def list_inputs() -> list[str]:
        try:
            import rtmidi

            return rtmidi.MidiIn().get_ports()
        except Exception as e:  # noqa: BLE001
            log.warning("MIDI unavailable: %s", e)
            return []

    def open(self, port_name: str) -> bool:
        import rtmidi

        self.close()
        mi = rtmidi.MidiIn()
        ports = mi.get_ports()
        if port_name not in ports:
            return False
        mi.open_port(ports.index(port_name))
        mi.ignore_types(sysex=True, timing=True, active_sense=True)
        mi.set_callback(self._on_midi)
        self._in = mi
        try:
            mo = rtmidi.MidiOut()
            outs = mo.get_ports()
            base = port_name.split(":")[0].split(" MIDI")[0]
            match = [i for i, p in enumerate(outs) if base and base in p]
            if match:
                mo.open_port(match[0])
                self._out = mo
        except Exception:  # noqa: BLE001
            self._out = None
        log.info("MIDI connected: %s", port_name)
        return True

    def close(self):
        for p in (self._in, self._out):
            try:
                if p is not None:
                    p.close_port()
            except Exception:  # noqa: BLE001
                pass
        self._in = self._out = None

    @property
    def connected(self) -> bool:
        return self._in is not None

    def _on_midi(self, event, _data=None):
        msg, _dt = event
        try:
            self.handle_raw(list(msg))
        except Exception:  # noqa: BLE001
            log.exception("MIDI handling failed for %s", msg)

    # ---- dispatch -------------------------------------------------------------------
    def handle_raw(self, msg: list[int]):
        ev = self.decoder.feed(msg)
        if ev is None:
            return
        if self.monitor_cb:
            self.monitor_cb(ev)
        if self.learn_cb:
            self.learn_cb(ev)
            return
        b = self._map.get((ev.type, ev.ch, ev.num))
        if b is None and ev.type == "cc14":
            b = self._map.get(("cc", ev.ch, ev.num))
        if b is not None:
            self.dispatch(b, ev)

    def dispatch(self, b: Binding, ev: Event):
        e = self.eng
        a = b.action
        deck = b.deck
        full = 16383.0 if ev.type == "cc14" else 127.0
        x = ev.value / full
        if b.invert:
            x = 1.0 - x
        pressed = ev.value > 0
        am = self.automix
        manual = a in ("volume", "eq_low", "eq_mid", "eq_high", "crossfader")
        if manual and am is not None and am.tr is not None and am.tr.started:
            if deck is None or deck in (am.tr.in_deck, am.tr.out_deck):
                am.user_took_over()

        if a == "play" and pressed:
            e.toggle_play(deck)
        elif a == "cue":
            e.cue_press(deck) if pressed else e.cue_release(deck)
        elif a == "sync" and pressed:
            e.set(deck, sync=not e.decks[deck].sync)
        elif a == "keylock" and pressed:
            e.set(deck, keylock=not e.decks[deck].keylock)
        elif a == "tempo":
            e.set(deck, tempo=float(x * 2.0 - 1.0))
        elif a == "volume":
            e.set(deck, volume=float(x))
        elif a in ("eq_low", "eq_mid", "eq_high"):
            i = {"eq_low": 0, "eq_mid": 1, "eq_high": 2}[a]
            self._eq[deck][i] = eq_curve(x)
            e.set(deck, eq=tuple(self._eq[deck]))
        elif a == "gain":
            e.set(deck, trim=float(10 ** ((x - 0.5) * 24 / 20)))     # +-12 dB
        elif a == "phones_cue" and pressed:
            e.set(deck, cue_phones=not e.decks[deck].cue_phones)
        elif a == "jog_touch":
            self._touch[deck] = pressed
            if pressed:
                e.scratch(deck, 0.0, True)
            else:
                e.jog_release(deck)
        elif a == "jog":
            delta = rel14(ev.value) if ev.type == "cc14" else rel7(ev.value)
            revs = delta / self.counts_per_rev
            now = ev.t
            dt = max(1e-3, min(0.05, now - self._jog_last.get(deck, now - 0.01)))
            self._jog_last[deck] = now
            self.jog(deck, revs, dt)
        elif a == "loop" and pressed:
            e.loop_beats(deck, None if e.decks[deck].loop else 4)
        elif a == "load" and pressed and self.ui is not None:
            self.ui.load(deck)
        elif a == "crossfader":
            e.crossfader = float(x * 2.0 - 1.0)
        elif a == "master_volume":
            e.master_volume = float(x)
        elif a == "cue_mix":
            e.cue_mix = float(x)
        elif a == "phones_volume":
            e.phones_volume = float(x)
        elif a == "browse" and self.ui is not None:
            d = rel14(ev.value) if ev.type == "cc14" else rel7(ev.value)
            self.ui.browse(d)
        elif a == "filter":
            e.set(deck, filter=float(x * 2.0 - 1.0))
        elif a in ("fx1_assign", "fx2_assign") and pressed:
            u = 0 if a == "fx1_assign" else 1
            cur = set(e.decks[deck].fx)
            cur.symmetric_difference_update({u})
            e.set(deck, fx=cur)
        elif a == "loop_in" and pressed:
            e.loop_in(deck)
        elif a == "loop_out" and pressed:
            e.loop_out(deck)
        elif a == "loop_halve" and pressed:
            e.loop_resize(deck, 0.5)
        elif a == "loop_double" and pressed:
            e.loop_resize(deck, 2.0)
        elif a == "beatjump_back" and pressed:
            e.beatjump(deck, -self.jump_beats)
        elif a == "beatjump_fwd" and pressed:
            e.beatjump(deck, self.jump_beats)
        elif a == "key_sync" and pressed:
            e.key_sync(deck)
        elif a.startswith("hotcue") and pressed:
            e.hotcue(deck, int(a[6:]) - 1)
            if self.ui is not None and hasattr(self.ui, "cues_changed"):
                self.ui.cues_changed(deck)
        elif a in ("fx1_on", "fx2_on") and pressed:
            u = e.fx_units[0 if a == "fx1_on" else 1]
            u.on = not u.on
        elif a in ("fx1_wet", "fx2_wet"):
            e.fx_units[0 if a == "fx1_wet" else 1].wet = float(x)
        elif a in ("fx1_amount", "fx2_amount"):
            e.fx_units[0 if a == "fx1_amount" else 1].amount = float(x)
        elif a == "rec" and pressed and self.ui is not None and hasattr(self.ui, "toggle_record"):
            self.ui.toggle_record()
        elif a == "automix" and pressed and am is not None:
            am.stop() if am.active else am.start()
        elif a == "transition" and pressed and am is not None:
            am.transition_now()

    def jog(self, deck: str, revs: float, dt: float):
        e = self.eng
        d = e.decks[deck]
        if self._touch.get(deck) and d.playing:
            # platter held: the record follows your hand (scratch)
            e.scratch(deck, (revs / dt) * REV_SECONDS, True)
        elif d.playing:
            # outer ring: temporary pitch bend to nudge the phase
            e.set(deck, nudge=float(max(-0.15, min(0.15, (revs / dt) * 0.25))))
        else:
            e.seek(deck, d.time + revs * REV_SECONDS)

    def tick(self):
        """Called ~60x/s from the UI: stop scratch motion when the hand stops."""
        now = time.perf_counter()
        for deck, held in self._touch.items():
            if held and now - self._jog_last.get(deck, 0) > 0.03:
                self.eng.scratch(deck, 0.0, True)

    # ---- LEDs -------------------------------------------------------------------------
    def update_leds(self):
        if self._out is None:
            return
        for b in self.bindings:
            if b.type != "note" or b.deck is None:
                continue
            d = self.eng.decks.get(b.deck)
            if d is None:
                continue
            on = {"play": d.playing, "sync": d.sync, "keylock": d.keylock, "phones_cue": d.cue_phones,
                  "cue": d.loaded and not d.playing, "loop": d.loop is not None,
                  "fx1_assign": 0 in d.fx, "fx2_assign": 1 in d.fx}.get(b.action)
            if b.action.startswith("hotcue"):
                on = d.hotcues[int(b.action[6:]) - 1] is not None
            if on is None:
                continue
            v = 127 if on else 0
            key = (b.ch, b.num)
            if self._led_state.get(key) != v:
                self._led_state[key] = v
                try:
                    self._out.send_message([0x90 | b.ch, b.num, v])
                except Exception:  # noqa: BLE001
                    pass


# --------------------------------------------------------------------------
# learn wizard logic (UI independent)
# --------------------------------------------------------------------------

def learn_steps(decks=("A", "B"), advanced: bool = True) -> list[tuple[str, str | None, str]]:
    steps = []
    for deck in decks:
        for a, (_kind, desc) in DECK_ACTIONS.items():
            if advanced or a not in ADVANCED:
                steps.append((a, deck, f"Deck {deck}: {desc}"))
    for a, (_kind, desc) in GLOBAL_ACTIONS.items():
        if advanced or a not in ADVANCED:
            steps.append((a, None, desc))
    return steps


class Learner:
    """Collects one binding per step from incoming events."""

    def __init__(self):
        self.pending: list[Event] = []
        self.first_value: int | None = None

    def offer(self, action: str, ev: Event) -> Binding | None:
        kind = (DECK_ACTIONS.get(action) or GLOBAL_ACTIONS.get(action))[0]
        if kind == "button":
            if ev.type == "note" and ev.value > 0:
                return Binding("note", ev.ch, ev.num, action)
            if ev.type == "cc" and ev.value > 0:
                return Binding("cc", ev.ch, ev.num, action)
            return None
        if ev.type == "note":
            return None
        self.pending.append(ev)
        # need a few messages of movement to tell 7-bit / 14-bit / relative apart
        if len(self.pending) < 6:
            return None
        nums = {(p.ch, p.num) for p in self.pending}
        msb = sorted(n for n in nums if (n[0], n[1] + 32) in nums)
        if msb:                                   # MSB n + LSB n+32 => 14-bit control
            ch, num = msb[0]
            is14 = True
        else:
            first = self.pending[0]
            ch, num, is14 = first.ch, first.num, first.type == "cc14"
        first = next(p for p in self.pending if (p.ch, p.num) == (ch, num))
        b = Binding("cc14" if is14 else "cc", ch, num, action, relative=(kind == "relative"))
        if kind == "fader" and action == "tempo":
            # ask the user to move it from top (slow) to bottom; top value > bottom => invert
            vals = [p.value for p in self.pending if p.num == first.num]
            if len(vals) >= 2 and vals[0] > vals[-1]:
                b.invert = True
        self.pending = []
        return b
