"""Time-shift mixing: rehearse a transition ahead of time, play it live later.

While the current song plays to the room, the DJ mixes the next transition
on headphones in a preview player (its own decks, not the live ones), with
the mouse or the controller. Every fader, crossfader, EQ and filter move is
recorded against the beat. When the live set reaches that pair of tracks,
auto-mix starts the incoming track on the same beat and replays the moves.

Lanes are recorded on the incoming track's beat clock (x = beats since it
started), the same clock auto-mix uses, so tempo changes do not matter.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

import numpy as np

from .engine import SR

LANES = ("xf", "out_vol", "in_vol", "out_low", "out_mid", "out_high", "in_low", "in_mid", "in_high",
         "out_filter", "in_filter")
META_KEY = "rehearsals"
TEMPLATE_KEY = "rehearsal_templates"       # style group -> key of the recording used for every mix of it
MIN_BEATS = 8.0                      # shorter than 2 bars is not a transition (ignored)
DEFAULTS = {"xf": 0.0, "out_vol": 1.0, "in_vol": 1.0, "out_low": 1.0, "out_mid": 1.0, "out_high": 1.0,
            "in_low": 1.0, "in_mid": 1.0, "in_high": 1.0, "out_filter": 0.0, "in_filter": 0.0}


def xf_gain(xf: float, side: int) -> float:
    """The engine's crossfader curve (full to the centre, then a constant-power fade)."""
    x = float(np.clip((xf * side + 1.0) / 2.0, 0.0, 1.0))
    return 1.0 if x >= 0.5 else float(np.sin(x * np.pi))


@dataclass
class Rehearsal:
    out_path: str
    in_path: str
    out_start_beat: float            # outgoing beat at which the incoming starts
    in_start_beat: float             # incoming beat it starts from
    beats: float                     # length on the incoming clock until the outgoing is gone
    lanes: dict = field(default_factory=dict)   # lane -> [[x, value], ...]
    out_title: str = ""
    in_title: str = ""
    created: float = field(default_factory=time.time)
    enabled: bool = True

    def touched(self, lane: str) -> bool:
        """Did the DJ move this control during the recording?"""
        pts = self.lanes.get(lane) or []
        if not pts:
            return False
        vs = [p[1] for p in pts]
        return max(vs) - min(vs) > 0.02 or abs(vs[0] - DEFAULTS[lane]) > 0.02

    def value(self, lane: str, x: float) -> float:
        pts = self.lanes.get(lane)
        if not pts:
            return DEFAULTS[lane]
        a = np.asarray(pts, np.float64)
        return float(np.interp(x, a[:, 0], a[:, 1]))

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in ("out_path", "in_path", "out_start_beat", "in_start_beat", "beats",
                                              "lanes", "out_title", "in_title", "created", "enabled")}

    @classmethod
    def from_dict(cls, d: dict) -> Rehearsal:
        return cls(**{k: d[k] for k in ("out_path", "in_path", "out_start_beat", "in_start_beat", "beats")},
                   lanes={k: [list(map(float, p)) for p in v] for k, v in d.get("lanes", {}).items()},
                   out_title=d.get("out_title", ""), in_title=d.get("in_title", ""),
                   created=float(d.get("created", 0)), enabled=bool(d.get("enabled", True)))


def apply(r: Rehearsal, x: float, din, dout, eng=None, base: dict | None = None):
    """Put the recorded moves at beat x onto the decks. Controls the DJ did
    not touch follow `base` (auto-mix's own blend at that beat), so a mix
    recorded with only the crossfader still gets auto-mix's EQ work.
    The crossfader really moves (mirrored when the outgoing is on the right);
    with both decks on the same side it is folded into the channel levels."""
    lv = any(r.touched(k) for k in ("xf", "out_vol", "in_vol"))
    if lv:
        xf = r.value("xf", x)
        ov, iv = r.value("out_vol", x), r.value("in_vol", x)
        if eng is not None and dout.xf_side != din.xf_side and dout.xf_side != 0:
            eng.crossfader = float(np.clip(xf * -dout.xf_side, -1.0, 1.0))
            dout.volume, din.volume = float(np.clip(ov, 0, 1)), float(np.clip(iv, 0, 1))
        else:                   # engine gain = volume^2 * crossfader: fold it into the faders
            dout.volume = float(np.clip(ov * np.sqrt(xf_gain(xf, -1)), 0.0, 1.0))
            din.volume = float(np.clip(iv * np.sqrt(xf_gain(xf, 1)), 0.0, 1.0))
    elif base is not None:
        dout.volume, din.volume = base["out_vol"], base["in_vol"]
    eq_out = list(base["out_eq"]) if base is not None else [1.0, 1.0, 1.0]
    eq_in = list(base["in_eq"]) if base is not None else [1.0, 1.0, 1.0]
    for i, band in enumerate(("low", "mid", "high")):
        if r.touched(f"out_{band}") or base is None:
            eq_out[i] = r.value(f"out_{band}", x)
        if r.touched(f"in_{band}") or base is None:
            eq_in[i] = r.value(f"in_{band}", x)
    dout.eq, din.eq = np.array(eq_out), np.array(eq_in)
    dout.filter = r.value("out_filter", x) if r.touched("out_filter") else 0.0
    din.filter = r.value("in_filter", x) if r.touched("in_filter") else 0.0


class RehearsalStore:
    """Rehearsed transitions saved with the library (one per pair of tracks)."""

    def __init__(self, db):
        self.db = db
        self._cache: dict[str, Rehearsal] | None = None
        self._templates: dict[str, str] | None = None
        self.on_change = None            # called after every put / delete

    @staticmethod
    def key(out_path: str, in_path: str) -> str:
        return f"{out_path}\n{in_path}"

    def _all(self) -> dict[str, Rehearsal]:
        if self._cache is None:
            try:
                raw = self.db.get_meta(META_KEY, {}) or {}
                self._cache = {k: Rehearsal.from_dict(v) for k, v in raw.items()}
            except Exception:  # noqa: BLE001
                self._cache = {}
        return self._cache

    def get(self, out_path: str, in_path: str) -> Rehearsal | None:
        r = self._all().get(self.key(out_path, in_path))
        return r if r is not None and r.enabled and r.beats >= MIN_BEATS else None

    def find(self, out_path: str, in_path: str) -> Rehearsal | None:
        return self._all().get(self.key(out_path, in_path))

    def put(self, r: Rehearsal):
        self._all()[self.key(r.out_path, r.in_path)] = r
        self._save()

    def delete(self, out_path: str, in_path: str):
        self._all().pop(self.key(out_path, in_path), None)
        self._save()

    def items(self) -> list[Rehearsal]:
        return sorted(self._all().values(), key=lambda r: -r.created)

    # ---- templates: one recording stands for a whole style ------------------------------
    def templates(self) -> dict[str, str]:
        if self._templates is None:
            try:
                self._templates = dict(self.db.get_meta(TEMPLATE_KEY, {}) or {})
            except Exception:  # noqa: BLE001
                self._templates = {}
        return self._templates

    def set_template(self, group: str, r: Rehearsal | None):
        if r is None:
            self.templates().pop(group, None)
        else:
            self.templates()[group] = self.key(r.out_path, r.in_path)
        self._save()

    def template(self, group: str) -> Rehearsal | None:
        k = self.templates().get(group)
        r = self._all().get(k) if k else None
        return r if r is not None and r.beats >= MIN_BEATS else None

    def template_group(self, r: Rehearsal) -> str | None:
        k = self.key(r.out_path, r.in_path)
        return next((g for g, kk in self.templates().items() if kk == k), None)

    def export_json(self) -> str:
        return json.dumps({"version": 1, "rehearsals": [r.to_dict() for r in self._all().values()],
                           "templates": self.templates()}, indent=1)

    def import_json(self, text: str) -> int:
        data = json.loads(text)
        n = 0
        for row in data.get("rehearsals", []):
            try:
                r = Rehearsal.from_dict(row)
            except (KeyError, TypeError):
                continue
            if self.key(r.out_path, r.in_path) not in self._all():
                n += 1
            self._all()[self.key(r.out_path, r.in_path)] = r
        for g, k in (data.get("templates") or {}).items():
            if k in self._all():
                self.templates()[g] = k
        self._save()
        return n

    def _save(self):
        try:
            self.db.set_meta(META_KEY, {k: r.to_dict() for k, r in self._all().items()})
            self.db.set_meta(TEMPLATE_KEY, self.templates())
        except Exception:  # noqa: BLE001
            pass
        if self.on_change is not None:
            try:
                self.on_change()
            except Exception:  # noqa: BLE001
                pass


class Session:
    """Runs on the preview engine (as one of its hooks): rehearse a transition
    from deck A (outgoing) into deck B (incoming), or play a saved one back.

    rehearse(): the outgoing plays from a lead-in before its OUT marker; the
    incoming drops in on a bar line (when the DJ asks, or at the OUT marker)
    in phase; from then on every move is recorded until the outgoing is gone
    (or stop()). replay(r): the same, but the moves come from r."""

    OUT, IN = "A", "B"
    LEAD_BEATS = 32.0

    def __init__(self, eng):
        self.eng = eng
        self.mode = "idle"              # idle | lead | recording | replay | done
        self.drop_beat: float | None = None
        self.in_start_beat = 0.0
        self.out_start_beat = 0.0
        self.lanes: dict[str, list[list[float]]] = {}
        self.x = 0.0
        self.replaying: Rehearsal | None = None
        self.result: Rehearsal | None = None
        self._quiet = 0.0
        self._last: dict[str, float] = {}
        self._b_was = False
        self.auto_record = True          # mixing the two preview decks records the transition
        self.fill = True                 # replay: controls not touched follow auto-mix
        eng.hooks.append(self._hook)

    # ---- control (UI thread; applied in the audio thread) ---------------------------
    def rehearse(self, out_beat: float, in_beat: float, lead_beats: float | None = None):
        """out_beat: where the incoming is planned to start (the OUT marker);
        in_beat: where the incoming starts from (its intro marker)."""
        lead = self.LEAD_BEATS if lead_beats is None else lead_beats

        def f(e):
            a, b = e.decks[self.OUT], e.decks[self.IN]
            if not (a.loaded and b.loaded and a.grid and b.grid):
                return
            self.replaying = None
            self.result = None
            self.lanes = {k: [] for k in LANES}
            self._last = {}
            self.drop_beat = out_beat
            self.in_start_beat = in_beat
            start = max(out_beat - lead, a.grid.beat_at(0.0))
            e._seek(a, a.grid.time_of_beat(start) * SR)
            a.volume, a.eq, a.filter, a.playing = 1.0, np.ones(3), 0.0, True
            e.master_deck = self.OUT
            b.playing = False
            b.sync = True
            b.keylock = True
            e._seek(b, b.grid.time_of_beat(in_beat) * SR)
            b.eq, b.filter = np.ones(3), 0.0
            self.mode = "lead"
        self.eng.cmd(f)

    def drop_now(self):
        """Bring the incoming in on the next bar line of the outgoing."""
        def f(e):
            a = e.decks[self.OUT]
            if self.mode == "lead" and a.grid is not None:
                self.drop_beat = float(np.ceil((a.grid.beat_at(a.time) + 0.05) / 4.0) * 4.0)
        self.eng.cmd(f)

    def replay(self, r: Rehearsal, lead_beats: float = 16.0):
        self.rehearse(r.out_start_beat, r.in_start_beat, lead_beats)

        def f(e):
            self.replaying = r
        self.eng.cmd(f)

    def stop(self):
        """End the rehearsal now; what was recorded appears in .result
        (mode becomes "done") once the audio thread has applied it."""
        def f(e):
            self._finish(e)
            for d in e.decks.values():
                d.playing = False
        self.eng.cmd(f)

    def close(self):
        if self._hook in self.eng.hooks:
            self.eng.hooks.remove(self._hook)

    # ---- audio thread ----------------------------------------------------------------
    def _hook(self, e, n: int):
        a, b = e.decks[self.OUT], e.decks[self.IN]
        started = b.playing and not self._b_was
        self._b_was = b.playing
        if self.mode in ("idle", "done"):
            # just mixing on the preview records it: the incoming starting while
            # the outgoing plays is the start of a transition
            if started and a.playing and a.grid and b.grid and self.auto_record:
                self.replaying = None
                self.result = None
                self.lanes = {k: [] for k in LANES}
                self._last = {}
                self.out_start_beat = a.grid.beat_at(a.time)
                self.in_start_beat = b.grid.beat_at(b.time)
                self.mode = "recording"
                self._quiet = 0.0
            else:
                return
        if not (a.grid and b.grid):
            return
        ab = a.grid.beat_at(a.time)
        if self.mode == "lead":
            if self.drop_beat is not None and ab >= self.drop_beat - 1e-6:
                # in phase: the incoming beat = its start + beats since the drop
                e._seek(b, b.grid.time_of_beat(self.in_start_beat + (ab - self.drop_beat)) * SR)
                b.playing = True
                self.out_start_beat = self.drop_beat
                self.mode = "replay" if self.replaying is not None else "recording"
                self._quiet = 0.0
            else:
                return
        self.x = b.grid.beat_at(b.time) - self.in_start_beat
        if self.mode == "replay":
            from .automix import blend_values

            r = self.replaying
            sw = float(np.floor(r.beats / 8.0) * 4.0) or r.beats / 2
            apply(r, self.x, b, a, e, blend_values(self.x, r.beats, sw, 0.0) if self.fill else None)
            if self.x >= self.replaying.beats:
                a.playing = False
                self.mode = "done"
            return
        self._sample(e, a, b)
        gone = a.volume ** 2 * e._xf_gain(a.xf_side) < 1e-3 or not a.playing
        self._quiet = self._quiet + n / SR * b.bpm / 60.0 if gone else 0.0
        if self.x > 1.0 and self._quiet >= 4.0:       # the outgoing has been silent for a bar
            self._finish(e)

    def _sample(self, e, a, b):
        v = {"xf": e.crossfader, "out_vol": a.volume, "in_vol": b.volume,
             "out_low": a.eq[0], "out_mid": a.eq[1], "out_high": a.eq[2],
             "in_low": b.eq[0], "in_mid": b.eq[1], "in_high": b.eq[2],
             "out_filter": a.filter, "in_filter": b.filter}
        for k, val in v.items():
            val = float(val)
            last = self._last.get(k)
            pts = self.lanes[k]
            # keep the curve, not every block: a point when it moves, plus one per beat
            if last is None or abs(val - last) > 0.004 or (pts and self.x - pts[-1][0] >= 1.0):
                pts.append([round(self.x, 4), round(val, 4)])
                self._last[k] = val

    def _finish(self, e):
        if self.mode != "recording":
            self.mode = "done" if self.mode != "idle" else "idle"
            return
        a, b = e.decks[self.OUT], e.decks[self.IN]
        if self.x < MIN_BEATS:                 # a blip (started and stopped at once): nothing recorded
            self.result = None
            self.mode = "done"
            return
        x = max(self.x, 1.0)
        for k, pts in self.lanes.items():
            last = self._last.get(k, DEFAULTS[k])
            if not pts or pts[-1][0] < x:
                pts.append([round(x, 4), round(last, 4)])
        self.result = Rehearsal(a.info.path, b.info.path, self.out_start_beat, self.in_start_beat, round(x, 3),
                                {k: v for k, v in self.lanes.items()}, a.info.title, b.info.title)
        self.mode = "done"
