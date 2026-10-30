"""Learning from the DJ: every transition you mix by hand is recorded
(where in each track's structure you mixed, how long, which genre), and
auto-mix copies your habits once it has a few examples per style.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass

import numpy as np

MIN_EXAMPLES = 3


@dataclass
class TransitionExample:
    ts: float
    style_group: str          # "long" (trance/progressive) | "club" | "short" (hip-hop/pop...)
    out_title: str
    in_title: str
    bars: float               # how long both tracks were audible together
    out_offset: float         # bars after the outgoing track's last drop ended (negative = before)
    in_lead: float            # bars between where the incoming started and its first drop
    out_section: str          # section kind of the outgoing when the blend started
    in_section: str


def style_group(style: str, genre: str) -> str:
    s = (style or "").lower()
    if any(w in s for w in ("trance", "progressive", "psy")):
        return "long"
    from .automix import SHORT_BLEND_GENRES

    if genre in SHORT_BLEND_GENRES:
        return "short"
    return "club"


class MixLearner:
    """Watches the decks (called ~30x/s from the UI) and records manual
    transitions; provides learned preferences to the planner."""

    def __init__(self, db, automix):
        self.db = db
        self.am = automix
        self.enabled = True
        self._active = None      # ongoing overlap
        self._quiet_since = None
        self.examples: list[TransitionExample] = self._load()

    # ---- storage ---------------------------------------------------------------
    def _load(self) -> list[TransitionExample]:
        try:
            rows = self.db.get_meta("learned_transitions", [])
            return [TransitionExample(**r) for r in rows]
        except Exception:  # noqa: BLE001
            return []

    def _save(self):
        try:
            self.db.set_meta("learned_transitions", [asdict(x) for x in self.examples[-500:]])
        except Exception:  # noqa: BLE001
            pass

    def forget(self):
        self.examples = []
        self._save()

    # ---- observing -------------------------------------------------------------------
    @staticmethod
    def _audible(eng, d) -> float:
        return d.volume ** 2 * eng._xf_gain(d.xf_side) if d.playing else 0.0

    def tick(self, eng):
        if not self.enabled:
            return
        # auto-mix's own transitions are not lessons - unless the DJ takes one over
        auto_now = self.am.active and self.am.tr is not None and self.am.tr.started
        if auto_now and self._active is not None:
            self._active["auto"] = True
        live = [d for d in eng.decks.values() if d.loaded and d.grid is not None and self._audible(eng, d) > 0.2]
        now = time.time()
        if self._active is None:
            if len(live) == 2:
                a, b = live
                # the one that has been playing longer is the outgoing track
                out, inc = (a, b) if a.time - a.grid.first_beat > b.time - b.grid.first_beat else (b, a)
                self._active = {"out": out.name, "in": inc.name, "t0": now,
                                "out_bar0": out.grid.beat_at(out.time) / 4.0,
                                "in_bar0": inc.grid.beat_at(inc.time) / 4.0, "bpm": out.bpm,
                                "auto": auto_now}
                self._quiet_since = None
            return
        st = self._active
        out, inc = eng.decks[st["out"]], eng.decks[st["in"]]
        out_gone = self._audible(eng, out) < 0.08
        if not out_gone:
            self._quiet_since = None
            if not inc.playing:                    # incoming stopped: not a transition
                self._active = None
            return
        if self._quiet_since is None:
            self._quiet_since = now
            return
        if now - self._quiet_since < 1.0:
            return
        bars = (self._quiet_since - st["t0"]) * st["bpm"] / 60.0 / 4.0
        self._active = None
        if 2.0 <= bars <= 96 and not st.get("auto"):          # a blip is not a transition
            self._record(eng, out, inc, st, bars)

    def took_over(self):
        """The DJ grabbed an auto-mix transition: from here it is their mix,
        timed from when the blend began."""
        if self._active is not None:
            self._active["auto"] = False

    def _record(self, eng, out, inc, st, bars):
        try:
            mo = self.am._analyse_deck(out).structure
            mi = self.am._analyse_deck(inc).structure
        except Exception:  # noqa: BLE001
            return
        drop_end = mo.last_drop_end if mo.last_drop_end is not None else mo.last_beat_bar + 1
        drop_in = mi.first_drop_bar if mi.first_drop_bar is not None else mi.first_beat_bar
        sec_o = mo.section_at(st["out_bar0"])
        sec_i = mi.section_at(st["in_bar0"])
        ex = TransitionExample(
            ts=time.time(), style_group=style_group(out.info.style, out.info.genre),
            out_title=out.info.title, in_title=inc.info.title, bars=round(float(bars), 1),
            out_offset=round(float(st["out_bar0"] - drop_end), 1), in_lead=round(float(drop_in - st["in_bar0"]), 1),
            out_section=sec_o.kind if sec_o else "?", in_section=sec_i.kind if sec_i else "?")
        self.examples.append(ex)
        self._save()

    # ---- what auto-mix uses ------------------------------------------------------------
    def count(self, group: str | None = None) -> int:
        return sum(1 for x in self.examples if group is None or x.style_group == group)

    def preferences(self, group: str) -> dict | None:
        usable = [x for x in self.examples if x.bars >= 2.0]     # ignore blips from older versions
        ex = [x for x in usable if x.style_group == group]
        if len(ex) < MIN_EXAMPLES:
            ex = usable if len(usable) >= MIN_EXAMPLES * 2 else []
        if not ex:
            return None
        bars = float(np.median([x.bars for x in ex]))
        return {
            "bars": int(max(4, round(bars / 4) * 4)),              # never shorter than a phrase-ish 4 bars
            "out_offset": float(np.median([x.out_offset for x in ex])),
            "in_lead": float(np.median([x.in_lead for x in ex])),
            "n": len(ex),
        }


def describe(prefs: dict | None) -> str:
    if not prefs:
        return ""
    return (f"learned from {prefs['n']} of your mixes: {prefs['bars']}-bar blends, "
            f"incoming {prefs['in_lead']:+.0f} bars before its drop")


def dumps(examples) -> str:
    return json.dumps([asdict(x) for x in examples], indent=1)
