"""Dance-music style families: which AI sub-genres (Discogs styles) blend.

The AI model names 400 Discogs styles; a DJ thinks in families (house, techno,
trance, ...) and knows which of those sit well next to each other in a set.
This is that knowledge, used by the suggestions, the auto-mix queue and the
library's Style filter. Tempo and key still decide the final order; the
family only says what belongs in the same set at all.
"""
from __future__ import annotations

FAMILIES: dict[str, tuple[str, ...]] = {
    "House": ("House", "Deep House", "Tech House", "Progressive House", "Electro House", "Tropical House",
              "Garage House", "Italo House", "Euro House", "Hip-House", "Tribal House", "Acid House",
              "Ghetto House", "Jazzdance", "Beatdown", "Future House", "Bass House", "Funky House",
              "Afro House", "Melodic House", "Organic House", "Soulful House", "Jackin House"),
    "Disco": ("Disco", "Nu-Disco", "Italo-Disco", "Euro-Disco", "Hi NRG", "Boogie", "Space Disco",
              "Cosmic Disco", "Disco Polo", "Funk"),
    "Techno": ("Techno", "Minimal Techno", "Deep Techno", "Hard Techno", "Dub Techno", "Minimal", "Schranz",
               "Acid", "Industrial", "EBM", "New Beat", "Tribal", "Detroit Techno", "Melodic Techno",
               "Peak Time Techno", "Hypnotic Techno", "Bleep", "Rhythmic Noise"),
    "Trance": ("Trance", "Progressive Trance", "Psy-Trance", "Goa Trance", "Tech Trance", "Hard Trance",
               "Uplifting Trance", "Vocal Trance", "Psytrance", "Full-On", "Dark Psy"),
    "Bass": ("Dubstep", "Grime", "Halftime", "Bassline", "UK Garage", "Speed Garage", "Garage", "2-Step",
             "Future Bass", "Trap", "UK Funky", "Riddim", "Brostep", "Wave", "Phonk", "Jersey Club",
             "Baltimore Club", "Juke"),
    "Drum & Bass": ("Drum n Bass", "Jungle", "Breakcore", "Liquid Funk", "Neurofunk", "Jump Up", "Drumfunk",
                    "Footwork", "Drum and Bass"),
    "Breaks": ("Breakbeat", "Breaks", "Big Beat", "Progressive Breaks", "Broken Beat", "Electro", "Freestyle",
               "Miami Bass", "Nu Skool Breaks", "Ghetto", "Glitch Hop"),
    "Hard Dance": ("Hardstyle", "Hardcore", "Happy Hardcore", "Gabber", "Jumpstyle", "Hard House", "Hands Up",
                   "Makina", "Speedcore", "Donk", "Rawstyle", "Frenchcore", "UK Hardcore", "Terror",
                   "Hard Bounce", "Hard NRG"),
    "Dance / Pop": ("Eurodance", "Dance-pop", "Italodance", "Eurobeat", "Synth-pop", "Synthwave", "Electroclash",
                    "Chiptune", "Electropop", "Future Pop", "Pop Rap", "Reggaeton", "Moombahton", "Dancehall",
                    "Latin"),
    "Downtempo": ("Downtempo", "Trip Hop", "Chillwave", "Ambient", "IDM", "Future Jazz", "Acid Jazz", "Dub",
                  "Glitch", "Leftfield", "Abstract", "Vaporwave", "Experimental", "Dark Ambient", "Drone",
                  "New Age", "Berlin-School", "Illbient", "Lo-Fi", "Chillout", "Lounge", "Balearic",
                  "Modern Classical", "Musique Concrète", "Sound Collage", "Noise", "Power Electronics",
                  "Dungeon Synth", "Darkwave", "Neofolk"),
}

# families that sit well next to each other in a set (symmetric)
_NEIGHBOURS = {
    ("House", "Disco"), ("House", "Techno"), ("House", "Trance"), ("House", "Breaks"), ("House", "Dance / Pop"),
    ("House", "Bass"), ("House", "Downtempo"),
    ("Techno", "Hard Dance"), ("Techno", "Trance"), ("Techno", "Breaks"),
    ("Trance", "Hard Dance"), ("Trance", "Dance / Pop"),
    ("Bass", "Drum & Bass"), ("Bass", "Breaks"), ("Drum & Bass", "Breaks"),
    ("Hard Dance", "Dance / Pop"), ("Disco", "Dance / Pop"), ("Disco", "Downtempo"), ("Breaks", "Downtempo"),
}
NEIGHBOURS: dict[str, tuple[str, ...]] = {
    f: tuple(sorted({b for a, b in _NEIGHBOURS if a == f} | {a for a, b in _NEIGHBOURS if b == f}))
    for f in FAMILIES
}

_FAMILY_OF = {style.lower(): fam for fam, styles in FAMILIES.items() for style in styles}


def family_of(style: str | None) -> str | None:
    """The family of an AI style ("Electronic---Tech House" or "Tech House"); None if unknown."""
    if not style:
        return None
    return _FAMILY_OF.get(style.split("---")[-1].strip().lower())


def blend_score(style_a: str | None, style_b: str | None, parent_a: str = "", parent_b: str = "") -> float:
    """How well two styles sit next to each other (0..1): the same style 1,
    the same family 0.85, neighbouring families 0.55, the same parent genre
    (both electronic, say) 0.2, anything else 0. Unknown on either side: 0.5
    (no opinion, not a penalty)."""
    fa, fb = family_of(style_a), family_of(style_b)
    if fa is None or fb is None:
        return 0.5
    sa = style_a.split("---")[-1].strip().lower()
    sb = style_b.split("---")[-1].strip().lower()
    if sa == sb:
        return 1.0
    if fa == fb:
        return 0.85
    if fb in NEIGHBOURS[fa]:
        return 0.55
    if parent_a and parent_a == parent_b:
        return 0.2
    return 0.0


def blends_with(style: str | None) -> dict[str, tuple[str, ...]]:
    """What to reach for after a track of this style: {family: styles},
    the track's own family first, then its neighbours."""
    fam = family_of(style)
    if fam is None:
        return {}
    out = {fam: FAMILIES[fam]}
    for n in NEIGHBOURS[fam]:
        out[n] = FAMILIES[n]
    return out


def reference_text() -> str:
    """The whole map as plain text (the "What blends with what" window)."""
    lines = []
    for fam, styles in FAMILIES.items():
        lines.append(f"{fam}: {', '.join(styles)}")
        lines.append(f"    blends with: {', '.join(NEIGHBOURS[fam]) or '-'}")
    return "\n".join(lines)
