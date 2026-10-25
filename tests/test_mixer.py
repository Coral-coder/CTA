import numpy as np

from cta.db import Database, vec_to_blob
from cta.library import Library
from cta.mixer import MixSettings, SetlistOptions, build_setlist, energy_target, suggest_next


def _fake_db(path, n=400, seed=0):
    rng = np.random.default_rng(seed)
    db = Database(path)
    rows = []
    for i in range(n):
        emb = rng.random(1280).astype(np.float32)
        sv = rng.dirichlet(np.ones(400) * 0.05).astype(np.float32)
        rows.append(dict(path=f"/m/{i}.mp3", mtime=0, size=0, title=f"T{i}", artist=f"A{i % 50}", album="",
                         tag_genre="", tag_bpm=None, tag_key=-1, duration=300, bpm=float(rng.uniform(118, 132)),
                         bpm_conf=1, key=int(rng.integers(0, 24)), key_conf=1, loudness=float(rng.uniform(-14, -6)),
                         centroid=2000, onset_rate=3, genre="House", styles=None, style_vec=vec_to_blob(sv),
                         embedding=vec_to_blob(emb), status="ok", error=None, analyzed_at=0, version=1))
    db.upsert_many(rows)
    return db


def test_suggestions_respect_limits(tmp_path):
    lib = Library(_fake_db(str(tmp_path / "x.db")))
    s = MixSettings(max_pitch_pct=3, strict_key=True)
    for a in range(0, 50, 7):
        for sug in suggest_next(lib, a, s, n=30):
            assert abs(sug.pitch_pct) <= 3 + 1e-9
            assert sug.key >= 0.5
            assert sug.idx != a


def test_setlist_unique_and_ramp(tmp_path):
    lib = Library(_fake_db(str(tmp_path / "x.db")))
    seed = int(np.argmin(np.abs(lib.bpm - 120)))
    seq = build_setlist(lib, seed, SetlistOptions(length=25, end_bpm=130, mix=MixSettings(max_pitch_pct=4)))
    assert len(seq) == 25 and len(set(seq)) == 25 and seq[0] == seed
    assert lib.bpm[seq[-1]] > lib.bpm[seq[0]] + 4
    for a, b in zip(seq, seq[1:]):
        assert abs(lib.bpm[a] / lib.bpm[b] - 1) * 100 <= 4 + 1e-6


def test_setlist_variety_changes_result(tmp_path):
    lib = Library(_fake_db(str(tmp_path / "x.db")))
    a = build_setlist(lib, 0, SetlistOptions(length=15, variety=1.0, seed=1))
    b = build_setlist(lib, 0, SetlistOptions(length=15, variety=1.0, seed=2))
    assert a != b


def test_energy_curves():
    assert energy_target("steady", 5, 0.5) == 5
    assert energy_target("build", 5, 1.0) == 9
    assert energy_target("cooldown", 5, 1.0) == 1
    assert energy_target("peak", 5, 0.65) == 9


def test_setlist_never_repeats_played_tracks(tmp_path):
    lib = Library(_fake_db(str(tmp_path / "x.db")))
    seed = 5
    first = build_setlist(lib, seed, SetlistOptions(length=10))
    played = set(first[1:6]) | {seed}
    again = build_setlist(lib, seed, SetlistOptions(length=10), exclude=played)
    assert again[0] == seed                      # the chosen seed still starts the set
    assert not (played - {seed}) & set(again)


def test_sets_stay_in_the_seed_genre(tmp_path):
    """A chain of tracks from electro house to trance where every neighbour
    mixes well: without an anchor a long set walks down the chain."""
    from cta.mixer import anchor_mask

    rng = np.random.default_rng(3)
    db = Database(str(tmp_path / "g.db"))
    rows = []
    a_style = np.zeros(400, np.float32)
    a_style[10] = 1.0                        # "electro house"
    b_style = np.zeros(400, np.float32)
    b_style[20] = 1.0                        # "trance"
    base = rng.random(1280).astype(np.float32)
    far = rng.random(1280).astype(np.float32)
    n = 100
    for i in range(n):
        t = i / (n - 1)
        sv = (1 - t) * a_style + t * b_style + 0.001
        emb = (1 - t) * base + t * far
        rows.append(dict(path=f"/m/{i}.mp3", mtime=0, size=0, title=f"T{i}", artist=f"A{i}", album="",
                         tag_genre="", tag_bpm=None, tag_key=-1, duration=300, bpm=126.0 + 10.0 * t,
                         bpm_conf=1, key=15, key_conf=1, loudness=-8.0,
                         centroid=2000, onset_rate=3, genre="Electronic", styles=None,
                         style_vec=vec_to_blob(sv / sv.sum()), embedding=vec_to_blob(emb), status="ok", error=None,
                         analyzed_at=0, version=1))
    db.upsert_many(rows)
    lib = Library(db)
    pos = {i: int(lib.paths[i].split("/")[-1].split(".")[0]) for i in range(lib.n)}
    seed = next(i for i in range(lib.n) if pos[i] == 0)
    house = {i for i in range(lib.n) if pos[i] < 35}
    opt = SetlistOptions(length=40, mix=MixSettings(max_pitch_pct=6))
    drift = build_setlist(lib, seed, opt)                         # the old behaviour
    assert set(drift) - house, "the test library must tempt a drift"
    seq = build_setlist(lib, seed, opt, anchor_w=0.45)
    assert len(seq) >= 15 and set(seq) <= house, sorted(pos[i] for i in seq)
    sug = suggest_next(lib, seq[5], MixSettings(max_pitch_pct=6), n=10, exclude=set(seq[:6]), anchor=seed)
    assert sug and all(s.idx in house for s in sug)
    # every house track played: suggestions widen to the nearest tracks, not the far end of the chain
    late = suggest_next(lib, seq[-1], MixSettings(max_pitch_pct=6), n=5, exclude=house, anchor=seed)
    assert late and max(pos[s.idx] for s in late) < 60
    assert anchor_mask(lib, seed).sum() < lib.n
