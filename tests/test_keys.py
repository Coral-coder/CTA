from cta.keys import camelot_str, compatibility, from_camelot, key_name, parse_key


def test_camelot_mapping_known_values():
    assert camelot_str(0) == "8B"       # C major
    assert camelot_str(7) == "9B"       # G major
    assert camelot_str(12 + 9) == "8A"  # A minor
    assert camelot_str(12 + 0) == "5A"  # C minor
    assert camelot_str(12 + 4) == "9A"  # E minor
    assert camelot_str(11) == "1B"      # B major
    assert camelot_str(12 + 8) == "1A"  # G# minor


def test_camelot_roundtrip():
    for k in range(24):
        c = camelot_str(k)
        assert from_camelot(int(c[:-1]), c[-1]) == k


def test_parse_key_notations():
    assert parse_key("Am") == 21
    assert parse_key("A minor") == 21
    assert parse_key("F#m") == 18
    assert parse_key("Gbm") == 18
    assert parse_key("Eb") == 3
    assert parse_key("8A") == 21
    assert parse_key("11B") == 9
    assert parse_key("1d") == 0       # Open Key: C major
    assert parse_key("1m") == 21      # Open Key: A minor
    assert parse_key("") == -1
    assert parse_key("garbage") == -1
    assert key_name(21) == "Am"


def test_compatibility_rules():
    am, em, c, dm = 21, 16, 0, 14
    assert compatibility(am, am) == 1.0
    assert compatibility(am, em) == 0.9          # 8A -> 9A
    assert compatibility(am, dm) == 0.9          # 8A -> 7A
    assert compatibility(am, c) == 0.85          # relative major
    assert compatibility(am, 1) < 0.2            # Am -> Db major: clash
    assert compatibility(-1, am) == 0.5


def test_style_tempo_fix():
    from cta.analyzer import style_tempo_fix

    dnb = [("Electronic---Drum n Bass", 0.6), ("Electronic---Jungle", 0.2)]
    assert style_tempo_fix(87.0, dnb) == 174.0
    assert style_tempo_fix(174.0, dnb) == 174.0
    hh = [("Hip Hop---Boom Bap", 0.5)]
    assert style_tempo_fix(180.0, hh) == 90.0
    house = [("Electronic---House", 0.5)]
    assert style_tempo_fix(124.0, house) == 124.0
    assert style_tempo_fix(124.0, [("Electronic---House", 0.05)]) == 124.0
