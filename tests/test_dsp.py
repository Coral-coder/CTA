import os

import numpy as np
import pytest
from conftest import synth_track

from cta import dsp


@pytest.mark.parametrize("bpm,pattern", [(128.0, "4x4"), (124.5, "4x4"), (174.0, "break"), (90.0, "hiphop"),
                                         (140.0, "4x4"), (100.0, "4x4")])
def test_tempo(bpm, pattern):
    y = synth_track(bpm, 40, pattern=pattern)
    r = dsp.estimate_tempo(y)
    assert abs(r.bpm - bpm) < 0.1, r


def test_tempo_silence():
    assert dsp.estimate_tempo(np.zeros(dsp.SR * 10, np.float32)).bpm == 0.0


@pytest.mark.parametrize("chord,expected", [((57, 60, 64), 21), ((60, 64, 67), 0), ((52, 55, 59), 16),
                                            ((62, 65, 69), 14), ((57, 61, 64), 9)])
def test_key(chord, expected):
    y = synth_track(120, 30, chord=chord)
    assert dsp.estimate_key(y).key == expected


def test_key_silence():
    assert dsp.estimate_key(np.zeros(dsp.SR * 5, np.float32)).key == -1


def test_model_input_matches_essentia():
    """Our numpy log-mel must equal Essentia's TensorflowInputMusiCNN, which
    the Discogs-EffNet model was trained with."""
    ref = np.load(os.path.join(os.path.dirname(__file__), "data", "essentia_musicnn_input_ref.npy"))
    sr = 16000
    t = np.arange(sr * 2) / sr
    rng = np.random.default_rng(42)
    x = (0.3 * np.sin(2 * np.pi * 330 * t) + 0.05 * rng.standard_normal(t.size)
         + 0.2 * np.sin(2 * np.pi * 2500 * t) * (t % 0.25 < 0.05)).astype(np.float32)
    ours = dsp.model_melspectrogram(x)
    assert ours.shape == ref.shape
    assert np.abs(ours - ref).max() < 1e-3


def test_model_patches_shape():
    p = dsp.model_patches(np.random.default_rng(0).standard_normal(16000 * 200).astype(np.float32) * 0.1)
    assert p.shape == (32, 128, 96)
    short = dsp.model_patches(np.zeros(16000, np.float32))
    assert short.shape == (1, 128, 96)


def test_energy_features():
    f = dsp.energy_features(synth_track(128, 20))
    assert -40 < f["loudness"] < 0
    assert f["centroid"] > 0


def test_round_bpm():
    assert dsp.round_bpm(127.97) == 128.0
    assert dsp.round_bpm(124.5) == 124.5
