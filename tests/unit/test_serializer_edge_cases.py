import json
import pytest
import numpy as np

from src.detector.FastDetector import FastDetector
from src.detector.DetectorSerializer import DetectorSerializer


def test_serializer_with_active_levels_keeps_features():
    """Active levels retain ml_features unchanged."""
    d = FastDetector()
    d._level_id_seq = 1
    d.confirmed_support_levels = [
        {
            'level_id': 1, 'type': 0,
            'center': 100.0, 'zone_low': 99.0, 'zone_high': 101.0,
            'count': 3,
            'first_touch_time': 1789000000,
            'last_touch_time': 1789343000,
            'touch_times': [1789000000, 1789200000, 1789343000],
            'ml_features': {
                'level_type': 'support',
                'touch_count': 3,
                'ticker': 'ETHUSDT',
                'return_24h': 0.0123,
                'atr_pct': 0.0005,
            },
        }
    ]
    d.levels = d.confirmed_support_levels

    blob = DetectorSerializer.serialize(d)
    restored = DetectorSerializer.deserialize(blob, FastDetector)

    assert len(restored.confirmed_support_levels) == 1
    lvl = restored.confirmed_support_levels[0]
    assert lvl['ml_features']['ticker'] == 'ETHUSDT'
    assert lvl['ml_features']['return_24h'] == 0.0123


def test_serializer_nan_inf_in_ml_features_are_sanitized():
    """NaN/Inf inside ml_features are replaced with None."""
    d = FastDetector()
    d._level_id_seq = 5
    d.confirmed_support_levels = [
        {
            'level_id': 1, 'type': 0,
            'center': 100.0, 'zone_low': 99.0, 'zone_high': 101.0,
            'count': 3,
            'first_touch_time': 1789000000,
            'last_touch_time': 1789343000,
            'touch_times': [1789000000, 1789343000],
            'ml_features': {
                'level_type': 'support',
                'touch_count': 3,
                'ticker': 'BTCUSDT',
                'return_24h': float('nan'),
                'atr_pct': float('inf'),
            },
        }
    ]
    d.levels = d.confirmed_support_levels

    blob = DetectorSerializer.serialize(d)
    json_str = json.dumps(blob)
    assert json_str is not None

    restored = DetectorSerializer.deserialize(json.loads(json_str), FastDetector)
    ml = restored.confirmed_support_levels[0]['ml_features']
    assert ml['return_24h'] is None
    assert ml['atr_pct'] is None


def test_deserialize_corrupted_blob_raises():
    """Damaged state_blob -> ValueError."""
    corrupted = {"invalid_key": 123}
    with pytest.raises(ValueError, match="Invalid or corrupted state_dict format"):
        DetectorSerializer.deserialize(corrupted, FastDetector)


def test_deserialize_accepts_iso_strings_legacy():
    """The old format in the database: times as ISO strings — they are readable."""
    legacy = {
        '_level_id_seq': 2,
        'last_processed_time': '2026-07-02T07:46:00.000000',
        'stack_max': [[100.5, '2026-07-02T07:40:00.000000']],
        'stack_min': [],
        'support_clusters': [],
        'resistance_clusters': [],
        'confirmed_support_levels': [
            {
                'level_id': 1, 'type': 0,
                'center': 100.0, 'zone_low': 99.0, 'zone_high': 101.0,
                'count': 3,
                'first_touch_time': '2026-07-01T00:00:00.000000',
                'last_touch_time':  '2026-07-02T07:46:00.000000',
                'touch_times': ['2026-07-01T00:00:00.000000',
                                '2026-07-02T07:46:00.000000'],
            }
        ],
        'confirmed_resistance_levels': [],
    }
    d = DetectorSerializer.deserialize(legacy, FastDetector)

    # We check that the times have become integers and haven’t gone to NaN.
    assert isinstance(d.last_processed_time, int)
    assert d.last_processed_time > 1_700_000_000
    assert isinstance(d.stack_max[0][1], int)
    assert d.stack_max[0][1] > 1_700_000_000
    assert d.confirmed_support_levels[0]['last_touch_time'] > 1_700_000_000


def test_deserialize_handles_microseconds_and_nanoseconds():
    """Auto‑scale: int in microseconds/nanoseconds is also readable."""
    # 2026-09-14 00:00 UTC in different units
    sec = 1789344000
    legacy = {
        '_level_id_seq': 0,
        'last_processed_time': sec * 10**6,           # microseconds
        'stack_max': [[100.5, sec * 10**9]],          # nanoseconds
        'stack_min': [[99.5,  sec * 10**3]],          # milliseconds
        'support_clusters': [],
        'resistance_clusters': [],
        'confirmed_support_levels': [],
        'confirmed_resistance_levels': [],
    }
    d = DetectorSerializer.deserialize(legacy, FastDetector)

    assert d.last_processed_time == sec
    assert d.stack_max[0][1] == sec
    assert d.stack_min[0][1] == sec