import json
import pytest
import numpy as np

from src.detector.FastDetector import FastDetector
from src.detector.DetectorSerializer import DetectorSerializer


@pytest.fixture
def populated_detector():
    """Detector with an active level in a new format."""
    d = FastDetector()
    d.symbol = "BTCUSDT"
    d.last_processed_time = 1789343000
    d._level_id_seq = 42

     # Stacks in the format [(price, time_sec), ...]
    d.stack_max = [(105.0, 1789343000), (110.0, 1789344000)]
    d.stack_min = [(95.0,  1789343000), (90.0,  1789344000)]

    # Cluster
    d.support_clusters = [
        {
            'type': 0,
            'sum': 200.0,
            'count': 2,
            'center': 100.0,
            'zone_low': 99.0,
            'zone_high': 101.0,
            'touch_times': [1789343000, 1789343060],
        }
    ]
    d.resistance_clusters = []

    # Confirmed level with ml_features
    d.confirmed_support_levels = [
        {
            'level_id': 1,
            'type': 0,
            'center': 100.0,
            'zone_low': 99.0,
            'zone_high': 101.0,
            'count': 3,
            'first_touch_time': 1789000000,
            'last_touch_time': 1789343000,
            'touch_times': [1789000000, 1789200000, 1789343000],
            'ml_features': {
                'level_type': 'support',
                'touch_count': 3,
                'ticker': 'BTCUSDT',
                'return_24h': 0.0123,
            },
        }
    ]
    d.confirmed_resistance_levels = []
    d.levels = d.confirmed_support_levels + d.confirmed_resistance_levels

    return d


def test_serializer_json_compatibility(populated_detector):
    """The serialization result is fully JSON‑compatible."""
    state_dict = DetectorSerializer.serialize(populated_detector)

    json_str = json.dumps(state_dict)
    assert isinstance(json_str, str)

    assert 'closes_tail' not in state_dict
    assert 'pending_samples' not in state_dict
    assert 'breakout_dataset' not in state_dict

    assert 'last_processed_time' in state_dict
    assert 'confirmed_support_levels' in state_dict
    assert 'confirmed_resistance_levels' in state_dict
    assert 'stack_max' in state_dict
    assert 'stack_min' in state_dict

    # Times in ISO
    assert state_dict['last_processed_time'].startswith('20')
    assert 'T' in state_dict['last_processed_time']

    # Level saved, ml_features in place
    lvl = state_dict['confirmed_support_levels'][0]
    assert lvl['touch_times'][0].startswith('20')
    assert 'ml_features' in lvl
    assert lvl['ml_features']['ticker'] == 'BTCUSDT'

    # touch_indices are not serialized
    assert 'touch_indices' not in lvl


def test_serializer_roundtrip(populated_detector):
    """The full Serialize -> Deserialize cycle saves the data."""
    state_dict = DetectorSerializer.serialize(populated_detector)
    restored = DetectorSerializer.deserialize(state_dict, FastDetector)

    assert restored._level_id_seq == 42
    assert restored.last_processed_time == 1789343000

    # The stacks are restored as a list of tuples. (price, time_sec)
    assert restored.stack_max == [(105.0, 1789343000), (110.0, 1789344000)]
    assert restored.stack_min == [(95.0, 1789343000), (90.0, 1789344000)]

    # The levels have been restored.
    assert len(restored.confirmed_support_levels) == 1
    lvl = restored.confirmed_support_levels[0]
    assert lvl['level_id'] == 1
    assert lvl['first_touch_time'] == 1789000000
    assert lvl['last_touch_time'] == 1789343000
    assert lvl['touch_times'] == [1789000000, 1789200000, 1789343000]
    assert lvl['ml_features']['ticker'] == 'BTCUSDT'
    assert lvl['ml_features']['return_24h'] == 0.0123

    # levels — union confirmed_*
    assert restored.levels == (
        restored.confirmed_support_levels + restored.confirmed_resistance_levels
    )

    # Clusters — times in int
    assert restored.support_clusters[0]['touch_times'] == [1789343000, 1789343060]


def test_serializer_empty_detector():
    """The empty detector serializes and deserializes without errors."""
    empty_detector = FastDetector()
    state_dict = DetectorSerializer.serialize(empty_detector)

    # Empty lists, nothing superfluous.
    assert state_dict['confirmed_support_levels'] == []
    assert state_dict['confirmed_resistance_levels'] == []
    assert state_dict['stack_max'] == []
    assert state_dict['stack_min'] == []
    assert state_dict['last_processed_time'] is None

    restored = DetectorSerializer.deserialize(state_dict, FastDetector)
    assert restored.confirmed_support_levels == []
    assert restored.confirmed_resistance_levels == []
    assert restored.stack_max == []
    assert restored.stack_min == []
    assert restored.last_processed_time is None
    assert restored.levels == []