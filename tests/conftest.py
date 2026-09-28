# tests/conftest.py
import numpy as np
import pytest

from src.detector.FastDetector import FastDetector


@pytest.fixture
def populated_detector():
    """
    Detector with an active level..

    Important:
      - stack_max / stack_min — list [(price, time_sec), ...]
      - times_np — int, UTC seconds
      - no pending_samples / breakout_dataset / *_tail
      - ml_features is required for the confirmed level
    """
    d = FastDetector()
    d.symbol = "BTCUSDT"
    d.last_processed_time = 1789343000
    d._level_id_seq = 42

    # Stacks: (price, time_sec)
    d.stack_max = [(105.0, 1789343000), (110.0, 1789344000)]
    d.stack_min = [(95.0,  1789343000), (90.0,  1789344000)]

    # Clusters
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
                'atr_pct': 0.0005,
            },
        }
    ]
    d.confirmed_resistance_levels = []
    d.levels = d.confirmed_support_levels + d.confirmed_resistance_levels

    return d


@pytest.fixture
def empty_detector():
    return FastDetector()