"""
  Sample lifecycle:
      - 3 touches produce 3 samples, exactly one flagged as confirmation
      - first_touch_time <= last_touch_time <= prediction_time
  Targets:
      - 1h => 4h => 24h nesting
      - samples without full forward horizon are dropped by get_dataset()
"""
import pytest
import numpy as np

from src.detector.FastDetector import FastDetector


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def detector():
    """Fresh detector with small batch/window for fast tests."""
    return FastDetector(
        batch_size=10,
        window_size=50,
        atr_period=14,
        min_touches=3,
        bars_per_hour=10,       # h1=10, h4=40, h24=240
        recent_touch_lookback_bars=20,
    )


@pytest.fixture
def primed_detector(detector):
    """
    Detector with all market arrays primed for direct
    _assign_point_to_cluster / _build_level_features tests.
    Flat market: high=101, low=99, close=100 -> TR=2 for every bar.
    ATR is preset to 1.0 so tolerance = 1.0 with tolerance_k=1.
    """
    n = 500
    detector.closes_np = np.full(n, 100.0)
    detector.high = np.full(n, 101.0)
    detector.low = np.full(n, 99.0)
    detector.volume_np = np.full(n, 50.0)
    detector.quote_volume_np = np.full(n, 5000.0)
    detector.times_np = np.arange(n, dtype=np.int64) * 60   # 1m bars
    detector.atr_values = np.full(n, 1.0)
    detector.symbol = "TESTUSDT"
    detector.num_bars = n
    return detector


# ============================================================
# 1. ATR
# ============================================================

def test_atr_converges_on_flat_market(detector):
    """Flat market with fixed high-low range: ATR must converge to that range."""
    n = 100
    closes = np.full(n, 100.0)
    highs = closes + 1.0
    lows = closes - 1.0

    atr = detector._compute_causal_atr(highs, lows, closes)

    assert len(atr) == n
    assert np.all(np.isfinite(atr))
    assert atr[-1] == pytest.approx(2.0, abs=1e-3)


def test_atr_floor_prevents_degenerate_values(detector):
    """
    BUG #3 regression: empty candles (h=l=c) would give ATR -> 0.
    Floor = close * ATR_FLOOR_PCT must keep ATR strictly positive.
    """
    n = 50
    closes = np.full(n, 100.0)
    highs = closes.copy()
    lows = closes.copy()

    atr = detector._compute_causal_atr(highs, lows, closes)

    floor = 100.0 * 1e-5
    assert np.all(atr >= floor - 1e-12)
    assert atr[-1] == pytest.approx(floor, rel=1e-6)


# ============================================================
# 2. Window extrema (BUG #2)
# ============================================================

def test_window_extrema_indices_are_paired_with_correct_value(detector):
    """
    BUG #2 regression: previously max index was taken from min position and
    vice versa, corrupting (price, idx) pairs pushed into the stacks.
    """
    detector._update_window_extrema(
        batch_min=95.0,
        batch_max=110.0,
        min_local_idx=2,
        max_local_idx=7,
        index_glob=100,
    )
    assert detector.w_batch_min == 95.0
    assert detector.w_batch_min_idx == 102          # 100 + 2
    assert detector.w_batch_max == 110.0
    assert detector.w_batch_max_idx == 107          # 100 + 7


# ============================================================
# 3. Sample lifecycle
# ============================================================

def test_three_touches_confirm_level_and_create_three_samples(primed_detector):
    d = primed_detector
    assert len(d.samples) == 0

    d._assign_point_to_cluster(100.00, 100, 10, ptype=0)
    d._assign_point_to_cluster(100.05, 200, 20, ptype=0)
    # not confirmed yet
    assert len(d.samples) == 0
    assert len(d.support_clusters) == 1

    d._assign_point_to_cluster(99.95, 300, 30, ptype=0)
    # confirmation -> 3 retrospective samples
    assert len(d.samples) == 3
    assert len(d.confirmed_support_levels) == 1

    level_id = d.confirmed_support_levels[0]["level_id"]
    per_level = [s for s in d.samples if s["level_id"] == level_id]
    assert len(per_level) == 3

    flags = [s["is_confirmation_touch"] for s in per_level]
    assert flags.count(True) == 1
    assert per_level[2]["is_confirmation_touch"] is True


def test_existing_samples_are_never_mutated_on_new_touch(primed_detector):
    """
    BUG #1 regression: adding a 4th touch must create a NEW sample
    and must not touch previously created ones.
    """
    d = primed_detector
    d._assign_point_to_cluster(100.00, 100, 10, ptype=0)
    d._assign_point_to_cluster(100.05, 200, 20, ptype=0)
    d._assign_point_to_cluster(99.95, 300, 30, ptype=0)

    # snapshot baseline
    assert d.samples[0]["touch_count"] == 1
    assert d.samples[0]["last_touch_time"] == 100
    assert d.samples[2]["touch_count"] == 3
    assert d.samples[2]["last_touch_time"] == 300

    # 4th touch
    d._assign_point_to_cluster(100.10, 400, 40, ptype=0)

    assert len(d.samples) == 4
    # old samples untouched
    assert d.samples[0]["touch_count"] == 1
    assert d.samples[0]["last_touch_time"] == 100
    assert d.samples[2]["touch_count"] == 3
    assert d.samples[2]["last_touch_time"] == 300
    # new sample sees full history up to itself
    assert d.samples[3]["touch_count"] == 4
    assert d.samples[3]["last_touch_time"] == 400


def test_sample_time_ordering(primed_detector):
    """first_touch_time <= last_touch_time <= prediction_time for every sample."""
    d = primed_detector
    for ts, idx in ((100, 10), (200, 20), (300, 30), (400, 40)):
        d._assign_point_to_cluster(100.0, ts, idx, ptype=0)

    for s in d.samples:
        assert s["first_touch_time"] <= s["last_touch_time"]
        assert s["last_touch_time"] <= s["prediction_time"]


# ============================================================
# 4. Feature normalization (BUG #4)
# ============================================================

def test_range_features_are_normalized_by_close(primed_detector):
    """
    BUG #4 regression: range_* was absolute price. Now it must be a fraction
    of close. Flat market: high=101, low=99, close=100 -> range == 0.02.
    """
    d = primed_detector
    for ts, idx in ((100, 10), (200, 20), (300, 30)):
        d._assign_point_to_cluster(100.0, ts, idx, ptype=0)

    for s in d.samples:
        for k in ("range_1h", "range_4h", "range_24h"):
            v = s[k]
            if v is None:
                continue
            assert v == pytest.approx(0.02, abs=1e-6)


# ============================================================
# 5. Full pipeline
# ============================================================

def test_full_process_pipeline(detector):
    n = 500
    rng = np.random.default_rng(42)
    closes = 100.0 + np.cumsum(rng.normal(0, 0.1, size=n))
    highs = closes + 0.5
    lows = closes - 0.5
    volumes = rng.uniform(10, 100, size=n)
    times = np.arange(n, dtype=np.int64) * 60

    points = detector.process(
        closes_np=closes,
        times_np=times,
        high_np=highs,
        low_np=lows,
        volume_np=volumes,
        quote_volume_np=volumes * closes,
        symbol="TESTUSDT",
    )

    assert isinstance(points, np.ndarray)
    assert len(detector.atr_values) == n
    assert detector.last_processed_time == int(times[-1])


def test_targets_are_binary_and_nested(detector):
    """breakout_1h => breakout_4h => breakout_24h for every exported sample."""
    n = 1000
    rng = np.random.default_rng(0)
    closes = 100.0 + np.cumsum(rng.normal(0, 0.2, size=n))
    highs = closes + 0.3
    lows = closes - 0.3

    detector.process(
        closes_np=closes,
        times_np=np.arange(n, dtype=np.int64) * 60,
        high_np=highs,
        low_np=lows,
        volume_np=np.full(n, 10.0),
        quote_volume_np=np.full(n, 1000.0),
        symbol="TESTUSDT",
    )

    dataset = detector.get_dataset()
    assert len(dataset) >= 0  # at minimum doesn't crash

    for s in dataset:
        assert s["breakout_1h"] in (0, 1)
        assert s["breakout_4h"] in (0, 1)
        assert s["breakout_24h"] in (0, 1)
        if s["breakout_1h"] == 1:
            assert s["breakout_4h"] == 1
        if s["breakout_4h"] == 1:
            assert s["breakout_24h"] == 1


def test_get_dataset_drops_incomplete_tail_samples(detector):
    """
    Samples too close to the end of the array cannot have a full 24h horizon
    and must be dropped from get_dataset().
    """
    n = 300
    closes = 100.0 + np.sin(np.linspace(0, 20, n)) * 5.0
    highs = closes + 1.0
    lows = closes - 1.0

    detector.process(
        closes_np=closes,
        times_np=np.arange(n, dtype=np.int64) * 60,
        high_np=highs,
        low_np=lows,
        volume_np=np.full(n, 10.0),
        quote_volume_np=np.full(n, 1000.0),
        symbol="TESTUSDT",
    )

    dataset = detector.get_dataset()
    for s in dataset:
        assert s["breakout_1h"] is not None
        assert s["breakout_4h"] is not None
        assert s["breakout_24h"] is not None


# ============================================================
# 6. get_active_levels
# ============================================================

def test_get_active_levels_shape(primed_detector):
    d = primed_detector
    for ts, idx in ((100, 10), (200, 20), (300, 30)):
        d._assign_point_to_cluster(100.0, ts, idx, ptype=0)

    active = d.get_active_levels()
    assert len(active) == 1
    lvl = active[0]
    assert lvl["level_type"] == "support"
    assert lvl["touch_count"] == 3
    assert lvl["center"] == pytest.approx(100.0, abs=0.2)
    assert lvl["symbol"] == "TESTUSDT"
    assert "touch_times" in lvl
    assert len(lvl["touch_times"]) == 3