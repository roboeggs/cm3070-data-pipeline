import logging
import numpy as np

from datetime import datetime, timezone

logger = logging.getLogger(__name__)


# The minimum allowable ATR as a fraction of close.
# Protects against atr_pct ~ 1e-54 on “empty” candles of small altcoins.
ATR_FLOOR_PCT = 1e-5


def _iso(ts):
    """int epoch seconds -> ISO string. None / str remain as they are."""
    if ts is None or isinstance(ts, str):
        return ts
    return datetime.fromtimestamp(int(ts), tz=timezone.utc).isoformat()


class FastDetector:
    """
    Batch-wise extremum detector with sliding window, ATR, stack filtering
    and online level clustering.

    Sample lifecycle
    ----------------
-   - Each touch of the level -> a separate sample (a snapshot of the market + level metrics
      on the bar of this touch). Existing samples are NEVER mutated.
    - On confirmation (min_touches-th touch), samples are built retrospectively
      for all previous touches of this level. The is_confirmation_touch=True flag
      is set for the touch that moved the level to confirmed.
    - Targets (breakout_1h/4h/24h) are calculated strictly based on the future:
      [prediction_idx+1 ... prediction_idx+horizon]. If at least one
      target does not fit within the available data horizon, the sample is discarded.
    - Upon a breakout, the level is removed from confirmed. Already created samples remain.
    """

    def __init__(
        self,
        batch_size=144,
        window_size=720,
        atr_period=14,
        stack_multiplier=2.0,
        tolerance_k=1.0,
        min_touches=3,
        breakout_atr_mult=1.5,
        bars_per_hour=60,
        recent_touch_lookback_bars=240,
    ):
        self.day_bars = 1440

        self.batch_size = batch_size
        self.window_size = window_size
        self.period = atr_period
        self.stack_multiplier = stack_multiplier
        self.tolerance_k = tolerance_k
        self.min_touches = min_touches

        self.breakout_atr_mult = breakout_atr_mult
        self.bars_per_hour = bars_per_hour
        self.h1 = 1 * bars_per_hour
        self.h4 = 4 * bars_per_hour
        self.h24 = 24 * bars_per_hour
        self.recent_touch_lookback_bars = recent_touch_lookback_bars

        self.symbol = None
        self.last_processed_time = None

        self.atr_values = None
        self.points = np.empty((0, 4), dtype=np.float64)
        self._reset_window()

        self.stack_max = []   # (price, idx)
        self.stack_min = []

        self.support_clusters = []
        self.resistance_clusters = []
        self.confirmed_support_levels = []
        self.confirmed_resistance_levels = []
        self.levels = []

        self._level_id_seq = 0

        # samples: all generated samples (including those whose targets are still None)
        self.samples = []
        # pending_by_level_id: level_id -> [samples], convenient for analysis
        self.pending_by_level_id = {}

        # market arrays (set in process)
        self.closes_np = None
        self.high = None
        self.low = None
        self.volume_np = None
        self.quote_volume_np = None
        self.times_np = None
        self.num_bars = 0

        self.btc_close_np = None
        self.btc_volume_np = None
        self.btc_quote_volume_np = None
        self.btc_high_np = None
        self.btc_low_np = None

        self.processed_indices = set()

    # --------------------------------------------------
    # Window reset
    # --------------------------------------------------
    def _reset_window(self):
        self.w_batch_min = float("inf")
        self.w_batch_max = float("-inf")
        self.w_batch_min_idx = 0
        self.w_batch_max_idx = 0

    # --------------------------------------------------
    # ATR (causal)
    # --------------------------------------------------
    def _compute_causal_atr(self, high_np, low_np, closes_np):
        n = len(closes_np)
        atr = np.full(n, np.nan, dtype=np.float64)
        if n == 0:
            return atr

        tr = np.zeros(n, dtype=np.float64)
        tr[0] = high_np[0] - low_np[0]

        prev_close = closes_np[:-1]
        tr1 = high_np[1:] - low_np[1:]
        tr2 = np.abs(high_np[1:] - prev_close)
        tr3 = np.abs(low_np[1:] - prev_close)
        tr[1:] = np.maximum(tr1, np.maximum(tr2, tr3))

        p = int(self.period)
        if p <= 1:
            atr[:] = tr
        elif n < p:
            csum = np.cumsum(tr)
            atr[:] = csum / (np.arange(n) + 1.0)
        else:
            atr[p - 1] = np.mean(tr[:p])
            for i in range(p, n):
                atr[i] = (atr[i - 1] * (p - 1) + tr[i]) / p
            csum = np.cumsum(tr[: p - 1])
            atr[: p - 1] = csum / (np.arange(p - 1) + 1.0)

        floor = closes_np * ATR_FLOOR_PCT
        atr = np.maximum(atr, floor)

        return atr

    # --------------------------------------------------
    # Stacks
    # --------------------------------------------------
    def _push_to_stack(self, price, idx, ptype):
        if idx >= len(self.atr_values):
            return
        atr_i = self.atr_values[idx]
        if not np.isfinite(atr_i):
            return
        threshold = self.stack_multiplier * atr_i
        stack = self.stack_max if ptype == 1 else self.stack_min

        if ptype == 1:  # maximum
            while stack and price > stack[-1][0] + threshold:
                old_price, _ = stack.pop()
                self._remove_cluster_by_point(old_price, 1)
        else:  # minimum
            while stack and price < stack[-1][0] - threshold:
                old_price, _ = stack.pop()
                self._remove_cluster_by_point(old_price, 0)

        time_point = (
            int(self.times_np[idx])
            if self.times_np is not None and idx < len(self.times_np)
            else idx
        )
        stack.append((price, time_point))

    def _update_window_extrema(self, batch_min, batch_max,
                               min_local_idx, max_local_idx, index_glob):
        if batch_max > self.w_batch_max:
            self.w_batch_max = batch_max
            self.w_batch_max_idx = index_glob + max_local_idx
        if batch_min < self.w_batch_min:
            self.w_batch_min = batch_min
            self.w_batch_min_idx = index_glob + min_local_idx

    def _remove_cluster_by_point(self, price, ptype, tolerance=None):
        clusters = self.support_clusters if ptype == 0 else self.resistance_clusters
        for i, c in enumerate(clusters):
            if tolerance is not None:
                if abs(price - c["center"]) <= tolerance:
                    del clusters[i]
                    return c
            else:
                if c["zone_low"] <= price <= c["zone_high"]:
                    del clusters[i]
                    return c
        return None

    # --------------------------------------------------
    # Time helpers
    # --------------------------------------------------
    def _to_epoch_seconds(self, t):
        if t is None:
            return None
        if isinstance(t, (int, np.integer)):
            return int(t)
        if isinstance(t, str):
            s = t.strip()
            if not s:
                return None
            try:
                dt = datetime.fromisoformat(s)
            except ValueError:
                try:
                    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
                except ValueError:
                    return None
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return int(dt.timestamp())
        if hasattr(t, "timestamp"):
            return int(t.timestamp())
        return None

    def _time_to_idx(self, t):
        if self.times_np is None or len(self.times_np) == 0:
            return 0
        return int(np.searchsorted(self.times_np, int(t), side="left"))

    def _rehydrate_state(self):
        """
        After loading from the database, touch_indices are empty — we restore 
        them from  touch_times, because the indices are not saved between runs.
        """
        if self.times_np is None or len(self.times_np) == 0:
            return

        for clusters in (self.support_clusters, self.resistance_clusters):
            for c in clusters:
                c["touch_indices"] = [
                    self._time_to_idx(t) for t in c.get("touch_times", [])
                ]
                # snapshots don’t need to be re‑built: they’re for the building-cluster
                # only for the current run; after resume, we don’t restore
                # partially built levels 1-to-1 — they’ll continue as they are.

        for lvls in (self.confirmed_support_levels, self.confirmed_resistance_levels):
            for lvl in lvls:
                lvl["touch_indices"] = [
                    self._time_to_idx(t) for t in lvl.get("touch_times", [])
                ]

        self.levels = self.confirmed_support_levels + self.confirmed_resistance_levels

    # --------------------------------------------------
    # Feature helpers (causal)
    # --------------------------------------------------
    def _safe_div(self, a, b):
        if b is None:
            return None
        if np.isscalar(b):
            if not np.isfinite(b) or b == 0:
                return None
            return a / b
        b_arr = np.asarray(b)
        if b_arr.size == 0:
            return None
        if b_arr.size == 1:
            bv = float(b_arr.reshape(-1)[0])
            if not np.isfinite(bv) or bv == 0:
                return None
            return a / bv
        return None

    def _return_over_bars(self, arr, t, bars):
        if arr is None or t - bars < 0:
            return None
        prev = arr[t - bars]
        curr = arr[t]
        if prev == 0 or not np.isfinite(prev) or not np.isfinite(curr):
            return None
        return (curr - prev) / prev

    def _range_over_bars(self, high, low, ref_close, t, bars):
        if high is None or low is None or ref_close is None:
            return None
        if t - bars + 1 < 0:
            return None
        h = np.max(high[t - bars + 1: t + 1])
        l = np.min(low[t - bars + 1: t + 1])
        c = ref_close[t]
        if not np.isfinite(h) or not np.isfinite(l):
            return None
        if not np.isfinite(c) or c <= 0:
            return None
        return (h - l) / c

    def _mean_prev_window_excl_current(self, arr, t, bars):
        if arr is None or t - bars < 0:
            return None
        w = arr[t - bars: t]
        if w.size == 0:
            return None
        m = np.mean(w)
        if not np.isfinite(m):
            return None
        return m

    def _volume_ratio(self, vol_arr, t, bars):
        if vol_arr is None or t < 0 or t >= len(vol_arr):
            return None
        curr = vol_arr[t]
        if not np.isfinite(curr):
            return None
        mean_hist = self._mean_prev_window_excl_current(vol_arr, t, bars)
        if mean_hist is None or mean_hist <= 0:
            return None
        return curr / mean_hist

    # --------------------------------------------------
    # Feature builder
    # --------------------------------------------------
    def _build_level_features_at_prediction(self, level_snap, prediction_idx):
        """
        Builds a sample on the bar prediction_idx strictly based on the data <= prediction_idx.
        level_snap — a “frozen” snapshot of the level at the moment of contact:
            {
              'type': 0/1,
              'level_id': int,
              'center': float,
              'zone_low': float,
              'zone_high': float,
              'count': int,
              'touch_indices': [int, ...],
              'touch_times': [int, ...],
            }
        """
        t = prediction_idx
        close_t = self.closes_np[t]
        atr_t = self.atr_values[t]

        center = level_snap["center"]
        zone_low = level_snap["zone_low"]
        zone_high = level_snap["zone_high"]
        touch_indices = level_snap["touch_indices"]
        touch_times = level_snap["touch_times"]

        touch_count = len(touch_indices)
        first_touch_idx = touch_indices[0] if touch_count > 0 else None
        last_touch_idx = touch_indices[-1] if touch_count > 0 else None

        level_age_bars = (t - first_touch_idx) if first_touch_idx is not None else None
        time_since_last_touch = (t - last_touch_idx) if last_touch_idx is not None else None

        if first_touch_idx is not None and t >= first_touch_idx:
            span = max(1, t - first_touch_idx + 1)
            touch_density = touch_count / span
        else:
            touch_density = None

        recent_start = max(0, t - self.recent_touch_lookback_bars + 1)
        if touch_count > 0:
            ti = np.asarray(touch_indices)
            recent_touch_count = int(np.sum((ti >= recent_start) & (ti <= t)))
        else:
            recent_touch_count = 0
        recent_span = max(1, t - recent_start + 1)
        recent_touch_density = recent_touch_count / recent_span

        zone_width = zone_high - zone_low
        zone_width_pct = self._safe_div(zone_width, center)
        zone_width_atr = (
            self._safe_div(zone_width, atr_t)
            if (atr_t is not None and np.isfinite(atr_t) and atr_t > 0) else None
        )

        price_position_in_zone = (
            (close_t - zone_low) / zone_width if zone_width > 0 else None
        )

        dist_abs = abs(close_t - center)
        distance_to_level_pct = self._safe_div(dist_abs, center)
        distance_to_level_atr = (
            self._safe_div(dist_abs, atr_t)
            if (atr_t is not None and np.isfinite(atr_t) and atr_t > 0) else None
        )

        return_1h = self._return_over_bars(self.closes_np, t, self.h1)
        return_4h = self._return_over_bars(self.closes_np, t, self.h4)
        return_12h = self._return_over_bars(self.closes_np, t, 12 * self.bars_per_hour)
        return_24h = self._return_over_bars(self.closes_np, t, self.h24)

        range_1h = self._range_over_bars(self.high, self.low, self.closes_np, t, self.h1)
        range_4h = self._range_over_bars(self.high, self.low, self.closes_np, t, self.h4)
        range_24h = self._range_over_bars(self.high, self.low, self.closes_np, t, self.h24)

        atr_pct = self._safe_div(atr_t, close_t)

        volume_ratio_1h = self._volume_ratio(self.volume_np, t, self.h1)
        volume_ratio_4h = self._volume_ratio(self.volume_np, t, self.h4)
        volume_ratio_24h = self._volume_ratio(self.volume_np, t, self.h24)

        quote_volume_ratio_1h = self._volume_ratio(self.quote_volume_np, t, self.h1)
        quote_volume_ratio_4h = self._volume_ratio(self.quote_volume_np, t, self.h4)
        quote_volume_ratio_24h = self._volume_ratio(self.quote_volume_np, t, self.h24)

        distance_change_1h = None
        distance_change_4h = None
        approach_velocity_1h = None
        approach_velocity_4h = None
        if t - self.h1 >= 0:
            d_prev = abs(self.closes_np[t - self.h1] - center)
            distance_change_1h = d_prev - dist_abs
            approach_velocity_1h = distance_change_1h / self.h1
        if t - self.h4 >= 0:
            d_prev = abs(self.closes_np[t - self.h4] - center)
            distance_change_4h = d_prev - dist_abs
            approach_velocity_4h = distance_change_4h / self.h4

        btc_return_1h = self._return_over_bars(self.btc_close_np, t, self.h1)
        btc_return_4h = self._return_over_bars(self.btc_close_np, t, self.h4)
        btc_return_24h = self._return_over_bars(self.btc_close_np, t, self.h24)

        btc_volume_ratio_1h = self._volume_ratio(self.btc_volume_np, t, self.h1)
        btc_volume_ratio_4h = self._volume_ratio(self.btc_volume_np, t, self.h4)
        btc_volume_ratio_24h = self._volume_ratio(self.btc_volume_np, t, self.h24)

        btc_quote_volume_ratio_1h = self._volume_ratio(self.btc_quote_volume_np, t, self.h1)
        btc_quote_volume_ratio_4h = self._volume_ratio(self.btc_quote_volume_np, t, self.h4)
        btc_quote_volume_ratio_24h = self._volume_ratio(self.btc_quote_volume_np, t, self.h24)

        btc_volatility_1h = self._range_over_bars(
            self.btc_high_np, self.btc_low_np, self.btc_close_np, t, self.h1
        )
        btc_volatility_4h = self._range_over_bars(
            self.btc_high_np, self.btc_low_np, self.btc_close_np, t, self.h4
        )

        sample = {
            # internal (stripped on export)
            "_prediction_idx": t,

            # metadata
            "symbol": self.symbol,
            "level_id": level_snap["level_id"],
            "level_type": "resistance" if level_snap["type"] == 1 else "support",
            "prediction_time": int(self.times_np[t]) if self.times_np is not None else None,

            # level geometry at this touch
            "center": center,
            "zone_low": zone_low,
            "zone_high": zone_high,

            # touch history up to and including this touch
            "first_touch_time": touch_times[0] if touch_times else None,
            "last_touch_time": touch_times[-1] if touch_times else None,
            "touch_times": list(touch_times),
            "touch_count": touch_count,

            # features
            "recent_touch_count": recent_touch_count,
            "level_age_bars": level_age_bars,
            "time_since_last_touch": time_since_last_touch,
            "touch_density": touch_density,
            "recent_touch_density": recent_touch_density,

            "zone_width_pct": zone_width_pct,
            "zone_width_atr": zone_width_atr,
            "price_position_in_zone": price_position_in_zone,

            "distance_to_level_pct": distance_to_level_pct,
            "distance_to_level_atr": distance_to_level_atr,

            "return_1h": return_1h,
            "return_4h": return_4h,
            "return_12h": return_12h,
            "return_24h": return_24h,

            "range_1h": range_1h,
            "range_4h": range_4h,
            "range_24h": range_24h,

            "atr_pct": atr_pct,

            "volume_ratio_1h": volume_ratio_1h,
            "volume_ratio_4h": volume_ratio_4h,
            "volume_ratio_24h": volume_ratio_24h,

            "quote_volume_ratio_1h": quote_volume_ratio_1h,
            "quote_volume_ratio_4h": quote_volume_ratio_4h,
            "quote_volume_ratio_24h": quote_volume_ratio_24h,

            "distance_change_1h": distance_change_1h,
            "distance_change_4h": distance_change_4h,
            "approach_velocity_1h": approach_velocity_1h,
            "approach_velocity_4h": approach_velocity_4h,

            "btc_return_1h": btc_return_1h,
            "btc_return_4h": btc_return_4h,
            "btc_return_24h": btc_return_24h,
            "btc_volume_ratio_1h": btc_volume_ratio_1h,
            "btc_volume_ratio_4h": btc_volume_ratio_4h,
            "btc_volume_ratio_24h": btc_volume_ratio_24h,
            "btc_quote_volume_ratio_1h": btc_quote_volume_ratio_1h,
            "btc_quote_volume_ratio_4h": btc_quote_volume_ratio_4h,
            "btc_quote_volume_ratio_24h": btc_quote_volume_ratio_24h,
            "btc_volatility_1h": btc_volatility_1h,
            "btc_volatility_4h": btc_volatility_4h,

            # targets (filled later)
            "breakout_1h": None,
            "breakout_4h": None,
            "breakout_24h": None,

            # flag: is this the touch that confirmed the level?
            "is_confirmation_touch": False,
        }
        return sample

    # --------------------------------------------------
    # Sample creation
    # --------------------------------------------------
    def _make_sample(self, level_snap, prediction_idx, is_confirmation_touch):
        s = self._build_level_features_at_prediction(level_snap, prediction_idx)
        s["is_confirmation_touch"] = bool(is_confirmation_touch)
        self.samples.append(s)
        self.pending_by_level_id.setdefault(level_snap["level_id"], []).append(s)
        return s

    # --------------------------------------------------
    # Targets
    # --------------------------------------------------
    def _compute_breakout_target(self, level_type, center, prediction_idx,
                                 horizon_bars, atr_t):
        end_idx = prediction_idx + horizon_bars
        if end_idx >= self.num_bars:
            return None
        if atr_t is None or not np.isfinite(atr_t) or atr_t <= 0:
            return None

        threshold = atr_t * self.breakout_atr_mult
        left = prediction_idx + 1
        right = end_idx + 1

        if level_type == "resistance":
            fh = np.max(self.high[left:right])
            if not np.isfinite(fh):
                return None
            return int(fh >= (center + threshold))
        else:
            fl = np.min(self.low[left:right])
            if not np.isfinite(fl):
                return None
            return int(fl <= (center - threshold))

    def _finalize_pending_samples(self):
        """Called once at the end of process(). Fills targets where horizon is
        fully available; leaves others with None.

        Samples, restored from the database, do not have `_prediction_idx` — their
        skip it (targets have already been calculated before saving or will not be
        calculated in this run).
        """
        for s in self.samples:
            if s.get("breakout_1h") is not None:
                continue

            pidx = s.get("_prediction_idx")
            if pidx is None:
                # sample restored from the database — nothing to finalize
                s.setdefault("breakout_1h",  None)
                s.setdefault("breakout_4h",  None)
                s.setdefault("breakout_24h", None)
                continue

            atr_t = self.atr_values[pidx] if pidx < len(self.atr_values) else None
            s["breakout_1h"] = self._compute_breakout_target(
                s["level_type"], s["center"], pidx, self.h1, atr_t
            )
            s["breakout_4h"] = self._compute_breakout_target(
                s["level_type"], s["center"], pidx, self.h4, atr_t
            )
            s["breakout_24h"] = self._compute_breakout_target(
                s["level_type"], s["center"], pidx, self.h24, atr_t
            )

    # --------------------------------------------------
    # Touch assignment
    # --------------------------------------------------
    def _assign_point_to_cluster(self, price, time_point, idx, ptype):
        if idx >= len(self.atr_values):
            return
        atr_i = self.atr_values[idx]
        if not np.isfinite(atr_i) or atr_i <= 0:
            return

        half_tol = self.tolerance_k * atr_i

        # confirmed levels
        confirmed = (
            self.confirmed_support_levels if ptype == 0
            else self.confirmed_resistance_levels
        )

        best_confirmed = None
        best_dist = float("inf")
        for lvl in confirmed:
            dist = abs(price - lvl["center"])
            if dist <= half_tol and dist < best_dist:
                best_confirmed = lvl
                best_dist = dist

        if best_confirmed is not None:
            best_confirmed["touch_times"].append(time_point)
            best_confirmed["touch_indices"].append(idx)
            best_confirmed["count"] += 1
            best_confirmed["end_idx"] = idx
            best_confirmed["last_touch_time"] = time_point

            # create a new sample for this touch
            level_snap = {
                "type": best_confirmed["type"],
                "level_id": best_confirmed["level_id"],
                "center": best_confirmed["center"],
                "zone_low": best_confirmed["zone_low"],
                "zone_high": best_confirmed["zone_high"],
                "count": best_confirmed["count"],
                "touch_indices": list(best_confirmed["touch_indices"]),
                "touch_times": list(best_confirmed["touch_times"]),
            }
            self._make_sample(level_snap, idx, is_confirmation_touch=False)
            return

        # building clusters
        clusters = self.support_clusters if ptype == 0 else self.resistance_clusters

        best = None
        min_dist = float("inf")
        for c in clusters:
            dist = abs(price - c["center"])
            if dist <= half_tol and dist < min_dist:
                min_dist = dist
                best = c

        if best is not None:
            best["sum"] += price
            best["count"] += 1
            best["center"] = best["sum"] / best["count"]
            if price < best["zone_low"]:
                best["zone_low"] = price
            if price > best["zone_high"]:
                best["zone_high"] = price
            best["touch_times"].append(time_point)
            best["touch_indices"].append(idx)
            best["snapshots"].append((
                best["center"], best["zone_low"], best["zone_high"], best["count"]
            ))

            # confirmation reached
            if best["count"] == self.min_touches:
                self._level_id_seq += 1
                level_id = self._level_id_seq

                # retrospective samples for every touch that built this level
                for k, (c_k, zl_k, zh_k, cnt_k) in enumerate(best["snapshots"]):
                    level_snap = {
                        "type": best["type"],
                        "level_id": level_id,
                        "center": c_k,
                        "zone_low": zl_k,
                        "zone_high": zh_k,
                        "count": cnt_k,
                        "touch_indices": best["touch_indices"][: k + 1],
                        "touch_times": best["touch_times"][: k + 1],
                    }
                    is_conf = (k + 1 == self.min_touches)
                    self._make_sample(level_snap, best["touch_indices"][k], is_conf)

                # finalize confirmed level state
                best["level_id"] = level_id
                best["end_idx"] = best["touch_indices"][-1]
                best["last_touch_time"] = best["touch_times"][-1]
                best.pop("snapshots", None)

                clusters.remove(best)
                if best["type"] == 0:
                    self.confirmed_support_levels.append(best)
                else:
                    self.confirmed_resistance_levels.append(best)
                self.levels = (
                    self.confirmed_support_levels + self.confirmed_resistance_levels
                )
        else:
            new_c = {
                "type": ptype,
                "sum": price,
                "count": 1,
                "center": price,
                "zone_low": price,
                "zone_high": price,
                "touch_times": [time_point],
                "touch_indices": [idx],
                "snapshots": [(price, price, price, 1)],
            }
            clusters.append(new_c)

    # --------------------------------------------------
    # Breakout / removal
    # --------------------------------------------------
    def _is_broken(self, lvl, bars_from, bars_to, is_resistance):
        if bars_from >= bars_to:
            return False
        if self.times_np is None or len(self.times_np) == 0:
            return False
        ltt = lvl.get("last_touch_time")
        if ltt is None:
            return False

        ltt = int(ltt)
        ltt_idx = int(np.searchsorted(self.times_np, ltt, side="left"))
        if ltt_idx >= len(self.atr_values):
            return False
        atr_at_touch = self.atr_values[ltt_idx]
        if not np.isfinite(atr_at_touch) or atr_at_touch <= 0:
            return False
        threshold = self.breakout_atr_mult * atr_at_touch

        cutoff_idx = int(np.searchsorted(self.times_np, ltt, side="right"))
        left = max(bars_from, cutoff_idx)
        right = bars_to
        if left >= right:
            return False

        closes = self.closes_np[left:right]
        if closes.size == 0:
            return False

        if is_resistance:
            return bool(np.any(closes > lvl["zone_high"] + threshold))
        return bool(np.any(closes < lvl["zone_low"] - threshold))

    def _check_breakouts(self, bars_from, bars_to):
        if bars_from >= bars_to:
            return
        removed = []
        kept_r = []
        for lvl in self.confirmed_resistance_levels:
            if self._is_broken(lvl, bars_from, bars_to, is_resistance=True):
                removed.append(lvl.get("level_id"))
                continue
            kept_r.append(lvl)
        self.confirmed_resistance_levels = kept_r

        kept_s = []
        for lvl in self.confirmed_support_levels:
            if self._is_broken(lvl, bars_from, bars_to, is_resistance=False):
                removed.append(lvl.get("level_id"))
                continue
            kept_s.append(lvl)
        self.confirmed_support_levels = kept_s

        if removed:
            self.levels = (
                self.confirmed_support_levels + self.confirmed_resistance_levels
            )
            logger.info("Removed %d broken levels: %s", len(removed), removed)

    # --------------------------------------------------
    # Params
    # --------------------------------------------------
    def get_params_dict(self):
        return {
            "batch_size": self.batch_size,
            "window_size": self.window_size,
            "period": self.period,
            "stack_multiplier": self.stack_multiplier,
            "tolerance_k": self.tolerance_k,
            "min_touches": self.min_touches,
            "bars_per_hour": self.bars_per_hour,
            "recent_touch_lookback_bars": self.recent_touch_lookback_bars,
            "breakout_atr_mult": self.breakout_atr_mult,
            "day_bars": self.day_bars,
        }

    # --------------------------------------------------
    # Output API
    # --------------------------------------------------
    def _serialize_sample(self, s):
        out = {}
        for k, v in s.items():
            if k.startswith("_"):
                continue
            if k == "touch_times" and isinstance(v, list):
                out[k] = [_iso(t) for t in v]
            elif k in ("prediction_time", "first_touch_time", "last_touch_time"):
                out[k] = _iso(v)
            else:
                out[k] = v
        return out

    def get_dataset(self):
        """Return samples with all 3 targets filled (i.e. with complete
        forward horizon). Samples with incomplete targets are dropped."""
        out = []
        for s in self.samples:
            if (s.get("breakout_1h")  is None
                    or s.get("breakout_4h")  is None
                    or s.get("breakout_24h") is None):
                continue
            out.append(self._serialize_sample(s))
        return out

    def get_active_levels(self):
        """Currently confirmed, not-yet-broken levels."""
        out = []
        for lvl in self.confirmed_support_levels + self.confirmed_resistance_levels:
            out.append({
                "symbol": self.symbol,
                "level_id": lvl.get("level_id"),
                "level_type": "support" if lvl["type"] == 0 else "resistance",
                "center": lvl["center"],
                "zone_low": lvl["zone_low"],
                "zone_high": lvl["zone_high"],
                "touch_count": lvl["count"],
                "first_touch_time": _iso(lvl["touch_times"][0]) if lvl["touch_times"] else None,
                "last_touch_time": _iso(lvl["touch_times"][-1]) if lvl["touch_times"] else None,
                "touch_times": [_iso(t) for t in lvl["touch_times"]],
            })
        return out


    def _export_levels(self, levels):
        """Groups samples by level_id and places them in the level."""
        def _key(s):
            t = s.get("prediction_time")
            return t if t is not None else 0

        out = []
        for lvl in levels:
            lid = lvl.get("level_id")
            snaps = list(self.pending_by_level_id.get(lid, []))
            snaps.sort(key=_key)

            samples_out = []
            for n, s in enumerate(snaps, start=1):
                item = {k: v for k, v in s.items() if not k.startswith("_")}
                item["touch_number"] = n
                samples_out.append(item)

            out.append({
                "level_id":         lid,
                "level_type":       "support" if lvl["type"] == 0 else "resistance",
                "center":           lvl["center"],
                "zone_low":         lvl["zone_low"],
                "zone_high":        lvl["zone_high"],
                "first_touch_time": lvl["touch_times"][0] if lvl["touch_times"] else None,
                "last_touch_time":  lvl["touch_times"][-1] if lvl["touch_times"] else None,
                "touch_count":      lvl["count"],
                "samples":          samples_out,
            })
        return out


    def get_confirmed_levels_export(self):
        """Returns support/resistance levels, each with nested samples."""
        return {
            "support":    self._export_levels(self.confirmed_support_levels),
            "resistance": self._export_levels(self.confirmed_resistance_levels),
        }

    def _to_float64_array(self, data):
        """Converts the input data to a NumPy float64 array if it is not None."""
        return np.asarray(data, dtype=np.float64) if data is not None else None

    # --------------------------------------------------
    # Main
    # --------------------------------------------------
    def process(
        self,
        closes_np,
        times_np=None,
        volume_np=None,
        quote_volume_np=None,
        high_np=None,
        low_np=None,
        btc_close_np=None,
        btc_volume_np=None,
        btc_quote_volume_np=None,
        btc_high_np=None,
        btc_low_np=None,
        symbol=None,
        resume_from_time=None,
    ):
        if closes_np is None or high_np is None or low_np is None:
            raise ValueError("closes_np, high_np, low_np are required.")

        self.symbol = symbol
        self.closes_np = self._to_float64_array(closes_np)
        self.times_np = times_np
        self.volume_np = self._to_float64_array(volume_np)
        self.quote_volume_np = self._to_float64_array(quote_volume_np)
        self.high = self._to_float64_array(high_np)
        self.low = self._to_float64_array(low_np)

        self.btc_close_np = self._to_float64_array(btc_close_np)
        self.btc_volume_np = self._to_float64_array(btc_volume_np)
        self.btc_quote_volume_np = self._to_float64_array(btc_quote_volume_np)
        self.btc_high_np = self._to_float64_array(btc_high_np)
        self.btc_low_np = self._to_float64_array(btc_low_np)

        self.num_bars = len(self.closes_np)
        self.atr_values = self._compute_causal_atr(self.high, self.low, self.closes_np)

        # restore touch_indices from touch_times (after loading from the database)
        self._rehydrate_state()

        # From which index should we continue?
        if (
            resume_from_time is not None
            and self.times_np is not None
            and len(self.times_np) > 0
        ):
            start_idx = int(
                np.searchsorted(self.times_np, int(resume_from_time), side="right")
            )
        else:
            start_idx = 0

        logger.info(
            "fastDetector.process: symbol=%s bars=%d start_idx=%d resume_from_time=%s",
            self.symbol, self.num_bars, start_idx, resume_from_time,
        )

        self.processed_indices = set()
        index_glob = 0

        for i in range(self.batch_size, self.num_bars, self.batch_size):
            is_window_end = (i - index_glob) >= self.window_size
            batch_closes = self.closes_np[index_glob:i]

            min_local_idx = int(batch_closes.argmin())
            max_local_idx = int(batch_closes.argmax())
            batch_min = batch_closes[min_local_idx]
            batch_max = batch_closes[max_local_idx]

            min_global_idx = index_glob + min_local_idx
            max_global_idx = index_glob + max_local_idx

            if not is_window_end:
                self._update_window_extrema(
                    batch_min, batch_max,
                    min_local_idx, max_local_idx, index_glob,
                )

            points_batch = []
            added_indices = set()

            points_batch.append((batch_min, min_global_idx, 0))
            added_indices.add(min_global_idx)
            points_batch.append((batch_max, max_global_idx, 1))
            added_indices.add(max_global_idx)

            if self.w_batch_max != float("-inf") and self.w_batch_max_idx not in added_indices:
                points_batch.append((self.w_batch_max, self.w_batch_max_idx, 1))
                added_indices.add(self.w_batch_max_idx)
            if self.w_batch_min != float("inf") and self.w_batch_min_idx not in added_indices:
                points_batch.append((self.w_batch_min, self.w_batch_min_idx, 0))
                added_indices.add(self.w_batch_min_idx)

            for price, x, ptype in points_batch:
                idx = int(x)
                if idx in self.processed_indices:
                    continue
                self.processed_indices.add(idx)
                if idx < start_idx:
                    continue

                self._push_to_stack(price, idx, ptype)
                time_point = int(self.times_np[idx]) if self.times_np is not None else idx
                self._assign_point_to_cluster(price, time_point, idx, ptype)

                new_row = np.array([[price, idx, ptype, 1]], dtype=np.float64)
                self.points = np.vstack([self.points, new_row])

            # breakout check on the processed range
            lo = index_glob
            hi = i
            if hi > lo:
                self._check_breakouts(lo, hi)

            if is_window_end:
                self._reset_window()
                index_glob = i

        # final tail
        lo = index_glob
        if self.num_bars > lo:
            self._check_breakouts(lo, self.num_bars)

        # compute targets for all samples with full forward horizon
        self._finalize_pending_samples()

        # We record the time of the last processed bar.
        if self.times_np is not None and self.num_bars > 0:
            self.last_processed_time = int(self.times_np[-1])

        return self.points