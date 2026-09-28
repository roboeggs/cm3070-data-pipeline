# src/detector/DetectorSerializer.py

import json
import math
from datetime import datetime, timezone

import numpy as np


class DetectorSerializer:
    """
    Serialization of the FastDetector state.

    The following is saved in the database:
        - last_processed_time
        - _level_id_seq
        - stack_max / stack_min
        - support_clusters / resistance_clusters
        - confirmed_support_levels / confirmed_resistance_levels
        - samples — snapshots for ML

    Not saved:
      - *_tail arrays
      - levels (restored from confirmed_*)
      - w_batch_* states
      - touch_indices inside levels/clusters
        (restored from touch_times)
      - internal _prediction_idx inside samples
    """

    _LEVEL_FIELDS = (
        'level_id',
        'type',
        'center',
        'zone_low',
        'zone_high',
        'count',
        'first_touch_time',
        'last_touch_time',
    )

    _CLUSTER_FIELDS = (
        'type',
        'sum',
        'count',
        'center',
        'zone_low',
        'zone_high',
    )

    _ISO_FMT = '%Y-%m-%dT%H:%M:%S.%f'

    # ---------------------------------------------------------
    # Time conversion
    # ---------------------------------------------------------

    @staticmethod
    def _to_seconds(v):
        """
        Converts numeric time to epoch seconds.

        < 1e11        -> seconds
        1e11..1e14    -> milliseconds
        1e14..1e17    -> microseconds
        >= 1e17       -> nanoseconds
        """
        if v is None:
            return None

        try:
            iv = int(v)
        except (TypeError, ValueError):
            return None

        av = abs(iv)

        if av < 10**11:
            return iv

        if av < 10**14:
            return iv // 10**3

        if av < 10**17:
            return iv // 10**6

        return iv // 10**9

    @staticmethod
    def _to_epoch_seconds(v):
        if v is None:
            return None

        if isinstance(v, (int, float, np.integer, np.floating)):
            return DetectorSerializer._to_seconds(v)

        if isinstance(v, datetime):
            if v.tzinfo is None:
                v = v.replace(tzinfo=timezone.utc)
            return int(v.timestamp())

        if isinstance(v, str):
            try:
                value = v.strip()

                if value.endswith("Z"):
                    value = value[:-1] + "+00:00"

                dt = datetime.fromisoformat(value)

                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)

                return int(dt.timestamp())

            except ValueError:
                return DetectorSerializer._to_seconds(v)

        return None

    @staticmethod
    def _sec_to_iso(v):
        if v is None:
            return None

        sec = DetectorSerializer._to_epoch_seconds(v)

        if sec is None:
            return None

        return datetime.fromtimestamp(
            sec,
            tz=timezone.utc
        ).isoformat()

    # ---------------------------------------------------------
    # JSON conversion
    # ---------------------------------------------------------

    @staticmethod
    def _to_py(v):
        if v is None:
            return None

        if isinstance(v, np.bool_):
            return bool(v)

        if isinstance(v, np.integer):
            return int(v)

        if isinstance(v, np.floating):
            f = float(v)
            return None if (math.isnan(f) or math.isinf(f)) else f

        if isinstance(v, bool):
            return v

        if isinstance(v, int):
            return v

        if isinstance(v, float):
            return None if (math.isnan(v) or math.isinf(v)) else v

        if isinstance(v, (list, tuple)):
            return [
                DetectorSerializer._to_py(x)
                for x in v
            ]

        if isinstance(v, dict):
            return {
                str(k): DetectorSerializer._to_py(value)
                for k, value in v.items()
            }

        return v

    # ---------------------------------------------------------
    # Level serialization
    # ---------------------------------------------------------

    @staticmethod
    def _slim_level(lvl):
        slim = {}

        for k in DetectorSerializer._LEVEL_FIELDS:
            value = lvl.get(k)

            if k in (
                'first_touch_time',
                'last_touch_time',
            ):
                slim[k] = DetectorSerializer._sec_to_iso(value)
            else:
                slim[k] = DetectorSerializer._to_py(value)

        slim['touch_times'] = [
            DetectorSerializer._sec_to_iso(t)
            for t in lvl.get('touch_times', [])
        ]

        ml = lvl.get('ml_features')

        if isinstance(ml, dict):
            slim['ml_features'] = DetectorSerializer._to_py(ml)

        return slim

    @staticmethod
    def _slim_cluster(c):
        slim = {
            k: DetectorSerializer._to_py(c.get(k))
            for k in DetectorSerializer._CLUSTER_FIELDS
        }

        slim['touch_times'] = [
            DetectorSerializer._sec_to_iso(t)
            for t in c.get('touch_times', [])
        ]

        return slim

    # ---------------------------------------------------------
    # Sample serialization
    # ---------------------------------------------------------

    @staticmethod
    def _serialize_sample(sample):
        """
        Saves the ML snapshot.

        _prediction_idx is NOT saved:
        this is an index within the specific array of candles of the current run
        and after a restart, it is no longer a valid index.

        prediction_time is saved — this is a stable identifier
        of the moment the snapshot was created.
        """

        out = {}

        for key, value in sample.items():

            # We do not save the internal fields of the detector.
            if key.startswith("_"):
                continue

            # List of touch timestamps.
            if key == "touch_times":
                if isinstance(value, list):
                    out[key] = [
                        DetectorSerializer._sec_to_iso(t)
                        for t in value
                    ]
                else:
                    out[key] = []

                continue

            # Separate timestamp fields.
            if key in (
                "prediction_time",
                "first_touch_time",
                "last_touch_time",
            ):
                out[key] = DetectorSerializer._sec_to_iso(value)
                continue

            out[key] = DetectorSerializer._to_py(value)

        return out

    @staticmethod
    def _serialize_samples(samples):
        if not samples:
            return []

        return [
            DetectorSerializer._serialize_sample(sample)
            for sample in samples
            if isinstance(sample, dict)
        ]

    @staticmethod
    def _serialize_level_with_samples(lvl, ml_prediction=None):
        """JSON-safe conversion of the level collected by FastDetector."""
        level_type = lvl.get("level_type")
        if level_type is None:
            level_type = "support" if lvl.get("type") == 0 else "resistance"

        touch_count = lvl.get("touch_count")
        if touch_count is None:
            touch_count = lvl.get("count")

        serialized = {
            "level_id":         DetectorSerializer._to_py(lvl.get("level_id")),
            "level_type":       level_type,
            "type":             DetectorSerializer._to_py(lvl.get("type")),
            "center":           DetectorSerializer._to_py(lvl.get("center")),
            "zone_low":         DetectorSerializer._to_py(lvl.get("zone_low")),
            "zone_high":        DetectorSerializer._to_py(lvl.get("zone_high")),
            "first_touch_time": DetectorSerializer._sec_to_iso(lvl.get("first_touch_time")),
            "last_touch_time":  DetectorSerializer._sec_to_iso(lvl.get("last_touch_time")),
            "touch_count":      DetectorSerializer._to_py(touch_count),
            "count":            DetectorSerializer._to_py(lvl.get("count", touch_count)),
            "touch_times": [
                DetectorSerializer._sec_to_iso(t)
                for t in lvl.get("touch_times", [])
            ],
            "ml_features": DetectorSerializer._to_py(lvl.get("ml_features")),
            "samples": [
                DetectorSerializer._serialize_sample(s)
                for s in (lvl.get("samples") or [])
                if isinstance(s, dict)
            ],
        }
        if ml_prediction is not None:
            serialized['ml_prediction'] = DetectorSerializer._to_py(ml_prediction)
        return serialized

    # ---------------------------------------------------------
    # Sample deserialization
    # ---------------------------------------------------------

    @staticmethod
    def _deserialize_sample(sample):
        """
        Restores the snapshot after loading from the database.

        _prediction_idx is NOT restored here.

        It refers to the old candles array and should not be used
        after a new run of detector.process().
        """

        if not isinstance(sample, dict):
            return None

        restored = dict(sample)

        for key in (
            "prediction_time",
            "first_touch_time",
            "last_touch_time",
        ):
            if key in restored:
                restored[key] = DetectorSerializer._to_epoch_seconds(
                    restored[key]
                )

        if isinstance(restored.get("touch_times"), list):
            restored["touch_times"] = [
                DetectorSerializer._to_epoch_seconds(t)
                for t in restored["touch_times"]
            ]

        return restored

    @staticmethod
    def _deserialize_samples(raw_samples):
        if not raw_samples:
            return []

        result = []

        for sample in raw_samples:
            restored = DetectorSerializer._deserialize_sample(sample)

            if restored is not None:
                result.append(restored)

        return result

    @staticmethod
    def _deserialize_level_with_samples(lvl):
        """
        Restores:
        - level: internal dict for detector.confirmed_*_levels
        - samples: flat list (with touch_number removed)
        """
        if not isinstance(lvl, dict):
            return None, []

        ltype  = 0 if lvl.get("level_type") == "support" else 1
        lid    = int(lvl.get("level_id") or 0)
        center = float(lvl.get("center"))
        count  = int(lvl.get("touch_count") or 0)

        touch_times = []
        for t in (lvl.get("touch_times") or []):
            sec = DetectorSerializer._to_epoch_seconds(t)
            if sec is not None:
                touch_times.append(sec)

        # if touch_times hasn’t been explicitly provided, we collect from samples
        if not touch_times:
            for s in (lvl.get("samples") or []):
                sec = DetectorSerializer._to_epoch_seconds(s.get("prediction_time"))
                if sec is not None:
                    touch_times.append(sec)

        level = {
            "level_id":         lid,
            "type":             ltype,
            "sum":              center * count if count > 0 else center,
            "count":            count,
            "center":           center,
            "zone_low":         float(lvl.get("zone_low")),
            "zone_high":        float(lvl.get("zone_high")),
            "first_touch_time": DetectorSerializer._to_epoch_seconds(lvl.get("first_touch_time")),
            "last_touch_time":  DetectorSerializer._to_epoch_seconds(lvl.get("last_touch_time")),
            "touch_times":      touch_times,
            "touch_indices":    [],
            "end_idx":          None,
        }

        if isinstance(lvl.get("ml_features"), dict):
            level["ml_features"] = DetectorSerializer._to_py(lvl["ml_features"])

        samples = []
        for s in (lvl.get("samples") or []):
            if not isinstance(s, dict):
                continue
            restored = DetectorSerializer._deserialize_sample(s)
            if restored is None:
                continue
            restored.pop("touch_number", None)
            # We guarantee the presence of target fields.
            restored.setdefault("breakout_1h",  None)
            restored.setdefault("breakout_4h",  None)
            restored.setdefault("breakout_24h", None)
            samples.append(restored)

        return level, samples

    # ---------------------------------------------------------
    # Serialize
    # ---------------------------------------------------------

    @staticmethod
    def serialize(detector, ml_predictions=None) -> dict:

        def _pairs(raw):
            out = []

            for item in (raw or []):

                if isinstance(item, (list, tuple)) and len(item) == 2:

                    out.append([
                        DetectorSerializer._to_py(item[0]),
                        DetectorSerializer._sec_to_iso(item[1]),
                    ])

            return out

        predictions = ml_predictions or []
        prediction_by_key = {
            (item.get('level_type'), item.get('level_id')): item
            for item in predictions
            if isinstance(item, dict)
        }
        exported_levels = detector.get_confirmed_levels_export()

        def _enrich_levels(exported, internal):
            internal_by_id = {
                level.get('level_id'): level
                for level in (internal or [])
            }
            enriched = []
            for level in exported:
                source = internal_by_id.get(level.get('level_id'), {})
                merged = dict(source)
                merged.update(level)
                if 'touch_times' not in level:
                    merged['touch_times'] = source.get('touch_times', [])
                if 'ml_features' not in level:
                    merged['ml_features'] = source.get('ml_features')
                enriched.append(merged)
            return enriched

        support_levels = _enrich_levels(
            exported_levels["support"],
            getattr(detector, 'confirmed_support_levels', []),
        )
        resistance_levels = _enrich_levels(
            exported_levels["resistance"],
            getattr(detector, 'confirmed_resistance_levels', []),
        )
        state = {
            # ---------------------------------------------
            # Detector continuation state
            # ---------------------------------------------

            'last_processed_time':
                DetectorSerializer._sec_to_iso(
                    getattr(
                        detector,
                        'last_processed_time',
                        None
                    )
                ),

            '_level_id_seq':
                int(
                    getattr(
                        detector,
                        '_level_id_seq',
                        0
                    )
                ),

            'stack_max':
                _pairs(
                    getattr(
                        detector,
                        'stack_max',
                        []
                    )
                ),

            'stack_min':
                _pairs(
                    getattr(
                        detector,
                        'stack_min',
                        []
                    )
                ),

            'support_clusters': [
                DetectorSerializer._slim_cluster(c)
                for c in getattr(
                    detector,
                    'support_clusters',
                    []
                )
            ],

            'resistance_clusters': [
                DetectorSerializer._slim_cluster(c)
                for c in getattr(
                    detector,
                    'resistance_clusters',
                    []
                )
            ],

            'confirmed_support_levels': [
                DetectorSerializer._serialize_level_with_samples(
                    l,
                    prediction_by_key.get(('support', l.get('level_id'))),
                )
                for l in support_levels
            ],

            'confirmed_resistance_levels': [
                DetectorSerializer._serialize_level_with_samples(
                    l,
                    prediction_by_key.get(('resistance', l.get('level_id'))),
                )
                for l in resistance_levels
            ],

            'ml_breakout_predictions': predictions,
            'ml_candidate_level_ids': [
                item.get('level_id')
                for item in predictions
                if item.get('is_candidate')
            ],

        }

        return json.loads(
            json.dumps(
                state,
                default=str
            )
        )

    # ---------------------------------------------------------
    # Deserialize
    # ---------------------------------------------------------

    @staticmethod
    def deserialize(state_dict, detector_class):

        if not isinstance(state_dict, dict):
            raise ValueError(
                "Invalid or corrupted state_dict format"
            )

        if '_level_id_seq' not in state_dict:
            raise ValueError(
                "Invalid or corrupted state_dict format"
            )

        detector = detector_class()

        # ---------------------------------------------
        # Basic state
        # ---------------------------------------------

        detector.last_processed_time = (
            DetectorSerializer._to_epoch_seconds(
                state_dict.get(
                    'last_processed_time'
                )
            )
        )

        detector._level_id_seq = int(
            state_dict.get(
                '_level_id_seq',
                0
            )
        )

        # ---------------------------------------------
        # Stacks
        # ---------------------------------------------

        def _pair_list(raw):
            out = []

            for item in (raw or []):

                if (
                    isinstance(item, (list, tuple))
                    and len(item) == 2
                ):
                    price, t = item

                    sec = DetectorSerializer._to_epoch_seconds(t)

                    if sec is None:
                        continue

                    out.append(
                        (
                            float(price),
                            sec
                        )
                    )

            return out

        detector.stack_max = _pair_list(
            state_dict.get(
                'stack_max'
            )
        )

        detector.stack_min = _pair_list(
            state_dict.get(
                'stack_min'
            )
        )

        # ---------------------------------------------
        # Touch times
        # ---------------------------------------------

        def _norm_times(raw):
            out = []

            for t in (raw or []):

                sec = DetectorSerializer._to_epoch_seconds(t)

                if sec is not None:
                    out.append(sec)

            return out

        # ---------------------------------------------
        # Clusters
        # ---------------------------------------------

        def _restore_cluster(c):

            restored = {
                k: DetectorSerializer._to_py(
                    c.get(k)
                )
                for k in DetectorSerializer._CLUSTER_FIELDS
            }

            restored['touch_times'] = _norm_times(
                c.get('touch_times')
            )

            # We will restore the indices later.
            restored['touch_indices'] = []

            return restored

        detector.support_clusters = [
            _restore_cluster(c)
            for c in state_dict.get(
                'support_clusters',
                []
            )
        ]

        detector.resistance_clusters = [
            _restore_cluster(c)
            for c in state_dict.get(
                'resistance_clusters',
                []
            )
        ]

        # ---------------------------------------------
        # Confirmed levels
        # ---------------------------------------------

        def _restore_level(level):

            restored = {
                k: DetectorSerializer._to_py(
                    level.get(k)
                )
                for k in DetectorSerializer._LEVEL_FIELDS
            }

            restored['touch_times'] = _norm_times(
                level.get('touch_times')
            )

            restored['touch_indices'] = []

            ml = level.get('ml_features')

            if isinstance(ml, dict):
                restored['ml_features'] = (
                    DetectorSerializer._to_py(ml)
                )

            return restored

        detector.confirmed_support_levels = []
        detector.confirmed_resistance_levels = []
        all_samples = []

        for raw in state_dict.get('confirmed_support_levels', []):
            lvl, samples = DetectorSerializer._deserialize_level_with_samples(raw)
            if lvl is not None:
                detector.confirmed_support_levels.append(lvl)
            all_samples.extend(samples)

        for raw in state_dict.get('confirmed_resistance_levels', []):
            lvl, samples = DetectorSerializer._deserialize_level_with_samples(raw)
            if lvl is not None:
                detector.confirmed_resistance_levels.append(lvl)
            all_samples.extend(samples)

        detector.levels = (
            detector.confirmed_support_levels
            + detector.confirmed_resistance_levels
        )

        detector.samples = all_samples


        detector.pending_by_level_id = {}

        for sample in detector.samples:

            level_id = sample.get(
                'level_id'
            )

            if level_id is None:
                continue

            detector.pending_by_level_id.setdefault(
                level_id,
                []
            ).append(sample)

        return detector