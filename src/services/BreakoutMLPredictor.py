import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.configs.config import PROJECT_ROOT


logger = logging.getLogger(__name__)


class BreakoutMLPredictor:
    """Run the exported breakout models on the latest level snapshot."""

    DEFAULT_BUNDLE_PATH = PROJECT_ROOT / "artifacts" / "backend_model_bundle.joblib"

    def __init__(self, bundle_path=None, thresholds=None):
        self.bundle_path = Path(bundle_path or self.DEFAULT_BUNDLE_PATH)
        self.thresholds = {
            "breakout_1h": 0.30,
            "breakout_4h": 0.50,
            "breakout_24h": 0.80,
        }
        if thresholds:
            self.thresholds.update(thresholds)
        self.bundle = None

    def _load(self):
        if self.bundle is None:
            if not self.bundle_path.exists():
                raise FileNotFoundError(
                    f"Breakout model bundle was not found: {self.bundle_path}"
                )
            self.bundle = joblib.load(self.bundle_path)
            logger.info(
                "ML bundle loaded: path=%s targets=%s features=%d thresholds=%s",
                self.bundle_path,
                ",".join(self.bundle.get("targets", self.bundle.get("models", {}).keys())),
                len(self.bundle.get("feature_cols", [])),
                self.thresholds,
            )
        return self.bundle

    @staticmethod
    def _latest_sample(level):
        samples = [sample for sample in level.get("samples", []) if isinstance(sample, dict)]
        if not samples:
            return None

        def sample_time(sample):
            value = sample.get("prediction_time") or sample.get("last_touch_time")
            try:
                if isinstance(value, (int, float, np.integer, np.floating)):
                    return pd.to_datetime(value, unit="s", utc=True).timestamp()
                return pd.Timestamp(value).timestamp()
            except (TypeError, ValueError):
                return float("-inf")

        return max(enumerate(samples), key=lambda item: (sample_time(item[1]), item[0]))[1]

    def _prepare_features(self, sample):
        bundle = self._load()
        row = dict(sample)
        level_type = str(row.get("level_type", "missing"))
        row["level_type"] = bundle["level_type_mapping"].get(level_type, -1)

        position = pd.to_numeric(row.get("price_position_in_zone"), errors="coerce")
        row["is_degenerate_zone"] = int(pd.isna(position))
        row["price_position_in_zone"] = float(np.clip(position, 0.0, 1.0)) if pd.notna(position) else 0.5

        feature_cols = bundle["feature_cols"]
        missing = [column for column in feature_cols if column not in row]
        if missing:
            raise ValueError(f"Missing model features: {', '.join(missing)}")

        features = pd.DataFrame([{column: row[column] for column in feature_cols}], columns=feature_cols)
        for column in feature_cols:
            features[column] = pd.to_numeric(features[column], errors="coerce")
        non_finite = np.isinf(features.to_numpy(dtype="float64"))
        if non_finite.any():
            invalid = features.columns[non_finite.any(axis=0)].tolist()
            raise ValueError(f"Invalid model features: {invalid}")
        return features

    def predict_level(self, ticker, level):
        sample = self._latest_sample(level)
        if sample is None:
            logger.warning("No samples available for %s level %s", ticker, level.get("level_id"))
            return None

        features = self._prepare_features(sample)
        predictions = {}
        for target, model in self._load()["models"].items():
            probability = float(model.predict_proba(features)[0, 1])
            predictions[target] = {
                "probability": probability,
                "threshold": float(self.thresholds.get(target, 0.5)),
                "is_candidate": probability >= self.thresholds.get(target, 0.5),
            }

        recommended = predictions["breakout_4h"]
        return {
            "model_version": self.bundle_path.name,
            "source_prediction_time": sample.get("prediction_time"),
            "level_id": level.get("level_id"),
            "level_type": level.get("level_type"),
            "breakout_1h": predictions["breakout_1h"],
            "breakout_4h": predictions["breakout_4h"],
            "breakout_24h": predictions["breakout_24h"],
            "recommended_horizon": "breakout_4h",
            "is_candidate": bool(recommended["is_candidate"]),
        }

    def predict_levels(self, ticker, levels):
        logger.info(
            "ML scoring started: ticker=%s active_levels=%d",
            ticker,
            len(levels),
        )
        predictions = []
        for level in levels:
            try:
                prediction = self.predict_level(ticker, level)
            except (TypeError, ValueError, KeyError) as error:
                logger.exception("ML prediction failed for %s level %s: %s", ticker, level.get("level_id"), error)
                continue
            if prediction is not None:
                predictions.append(prediction)
                if prediction["is_candidate"]:
                    logger.info(
                        "ML candidate: ticker=%s level_id=%s type=%s "
                        "p1h=%.4f p4h=%.4f p24h=%.4f",
                        ticker,
                        prediction["level_id"],
                        prediction["level_type"],
                        prediction["breakout_1h"]["probability"],
                        prediction["breakout_4h"]["probability"],
                        prediction["breakout_24h"]["probability"],
                    )

        candidates = [item for item in predictions if item["is_candidate"]]
        logger.info(
            "ML scoring finished: ticker=%s scored=%d candidates=%d candidate_level_ids=%s",
            ticker,
            len(predictions),
            len(candidates),
            [item["level_id"] for item in candidates],
        )
        return predictions