import joblib
import numpy as np
import pandas as pd

from src.services.BreakoutMLPredictor import BreakoutMLPredictor


class FakeModel:
    def predict_proba(self, frame):
        return np.array([[0.2, 0.8]])


def test_predictor_uses_latest_sample_and_returns_level_id(tmp_path):
    bundle_path = tmp_path / 'bundle.joblib'
    joblib.dump(
        {
            'feature_cols': ['level_type', 'price_position_in_zone'],
            'level_type_mapping': {'support': 1, 'resistance': 0},
            'missing_level_type_value': 'missing',
            'models': {
                'breakout_1h': FakeModel(),
                'breakout_4h': FakeModel(),
                'breakout_24h': FakeModel(),
            },
        },
        bundle_path,
    )

    predictor = BreakoutMLPredictor(bundle_path=bundle_path, thresholds={'breakout_4h': 0.75})
    level = {
        'level_id': 17,
        'level_type': 'support',
        'samples': [
            {
                'prediction_time': '2026-09-01T00:00:00+00:00',
                'level_type': 'support',
                'price_position_in_zone': None,
            },
            {
                'prediction_time': '2026-09-02T00:00:00+00:00',
                'level_type': 'support',
                'price_position_in_zone': 0.8,
            },
        ],
    }

    result = predictor.predict_level('BTCUSDT', level)

    assert result['level_id'] == 17
    assert result['source_prediction_time'] == '2026-09-02T00:00:00+00:00'
    assert result['breakout_4h']['probability'] == 0.8
    assert result['breakout_4h']['is_candidate'] is True


def test_predictor_allows_missing_features_supported_by_lightgbm(tmp_path):
    bundle_path = tmp_path / 'bundle.joblib'
    joblib.dump(
        {
            'feature_cols': ['level_type', 'return_1h'],
            'level_type_mapping': {'support': 1, 'resistance': 0},
            'missing_level_type_value': 'missing',
            'models': {
                'breakout_1h': FakeModel(),
                'breakout_4h': FakeModel(),
                'breakout_24h': FakeModel(),
            },
        },
        bundle_path,
    )

    predictor = BreakoutMLPredictor(bundle_path=bundle_path)
    level = {
        'level_id': 18,
        'level_type': 'support',
        'samples': [{
            'prediction_time': '2026-09-02T00:00:00+00:00',
            'level_type': 'support',
            'return_1h': None,
        }],
    }

    result = predictor.predict_level('PARTIUSDT', level)

    assert result['level_id'] == 18
