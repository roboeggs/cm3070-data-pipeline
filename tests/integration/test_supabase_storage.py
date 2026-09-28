import pytest
from unittest.mock import MagicMock
from src.detector.FastDetector import FastDetector
from src.detector.DetectorSerializer import DetectorSerializer
from src.storage.SupabaseStorage import SupabaseStorage


@pytest.fixture
def mock_supabase_client():
    mock = MagicMock()
    # Configuring the call chain for saving (upsert)
    mock.table.return_value.upsert.return_value.execute.return_value = MagicMock(
        data=[{"status": "ok"}]
    )
    
    # Configuring the call chain for loading (select -> eq -> execute)
    mock.table.return_value.select.return_value.eq.return_value.execute.return_value = MagicMock(
        data=[{"ticker": "BTCUSDT", "state_blob": {"_level_id_seq": 10}}]
    )
    return mock


def test_save_detector_state_calls_supabase(mock_supabase_client, populated_detector):
    """Verification: The save call correctly formats and sends data to Supabase."""
    storage = SupabaseStorage(mock_supabase_client)

    state_dict = DetectorSerializer.serialize(populated_detector)
    storage.save_detector_state("BTCUSDT", state_dict)

    # Specifying the exact table name used in SupabaseStorage
    mock_supabase_client.table.assert_called_with("detector_state")


def test_load_detector_state_and_reconstruct(mock_supabase_client):
    """Verification: Loading the dictionary from Supabase correctly restores the FastDetector instance."""
    storage = SupabaseStorage(mock_supabase_client)

    record = storage.load_detector_state("BTCUSDT")
    assert record is not None
    assert "state_blob" in record

    detector = DetectorSerializer.deserialize(record["state_blob"], FastDetector)
    assert isinstance(detector, FastDetector)
    assert detector._level_id_seq == 10