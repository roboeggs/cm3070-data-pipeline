import pytest
from unittest.mock import MagicMock
from src.storage.SupabaseStorage import SupabaseStorage


def test_supabase_save_network_error_handling():
    """Handling network error / timeout during save."""
    mock_client = MagicMock()
    # Simulate a network exception when calling execute()
    mock_client.table.return_value.upsert.return_value.execute.side_effect = Exception("Connection Timeout")

    storage = SupabaseStorage(mock_client)

    #  Check that a network failure triggers a controlled exception or does not drop the thread
    with pytest.raises(Exception) as exc_info:
        storage.save_detector_state("BTCUSDT", {"sample": "data"})
    
    assert "Connection Timeout" in str(exc_info.value)


def test_supabase_load_returns_none_on_missing_record():
    """5. Loading a non-existent ticker returns None or an empty result."""
    mock_client = MagicMock()
    # Supabase returns an empty data array
    mock_client.table.return_value.select.return_value.eq.return_value.execute.return_value = MagicMock(
        data=[]
    )

    storage = SupabaseStorage(mock_client)
    result = storage.load_detector_state("UNKNOWN_TICKER")

    assert result is None or result == {}