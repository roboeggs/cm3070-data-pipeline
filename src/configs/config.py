from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# --- service directories ---
UPLOAD_TMP_DIR = PROJECT_ROOT / "upload"
MODEL_BUNDLE_PATH = PROJECT_ROOT / "artifacts" / "backend_model_bundle.joblib"

# --- Binance data ---
BASE_PATH    = UPLOAD_TMP_DIR / "data" / "spot" / "daily"   / "klines"
DATASET_PATH = UPLOAD_TMP_DIR / "data" / "spot" / "monthly" / "klines"

# --- results ---
OUTPUT_PATH  = UPLOAD_TMP_DIR / "level_data"
INDEX_PATH   = UPLOAD_TMP_DIR / "btc_index"

#  --- external loader script ---
BINANCE_SCRIPT = PROJECT_ROOT / "binance-public-data" / "python" / "download-kline.py"

__all__ = [
    "PROJECT_ROOT", "UPLOAD_TMP_DIR",
    "BASE_PATH", "DATASET_PATH",
    "MODEL_BUNDLE_PATH",
    "OUTPUT_PATH", "INDEX_PATH",
    "BINANCE_SCRIPT",
]