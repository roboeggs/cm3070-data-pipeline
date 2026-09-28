import argparse
import logging
import os
import sys
import time

from dotenv import load_dotenv
from supabase import create_client

# ============================================================
# Logging
# ============================================================

LOG_LEVEL = logging.INFO

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True,
    handlers=[
        logging.FileHandler("app.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)

LOG_MODES = {
    "DOWNLOAD": {
        "src.services.BinanceDownloader": logging.DEBUG,
        "src.services.HistoryBackfill": logging.DEBUG,
        "src.services.download_kline": logging.DEBUG,
        "src.utils.utility": logging.WARNING,
    },
    "DETECTOR": {
        "src.services.HistoryBackfill": logging.INFO,
        "src.detector.FastDetector": logging.DEBUG,
        "src.detector.DetectorSerializer": logging.DEBUG,
    },
    "STORAGE": {
        "src.storage.SupabaseStorage": logging.DEBUG,
        "src.services.HistoryBackfill": logging.INFO,
    },
    "NORMAL": {
        "src.services.BinanceDownloader": logging.INFO,
        "src.services.HistoryBackfill": logging.INFO,
        "src.services.download_kline": logging.INFO,
        "src.detector.FastDetector": logging.WARNING,
        "src.detector.DetectorSerializer": logging.WARNING,
        "src.storage.SupabaseStorage": logging.INFO,
        "src.utils.utility": logging.WARNING,
    },
}

LOG_MODE = "NORMAL"
for module_name, level in LOG_MODES[LOG_MODE].items():
    logging.getLogger(module_name).setLevel(level)

logger = logging.getLogger(__name__)


# ============================================================
# Imports
# ============================================================

from src.configs.config import BASE_PATH, BINANCE_SCRIPT, DATASET_PATH, UPLOAD_TMP_DIR
from src.detector.DetectorSerializer import DetectorSerializer
from src.detector.FastDetector import FastDetector
from src.services.BinanceDownloader import BinanceDownloader
from src.services.HistoryBackfill import HistoryBackfill
from src.storage.SupabaseStorage import SupabaseStorage
from src.utils.utils import get_available_tickers

# ============================================================
# Environment
# ============================================================

load_dotenv()


# ============================================================
# Builders
# ============================================================

def _make_downloader() -> BinanceDownloader:
    return BinanceDownloader(
        BINANCE_SCRIPT,
        UPLOAD_TMP_DIR,
        "1m",
        int(time.time()),
    )


def _make_storage() -> SupabaseStorage:
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SECRET_KEY")
    if not url or not key:
        raise RuntimeError(
            "SUPABASE_URL / SUPABASE_SECRET_KEY are not set in environment"
        )
    return SupabaseStorage(create_client(url, key))


def build_backfill() -> HistoryBackfill:
    """
    Backfill: daily klines + recording the detector state in the database.
    data_path=None load_ticker_data uses BASE_PATH (daily).
    """
    return HistoryBackfill(
        downloader=_make_downloader(),
        storage=_make_storage(),
        detector_class=lambda: FastDetector(),
        serializer_class=DetectorSerializer,
        data_path=None,
    )


def build_dataset() -> HistoryBackfill:
    """
    Dataset: monthly klines, without writing to the database (state is neither read nor written).
    data_path=DATASET_PATH both coins and BTC are read from monthly.
    """
    return HistoryBackfill(
        downloader=_make_downloader(),
        storage=None,
        detector_class=lambda: FastDetector(),
        serializer_class=DetectorSerializer,
        data_path=DATASET_PATH,
    )


# ============================================================
# Mode: backfill  (download + detect + save state to DB)
# ============================================================

def run_backfill():
    logger.info(">>> MODE: backfill (download history + save state to DB)")
    logger.info("    BASE_PATH    = %s", BASE_PATH)
    logger.info("    DATASET_PATH = %s", DATASET_PATH)

    history = build_backfill()
    history.download_history()

    logger.info("<<< MODE: backfill finished")


# ============================================================
# Mode: dataset  (local monthly data only, no DB writes)
# ============================================================

def run_dataset(output_dir: str | None = None):
    logger.info(">>> MODE: dataset (build breakout dataset from local data)")
    logger.info("    DATASET_PATH = %s", DATASET_PATH)

    if not DATASET_PATH.is_dir():
        logger.error("DATASET_PATH does not exist: %s", DATASET_PATH)
        return

    history = build_dataset()

    symbols = get_available_tickers(base_path=str(DATASET_PATH))
    if not symbols:
        logger.error("No local tickers found — nothing to do")
        return

    logger.info("Dataset symbols: %d", len(symbols))

    history.create_dataset(
        symbols=symbols,
        output_dir=output_dir,
        load_state=False,
    )

    logger.info("<<< MODE: dataset finished")


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Crypto history pipeline — backfill or dataset builder"
    )
    parser.add_argument(
        "--mode",
        choices=["backfill", "dataset"],
        default="backfill",
        help=(
            "backfill (default): download history and persist detector state to DB; "
            "dataset: build breakout parquet from local data (no DB writes)"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override output directory for --mode dataset",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.mode == "backfill":
        run_backfill()
    elif args.mode == "dataset":
        run_dataset(output_dir=args.output_dir)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Pipeline failed")
        sys.exit(1)