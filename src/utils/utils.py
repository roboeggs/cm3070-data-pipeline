from src.configs.config import BASE_PATH, OUTPUT_PATH, INDEX_PATH, DATASET_PATH
import warnings
import zipfile
import glob
import os
import io
from typing import  List, Optional, Iterable
from tqdm import tqdm

import pandas as pd

import logging

logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore")


os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(INDEX_PATH, exist_ok=True)

# Required columns
USE_COLS = ["open_time", "open", "high", "low", "close", "volume", "quote_volume"]

# Full list of columns (as in the original CSV)
ALL_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "count", "taker_base", "taker_quote", "ignore"
]


def load_ticker_data(ticker: str, usecols: Optional[List[str]] = None,
    base_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Loads ticker kline data from local files, retaining only the required columns.

    Supports all layouts produced by Binance public data:

        daily   (zip):    base/{TICKER}/1m/YYYY-MM-DD_YYYY-MM-DD/{TICKER}-1m-YYYY-MM-DD.zip
        monthly (zip):    base/{TICKER}/1m/{TICKER}-1m-YYYY-MM.zip
        monthly (csv):    base/{TICKER}/1m/YYYY-MM-DD_YYYY-MM-DD/{TICKER}-1m-YYYY-MM/{TICKER}-1m-YYYY-MM.csv
        monthly (csv):    base/{TICKER}/1m/{TICKER}-1m-YYYY-MM/{TICKER}-1m-YYYY-MM.csv

    Detects a header row automatically (monthly CSVs have one, daily zips don't).
    """
    if usecols is None:
        usecols = USE_COLS
    if base_path is None:
        base_path = BASE_PATH

    base_path = str(base_path)
    col_indices = [ALL_COLUMNS.index(c) for c in usecols]

    ticker_dir = os.path.join(base_path, ticker)
    if not os.path.isdir(ticker_dir):
        logger.warning("Ticker directory not found: %s", ticker_dir)
        return pd.DataFrame(columns=usecols)

    # Recursively find both .zip and .csv, keep only 1m files of this ticker
    candidates = []
    for ext in ("*.zip", "*.csv"):
        candidates.extend(
            glob.glob(os.path.join(ticker_dir, "**", ext), recursive=True)
        )

    files = sorted(
        p for p in candidates
        if f"{ticker}-1m-" in os.path.basename(p)
    )

    if not files:
        logger.warning(
            "No 1m zip/csv files found for %s under %s", ticker, ticker_dir
        )
        return pd.DataFrame(columns=usecols)

    logger.info(
        "[%s] Found %d files. Reading %d columns...",
        ticker, len(files), len(usecols),
    )

    chunks = []
    for path in tqdm(files, desc=f"Loading {ticker}", leave=False):
        try:
            # --- read raw text from zip or plain csv ---
            if path.lower().endswith(".zip"):
                with zipfile.ZipFile(path, "r") as zf:
                    csv_name = next(
                        (n for n in zf.namelist() if n.lower().endswith(".csv")),
                        None,
                    )
                    if csv_name is None:
                        logger.warning("No CSV inside %s", path)
                        continue
                    with zf.open(csv_name) as f:
                        text = f.read().decode("utf-8", errors="replace")
            else:
                with open(path, "rb") as f:
                    text = f.read().decode("utf-8", errors="replace")

            if not text.strip():
                continue

            # --- detect header on the first non-empty line ---
            first_line = next(
                (ln for ln in text.splitlines() if ln.strip()), ""
            )
            first_cell = first_line.split(",", 1)[0].strip().strip('"')
            has_header = not first_cell.lstrip("-").isdigit()

            # --- read the file ---
            if has_header:
                raw = pd.read_csv(io.StringIO(text))
                # Take first 12 columns positionally, rename to canonical names
                raw = raw.iloc[:, : len(ALL_COLUMNS)]
                raw.columns = ALL_COLUMNS
                df = raw.iloc[:, col_indices].copy()
            else:
                df = pd.read_csv(
                    io.StringIO(text),
                    header=None,
                    names=ALL_COLUMNS,
                    usecols=col_indices,
                )

            # --- clean open_time ---
            df["open_time"] = pd.to_numeric(df["open_time"], errors="coerce")
            df = df.dropna(subset=["open_time"])

            df["open_time"] = df["open_time"].apply(
                lambda x: x * 1000 if x < 1e13 else x
            )
            df["open_time"] = pd.to_datetime(df["open_time"], unit="us")

            chunks.append(df)

        except Exception as e:
            logger.error(
                "Failed to process %s: %s", path, e, exc_info=True
            )

    if not chunks:
        return pd.DataFrame(columns=usecols)

    full_df = pd.concat(chunks, ignore_index=True)

    numeric_cols = [c for c in usecols if c != "open_time"]
    full_df[numeric_cols] = full_df[numeric_cols].apply(
        pd.to_numeric, errors="coerce"
    )

    full_df = full_df.sort_values("open_time").reset_index(drop=True)

    logger.info(
        "[%s] Successfully loaded %s rows", ticker, f"{len(full_df):,}"
    )
    return full_df


def get_available_tickers(base_path: str = DATASET_PATH, suffix: str = "USDT",
    exclude: Iterable[str] = ("BTCUSDT",),
) -> List[str]:
    
    """
    Scan `base_path` and return sorted list of ticker folders
    matching `suffix`, excluding those in `exclude`.

    Directory structure is expected to be:

        base_path/{TICKER}/...

    Parameters
    ----------
    base_path : str
        Root directory containing ticker subfolders.
    suffix : str
        Ticker suffix filter (default "USDT").
    exclude : Iterable[str]
        Tickers to skip (default excludes BTCUSDT).

    Returns
    -------
    List[str]
        Sorted list of ticker names.
    """
    if not os.path.isdir(base_path):
        logger.warning("Base path does not exist: %s", base_path)
        return []

    exclude_set = set(exclude)

    tickers = [
        item
        for item in os.listdir(base_path)
        if item.endswith(suffix)
        and item not in exclude_set
        and os.path.isdir(os.path.join(base_path, item))
    ]

    logger.info(
        "Found %d ticker folders in %s (suffix=%s, excluded=%d)",
        len(tickers), base_path, suffix, len(exclude_set),
    )

    return sorted(tickers)


def save_breakout_dataset(ticker: str, breakout_list,
    output_dir: str,
) -> pd.DataFrame:
    
    """
    Save breakout rows for a single ticker to a parquet file.
    Returns the (possibly empty) DataFrame that was written.
    """
    if breakout_list is None or len(breakout_list) == 0:
        return pd.DataFrame()

    df = pd.DataFrame(breakout_list)

    if 'symbol' not in df.columns:
        df['symbol'] = ticker
    df['ticker'] = ticker

    if 'end_idx' in df.columns and 'breakout_idx' not in df.columns:
        df.rename(columns={'end_idx': 'breakout_idx'}, inplace=True)

    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f"{ticker}_breakout_levels.parquet")
    df.to_parquet(filepath, index=False)
    logger.debug("Saved %d breakout rows for %s -> %s", len(df), ticker, filepath)
    return df


# ---------------------------------------------------------
# BTC cache (raw OHLCV snapshot for dataset creation)
# ---------------------------------------------------------

def save_btc_cache(df: pd.DataFrame, path: str) -> None:
    """Write raw BTC dataframe to parquet cache."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False)
    logger.info("BTC cache written: %s (%d rows)", path, len(df))


def load_btc_cache(path: str) -> Optional[pd.DataFrame]:
    """Read BTC cache if it exists, else None."""
    if not os.path.isfile(path):
        return None
    df = pd.read_parquet(path)
    logger.info("BTC cache loaded: %s (%d rows)", path, len(df))
    return df