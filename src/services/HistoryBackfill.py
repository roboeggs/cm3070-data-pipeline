import time
import json
import logging
import pandas as pd
import numpy as np
import os

from datetime import datetime, timezone

from ..utils.utility import get_all_symbols
from src.utils.utils import load_ticker_data, save_breakout_dataset, save_btc_cache, load_btc_cache
from src.configs.config import OUTPUT_PATH, INDEX_PATH, MODEL_BUNDLE_PATH
from src.services.BreakoutMLPredictor import BreakoutMLPredictor
from src.services.LevelsPublisher import LevelsPublisher



logger = logging.getLogger(__name__)


class HistoryBackfill:

    def __init__(
        self,
        downloader,
        storage,
        detector_class,
        serializer_class,
        symbols_provider=None,
        data_path=None,
        model_bundle_path=MODEL_BUNDLE_PATH,
        ml_thresholds=None,
    ):
        self.downloader = downloader
        self.storage = storage
        self.detector_class = detector_class
        self.serializer_class = serializer_class
        self.summary_publisher = LevelsPublisher(storage=storage)

        if symbols_provider is None:
            symbols_provider = self._get_all_symbols_usdt
        self.symbols = symbols_provider()

        self.data_path = data_path
        self.ml_predictor = BreakoutMLPredictor(
            bundle_path=model_bundle_path,
            thresholds=ml_thresholds,
        )
        self.btc_cache_path = os.path.join(str(INDEX_PATH), "BTCUSDT.parquet")

        self.limit_date = datetime(
            2026,9,1,
            tzinfo=timezone.utc
        )

        self.btc_df = pd.DataFrame()

        logger.debug( "HistoryBackfill initialized: %d USDT symbols", len(self.symbols) )

        logger.debug( "History limit date: %s", self.limit_date )

    # ---------------------------------------------------------
    # Get symbols
    # ---------------------------------------------------------

    def _get_all_symbols_usdt(self):

        logger.debug("Fetching all spot symbols from Binance")

        symbols = get_all_symbols("spot")

        logger.debug( "Received %d symbols from Binance", len(symbols) )

        usdt_symbols = [ s for s in symbols if s.endswith("USDT") ]

        logger.info( "Found %d USDT symbols", len(usdt_symbols) )

        return usdt_symbols

    # ---------------------------------------------------------
    # Synchronize ticker creation/update time
    # ---------------------------------------------------------

    def _sync_tickers_creation_time(self):

        logger.debug( "Starting ticker synchronization" )

        created = self.storage.get_processing_log()

        logger.debug( "Loaded processing log: %d tickers", len(created) )

        current_time = int(time.time())

        limit_ts = int( self.limit_date.timestamp() )

        exist_tickers = {}
        new_tickers = {}

        skipped_tickers = 0

        for ticker in self.symbols:

            ts = created.get(ticker)

            # -------------------------------------------------
            # Already processed recently
            # -------------------------------------------------

            if ts is not None and (current_time - ts) < 86400:

                skipped_tickers += 1

                logger.debug( "Skipping %s: processed less than 24h ago", ticker )

                continue

            # -------------------------------------------------
            # Existing ticker
            # -------------------------------------------------

            if ts is not None and ts != 0:

                exist_tickers[ticker] = ts

                logger.debug( "Existing ticker %s will be updated from timestamp %s", ticker, ts )

            # -------------------------------------------------
            # New ticker
            # -------------------------------------------------

            else:

                new_tickers[ticker] = limit_ts

                logger.debug( "New ticker %s will be downloaded from %s", ticker, self.limit_date )

        # -----------------------------------------------------
        # Determine BTC start timestamp
        # -----------------------------------------------------

        if new_tickers:

            min_ts = limit_ts

            logger.debug( "New tickers detected: BTC download starts from limit timestamp %s", min_ts )

        elif exist_tickers:

            min_ts = min( exist_tickers.values() )

            logger.debug( "No new tickers. BTC download starts from earliest existing timestamp %s", min_ts)

        else:

            min_ts = limit_ts

            logger.debug( "No tickers require update. Using limit timestamp %s", min_ts )

        logger.info(
            "Ticker synchronization complete: "
            "existing=%d, new=%d, skipped=%d, btc_start=%s",
            len(exist_tickers),
            len(new_tickers),
            skipped_tickers,
            min_ts
        )

        return (
            exist_tickers,
            new_tickers,
            min_ts
        )

    # ---------------------------------------------------------
    # BTC metrics
    # ---------------------------------------------------------

    def _get_btc_metrics_for_coin(self, coin_df):

        logger.debug( "Synchronizing BTC metrics: coin rows=%d, BTC rows=%d", len(coin_df), len(self.btc_df) )


        btc_indexed = self.btc_df.set_index( 'open_time')

        mask = coin_df['open_time'].isin( self.btc_df['open_time'] )

        coin_df_filtered = coin_df[mask].reset_index(drop=True)

        btc_slice = btc_indexed.loc[ coin_df_filtered['open_time'] ]

        logger.debug( "Coin rows after BTC synchronization: %d", len(coin_df_filtered) )

        return (
            coin_df_filtered,
            btc_slice['close'].values,          # btc_close_np
            btc_slice['volume'].values,         # btc_volume_np
            btc_slice['high'].values,           # btc_high_np
            btc_slice['low'].values,            # btc_low_np
            btc_slice['quote_volume'].values    # btc_quote_volume
        )

    # ---------------------------------------------------------
    # Normalize time
    # ---------------------------------------------------------

    def _normalize_time_value(self, value):

        if value is None:
            return None

        if isinstance(value, (int, float)):
            return int(value)

        if hasattr(value, 'timestamp'):
            return int(value.timestamp())

        ts = pd.Timestamp(value)

        if ts.tzinfo is not None:
            return int(ts.timestamp())

        return int(
            ts.value / 1_000_000_000
        )

    # ---------------------------------------------------------
    # Save processing log
    # ---------------------------------------------------------

    def _save_processing_log(self, ticker, detector):
        logger.debug("Saving processing log for %s", ticker)

        times = detector.times_np
        if times is not None and len(times) > 0:
            first_time = int(times[0])
            last_time  = int(times[-1])
        else:
            first_time = None
            last_time = None

        params_json = json.dumps(detector.get_params_dict())



        logger.debug("Processing log saved for %s: first=%s, last=%s",
                    ticker, first_time, last_time)
                    
        MIN_TS = 946684800      # 2000-01-01
        MAX_TS = 4102444800     # 2100-01-01

        if last_time is not None and not (MIN_TS <= last_time <= MAX_TS):
            logger.error(
                "Bad last_processed_time for %s: %s — refusing to save",
                ticker, last_time
            )
            return

        self.storage.save_processing_log(ticker, first_time, last_time, params_json)

    # ---------------------------------------------------------
    # Save detector state
    # ---------------------------------------------------------

    def _save_detector_state(self, ticker, detector, ml_predictions=None):

        logger.debug(
            "Serializing detector state for %s",
            ticker
        )

        state_json = self.serializer_class.serialize(
            detector,
            ml_predictions=ml_predictions,
        )

        samples_count = len(
            getattr(
                detector,
                "samples",
                []
            )
        )

        support_count = len(
            getattr(
                detector,
                "confirmed_support_levels",
                []
            )
        )

        resistance_count = len(
            getattr(
                detector,
                "confirmed_resistance_levels",
                []
            )
        )

        logger.info(
            "Detector state for %s: "
            "samples=%d, active_support=%d, active_resistance=%d",
            ticker,
            samples_count,
            support_count,
            resistance_count,
        )

        logger.debug(
            "Detector state serialized for %s: %d characters",
            ticker,
            len(state_json)
        )

        self.storage.save_detector_state(
            ticker,
            state_json
        )

        logger.debug(
            "Detector state saved for %s",
            ticker
        )


    # ---------------------------------------------------------
    # Core detection
    # ---------------------------------------------------------

    def _run_detection(self, ticker, load_state=True):
        """
        Run detector for a ticker and return the detector instance.
        Returns None if there is nothing to process.
        Touches storage only if load_state=True (read-only).
        """
        logger.info("Running detection for %s (load_state=%s)", ticker, load_state)

        coin_df = load_ticker_data(ticker, base_path=self.data_path)
        if coin_df is None or len(coin_df) == 0:
            logger.warning("No data for %s, skipping", ticker)
            return None

        (
            coin_df_filtered,
            btc_close_np,
            btc_volume_np,
            btc_high_np,
            btc_low_np,
            btc_quote_volume,
        ) = self._get_btc_metrics_for_coin(coin_df)

        if len(coin_df_filtered) == 0:
            logger.warning("No BTC-aligned rows for %s, skipping", ticker)
            return None

        times_sec = (
            coin_df_filtered['open_time']
            .values.astype('datetime64[s]')
            .astype('int64')
        )

        detector = None
        if load_state:
            row = self.storage.load_detector_state(ticker)
            if row:
                detector = self.serializer_class.deserialize(
                    row['state_blob'], self.detector_class
                )
                patched = 0
                for attr in ("support_clusters", "resistance_clusters"):
                    for c in getattr(detector, attr, None) or []:
                        if "snapshots" not in c:
                            c["snapshots"] = []
                            patched += 1
                logger.info(
                    "Loaded existing detector state for %s (%d building clusters migrated)",
                    ticker, patched,
                )

        if detector is None:
            detector = self.detector_class()
            logger.info("Created new detector for %s", ticker)

        resume_from_time = detector.last_processed_time
        logger.debug("Resume for %s from %s", ticker, resume_from_time)

        detector.process(
            closes_np=coin_df_filtered['close'].values,
            times_np=times_sec,
            high_np=coin_df_filtered['high'].values,
            low_np=coin_df_filtered['low'].values,
            volume_np=coin_df_filtered['volume'].values,
            quote_volume_np=coin_df_filtered['quote_volume'].values,
            btc_close_np=btc_close_np,
            btc_volume_np=btc_volume_np,
            btc_high_np=btc_high_np,
            btc_low_np=btc_low_np,
            btc_quote_volume_np=btc_quote_volume,
            symbol=ticker,
            resume_from_time=resume_from_time,
        )

        return detector

    def _predict_active_levels(self, ticker, detector):
        """Score the latest sample for every currently confirmed level."""
        exported = detector.get_confirmed_levels_export()
        levels = exported.get("support", []) + exported.get("resistance", [])
        logger.info(
            "Preparing ML scoring: ticker=%s support=%d resistance=%d total=%d",
            ticker,
            len(exported.get("support", [])),
            len(exported.get("resistance", [])),
            len(levels),
        )
        return self.ml_predictor.predict_levels(ticker, levels)


    # ---------------------------------------------------------
    # Detection + persistence (backfill path)
    # ---------------------------------------------------------

    def _level_detection(self, ticker):
        detector = self._run_detection(ticker, load_state=True)

        if detector is None:
            return None

        if detector.times_np is None or len(detector.times_np) == 0:
            logger.warning("No valid time range for %s, skipping log save", ticker)
            return detector

        self._save_processing_log(ticker, detector)
        ml_predictions = self._predict_active_levels(ticker, detector)
        logger.info(
            "ML scoring summary: ticker=%s predictions=%d candidates=%d candidate_level_ids=%s",
            ticker,
            len(ml_predictions),
            sum(1 for item in ml_predictions if item.get("is_candidate")),
            [item.get("level_id") for item in ml_predictions if item.get("is_candidate")],
        )
        self._save_detector_state(ticker, detector, ml_predictions)

        logger.info("Level detection completed for %s", ticker)
        return detector

    # ---------------------------------------------------------
    # Load data
    # ---------------------------------------------------------

    def _load_data(self, dict_tickers):

        if not dict_tickers:

            logger.debug( "No tickers to load" )

            return

        logger.info( "Starting data loading for %d tickers", len(dict_tickers))

        for index, (ticker, timestamp) in enumerate( dict_tickers.items(), start=1 ):

            logger.info( "[%d/%d] Downloading %s", index, len(dict_tickers), ticker )

            logger.debug( "Download timestamp for %s: %s", ticker, timestamp )

            try:
                self.downloader.download(ticker, timestamp)
                logger.debug("Download completed for %s", ticker)

                self._level_detection(ticker)

                # After successful processing and saving the state to the database
                # delete the downloaded ticker files, except for BTCUSDT
                if ticker != "BTCUSDT":
                    self.downloader.cleanup_downloaded(ticker)

                logger.info("[%d/%d] Finished %s", index, len(dict_tickers), ticker)

            except Exception:
                logger.exception("Failed processing ticker %s", ticker)
                # We don’t delete the files so that we can figure out the error.
                logger.info("Kept files under %s for debugging", self.data_path)

        logger.info( "Data loading completed for %d tickers", len(dict_tickers) )

    # ---------------------------------------------------------
    # Load BTC
    # ---------------------------------------------------------

    def _load_btc_local(self, use_cache: bool = True):
        """
        Load BTC for detection.

        use_cache=True  (default, create_dataset path):
            cache INDEX_PATH/BTCUSDT.parquet -> fallback to data_path -> write cache once.

        use_cache=False (download_history path):
            always read fresh BTC from data_path, never touch the cache.
        """
        btc_df_loc = None

        if use_cache:
            btc_df_loc = load_btc_cache(self.btc_cache_path)

        if btc_df_loc is None or len(btc_df_loc) == 0:
            logger.info(
                "BTC cache %s — loading from local data_path",
                "missing" if use_cache else "bypassed",
            )
            btc_df_loc = load_ticker_data("BTCUSDT", base_path=self.data_path)

            if btc_df_loc is None or len(btc_df_loc) == 0:
                raise RuntimeError(
                    "BTCUSDT not found in cache "
                    f"({self.btc_cache_path}) nor in data_path "
                    f"({self.data_path}) — run download_history first"
                )

            if use_cache:
                # write the frozen snapshot once
                save_btc_cache(btc_df_loc, self.btc_cache_path)

        btc_df_loc = btc_df_loc.sort_values('open_time').reset_index(drop=True)

        day_bars = 1440
        logger.debug("Calculating BTC rolling volume: window=%d", day_bars)

        btc_df_loc['volume_btc_day'] = (
            btc_df_loc['volume'].rolling(window=day_bars, min_periods=1).sum()
        )
        btc_df_loc['quote_volume_btc_day'] = (
            btc_df_loc['quote_volume'].rolling(window=day_bars, min_periods=1).sum()
        )

        self.btc_df = btc_df_loc
        logger.info("BTC data prepared: %d rows", len(self.btc_df))


    def _load_data_btc(self, min_timestamp):
        logger.info("Starting BTC download")
        logger.debug("BTC download timestamp: %s", min_timestamp)

        self.downloader.download("BTCUSDT", min_timestamp)
        logger.debug("BTC download completed")

        self._load_btc_local(use_cache=False)

    # ---------------------------------------------------------
    # Main history download
    # ---------------------------------------------------------

    def download_history(self, publish_summary: bool = True):

        logger.info("=" * 70)
        logger.info("STARTING BINANCE HISTORY DOWNLOAD")
        logger.info("=" * 70)

        start_time = time.time()

        try:

            # -------------------------------------------------
            # Synchronize tickers
            # -------------------------------------------------

            ( exist_tickers, new_tickers, min_ts ) = self._sync_tickers_creation_time()

            logger.info( "Download plan: existing=%d, new=%d",len(exist_tickers), len(new_tickers))

            # -------------------------------------------------
            # BTC
            # -------------------------------------------------

            self._load_data_btc( min_ts )

            # -------------------------------------------------
            # Existing tickers
            # -------------------------------------------------

            logger.info( "Processing existing tickers" )

            self._load_data( exist_tickers)

            # -------------------------------------------------
            # New tickers
            # -------------------------------------------------

            logger.info( "Processing new tickers")

            self._load_data( new_tickers )

            elapsed = time.time() - start_time

            # clear history for BTC
            self.downloader.cleanup_downloaded("BTCUSDT")


            logger.info("=" * 70)
            logger.info( "BINANCE HISTORY DOWNLOAD COMPLETED")
            logger.info( "Total execution time: %.2f seconds", elapsed)
            logger.info("=" * 70)

            if publish_summary:
                logger.info("Publishing active levels summary")
                self.summary_publisher.publish()

        except Exception:

            logger.exception( "BINANCE HISTORY DOWNLOAD FAILED" )

            raise


    def create_dataset(self, symbols=None, output_dir=None, load_state=False):
        """
        Build a breakout dataset from local data.
        Does NOT write to storage (processing log / detector state untouched).

        Parameters
        ----------
        symbols : list[str] | None
            Tickers to process. Defaults to self.symbols.
        output_dir : str | None
            Where to write per-ticker parquet files.
            Defaults to OUTPUT_PATH/breakout_dataset.
        load_state : bool
            Whether to resume from saved detector state (default False —
            start each detector fresh).
        """

        if output_dir is None:
            output_dir = os.path.join(OUTPUT_PATH, "breakout_dataset")
        os.makedirs(output_dir, exist_ok=True)

        if symbols is None:
            symbols = self.symbols

        logger.info("=" * 70)
        logger.info("STARTING DATASET CREATION")
        logger.info("symbols=%d, output_dir=%s, load_state=%s",
                    len(symbols), output_dir, load_state)
        logger.info("=" * 70)

        start_time = time.time()

        # BTC must be prepared before running detection on any ticker
        if self.btc_df.empty:
            self._load_btc_local()

        all_frames = []
        failed = []

        for index, ticker in enumerate(symbols, start=1):
            logger.info("[%d/%d] Processing %s", index, len(symbols), ticker)
            try:
                detector = self._run_detection(ticker, load_state=load_state)
                if detector is None:
                    continue


                breakout_list = detector.get_dataset()

                if breakout_list is None or len(breakout_list) == 0:
                    logger.debug("No breakouts for %s", ticker)
                    continue

                df = save_breakout_dataset(ticker, breakout_list, output_dir)
                if len(df) > 0:
                    all_frames.append(df)

                logger.info("[%d/%d] Finished %s (%d rows)",
                            index, len(symbols), ticker, len(df))

            except Exception:
                logger.exception("Failed create_dataset for %s", ticker)
                failed.append(ticker)
                continue

        # Combined file
        if all_frames:
            full = pd.concat(all_frames, ignore_index=True)
            full_path = os.path.join(output_dir, "_all_breakouts.parquet")
            full.to_parquet(full_path, index=False)
            logger.info("Combined dataset: %d rows -> %s", len(full), full_path)
        else:
            logger.warning("No breakout rows were produced")

        elapsed = time.time() - start_time
        logger.info("=" * 70)
        logger.info("DATASET CREATION COMPLETED")
        logger.info("Tickers processed: %d, failed: %d", len(symbols) - len(failed), len(failed))
        if failed:
            logger.warning("Failed tickers: %s", failed)
        logger.info("Total execution time: %.2f seconds", elapsed)
        logger.info("=" * 70)

        return all_frames 