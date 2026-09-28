import os
import glob
import shutil
import re 

import logging
from datetime import datetime, timezone, timedelta

from .download_kline import download_daily_klines
from ..core.enums import DAILY_INTERVALS
from ..utils.utility import get_path


logger = logging.getLogger(__name__)


class BinanceDownloader:

    def __init__(
        self,
        script_path,
        upload_dir,
        timeframe,
        end_date
    ):
        self.script_path = str(script_path)
        self.upload_dir = str(upload_dir)
        self.timeframe = timeframe
        self.end_date = self._timestamp_to_date_str(end_date)



    def cleanup_downloaded(self, symbol: str, keep_days: int = 3):
        """
        Removes klines older than keep_days days for the specified symbol.

        Structure on the disk:
            upload/data/spot/daily/klines/<SYMBOL>/1m/
                YYYY-MM-DD_YYYY-MM-DD/
                    <SYMBOL>-1m-YYYY-MM-DD.zip
                    <SYMBOL>-1m-YYYY-MM-DD.zip.CHECKSUM
        """

        rel_path = get_path("spot", "klines", "daily", symbol, self.timeframe)
        target_dir = os.path.join(self.upload_dir, rel_path)

        if not os.path.isdir(target_dir):
            logger.debug("No directory for %s at %s", symbol, target_dir)
            return

        cutoff = datetime.now(timezone.utc).date() - timedelta(days=keep_days)
        removed_files = 0
        kept_files = 0

        # recursive traversal: files can be located both in the root directory and in subfolders
        for root, dirs, files in os.walk(target_dir, topdown=False):
            for fname in files:
                m = re.search(r'(\d{4}-\d{2}-\d{2})', fname)
                if not m:
                    continue
                try:
                    file_date = datetime.strptime(m.group(1), "%Y-%m-%d").date()
                except ValueError:
                    continue

                if file_date < cutoff:
                    fpath = os.path.join(root, fname)
                    try:
                        os.remove(fpath)
                        removed_files += 1
                    except OSError:
                        logger.exception("Failed to remove %s", fpath)
                else:
                    kept_files += 1

            # delete empty subfolders after cleaning the files.
            for d in dirs:
                dpath = os.path.join(root, d)
                try:
                    if not os.listdir(dpath):
                        os.rmdir(dpath)
                except OSError:
                    pass

        logger.info(
            "Cleaned %s: removed=%d, kept=%d (cutoff=%s)",
            symbol, removed_files, kept_files, cutoff
        )

    def _timestamp_to_date_str(self, timestamp: int) -> str:
        """
        Convert Unix timestamp to YYYY-MM-DD in UTC.
        """
        return datetime.fromtimestamp( timestamp, tz=timezone.utc ).strftime("%Y-%m-%d")

    def download(self, symbol, start_date):

        start_date = self._timestamp_to_date_str(int(start_date))

        logger.info("=" * 60)
        logger.info("Downloading %s", symbol)
        logger.info("=" * 60)

        try:
            if self.timeframe not in DAILY_INTERVALS:
                logger.error(
                    "Timeframe %s is not a valid daily interval",
                    self.timeframe
                )
                return False
            # --------------------------------------------------
            # Convert dates to date objects
            # --------------------------------------------------

            start = datetime.strptime(
                start_date,
                "%Y-%m-%d"
            ).date()

            end = datetime.strptime(
                self.end_date,
                "%Y-%m-%d"
            ).date()

            # --------------------------------------------------
            # Generate list of dates
            # --------------------------------------------------

            dates = []

            current_date = start

            while current_date <= end:
                dates.append(
                    current_date.strftime("%Y-%m-%d")
                )

                current_date += timedelta(days=1)

            logger.debug(
                "Downloading dates from %s to %s (%d days)",
                start_date,
                self.end_date,
                len(dates)
            )

            # --------------------------------------------------
            # Direct call to download_daily_klines()
            # --------------------------------------------------

            download_daily_klines(
                trading_type="spot",
                symbols=[symbol],
                num_symbols=1,
                intervals=[self.timeframe],
                dates=dates,
                start_date=start_date,
                end_date=self.end_date,
                folder=self.upload_dir,
                checksum=1
            )

            logger.info(
                "%s downloaded successfully",
                symbol
            )

            return True

        except Exception:
            logger.exception(
                "Error downloading %s",
                symbol
            )

            return False