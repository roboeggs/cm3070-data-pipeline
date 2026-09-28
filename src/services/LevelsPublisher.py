import os
import json
import logging

import requests


logger = logging.getLogger(__name__)


class LevelsPublisher:

    def __init__(
        self,
        storage,
        local_path=None,
        gist_token=None,
        gist_id=None,
        file_name=None,
    ):
        self.storage = storage
        self.local_path = local_path or os.environ.get(
            "ACTIVE_LEVELS_LOCAL_PATH",
            "active_levels_summary.json",
        )
        self.gist_token = gist_token or os.environ.get("GISTS")
        self.gist_id = gist_id or os.environ.get("GIST_ID")
        self.file_name = file_name or os.environ.get("GIST_FILE_NAME")

    def build_summary(self):
        logger.info("Building active levels summary from storage")
        summary = self.storage.build_active_levels_summary()

        tickers_count = len(summary.get("tickers", []))
        levels_count = sum(
            len(t.get("levels", [])) for t in summary.get("tickers", [])
        )

        logger.info(
            "Active levels summary built: tickers=%d levels=%d",
            tickers_count,
            levels_count,
        )
        return summary

    def save_local(self, summary):
        with open(self.local_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        logger.info("Active levels summary saved locally: %s", self.local_path)

    def upload_to_gist(self, summary):
        if not (self.gist_token and self.gist_id and self.file_name):
            logger.warning(
                "Gist upload skipped: missing GISTS / GIST_ID / GIST_FILE_NAME"
            )
            return False

        headers = {
            "Authorization": f"Bearer {self.gist_token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        url = f"https://api.github.com/gists/{self.gist_id}"

        content_str = json.dumps(summary, ensure_ascii=False, indent=2)
        payload = {
            "files": {
                self.file_name: {
                    "content": content_str,
                }
            }
        }

        response = requests.patch(url, headers=headers, json=payload, timeout=30)

        if response.status_code == 200:
            logger.info("Gist updated: %s / %s", self.gist_id, self.file_name)
            return True

        logger.error(
            "Gist update failed: status=%s body=%s",
            response.status_code,
            response.text,
        )
        return False

    def publish(self):
        try:
            summary = self.build_summary()
        except Exception:
            logger.exception("Failed to build active levels summary")
            return False

        try:
            self.save_local(summary)
        except Exception:
            logger.exception("Failed to save active levels summary locally")

        try:
            return self.upload_to_gist(summary)
        except Exception:
            logger.exception("Failed to upload active levels summary to Gist")
            return False