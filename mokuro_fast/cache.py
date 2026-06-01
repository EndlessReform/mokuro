import os
from pathlib import Path
from typing import Any

import requests
from loguru import logger


class cache:
    def __init__(self):
        self.root = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "manga-ocr"
        self.root.mkdir(parents=True, exist_ok=True)

    @property
    def comic_text_detector(self):
        path = self.root / "comictextdetector.pt"
        url = "https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.2.1/comictextdetector.pt"

        self._download_if_needed(path, url)
        return path

    def _download_if_needed(self, path, url):
        if not path.is_file():
            logger.info(f"Downloading {url}")
            r = requests.get(url, stream=True, verify=True)
            if r.status_code != 200:
                raise RuntimeError(f"Failed downloading {url}")
            with path.open("wb") as f:
                for chunk in r.iter_content(1024):
                    if chunk:
                        f.write(chunk)
            logger.info(f"Finished downloading {url}")


    @property
    def comic_text_detector_mlx(self) -> Path:
        """Return the path to the MLX detector artifact, downloading from HF Hub if needed."""
        try:
            from huggingface_hub import snapshot_download
        except ModuleNotFoundError:
            raise RuntimeError(
                "huggingface-hub is required for MLX detector artifacts. "
                "Install mokuro[mlx] to enable it."
            )

        repo_id = "jkeisling/comictextdetector-mlx"
        cache_dir = self.root / "mlx"
        cache_dir.mkdir(parents=True, exist_ok=True)

        try:
            local_dir = snapshot_download(repo_id, cache_dir=str(cache_dir))
            return Path(local_dir)
        except Exception as exc:
            raise RuntimeError(f"Failed to download MLX detector artifact from {repo_id}: {exc}") from exc


cache = cache()
