from json import JSONDecodeError

from loguru import logger
from tqdm import tqdm

from mokuro import __version__
from mokuro.manga_page_ocr import MangaPageOcr
from mokuro.utils import dump_json, load_json
from mokuro.volume import Volume


class MokuroGenerator:
    def __init__(
        self,
        pretrained_model_name_or_path="kha-white/manga-ocr-base",
        force_cpu=False,
        disable_ocr=False,
        timings_fh=None,
        **kwargs,
    ):
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.force_cpu = force_cpu
        self.disable_ocr = disable_ocr
        self.timings_fh = timings_fh
        self.kwargs = kwargs
        self.mpocr = None

    def init_models(self):
        if self.mpocr is None:
            self.mpocr = MangaPageOcr(
                self.pretrained_model_name_or_path,
                force_cpu=self.force_cpu,
                disable_ocr=self.disable_ocr,
                **self.kwargs,
            )

    def process_volume(self, volume: Volume, ignore_errors=False, no_cache=False, page_limit=None):
        volume.path_ocr_cache.mkdir(parents=True, exist_ok=True)

        if volume.mokuro_data is not None:
            for page in volume.mokuro_data["pages"][:page_limit]:
                json_path = (volume.path_ocr_cache / page["img_path"]).with_suffix(".json")
                if json_path.is_file():
                    continue
                json_path.parent.mkdir(parents=True, exist_ok=True)
                page = page.copy()
                page.pop("img_path")
                dump_json(page, json_path)

        img_paths = volume.get_img_paths()
        img_path_items = list(img_paths.values())[:page_limit]
        pending_pages = []

        for page_idx, img_path_rel in enumerate(tqdm(img_path_items, desc="Processing pages...")):
            try:
                json_path = (volume.path_ocr_cache / img_path_rel).with_suffix(".json")

                try:
                    load_json(json_path)
                    already_processed = True
                except (FileNotFoundError, JSONDecodeError, UnicodeDecodeError):
                    already_processed = False

                if no_cache or not already_processed:
                    self.init_models()
                    pending_pages.append((page_idx, img_path_rel, json_path))
                    if len(pending_pages) >= self.mpocr.detector_batch_size:
                        self._process_pending_pages(volume, pending_pages, ignore_errors=ignore_errors)
                        pending_pages = []
            except Exception as e:
                if ignore_errors:
                    logger.error(e)
                else:
                    raise e

        if pending_pages:
            self._process_pending_pages(volume, pending_pages, ignore_errors=ignore_errors)

        self.generate_mokuro_file(volume, ignore_errors=ignore_errors, page_limit=page_limit)

    def _process_pending_pages(self, volume, pending_pages, ignore_errors=False):
        img_paths = [volume.path_in / img_path_rel for _page_idx, img_path_rel, _json_path in pending_pages]
        page_indices = [page_idx for page_idx, _img_path_rel, _json_path in pending_pages]
        json_paths = [json_path for _page_idx, _img_path_rel, json_path in pending_pages]

        try:
            results = self.mpocr.process_pages(img_paths, page_indices=page_indices, timings_fh=self.timings_fh)
        except Exception as e:
            if not ignore_errors:
                raise e
            logger.error(e)
            for page_idx, img_path_rel, json_path in pending_pages:
                try:
                    result = self.mpocr(
                        volume.path_in / img_path_rel,
                        page_idx=page_idx,
                        timings_fh=self.timings_fh,
                    )
                except Exception as page_error:
                    logger.error(page_error)
                else:
                    json_path.parent.mkdir(parents=True, exist_ok=True)
                    dump_json(result, json_path)
            return

        for result, json_path in zip(results, json_paths):
            json_path.parent.mkdir(parents=True, exist_ok=True)
            dump_json(result, json_path)

    @staticmethod
    def generate_mokuro_file(volume: Volume, ignore_errors=False, page_limit=None):
        json_paths = dict(list(volume.get_json_paths().items())[:page_limit])
        img_paths = volume.get_img_paths()

        out = {
            "version": __version__,
            "title": volume.title.name,
            "title_uuid": volume.title.uuid,
            "volume": volume.name,
            "volume_uuid": volume.uuid,
            "pages": [],
        }

        for key, json_path_rel in json_paths.items():
            try:
                img_path_rel = img_paths[key]
                page_json = load_json(volume.path_ocr_cache / json_path_rel)
                page_json["img_path"] = str(img_path_rel).replace("\\", "/")
                out["pages"].append(page_json)
            except Exception as e:
                if ignore_errors:
                    logger.error(e)
                else:
                    raise e

        dump_json(out, volume.path_mokuro)
