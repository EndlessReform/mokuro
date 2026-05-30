import cv2
import copy
import json
import time
import numpy as np
from dataclasses import dataclass
from PIL import Image
from loguru import logger
from scipy.signal.windows import gaussian

from comic_text_detector.inference import TextDetector
from manga_ocr import MangaOcr
from manga_ocr.ocr import post_process
from mokuro import __version__
from mokuro.cache import cache
from mokuro.utils import imread
import torch


@dataclass
class PageLayout:
    img: np.ndarray
    img_width: int
    img_height: int
    mask: np.ndarray | None
    mask_refined: np.ndarray | None
    blk_list: list


@dataclass(frozen=True)
class OcrCropRequest:
    page_idx: int
    blk_idx: int
    line_idx: int
    chunk_idx: int
    img: Image.Image
    crop_h: int
    crop_w: int


@dataclass(frozen=True)
class OcrCropResult:
    page_idx: int
    blk_idx: int
    line_idx: int
    chunk_idx: int
    text: str
    crop_h: int | None = None
    crop_w: int | None = None
    elapsed_ms: float | None = None
    tokens: int | None = None


class InvalidImage(Exception):
    def __init__(self, message="Animation file, Corrupted file or Unsupported type"):
        super().__init__(message)


class MangaPageOcr:
    def __init__(
        self,
        pretrained_model_name_or_path="kha-white/manga-ocr-base",
        force_cpu=False,
        detector_input_size=1024,
        text_height=64,
        max_ratio_vert=16,
        max_ratio_hor=8,
        anchor_window=2,
        disable_ocr=False,
        ocr_num_beams=None,
        dev_repeat_ocr_batch_size=1,
        detector_batch_size=4,
    ):
        self.text_height = text_height
        self.max_ratio_vert = max_ratio_vert
        self.max_ratio_hor = max_ratio_hor
        self.anchor_window = anchor_window
        self.disable_ocr = disable_ocr
        self.ocr_num_beams = ocr_num_beams
        self.dev_repeat_ocr_batch_size = dev_repeat_ocr_batch_size
        self.detector_batch_size = detector_batch_size

        if self.ocr_num_beams is not None and self.ocr_num_beams < 1:
            raise ValueError("ocr_num_beams must be at least 1")

        if self.dev_repeat_ocr_batch_size < 1:
            raise ValueError("dev_repeat_ocr_batch_size must be at least 1")

        if self.detector_batch_size < 1:
            raise ValueError("detector_batch_size must be at least 1")

        if not self.disable_ocr:
            if self.dev_repeat_ocr_batch_size > 1:
                logger.warning(
                    "DEV ONLY: running OCR with artificial repeated-crop batch size "
                    f"{self.dev_repeat_ocr_batch_size}; duplicate outputs are discarded."
                )
            if not force_cpu and torch.cuda.is_available():
                device = "cuda"
            elif not force_cpu and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
            logger.info(f"Initializing text detector, using device {device}")
            self.text_detector = TextDetector(
                model_path=cache.comic_text_detector, input_size=detector_input_size, device=device, act="leaky"
            )
            self.mocr = MangaOcr(pretrained_model_name_or_path, force_cpu)

    def __call__(self, img_path, page_idx=0, timings_fh=None):
        return self.process_pages([img_path], page_indices=[page_idx], timings_fh=timings_fh)[0]

    def process_pages(self, img_paths, page_indices=None, timings_fh=None):
        if page_indices is None:
            page_indices = list(range(len(img_paths)))
        if len(img_paths) != len(page_indices):
            raise ValueError("img_paths and page_indices must have the same length")

        layouts = [self.load_page_layout(img_path) for img_path in img_paths]
        if self.disable_ocr:
            return [self._empty_page_result(layout) for layout in layouts]

        layouts = self.detect_pages(layouts)
        return [
            self._recognize_page_layout(layout, page_idx=page_idx, timings_fh=timings_fh)
            for layout, page_idx in zip(layouts, page_indices)
        ]

    def load_page_layout(self, img_path):
        img = imread(img_path)
        if img is None:
            raise InvalidImage()
        H, W, *_ = img.shape
        return PageLayout(img=img, img_width=W, img_height=H, mask=None, mask_refined=None, blk_list=[])

    def detect_page(self, layout):
        mask, mask_refined, blk_list = self.text_detector(layout.img, refine_mode=1, keep_undetected_mask=True)
        return PageLayout(
            img=layout.img,
            img_width=layout.img_width,
            img_height=layout.img_height,
            mask=mask,
            mask_refined=mask_refined,
            blk_list=blk_list,
        )

    def detect_pages(self, layouts):
        if not layouts:
            return []

        if hasattr(self.text_detector, "detect_batch"):
            detector_results = self.text_detector.detect_batch(
                [layout.img for layout in layouts],
                refine_mode=1,
                keep_undetected_mask=True,
            )
            return [
                PageLayout(
                    img=layout.img,
                    img_width=layout.img_width,
                    img_height=layout.img_height,
                    mask=mask,
                    mask_refined=mask_refined,
                    blk_list=blk_list,
                )
                for layout, (mask, mask_refined, blk_list) in zip(layouts, detector_results)
            ]

        return [self.detect_page(layout) for layout in layouts]

    def _recognize_page_layout(self, layout, page_idx=0, timings_fh=None):
        page_result, requests = self._collect_ocr_crop_requests(layout, page_idx)
        results = self._recognize_crop_requests(requests)
        if timings_fh is not None:
            for result in results:
                self._write_timing(result, timings_fh)
        return self._reconstruct_page_result(layout, results, page_result=page_result)

    def _collect_ocr_crop_requests(self, layout, page_idx):
        page_result = self._empty_page_result(layout)
        requests = []
        for blk_idx, blk in enumerate(layout.blk_list):
            result_blk = {
                "box": list(blk.xyxy),
                "vertical": blk.vertical,
                "font_size": blk.font_size,
                "lines_coords": [],
                "lines": [],
            }

            for line_idx, _line in enumerate(blk.lines_array()):
                max_ratio = self.max_ratio_vert if blk.vertical else self.max_ratio_hor
                line_crops, _cut_points = self.split_into_chunks(
                    layout.img,
                    layout.mask_refined,
                    blk,
                    line_idx,
                    textheight=self.text_height,
                    max_ratio=max_ratio,
                    anchor_window=self.anchor_window,
                )
                result_blk["lines_coords"].append(blk.lines_array()[line_idx].tolist())
                result_blk["lines"].append("")

                for chunk_idx, line_crop in enumerate(line_crops):
                    crop_h, crop_w = line_crop.shape[:2]
                    if blk.vertical:
                        line_crop = cv2.rotate(line_crop, cv2.ROTATE_90_CLOCKWISE)

                    requests.append(
                        OcrCropRequest(
                            page_idx=page_idx,
                            blk_idx=blk_idx,
                            line_idx=line_idx,
                            chunk_idx=chunk_idx,
                            img=Image.fromarray(line_crop),
                            crop_h=crop_h,
                            crop_w=crop_w,
                        )
                    )
            page_result["blocks"].append(result_blk)
        return page_result, requests

    def _recognize_crop_requests(self, requests):
        results = []
        for request in requests:
            t0 = time.perf_counter()
            chunk_text = self._recognize_crop(request.img)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            num_tokens = len(self.mocr.tokenizer.encode(chunk_text)) - 2  # subtract CLS/SEP
            results.append(
                OcrCropResult(
                    page_idx=request.page_idx,
                    blk_idx=request.blk_idx,
                    line_idx=request.line_idx,
                    chunk_idx=request.chunk_idx,
                    text=chunk_text,
                    crop_h=request.crop_h,
                    crop_w=request.crop_w,
                    elapsed_ms=elapsed_ms,
                    tokens=num_tokens,
                )
            )
        return results

    @staticmethod
    def _empty_page_result(layout):
        return {"version": __version__, "img_width": layout.img_width, "img_height": layout.img_height, "blocks": []}

    @staticmethod
    def _reconstruct_page_result(layout, crop_results, page_result=None):
        result = copy.deepcopy(page_result) if page_result is not None else MangaPageOcr._empty_page_result(layout)
        line_texts = {}

        for crop_result in sorted(crop_results, key=lambda r: (r.blk_idx, r.line_idx, r.chunk_idx)):
            line_texts.setdefault((crop_result.blk_idx, crop_result.line_idx), "")
            line_texts[(crop_result.blk_idx, crop_result.line_idx)] += crop_result.text

        if page_result is not None:
            for blk_idx, result_blk in enumerate(result["blocks"]):
                for line_idx in range(len(result_blk["lines"])):
                    result_blk["lines"][line_idx] = line_texts.get((blk_idx, line_idx), "")
            return result

        for blk_idx, blk in enumerate(layout.blk_list):
            result_blk = {
                "box": list(blk.xyxy),
                "vertical": blk.vertical,
                "font_size": blk.font_size,
                "lines_coords": [],
                "lines": [],
            }

            for line_idx, line in enumerate(blk.lines_array()):
                result_blk["lines_coords"].append(line.tolist())
                result_blk["lines"].append(line_texts.get((blk_idx, line_idx), ""))

            result["blocks"].append(result_blk)

        return result

    def _write_timing(self, result, timings_fh):
        timings_fh.write(
            json.dumps({
                "page": result.page_idx,
                "blk": result.blk_idx,
                "line": result.line_idx,
                "chunk": result.chunk_idx,
                "crop_h": result.crop_h,
                "crop_w": result.crop_w,
                "area": result.crop_h * result.crop_w,
                "aspect": round(result.crop_w / result.crop_h, 2) if result.crop_h else 0,
                "tokens": result.tokens,
                "ocr_ms": round(result.elapsed_ms, 1),
                "ocr_batch_size": self.dev_repeat_ocr_batch_size,
            }) + "\n"
        )

    def _recognize_crop(self, img):
        if self.dev_repeat_ocr_batch_size == 1 and self.ocr_num_beams is None:
            return self.mocr(img)

        img = img.convert("L").convert("RGB")
        x = self.mocr._preprocess(img)
        x = x[None].repeat(self.dev_repeat_ocr_batch_size, 1, 1, 1).to(self.mocr.model.device)
        generate_kwargs = {"max_length": 300}
        if self.ocr_num_beams is not None:
            generate_kwargs["num_beams"] = self.ocr_num_beams
        x = self.mocr.model.generate(x, **generate_kwargs)[0].cpu()
        text = self.mocr.tokenizer.decode(x, skip_special_tokens=True)
        return post_process(text)

    @staticmethod
    def split_into_chunks(img, mask_refined, blk, line_idx, textheight, max_ratio=16, anchor_window=2):
        line_crop = blk.get_transformed_region(img, line_idx, textheight)

        h, w, *_ = line_crop.shape
        ratio = w / h

        if ratio <= max_ratio:
            return [line_crop], []

        else:
            k = gaussian(textheight * 2, textheight / 8)

            line_mask = blk.get_transformed_region(mask_refined, line_idx, textheight)
            num_chunks = int(np.ceil(ratio / max_ratio))

            anchors = np.linspace(0, w, num_chunks + 1)[1:-1]

            line_density = line_mask.sum(axis=0)
            line_density = np.convolve(line_density, k, "same")
            line_density /= line_density.max()

            anchor_window *= textheight

            cut_points = []
            for anchor in anchors:
                anchor = int(anchor)

                n0 = np.clip(anchor - anchor_window // 2, 0, w)
                n1 = np.clip(anchor + anchor_window // 2, 0, w)

                p = line_density[n0:n1].argmin()
                p += n0

                cut_points.append(p)

            return np.split(line_crop, cut_points, axis=1), cut_points
