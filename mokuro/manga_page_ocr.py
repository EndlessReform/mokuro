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
    ocr_batch_size: int = 1


OCR_BATCH_SUMMARY_SCHEMA = "mokuro.ocr_batch_summary.v1"


def _stats(values):
    values = [value for value in values if value is not None]
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None, "p50": None, "p90": None}

    values = sorted(values)
    return {
        "count": len(values),
        "min": values[0],
        "max": values[-1],
        "mean": sum(values) / len(values),
        "p50": _percentile(values, 50),
        "p90": _percentile(values, 90),
    }


def _percentile(sorted_values, percentile):
    if len(sorted_values) == 1:
        return sorted_values[0]

    pos = (len(sorted_values) - 1) * percentile / 100
    lo = int(np.floor(pos))
    hi = int(np.ceil(pos))
    if lo == hi:
        return sorted_values[lo]
    weight = pos - lo
    return sorted_values[lo] * (1 - weight) + sorted_values[hi] * weight


def _token_raggedness(token_counts):
    if not token_counts:
        return {
            "token_min": None,
            "token_max": None,
            "token_range": None,
            "token_ratio": None,
        }

    token_min = min(token_counts)
    token_max = max(token_counts)
    return {
        "token_min": token_min,
        "token_max": token_max,
        "token_range": token_max - token_min,
        "token_ratio": token_max / token_min if token_min > 0 else None,
    }


def _new_ocr_batch_stats():
    return {"pages": [], "batches": [], "reorder_buffers": []}


def summarize_ocr_batch_stats(stats, config):
    pages = stats["pages"]
    batches = stats["batches"]
    reorder_buffers = stats["reorder_buffers"]
    ocr_batch_size = config["ocr_batch_size"]

    return {
        "schema": OCR_BATCH_SUMMARY_SCHEMA,
        "config": config,
        "totals": {
            "pages": len(pages),
            "crops": sum(page["crop_count"] for page in pages),
            "batches": len(batches),
            "reorder_buffers": len(reorder_buffers),
        },
        "yield_rate": {
            "batches_per_page": _stats([page["batch_count"] for page in pages]),
            "crops_per_page": _stats([page["crop_count"] for page in pages]),
            "reorder_buffers_per_page": _stats([page["reorder_buffer_count"] for page in pages]),
            "batch_size": _stats([batch["crop_count"] for batch in batches]),
            "batch_fill_rate": _stats(
                [batch["crop_count"] / ocr_batch_size for batch in batches]
            ),
        },
        "raggedness": {
            "batches": {
                "token_min": _stats([batch["token_min"] for batch in batches]),
                "token_max": _stats([batch["token_max"] for batch in batches]),
                "token_range": _stats([batch["token_range"] for batch in batches]),
                "token_ratio": _stats([batch["token_ratio"] for batch in batches]),
            },
            "reorder_buffers": {
                "token_min": _stats([buffer["token_min"] for buffer in reorder_buffers]),
                "token_max": _stats([buffer["token_max"] for buffer in reorder_buffers]),
                "token_range": _stats([buffer["token_range"] for buffer in reorder_buffers]),
                "token_ratio": _stats([buffer["token_ratio"] for buffer in reorder_buffers]),
            },
        },
    }


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
        bf16=False,
        ocr_bf16=False,
        dev_repeat_ocr_batch_size=1,
        detector_batch_size=4,
        detector_backend="auto",
        detector_model_path=None,
        detector_compute_device=None,
        detector_compile=False,
        ocr_batch_size=1,
        ocr_reorder_buffer_size=None,
    ):
        self.text_height = text_height
        self.max_ratio_vert = max_ratio_vert
        self.max_ratio_hor = max_ratio_hor
        self.anchor_window = anchor_window
        self.disable_ocr = disable_ocr
        self.ocr_num_beams = ocr_num_beams
        self.bf16 = bf16
        self.ocr_bf16 = ocr_bf16 or bf16
        self.dev_repeat_ocr_batch_size = dev_repeat_ocr_batch_size
        self.detector_batch_size = detector_batch_size
        self.detector_backend = detector_backend
        self.detector_model_path = detector_model_path
        self.detector_compute_device = detector_compute_device
        self.detector_compile = detector_compile
        self.ocr_batch_size = ocr_batch_size
        self.ocr_reorder_buffer_size = (
            ocr_batch_size if ocr_reorder_buffer_size is None else ocr_reorder_buffer_size
        )
        self._ocr_batch_stats = _new_ocr_batch_stats()

        if self.ocr_num_beams is not None and self.ocr_num_beams < 1:
            raise ValueError("ocr_num_beams must be at least 1")

        if self.dev_repeat_ocr_batch_size < 1:
            raise ValueError("dev_repeat_ocr_batch_size must be at least 1")

        if self.detector_batch_size < 1:
            raise ValueError("detector_batch_size must be at least 1")

        if self.ocr_batch_size < 1:
            raise ValueError("ocr_batch_size must be at least 1")

        if self.ocr_reorder_buffer_size < 1:
            raise ValueError("ocr_reorder_buffer_size must be at least 1")

        if self.ocr_reorder_buffer_size < self.ocr_batch_size:
            raise ValueError("ocr_reorder_buffer_size must be at least ocr_batch_size")

        if self.dev_repeat_ocr_batch_size > 1 and self.ocr_batch_size > 1:
            raise ValueError("dev_repeat_ocr_batch_size cannot be combined with ocr_batch_size > 1")

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
            # MLX backend now has a default HF model, so no explicit path needed
            pass  # removed: detector_model_path requirement for mlx backend
            if self.bf16 and self.detector_model_path is None and self.detector_backend != "mlx":
                raise ValueError("--bf16 requires an MLX detector backend and artifact")
            if self.detector_compile and self.detector_model_path is None and self.detector_backend != "mlx":
                raise ValueError("--compile requires an MLX detector backend and artifact")

            # Resolve backend first (auto may pick mlx on Apple Silicon)
            tentative_backend = TextDetector._resolve_backend(
                None,  # no explicit path yet
                self.detector_backend,
            )
            # Pick model path: explicit > backend-specific default > torch checkpoint
            if self.detector_model_path is not None:
                detector_model_path = self.detector_model_path
            elif tentative_backend == "mlx":
                detector_model_path = None  # MLX resolves to default HF model
            else:
                detector_model_path = cache.comic_text_detector
            resolved_detector_backend = TextDetector._resolve_backend(
                detector_model_path,
                self.detector_backend,
            )
            if self.bf16 and resolved_detector_backend != "mlx":
                raise ValueError("--bf16 requires an MLX detector backend")
            if self.detector_compile and resolved_detector_backend != "mlx":
                raise ValueError("--compile requires an MLX detector backend")
            detector_compute_device = self.detector_compute_device
            if detector_compute_device is None and force_cpu:
                detector_compute_device = "cpu"
            detector_compute_dtype = "bf16" if self.bf16 else None
            logger.info(
                f"Initializing text detector, backend {resolved_detector_backend}, "
                f"using device {device}"
            )
            self.text_detector = TextDetector(
                model_path=detector_model_path,
                input_size=detector_input_size,
                device=device,
                act="leaky",
                backend=resolved_detector_backend,
                compute_device=detector_compute_device,
                compute_dtype=detector_compute_dtype,
                compile_model=self.detector_compile,
            )
            self.mocr = MangaOcr(pretrained_model_name_or_path, force_cpu)
            self._configure_ocr_dtype()

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
        return self._recognize_page_layouts(layouts, page_indices=page_indices, timings_fh=timings_fh)

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
        return self._recognize_page_layouts([layout], page_indices=[page_idx], timings_fh=timings_fh)[0]

    def _recognize_page_layouts(self, layouts, page_indices=None, timings_fh=None):
        self._ensure_ocr_batch_stats()
        if page_indices is None:
            page_indices = list(range(len(layouts)))
        if len(layouts) != len(page_indices):
            raise ValueError("layouts and page_indices must have the same length")

        page_results = []
        requests_by_page = {}
        all_requests = []
        for layout, page_idx in zip(layouts, page_indices):
            page_result, requests = self._collect_ocr_crop_requests(layout, page_idx)
            page_results.append(page_result)
            requests_by_page[page_idx] = requests
            all_requests.extend(requests)

        batch_count_before = len(self._ocr_batch_stats["batches"])
        reorder_buffer_count_before = len(self._ocr_batch_stats["reorder_buffers"])
        results = self._recognize_crop_requests(all_requests)
        page_batch_counts = self._count_containers_by_page(
            self._ocr_batch_stats["batches"][batch_count_before:],
            "page_set",
        )
        page_reorder_buffer_counts = self._count_containers_by_page(
            self._ocr_batch_stats["reorder_buffers"][reorder_buffer_count_before:],
            "page_set",
        )
        for page_idx in page_indices:
            self._ocr_batch_stats["pages"].append({
                "page": page_idx,
                "crop_count": len(requests_by_page[page_idx]),
                "batch_count": page_batch_counts.get(page_idx, 0),
                "reorder_buffer_count": page_reorder_buffer_counts.get(page_idx, 0),
            })

        if timings_fh is not None:
            for result in results:
                self._write_timing(result, timings_fh)

        results_by_page = {}
        for result in results:
            results_by_page.setdefault(result.page_idx, []).append(result)

        return [
            self._reconstruct_page_result(
                layout,
                results_by_page.get(page_idx, []),
                page_result=page_result,
            )
            for layout, page_idx, page_result in zip(layouts, page_indices, page_results)
        ]

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
        self._ensure_ocr_batch_stats()
        results = []
        for window_start, window in self._iter_ocr_request_windows(requests):
            window_results = []
            for batch_start in range(0, len(window), self.ocr_batch_size):
                batch = window[batch_start:batch_start + self.ocr_batch_size]
                t0 = time.perf_counter()
                chunk_texts = self._recognize_crop_batch([request.img for request in batch])
                elapsed_ms = (time.perf_counter() - t0) * 1000
                batch_size = len(batch)
                if len(chunk_texts) != batch_size:
                    raise ValueError("OCR backend returned a different number of results than requests")
                batch_results = []
                for request, chunk_text in zip(batch, chunk_texts):
                    num_tokens = len(self.mocr.tokenizer.encode(chunk_text)) - 2  # subtract CLS/SEP
                    batch_results.append(
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
                            ocr_batch_size=batch_size,
                        )
                    )
                results.extend(batch_results)
                window_results.extend(batch_results)
                self._record_ocr_batch_stats(batch, batch_results)
            self._record_ocr_reorder_buffer_stats(window_start, window, window_results)
        return results

    def _record_ocr_batch_stats(self, batch, batch_results):
        raggedness = _token_raggedness([result.tokens for result in batch_results])
        page_set = sorted({request.page_idx for request in batch})
        self._ocr_batch_stats["batches"].append(
            {
                "page": batch[0].page_idx if batch else None,
                "page_set": page_set,
                "crop_count": len(batch),
                **raggedness,
            }
        )

    def _record_ocr_reorder_buffer_stats(self, window_start, window, window_results):
        raggedness = _token_raggedness([result.tokens for result in window_results])
        page_set = sorted({request.page_idx for request in window})
        self._ocr_batch_stats["reorder_buffers"].append(
            {
                "page": window[0].page_idx if window else None,
                "page_set": page_set,
                "window_start": window_start,
                "crop_count": len(window),
                "batch_count": int(np.ceil(len(window) / self.ocr_batch_size)) if window else 0,
                **raggedness,
            }
        )

    @staticmethod
    def _count_containers_by_page(containers, page_set_key):
        counts = {}
        for container in containers:
            for page_idx in container.get(page_set_key, []):
                counts[page_idx] = counts.get(page_idx, 0) + 1
        return counts

    def _ensure_ocr_batch_stats(self):
        if not hasattr(self, "_ocr_batch_stats"):
            self._ocr_batch_stats = _new_ocr_batch_stats()

    def get_ocr_batch_summary(self):
        self._ensure_ocr_batch_stats()
        return summarize_ocr_batch_stats(
            self._ocr_batch_stats,
            {
                "ocr_batch_size": self.ocr_batch_size,
                "ocr_reorder_buffer_size": self.ocr_reorder_buffer_size,
                "ocr_bf16": getattr(self, "ocr_bf16", False),
                "scope": "detector_batch_sync",
                "reordering": "disabled",
            },
        )

    def _configure_ocr_dtype(self):
        if not getattr(self, "ocr_bf16", False):
            return

        device_type = self._ocr_device().type
        if device_type not in ("cuda", "mps"):
            logger.warning("--ocr-bf16 is only enabled on CUDA/MPS; leaving OCR model in default dtype")
            self.ocr_bf16 = False
            return

        logger.info(f"Casting OCR model to bfloat16 on {device_type}")
        self.mocr.model.to(dtype=torch.bfloat16)

    def _ocr_device(self):
        device = getattr(self.mocr.model, "device", None)
        if device is None:
            try:
                device = next(self.mocr.model.parameters()).device
            except StopIteration:
                device = torch.device("cpu")
        return torch.device(device)

    def _ocr_input_dtype(self):
        return torch.bfloat16 if getattr(self, "ocr_bf16", False) else None

    def _prepare_ocr_input(self, imgs):
        imgs = [img.convert("L").convert("RGB") for img in imgs]
        x = torch.stack([self.mocr._preprocess(img) for img in imgs])
        return x.to(self._ocr_device(), dtype=self._ocr_input_dtype())

    def _iter_ocr_request_windows(self, requests):
        if self.ocr_reorder_buffer_size < self.ocr_batch_size:
            raise ValueError("ocr_reorder_buffer_size must be at least ocr_batch_size")

        for window_start in range(0, len(requests), self.ocr_reorder_buffer_size):
            yield window_start, requests[window_start:window_start + self.ocr_reorder_buffer_size]

    def _iter_ocr_request_batches(self, requests):
        for _window_start, window in self._iter_ocr_request_windows(requests):
            # Future crop reordering belongs inside this window. For now, preserve detector/page order.
            for batch_start in range(0, len(window), self.ocr_batch_size):
                yield window[batch_start:batch_start + self.ocr_batch_size]

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
                "ocr_batch_size": result.ocr_batch_size,
            }) + "\n"
        )

    def _recognize_crop_batch(self, imgs):
        if len(imgs) == 1 and self.ocr_batch_size == 1:
            return [self._recognize_crop(imgs[0])]

        if self.dev_repeat_ocr_batch_size != 1:
            raise ValueError("dev_repeat_ocr_batch_size cannot be combined with batched OCR")

        x = self._prepare_ocr_input(imgs)
        generate_kwargs = {"max_length": 300}
        if self.ocr_num_beams is not None:
            generate_kwargs["num_beams"] = self.ocr_num_beams
        xs = self.mocr.model.generate(x, **generate_kwargs).cpu()
        return [
            post_process(self.mocr.tokenizer.decode(tokens, skip_special_tokens=True))
            for tokens in xs
        ]

    def _recognize_crop(self, img):
        if (
            self.dev_repeat_ocr_batch_size == 1
            and self.ocr_num_beams is None
            and not getattr(self, "ocr_bf16", False)
        ):
            return self.mocr(img)

        x = self._prepare_ocr_input([img])
        x = x.repeat(self.dev_repeat_ocr_batch_size, 1, 1, 1)
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
