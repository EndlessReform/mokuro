import cv2
import json
import time
import numpy as np
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
    ):
        self.text_height = text_height
        self.max_ratio_vert = max_ratio_vert
        self.max_ratio_hor = max_ratio_hor
        self.anchor_window = anchor_window
        self.disable_ocr = disable_ocr
        self.ocr_num_beams = ocr_num_beams
        self.dev_repeat_ocr_batch_size = dev_repeat_ocr_batch_size

        if self.ocr_num_beams is not None and self.ocr_num_beams < 1:
            raise ValueError("ocr_num_beams must be at least 1")

        if self.dev_repeat_ocr_batch_size < 1:
            raise ValueError("dev_repeat_ocr_batch_size must be at least 1")

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
        img = imread(img_path)
        if img is None:
            raise InvalidImage()
        H, W, *_ = img.shape
        result = {"version": __version__, "img_width": W, "img_height": H, "blocks": []}

        if self.disable_ocr:
            return result

        mask, mask_refined, blk_list = self.text_detector(img, refine_mode=1, keep_undetected_mask=True)
        for blk_idx, blk in enumerate(blk_list):
            result_blk = {
                "box": list(blk.xyxy),
                "vertical": blk.vertical,
                "font_size": blk.font_size,
                "lines_coords": [],
                "lines": [],
            }

            for line_idx, line in enumerate(blk.lines_array()):
                if blk.vertical:
                    max_ratio = self.max_ratio_vert
                else:
                    max_ratio = self.max_ratio_hor

                line_crops, cut_points = self.split_into_chunks(
                    img,
                    mask_refined,
                    blk,
                    line_idx,
                    textheight=self.text_height,
                    max_ratio=max_ratio,
                    anchor_window=self.anchor_window,
                )

                line_text = ""
                for chunk_idx, line_crop in enumerate(line_crops):
                    crop_h, crop_w = line_crop.shape[:2]
                    if blk.vertical:
                        line_crop = cv2.rotate(line_crop, cv2.ROTATE_90_CLOCKWISE)

                    t0 = time.perf_counter()
                    chunk_text = self._recognize_crop(Image.fromarray(line_crop))
                    elapsed_ms = (time.perf_counter() - t0) * 1000

                    if timings_fh is not None:
                        num_tokens = len(self.mocr.tokenizer.encode(chunk_text)) - 2  # subtract CLS/SEP
                        timings_fh.write(
                            json.dumps({
                                "page": page_idx,
                                "blk": blk_idx,
                                "line": line_idx,
                                "chunk": chunk_idx,
                                "crop_h": crop_h,
                                "crop_w": crop_w,
                                "area": crop_h * crop_w,
                                "aspect": round(crop_w / crop_h, 2) if crop_h else 0,
                                "tokens": num_tokens,
                                "ocr_ms": round(elapsed_ms, 1),
                                "ocr_batch_size": self.dev_repeat_ocr_batch_size,
                            }) + "\n"
                        )

                    line_text += chunk_text

                result_blk["lines_coords"].append(line.tolist())
                result_blk["lines"].append(line_text)

            result["blocks"].append(result_blk)

        return result

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
