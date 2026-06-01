import numpy as np
import pytest
import torch
from PIL import Image

from mokuro.mokuro_generator import MokuroGenerator
from mokuro.manga_page_ocr import MangaPageOcr, OcrCropRequest, OcrCropResult, PageLayout


class FakeBlock:
    def __init__(self, xyxy, lines):
        self.xyxy = xyxy
        self.vertical = False
        self.font_size = 12
        self._lines = lines

    def lines_array(self):
        return self._lines


def test_reconstructs_lines_by_block_line_and_chunk_order_not_result_order():
    layout = PageLayout(
        img=np.zeros((10, 10, 3), dtype=np.uint8),
        img_width=10,
        img_height=10,
        mask=None,
        mask_refined=None,
        blk_list=[
            FakeBlock([0, 0, 5, 5], [np.array([[0, 0], [1, 0], [1, 1], [0, 1]])]),
            FakeBlock(
                [5, 5, 10, 10],
                [
                    np.array([[5, 5], [6, 5], [6, 6], [5, 6]]),
                    np.array([[7, 7], [8, 7], [8, 8], [7, 8]]),
                ],
            ),
        ],
    )

    result = MangaPageOcr._reconstruct_page_result(
        layout,
        [
            OcrCropResult(page_idx=0, blk_idx=1, line_idx=0, chunk_idx=1, text="B"),
            OcrCropResult(page_idx=0, blk_idx=0, line_idx=0, chunk_idx=0, text="cover"),
            OcrCropResult(page_idx=0, blk_idx=1, line_idx=1, chunk_idx=0, text="next"),
            OcrCropResult(page_idx=0, blk_idx=1, line_idx=0, chunk_idx=0, text="A"),
        ],
    )

    assert [block["box"] for block in result["blocks"]] == [[0, 0, 5, 5], [5, 5, 10, 10]]
    assert result["blocks"][0]["lines"] == ["cover"]
    assert result["blocks"][1]["lines"] == ["AB", "next"]


def test_process_pages_preserves_page_order_after_batched_detection():
    class FakeMangaPageOcr(MangaPageOcr):
        def load_page_layout(self, img_path):
            marker = int(str(img_path))
            return PageLayout(
                img=np.full((2, 2, 3), marker, dtype=np.uint8),
                img_width=2,
                img_height=2,
                mask=None,
                mask_refined=None,
                blk_list=[],
            )

        def detect_pages(self, layouts):
            self.detected_markers = [int(layout.img[0, 0, 0]) for layout in layouts]
            return layouts

        def _recognize_page_layouts(self, layouts, page_indices=None, timings_fh=None):
            return [
                {"page_idx": page_idx, "marker": int(layout.img[0, 0, 0])}
                for layout, page_idx in zip(layouts, page_indices)
            ]

    mpocr = FakeMangaPageOcr.__new__(FakeMangaPageOcr)
    mpocr.disable_ocr = False

    results = mpocr.process_pages(["3", "1", "2"], page_indices=[30, 10, 20])

    assert mpocr.detected_markers == [3, 1, 2]
    assert results == [
        {"page_idx": 30, "marker": 3},
        {"page_idx": 10, "marker": 1},
        {"page_idx": 20, "marker": 2},
    ]


def test_manga_page_ocr_passes_mlx_detector_options(monkeypatch, tmp_path):
    seen = {}

    class FakeTextDetector:
        @staticmethod
        def _resolve_backend(_model_path, backend):
            return backend

        def __init__(self, **kwargs):
            seen.update(kwargs)

    class FakeMangaOcr:
        def __init__(self, *args, **kwargs):
            pass

    artifact_dir = tmp_path / "mlx-comictextdetector"
    artifact_dir.mkdir()

    monkeypatch.setattr("mokuro.manga_page_ocr.TextDetector", FakeTextDetector)
    monkeypatch.setattr("mokuro.manga_page_ocr.MangaOcr", FakeMangaOcr)
    monkeypatch.setattr("mokuro.manga_page_ocr.MangaPageOcr._configure_ocr_dtype", lambda self: None)

    mpocr = MangaPageOcr(
        force_cpu=True,
        bf16=True,
        detector_compile=True,
        detector_backend="mlx",
        detector_model_path=artifact_dir,
        detector_input_size=512,
    )

    assert seen["model_path"] == artifact_dir
    assert seen["input_size"] == 512
    assert seen["backend"] == "mlx"
    assert seen["compute_device"] == "cpu"
    assert seen["compute_dtype"] == "bf16"
    assert seen["compile_model"] is True
    assert mpocr.ocr_bf16 is True


def test_bf16_requires_mlx_detector_backend():
    with pytest.raises(ValueError, match="--bf16 requires an MLX detector backend"):
        MangaPageOcr(force_cpu=True, bf16=True, detector_backend="torch")


def test_compile_requires_mlx_detector_backend():
    with pytest.raises(ValueError, match="--compile requires an MLX detector backend"):
        MangaPageOcr(force_cpu=True, detector_compile=True, detector_backend="torch")


def test_collect_ocr_requests_captures_block_box_before_crop_extraction_mutates_block():
    class MutatingBlock(FakeBlock):
        def get_transformed_region(self, img, line_idx, textheight):
            self.xyxy[2] = 99
            return np.zeros((2, 2, 3), dtype=np.uint8)

    layout = PageLayout(
        img=np.zeros((10, 10, 3), dtype=np.uint8),
        img_width=10,
        img_height=10,
        mask=np.zeros((10, 10), dtype=np.uint8),
        mask_refined=np.zeros((10, 10), dtype=np.uint8),
        blk_list=[MutatingBlock([0, 0, 5, 5], [np.array([[0, 0], [1, 0], [1, 1], [0, 1]])])],
    )
    mpocr = MangaPageOcr.__new__(MangaPageOcr)
    mpocr.text_height = 64
    mpocr.max_ratio_hor = 8
    mpocr.max_ratio_vert = 16
    mpocr.anchor_window = 2

    page_result, requests = mpocr._collect_ocr_crop_requests(layout, page_idx=0)

    assert page_result["blocks"][0]["box"] == [0, 0, 5, 5]
    assert layout.blk_list[0].xyxy == [0, 0, 99, 5]
    assert len(requests) == 1


def test_ocr_crop_requests_are_decoder_batched_without_reordering():
    class FakeModel:
        device = torch.device("cpu")

        def __init__(self):
            self.input_shapes = []

        def generate(self, x, **kwargs):
            self.input_shapes.append(tuple(x.shape))
            start = sum(shape[0] for shape in self.input_shapes[:-1])
            return torch.tensor([[start + i] for i in range(x.shape[0])])

    class FakeTokenizer:
        def decode(self, tokens, skip_special_tokens):
            return f"text-{tokens.tolist()[0]}"

        def encode(self, text):
            return [101, *range(len(text)), 102]

    class FakeMangaOcr:
        def __init__(self):
            self.model = FakeModel()
            self.tokenizer = FakeTokenizer()

        def _preprocess(self, img):
            return torch.zeros(3, 2, 2)

    mpocr = MangaPageOcr.__new__(MangaPageOcr)
    mpocr.mocr = FakeMangaOcr()
    mpocr.ocr_num_beams = None
    mpocr.dev_repeat_ocr_batch_size = 1
    mpocr.ocr_batch_size = 2
    mpocr.ocr_reorder_buffer_size = 3

    requests = [
        OcrCropRequest(
            page_idx=0,
            blk_idx=idx,
            line_idx=0,
            chunk_idx=0,
            img=Image.new("RGB", (2, 2)),
            crop_h=2,
            crop_w=2,
        )
        for idx in range(5)
    ]

    results = mpocr._recognize_crop_requests(requests)

    assert [result.blk_idx for result in results] == [0, 1, 2, 3, 4]
    assert [result.text for result in results] == ["ｔｅｘｔ－０", "ｔｅｘｔ－１", "ｔｅｘｔ－２", "ｔｅｘｔ－３", "ｔｅｘｔ－４"]
    assert [result.ocr_batch_size for result in results] == [2, 2, 1, 2, 2]
    assert mpocr.mocr.model.input_shapes == [(2, 3, 2, 2), (1, 3, 2, 2), (2, 3, 2, 2)]


def test_process_pages_batches_ocr_requests_across_page_boundaries():
    class FakeModel:
        device = torch.device("cpu")

        def __init__(self):
            self.input_shapes = []

        def generate(self, x, **kwargs):
            self.input_shapes.append(tuple(x.shape))
            start = sum(shape[0] for shape in self.input_shapes[:-1])
            return torch.tensor([[start + i] for i in range(x.shape[0])])

    class FakeTokenizer:
        def decode(self, tokens, skip_special_tokens):
            return ["あ", "い", "う", "え"][tokens.tolist()[0]]

        def encode(self, text):
            return [101, *range(len(text)), 102]

    class FakeMangaOcr:
        def __init__(self):
            self.model = FakeModel()
            self.tokenizer = FakeTokenizer()

        def _preprocess(self, img):
            return torch.zeros(3, 2, 2)

    class FakeMangaPageOcr(MangaPageOcr):
        def load_page_layout(self, img_path):
            marker = int(str(img_path))
            return PageLayout(
                img=np.full((2, 2, 3), marker, dtype=np.uint8),
                img_width=2,
                img_height=2,
                mask=None,
                mask_refined=None,
                blk_list=[],
            )

        def detect_pages(self, layouts):
            return layouts

        def _collect_ocr_crop_requests(self, layout, page_idx):
            page_result = {
                "version": "test",
                "img_width": 2,
                "img_height": 2,
                "blocks": [{"box": [0, 0, 1, 1], "vertical": False, "font_size": 1, "lines_coords": [], "lines": [""]}],
            }
            requests = [
                OcrCropRequest(
                    page_idx=page_idx,
                    blk_idx=0,
                    line_idx=0,
                    chunk_idx=idx,
                    img=Image.new("RGB", (2, 2)),
                    crop_h=2,
                    crop_w=2,
                )
                for idx in range(int(layout.img[0, 0, 0]))
            ]
            return page_result, requests

    mpocr = FakeMangaPageOcr.__new__(FakeMangaPageOcr)
    mpocr.disable_ocr = False
    mpocr.mocr = FakeMangaOcr()
    mpocr.ocr_num_beams = None
    mpocr.dev_repeat_ocr_batch_size = 1
    mpocr.ocr_batch_size = 4
    mpocr.ocr_reorder_buffer_size = 4
    mpocr._ocr_batch_stats = {"pages": [], "batches": [], "reorder_buffers": []}

    results = mpocr.process_pages(["2", "2"], page_indices=[10, 11])

    assert mpocr.mocr.model.input_shapes == [(4, 3, 2, 2)]
    assert [page["blocks"][0]["lines"] for page in results] == [["あい"], ["うえ"]]
    summary = mpocr.get_ocr_batch_summary()
    assert summary["config"]["scope"] == "detector_batch_sync"
    assert summary["totals"] == {"pages": 2, "crops": 4, "batches": 1, "reorder_buffers": 1}
    assert summary["yield_rate"]["batches_per_page"]["max"] == 1


def test_ocr_reorder_buffer_size_must_cover_ocr_batch_size():
    mpocr = MangaPageOcr.__new__(MangaPageOcr)
    mpocr.ocr_batch_size = 4
    mpocr.ocr_reorder_buffer_size = 3

    try:
        list(mpocr._iter_ocr_request_batches([object()] * 4))
    except ValueError as e:
        assert str(e) == "ocr_reorder_buffer_size must be at least ocr_batch_size"
    else:
        raise AssertionError("expected reorder buffer validation to fail")


def test_zero_ocr_reorder_buffer_size_is_rejected(monkeypatch):
    monkeypatch.setattr("mokuro.manga_page_ocr.TextDetector", lambda *args, **kwargs: None)
    monkeypatch.setattr("mokuro.manga_page_ocr.MangaOcr", lambda *args, **kwargs: None)

    try:
        MangaPageOcr(ocr_reorder_buffer_size=0)
    except ValueError as e:
        assert str(e) == "ocr_reorder_buffer_size must be at least 1"
    else:
        raise AssertionError("expected reorder buffer validation to fail")


def test_ocr_bf16_casts_ocr_model_on_accelerated_device(monkeypatch):
    class FakeModel:
        device = torch.device("mps")

        def __init__(self):
            self.dtype = None

        def to(self, dtype=None):
            self.dtype = dtype
            return self

    class FakeMangaOcr:
        def __init__(self, *args, **kwargs):
            self.model = FakeModel()

    monkeypatch.setattr("mokuro.manga_page_ocr.TextDetector", lambda *args, **kwargs: None)
    monkeypatch.setattr("mokuro.manga_page_ocr.MangaOcr", FakeMangaOcr)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)

    mpocr = MangaPageOcr(ocr_bf16=True)

    assert mpocr.ocr_bf16 is True
    assert mpocr.mocr.model.dtype == torch.bfloat16


def test_ocr_backend_result_count_must_match_request_count():
    mpocr = MangaPageOcr.__new__(MangaPageOcr)
    mpocr.ocr_batch_size = 2
    mpocr.ocr_reorder_buffer_size = 2
    mpocr._recognize_crop_batch = lambda imgs: ["only one"]

    requests = [
        OcrCropRequest(
            page_idx=0,
            blk_idx=idx,
            line_idx=0,
            chunk_idx=0,
            img=Image.new("RGB", (2, 2)),
            crop_h=2,
            crop_w=2,
        )
        for idx in range(2)
    ]

    try:
        mpocr._recognize_crop_requests(requests)
    except ValueError as e:
        assert str(e) == "OCR backend returned a different number of results than requests"
    else:
        raise AssertionError("expected OCR result count validation to fail")


def test_ocr_batch_summary_reports_yield_and_raggedness():
    class FakeModel:
        device = torch.device("cpu")

        def __init__(self):
            self.next_token = 1

        def generate(self, x, **kwargs):
            tokens = torch.tensor([[self.next_token + i] for i in range(x.shape[0])])
            self.next_token += x.shape[0]
            return tokens

    class FakeTokenizer:
        def decode(self, tokens, skip_special_tokens):
            return "x" * tokens.tolist()[0]

        def encode(self, text):
            return [101, *range(len(text)), 102]

    class FakeMangaOcr:
        def __init__(self):
            self.model = FakeModel()
            self.tokenizer = FakeTokenizer()

        def _preprocess(self, img):
            return torch.zeros(3, 2, 2)

    layout = PageLayout(
        img=np.zeros((10, 10, 3), dtype=np.uint8),
        img_width=10,
        img_height=10,
        mask=None,
        mask_refined=None,
        blk_list=[],
    )
    page_result = MangaPageOcr._empty_page_result(layout)
    requests = [
        OcrCropRequest(
            page_idx=7,
            blk_idx=idx,
            line_idx=0,
            chunk_idx=0,
            img=Image.new("RGB", (2, 2)),
            crop_h=2,
            crop_w=2,
        )
        for idx in range(5)
    ]

    mpocr = MangaPageOcr.__new__(MangaPageOcr)
    mpocr.mocr = FakeMangaOcr()
    mpocr.ocr_num_beams = None
    mpocr.dev_repeat_ocr_batch_size = 1
    mpocr.ocr_batch_size = 2
    mpocr.ocr_reorder_buffer_size = 3
    mpocr._collect_ocr_crop_requests = lambda layout, page_idx: (page_result, requests)
    mpocr._reconstruct_page_result = lambda layout, results, page_result=None: page_result

    mpocr._recognize_page_layout(layout, page_idx=7)
    summary = mpocr.get_ocr_batch_summary()

    assert summary["schema"] == "mokuro.ocr_batch_summary.v1"
    assert summary["config"]["ocr_batch_size"] == 2
    assert summary["config"]["ocr_reorder_buffer_size"] == 3
    assert summary["totals"] == {"pages": 1, "crops": 5, "batches": 3, "reorder_buffers": 2}
    assert summary["yield_rate"]["batches_per_page"]["max"] == 3
    assert summary["yield_rate"]["crops_per_page"]["mean"] == 5
    assert summary["yield_rate"]["batch_size"]["min"] == 1
    assert summary["yield_rate"]["batch_size"]["max"] == 2
    assert summary["raggedness"]["batches"]["token_range"]["max"] == 1
    assert summary["raggedness"]["reorder_buffers"]["token_range"]["max"] == 2


def test_process_volume_batches_uncached_pages_before_detector_work(tmp_path, monkeypatch):
    class FakeVolume:
        path_in = tmp_path / "in"
        path_ocr_cache = tmp_path / "_ocr"
        mokuro_data = None

        def get_img_paths(self):
            return {
                "000a": "000a.jpg",
                "000b": "000b.jpg",
                "000c": "000c.jpg",
            }

    class FakeMangaPageOcr:
        detector_batch_size = 2

        def __init__(self):
            self.batches = []

        def process_pages(self, img_paths, page_indices=None, timings_fh=None):
            self.batches.append((list(img_paths), list(page_indices)))
            return [
                {"version": "test", "img_width": 1, "img_height": 1, "blocks": [], "page_idx": page_idx}
                for page_idx in page_indices
            ]

    monkeypatch.setattr(MokuroGenerator, "generate_mokuro_file", lambda *args, **kwargs: None)

    mpocr = FakeMangaPageOcr()
    generator = MokuroGenerator()
    generator.mpocr = mpocr

    generator.process_volume(FakeVolume())

    assert mpocr.batches == [
        ([tmp_path / "in" / "000a.jpg", tmp_path / "in" / "000b.jpg"], [0, 1]),
        ([tmp_path / "in" / "000c.jpg"], [2]),
    ]
