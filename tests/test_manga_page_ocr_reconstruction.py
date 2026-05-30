import numpy as np

from mokuro.mokuro_generator import MokuroGenerator
from mokuro.manga_page_ocr import MangaPageOcr, OcrCropResult, PageLayout


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

        def _recognize_page_layout(self, layout, page_idx=0, timings_fh=None):
            return {"page_idx": page_idx, "marker": int(layout.img[0, 0, 0])}

    mpocr = FakeMangaPageOcr.__new__(FakeMangaPageOcr)
    mpocr.disable_ocr = False

    results = mpocr.process_pages(["3", "1", "2"], page_indices=[30, 10, 20])

    assert mpocr.detected_markers == [3, 1, 2]
    assert results == [
        {"page_idx": 30, "marker": 3},
        {"page_idx": 10, "marker": 1},
        {"page_idx": 20, "marker": 2},
    ]


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
