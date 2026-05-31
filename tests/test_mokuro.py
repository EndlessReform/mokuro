import json
import shutil

import pytest
import torch
from loguru import logger
from PIL import Image

from mokuro.manga_page_ocr import MangaPageOcr
from mokuro.run import run


@pytest.mark.parametrize(
    "input_dir_name,disable_html",
    [
        ("test0", True),
        ("test0", False),
        ("test1_webp", True),
        ("test3_convert_legacy_ocr", False),
        ("test3_convert_legacy_ocr", True),
    ],
)
@pytest.mark.parametrize("disable_ocr", [True, False])
def test_mokuro(
    input_dir_name, disable_ocr, disable_html, tmp_path, input_data_root, expected_results_root, regenerate
):
    input_dir, expected_results_dir = _setup_and_run(
        input_dir_name, disable_ocr, False, disable_html, tmp_path, input_data_root, expected_results_root, regenerate
    )

    assert disable_html != (input_dir / "vol1.html").is_file()

    json_paths = sorted((input_dir / "_ocr/vol1").iterdir())
    expected_json_paths = sorted((expected_results_dir / "_ocr/vol1").iterdir())
    _validate_cache_jsons(json_paths, expected_json_paths)

    mokuro_paths = sorted(input_dir.glob("*.mokuro"))
    expected_mokuro_paths = sorted(expected_results_dir.glob("*.mokuro"))
    _validate_mokuro_files(mokuro_paths, expected_mokuro_paths)


@pytest.mark.parametrize("input_dir_name", ["test2_zip"])
@pytest.mark.parametrize("unzip", [True, False])
def test_mokuro_zip(input_dir_name, unzip, tmp_path, input_data_root, expected_results_root, regenerate):
    input_dir, expected_results_dir = _setup_and_run(
        input_dir_name, False, unzip, True, tmp_path, input_data_root, expected_results_root, regenerate
    )

    json_paths = sorted((input_dir / "_ocr/vol1").iterdir())
    expected_json_paths = sorted((expected_results_dir / "_ocr/vol1").iterdir())
    _validate_cache_jsons(json_paths, expected_json_paths)

    mokuro_paths = sorted(input_dir.glob("*.mokuro"))
    expected_mokuro_paths = sorted(expected_results_dir.glob("*.mokuro"))
    _validate_mokuro_files(mokuro_paths, expected_mokuro_paths)

    if unzip:
        assert (input_dir / "vol1").is_dir()

        mokuro = json.loads((input_dir / "vol1.mokuro").read_text(encoding="utf-8"))
        for page in mokuro["pages"]:
            assert (input_dir / "vol1" / page["img_path"]).is_file()

    else:
        assert not (input_dir / "vol1").exists()


def test_page_limit(tmp_path, input_data_root):
    input_dir = tmp_path / "test0"
    shutil.copytree(input_data_root / "test0", input_dir)

    run(
        parent_dir=input_dir,
        force_cpu=True,
        disable_confirmation=True,
        disable_ocr=True,
        legacy_html=False,
        page_limit=2,
    )

    json_paths = sorted((input_dir / "_ocr/vol1").iterdir())
    assert [path.name for path in json_paths] == ["000a.json", "000b.json"]

    mokuro = json.loads((input_dir / "vol1.mokuro").read_text(encoding="utf-8"))
    assert [page["img_path"] for page in mokuro["pages"]] == ["000a.jpg", "000b.jpg"]


def test_cli_int_options_accept_string_values(tmp_path, input_data_root):
    input_dir = tmp_path / "test0"
    shutil.copytree(input_data_root / "test0", input_dir)

    run(
        parent_dir=input_dir,
        force_cpu=True,
        disable_confirmation=True,
        disable_ocr=True,
        legacy_html=False,
        page_limit="2",
        ocr_num_beams="1",
        dev_repeat_ocr_batch_size="1",
        detector_batch_size="2",
        ocr_batch_size="2",
        ocr_reorder_buffer_size="4",
    )

    json_paths = sorted((input_dir / "_ocr/vol1").iterdir())
    assert [path.name for path in json_paths] == ["000a.json", "000b.json"]


def test_dev_repeat_ocr_batch_size_uses_first_output():
    class FakeModel:
        device = torch.device("cpu")

        def __init__(self):
            self.input_shape = None

        def generate(self, x, max_length, **kwargs):
            self.input_shape = tuple(x.shape)
            return torch.tensor([[1, 2, 3], [4, 5, 6]])

    class FakeTokenizer:
        def __init__(self):
            self.seen_tokens = None

        def decode(self, tokens, skip_special_tokens):
            self.seen_tokens = tokens.tolist()
            return "first"

    class FakeMangaOcr:
        def __init__(self):
            self.model = FakeModel()
            self.tokenizer = FakeTokenizer()

        def _preprocess(self, img):
            return torch.zeros(3, 2, 2)

    mpocr = MangaPageOcr.__new__(MangaPageOcr)
    mpocr.ocr_num_beams = None
    mpocr.dev_repeat_ocr_batch_size = 2
    mpocr.mocr = FakeMangaOcr()

    text = mpocr._recognize_crop(Image.new("RGB", (2, 2)))

    assert text
    assert mpocr.mocr.tokenizer.seen_tokens == [1, 2, 3]
    assert mpocr.mocr.model.input_shape == (2, 3, 2, 2)


def test_ocr_num_beams_is_passed_to_generate():
    class FakeModel:
        device = torch.device("cpu")

        def __init__(self):
            self.generate_kwargs = None

        def generate(self, x, **kwargs):
            self.generate_kwargs = kwargs
            return torch.tensor([[1, 2, 3]])

    class FakeTokenizer:
        def decode(self, tokens, skip_special_tokens):
            return "beamed"

    class FakeMangaOcr:
        def __init__(self):
            self.model = FakeModel()
            self.tokenizer = FakeTokenizer()

        def _preprocess(self, img):
            return torch.zeros(3, 2, 2)

    mpocr = MangaPageOcr.__new__(MangaPageOcr)
    mpocr.ocr_num_beams = 2
    mpocr.dev_repeat_ocr_batch_size = 1
    mpocr.mocr = FakeMangaOcr()

    text = mpocr._recognize_crop(Image.new("RGB", (2, 2)))

    assert text
    assert mpocr.mocr.model.generate_kwargs == {"max_length": 300, "num_beams": 2}


def _setup_and_run(
    input_dir_name, disable_ocr, unzip, disable_html, tmp_path, input_data_root, expected_results_root, regenerate
):
    input_dir = tmp_path / input_dir_name
    tag = input_dir_name
    if disable_ocr:
        tag += "_disable_ocr"
    if unzip:
        tag += "_unzip"
    if disable_html:
        tag += "_disable_html"
    expected_results_dir = expected_results_root / tag

    shutil.copytree(input_data_root / input_dir_name, input_dir)
    run(
        parent_dir=input_dir,
        force_cpu=True,
        disable_confirmation=True,
        disable_ocr=disable_ocr,
        unzip=unzip,
        legacy_html=not disable_html,
    )

    if regenerate:
        logger.warning("Regenerating expected results")
        shutil.rmtree(expected_results_dir, ignore_errors=True)
        shutil.copytree(input_dir, expected_results_dir, ignore=shutil.ignore_patterns("*.jpg", "*.webp", "*.zip"))

    return input_dir, expected_results_dir


def _validate_cache_jsons(json_paths, expected_json_paths):
    assert [path.name for path in expected_json_paths] == [path.name for path in json_paths]

    for json_path, expected_json_path in zip(json_paths, expected_json_paths):
        result = json.loads(json_path.read_text(encoding="utf-8"))
        expected_result = json.loads(expected_json_path.read_text(encoding="utf-8"))

        for json_ in (result, expected_result):
            json_.pop("version")

        assert result == expected_result


def _validate_mokuro_files(json_paths, expected_json_paths):
    assert [path.name for path in expected_json_paths] == [path.name for path in json_paths]

    for json_path, expected_json_path in zip(json_paths, expected_json_paths):
        result = json.loads(json_path.read_text(encoding="utf-8"))
        expected_result = json.loads(expected_json_path.read_text(encoding="utf-8"))

        for json_ in (result, expected_result):
            json_.pop("version")
            json_.pop("title_uuid")
            json_.pop("volume_uuid")

            for page in json_["pages"]:
                page.pop("version")

        assert result == expected_result
