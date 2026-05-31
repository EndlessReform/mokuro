from pathlib import Path

import torch

from comic_text_detector.scripts.convert_to_mlx import (
    build_safetensors_payload,
    collect_tensors,
    convert_tensor_for_mlx,
    render_keys_report,
)


def test_collect_tensors_flattens_nested_checkpoint():
    checkpoint = {
        "blk_det": {
            "weights": {
                "model.0.conv.weight": torch.zeros((16, 3, 3, 3), dtype=torch.float32),
                "model.0.bn.bias": torch.zeros((16,), dtype=torch.float32),
            }
        },
        "text_seg": {"upconv.weight": torch.zeros((8, 16, 2, 2), dtype=torch.float16)},
    }

    tensors = collect_tensors(checkpoint)

    assert [item.key for item in tensors] == [
        "blk_det.weights.model.0.bn.bias",
        "blk_det.weights.model.0.conv.weight",
        "text_seg.upconv.weight",
    ]
    assert tensors[0].shape == (16,)
    assert tensors[1].elements == 432
    assert tensors[2].bytes == 1024


def test_render_keys_report_ends_with_total_model_size():
    checkpoint = {
        "a": torch.zeros((2, 3), dtype=torch.float32),
        "b": torch.zeros((4,), dtype=torch.float16),
    }
    tensors = collect_tensors(checkpoint)

    report = render_keys_report(Path("comictextdetector.pt"), tensors)

    assert "a\tfloat32\t(2, 3)\t6\t24" in report
    assert "b\tfloat16\t(4)\t4\t8" in report
    assert "# total model size" in report
    assert "total_tensors\t2" in report
    assert "total_elements\t10" in report
    assert "total_bytes\t32" in report


def test_convert_tensor_for_mlx_transposes_conv2d_weights_to_ohwi():
    tensor = torch.arange(2 * 3 * 4 * 5, dtype=torch.float16).reshape(2, 3, 4, 5)

    converted, layout = convert_tensor_for_mlx("blk_det.weights.model.0.conv.weight", tensor)

    assert layout == "conv2d_oihw_to_ohwi"
    assert converted.shape == (2, 4, 5, 3)
    assert converted.dtype == torch.float32
    assert converted[1, 2, 3, 0] == tensor[1, 0, 2, 3].float()


def test_convert_tensor_for_mlx_transposes_conv_transpose_weights_to_ohwi():
    tensor = torch.arange(2 * 3 * 4 * 5, dtype=torch.float32).reshape(2, 3, 4, 5)

    converted, layout = convert_tensor_for_mlx("text_seg.upconv5.conv.1.weight", tensor)

    assert layout == "conv_transpose2d_iohw_to_ohwi"
    assert converted.shape == (3, 4, 5, 2)
    assert converted[1, 2, 3, 0] == tensor[0, 1, 2, 3]


def test_build_safetensors_payload_records_layout_metadata():
    checkpoint = {
        "conv": {"weight": torch.zeros((2, 3, 1, 1), dtype=torch.float16)},
        "text_det": {"binarize": {"3": {"weight": torch.zeros((4, 5, 2, 2), dtype=torch.float32)}}},
        "scalar": torch.tensor(7, dtype=torch.int64),
    }

    payload, infos = build_safetensors_payload(checkpoint)

    assert payload["conv.weight"].shape == (2, 1, 1, 3)
    assert payload["text_det.binarize.3.weight"].shape == (5, 2, 2, 4)
    assert payload["scalar"].dtype == torch.int64
    assert {info.key: info.layout for info in infos} == {
        "conv.weight": "conv2d_oihw_to_ohwi",
        "scalar": "none",
        "text_det.binarize.3.weight": "conv_transpose2d_iohw_to_ohwi",
    }
