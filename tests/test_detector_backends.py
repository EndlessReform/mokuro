import importlib

import numpy as np
import pytest
import torch

from comic_text_detector.backends import TextDetBackendUnavailable, TorchTextDetComputeBackend


def test_torch_compute_backend_returns_numpy_contract(monkeypatch):
    instances = []

    class FakeTextDetBase:
        def __init__(self, model_path, device="cpu", half=False, act="leaky"):
            self.model_path = model_path
            self.device = device
            self.half = half
            self.act = act
            instances.append(self)

        def __call__(self, input_tensor):
            batch, _channels, height, width = input_tensor.shape
            self.input_shape = tuple(input_tensor.shape)
            return (
                torch.ones((batch, 2, 6), dtype=torch.float32, device=input_tensor.device),
                torch.ones((batch, 1, height, width), dtype=torch.float32, device=input_tensor.device),
                torch.ones((batch, 2, height, width), dtype=torch.float32, device=input_tensor.device),
            )

    monkeypatch.setattr("comic_text_detector.backends.TextDetBase", FakeTextDetBase)

    backend = TorchTextDetComputeBackend("detector.pt", device="cpu", act="leaky")
    yolo_decoded, mask, lines = backend.forward(np.zeros((3, 3, 8, 8), dtype=np.float32))

    assert instances[0].input_shape == (3, 3, 8, 8)
    assert yolo_decoded.shape == (3, 2, 6)
    assert mask.shape == (3, 1, 8, 8)
    assert lines.shape == (3, 2, 8, 8)
    assert yolo_decoded.dtype == np.float32
    assert mask.dtype == np.float32
    assert lines.dtype == np.float32


def test_mlx_backend_module_imports_without_mlx_extra():
    importlib.import_module("comic_text_detector.mlx_backend")


def test_mlx_backend_constructor_reports_missing_extra(monkeypatch):
    mlx_backend = importlib.import_module("comic_text_detector.mlx_backend")

    def fake_import_module(name):
        if name == "mlx.core":
            raise ModuleNotFoundError("No module named 'mlx'", name="mlx")
        return importlib.import_module(name)

    monkeypatch.setattr(mlx_backend.importlib, "import_module", fake_import_module)

    with pytest.raises(TextDetBackendUnavailable, match="requires the optional mlx extra"):
        mlx_backend.MlxTextDetComputeBackend("model.safetensors")
