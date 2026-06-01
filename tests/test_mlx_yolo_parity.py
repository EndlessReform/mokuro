"""Parity test for the MLX YOLO block head against the Torch detector fixture.

The fixtures and converted MLX artifact live under the gitignored
``output/detector/`` tree, so this test skips cleanly when the optional ``mlx``
extra or generated assets are unavailable. Regenerate them with::

    uv run python -m comic_text_detector.scripts.dump_detector_fixture --out output/detector
    uv run --extra mlx python -m comic_text_detector.scripts.convert_to_mlx dump
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from comic_text_detector.scripts.compare_mlx_yolo_fixture import (
    YOLO_KEY,
    compare_yolo,
)

ARTIFACT_DIR = Path(__file__).parent.parent / "output" / "detector" / "mlx-comictextdetector"
FIXTURE_DIR = Path(__file__).parent.parent / "output" / "detector"

TOLERANCE = 1e-3

requires_mlx = pytest.mark.skipif(
    importlib.util.find_spec("mlx") is None,
    reason="requires the optional mlx extra (comic-text-detector[mlx])",
)


def _load_backend():
    from comic_text_detector.mlx_backend import MlxTextDetComputeBackend

    if not (ARTIFACT_DIR / "config.json").is_file():
        pytest.skip(f"missing MLX artifact: {ARTIFACT_DIR}")
    return MlxTextDetComputeBackend(ARTIFACT_DIR, compute_device="cpu")


def _load_fixture(name: str) -> np.lib.npyio.NpzFile:
    fixture_path = FIXTURE_DIR / name
    if not fixture_path.is_file():
        pytest.skip(f"missing detector fixture: {fixture_path}")
    return np.load(fixture_path)


@requires_mlx
@pytest.mark.parametrize("fixture_name", ["single.npz", "batch.npz"])
def test_mlx_yolo_head_matches_torch_fixture(fixture_name):
    fixture = _load_fixture(fixture_name)
    backend = _load_backend()

    actual = backend.forward_yolo_decoded(fixture["input.nchw.fp32"])
    row, passed = compare_yolo(fixture, actual, rtol=TOLERANCE, atol=TOLERANCE)

    assert passed, (
        "MLX YOLO decoded output diverges from Torch fixture: "
        f"{row['key']} max_abs={row['max_abs']:.3e}"
    )
    assert row["key"] == YOLO_KEY
    assert row["shape"] == list(fixture[YOLO_KEY].shape)
