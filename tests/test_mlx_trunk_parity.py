"""Parity test for the MLX YOLO trunk against the Torch detector fixture.

The fixtures (``single.npz`` / ``batch.npz``) and the converted MLX artifact live
under the gitignored ``output/detector/`` tree, so this test skips cleanly when the
optional ``mlx`` extra or the generated assets are unavailable. Regenerate them with::

    uv run python -m comic_text_detector.scripts.dump_detector_fixture --out output/detector
    uv run --extra mlx python -m comic_text_detector.scripts.convert_to_mlx dump

The comparison runs on the MLX CPU stream on purpose: the GPU conv2d uses a
Winograd fast path that loses fp32 accuracy on 3x3 stride-1 convs at >= 64 px,
which breaks tight parity (trunk.feature_7/9 drift to ~1.8e-2). The error is
bounded, not exponential -- it even attenuates through the heads -- but CPU is
required for numerical parity (see mlx_backend.py).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from comic_text_detector.scripts.compare_mlx_trunk_fixture import (
    TRUNK_KEYS,
    compare_features,
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
    # The precision-preserving CPU stream is required for trunk parity.
    return MlxTextDetComputeBackend(ARTIFACT_DIR, compute_device="cpu")


def _load_fixture(name: str) -> np.lib.npyio.NpzFile:
    fixture_path = FIXTURE_DIR / name
    if not fixture_path.is_file():
        pytest.skip(f"missing detector fixture: {fixture_path}")
    return np.load(fixture_path)


@requires_mlx
@pytest.mark.parametrize("fixture_name", ["single.npz", "batch.npz"])
def test_mlx_trunk_matches_torch_fixture(fixture_name):
    fixture = _load_fixture(fixture_name)
    backend = _load_backend()

    actual = backend.forward_trunk_features(fixture["input.nchw.fp32"])
    rows, passed = compare_features(fixture, actual, rtol=TOLERANCE, atol=TOLERANCE)

    failures = [row for row in rows if not row["allclose"]]
    assert passed, "MLX trunk features diverge from Torch fixture: " + ", ".join(
        f"{row['key']} max_abs={row['max_abs']:.3e}" for row in failures
    )

    assert {row["key"] for row in rows} == set(TRUNK_KEYS)
    for row in rows:
        assert row["max_abs"] <= TOLERANCE
