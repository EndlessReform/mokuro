"""Parity test for the MLX detector heads (mask + DB lines) against the fixture.

This injects the Torch-reference ``trunk.feature_{1,3,5,7,9}`` arrays from the
fixture straight into the MLX ``text_seg``/``text_det`` heads, isolating head
numerics from trunk drift: a failure here is a head-port bug, not a trunk issue.
Use ``compare_mlx_heads_fixture.py --trunk-source mlx`` to instead exercise the
full MLX trunk+heads pipeline.

Runs on the MLX CPU stream: the heads contain 3x3 stride-1 C3 convs that hit the
GPU Winograd fast path at high resolution and lose fp32 accuracy (same as the
trunk), so CPU is required for tight parity. CPU is exact (~1e-5) but slow at
1024x1024, so only the single fixture is exercised here.

The fixtures and converted MLX artifact live under the gitignored
``output/detector/`` tree, so this test skips cleanly when the optional ``mlx``
extra or the generated assets are unavailable. Regenerate them with::

    uv run python -m comic_text_detector.scripts.dump_detector_fixture --out output/detector
    uv run --extra mlx python -m comic_text_detector.scripts.convert_to_mlx dump
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from comic_text_detector.scripts.compare_mlx_heads_fixture import (
    HEAD_KEYS,
    TRUNK_KEYS,
    compare_heads,
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
    # The precision-preserving CPU stream is required for head parity.
    return MlxTextDetComputeBackend(ARTIFACT_DIR, compute_device="cpu")


def _load_fixture(name: str) -> np.lib.npyio.NpzFile:
    fixture_path = FIXTURE_DIR / name
    if not fixture_path.is_file():
        pytest.skip(f"missing detector fixture: {fixture_path}")
    return np.load(fixture_path)


@requires_mlx
def test_mlx_heads_match_torch_fixture_with_torch_trunk():
    fixture = _load_fixture("single.npz")
    backend = _load_backend()

    # Inject the Torch-reference trunk features to isolate the head port.
    trunk_features = {key: fixture[key] for key in TRUNK_KEYS}
    actual = backend.forward_detector_heads(trunk_features)
    rows, passed = compare_heads(fixture, actual, rtol=TOLERANCE, atol=TOLERANCE)

    failures = [row for row in rows if not row["allclose"]]
    assert passed, "MLX heads diverge from Torch fixture: " + ", ".join(
        f"{row['key']} max_abs={row['max_abs']:.3e}" for row in failures
    )

    assert {row["key"] for row in rows} == set(HEAD_KEYS)
    for row in rows:
        assert row["max_abs"] <= TOLERANCE
