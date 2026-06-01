# MLX Text Detector Design

## Goal

Add an optional MLX compute backend for `comic-text-detector`, then expose it through mokuro as `mokuro[mlx]`.

The backend boundary stays deliberately narrow:

```text
OpenCV/numpy preprocessing
-> NCHW detector tensor
-> compute backend
-> ndarray outputs
-> existing Python/OpenCV postprocess
```

MLX should accelerate the detector model forward pass only. Letterboxing, inverse scaling, YOLO NMS, DB contour extraction, mask resizing, `group_output(...)`, `refine_mask(...)`, and `TextBlock` construction remain in the current detector code unless benchmarks later show that moving a postprocess step is worth the extra complexity.

The public compute contract is:

```python
class TextDetComputeBackend(Protocol):
    name: str

    def forward(self, input_nchw: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return yolo_decoded, mask, and lines as numpy arrays."""
```

Output shapes:

```text
yolo_decoded: float32, shape (N, anchors, 5 + nc), currently (N, anchors, 7)
mask:         float32, shape (N, 1, H, W)
lines:        float32, shape (N, 2, H, W)
```

## Current State

Implemented:

- `comic_text_detector` is a workspace package with an optional `mlx` extra.
- `mokuro[mlx]` depends on `comic-text-detector[mlx]`.
- `TextDetector` has a backend selection path for `auto | torch | opencv | mlx`.
- Torch and MLX compute backends expose the same ndarray contract.
- `TextDetBase.forward(...)` returns the full decoded YOLO batch instead of a single item.
- The fixture dumper records Torch reference tensors and final detector JSON.
- The conversion script writes an fp32 `safetensors` artifact and a config file.
- The MLX backend ports:
  - YOLOv5 trunk layers `0..9`;
  - YOLO PAN neck and Detect decode layers `10..24`;
  - segmentation mask head;
  - DB line head.
- Optional fixture parity tests cover trunk features, mask/DB heads, and decoded YOLO output.

Not implemented yet:

- Mokuro does not auto-select or directly pass an MLX detector artifact yet.
- The MLX model code is still a porting workbench in one large file.
- The MLX forward path is eager and not compiled with `mx.compile`.
- The artifact config is too verbose and should be reshaped before publishing or wiring into `auto`.
- No official MLX artifact download/cache path exists yet.

## Files

Core detector boundary:

```text
comic_text_detector/backends.py
comic_text_detector/inference.py
comic_text_detector/mlx_backend.py
```

Fixture and conversion tools:

```text
comic_text_detector/scripts/dump_detector_fixture.py
comic_text_detector/scripts/convert_to_mlx.py
comic_text_detector/scripts/compare_mlx_trunk_fixture.py
comic_text_detector/scripts/compare_mlx_heads_fixture.py
comic_text_detector/scripts/compare_mlx_yolo_fixture.py
```

Optional parity tests:

```text
tests/test_detector_backends.py
tests/test_mlx_trunk_parity.py
tests/test_mlx_head_parity.py
tests/test_mlx_yolo_parity.py
```

Local generated assets live under ignored `output/detector/`:

```text
output/detector/
  manifest.json
  single.npz
  batch.npz
  single-final.json
  batch-final.json
  mlx-comictextdetector/
    config.json
    model.fp32.safetensors
```

## Fixture Contract

The fixture is the framework-neutral truth boundary. It records the exact tensors that the MLX backend must reproduce before we trust it inside mokuro.

Default generation:

```bash
uv run python -m comic_text_detector.scripts.dump_detector_fixture \
  --image tests/data/input/test0/vol1/000a.jpg \
  --batch-image tests/data/input/test0/vol1/001a.jpg \
  --input-size 1024 \
  --out output/detector
```

The script defaults to the same detector checkpoint mokuro uses:

```text
${XDG_CACHE_HOME:-~/.cache}/manga-ocr/comictextdetector.pt
```

If missing, it downloads the current mokuro detector checkpoint from the `zyddnys/manga-image-translator` `beta-0.2.1` release.

Important `.npz` keys:

```text
input.nchw.fp32
input.nhwc.fp32
preprocess.dw_dh
preprocess.resize_ratio
trunk.feature_1
trunk.feature_3
trunk.feature_5
trunk.feature_7
trunk.feature_9
head.yolo_decoded
head.mask
head.lines
post.mask_uint8
post.nms.values
post.nms.counts
post.lines.values
post.lines.scores
post.lines.counts
```

`single-final.json` and `batch-final.json` serialize final public detector results: mask checksums and `TextBlock.to_dict()` output.

## Artifact Conversion

Current command:

```bash
uv run --extra mlx python -m comic_text_detector.scripts.convert_to_mlx dump \
  --out-dir output/detector/mlx-comictextdetector
```

Current output:

```text
mlx-comictextdetector/
  config.json
  model.fp32.safetensors
```

Current conversion rules:

```text
Conv2d:           torch OIHW -> MLX OHWI
ConvTranspose2d:  torch IOHW -> MLX OHWI
Floating tensors: torch dtype -> fp32
Integer buffers:  preserved
```

The current artifact works for parity, but `config.json` is not publishable shape. It is about 177 KB and includes:

- local source checkpoint path;
- source SHA and URL;
- artifact totals;
- layout transform counts;
- all 670 tensor names and shapes;
- raw YOLOv5 cfg.

Runtime currently needs only:

```text
artifact.file
yolov5.cfg
```

So the next round should split runtime config from conversion audit data.

## MLX Implementation

Current implementation is intentionally direct:

```text
public NCHW np.ndarray
-> mx.array(...)
-> transpose to NHWC
-> MLX model math using converted OHWI weights
-> transpose mask/lines back to NCHW
-> np.ndarray outputs
```

Implemented model pieces:

- `MlxYoloTrunk` runs YOLOv5 block detector layers, including PAN neck and Detect decode.
- `MlxTextDetHeads` runs the segmentation and DB heads.
- `MlxTextDetComputeBackend.forward(...)` returns `(yolo_decoded, mask, lines)`.

The current code is correct enough for parity but not yet idiomatic MLX:

- it is one large `mlx_backend.py` file;
- model math and numpy boundary handling are mixed;
- weights are accessed from a flat dictionary;
- grid creation uses numpy;
- the forward path is eager;
- no `mx.compile` path exists yet;
- head Conv+BN fusion is recomputed during forward rather than cached or converted.

## Parity

CPU-stream parity is the strict correctness gate because MLX GPU uses a fast Winograd path for some larger 3x3 stride-1 convolutions.

Current passing tests:

```bash
uv run --extra mlx python -m pytest \
  tests/test_detector_backends.py \
  tests/test_mlx_trunk_parity.py \
  tests/test_mlx_head_parity.py \
  tests/test_mlx_yolo_parity.py \
  -q
```

Measured CPU-stream parity:

```text
trunk.feature_{1,3,5,7,9}: allclose at 1e-3
head.mask:                 max abs around 4e-6 with Torch trunk injection
head.lines:                max abs around 5e-6 with Torch trunk injection
head.yolo_decoded:          allclose at rtol=1e-3, atol=1e-3
full backend triple:        allclose at rtol=1e-3, atol=1e-3 on fixture
```

Measured MLX GPU vs Torch MPS on the two-image batch fixture:

```text
mask max abs:       ~0.021
mask mean abs:      ~4.8e-5
mask P99 abs:       ~0.0014
lines max abs:      ~0.016
yolo max abs:       ~3.68 px
yolo mean abs:      ~0.014 px
binary mask drift:  ~0.006% pixels at threshold 0.3 or 0.5
```

The GPU drift is small at the final mask decision level, but strict tensor parity should continue to run on the CPU stream.

## Round 1: Make The MLX Path Idiomatic

Before wiring MLX into mokuro auto-selection, clean up the MLX backend and artifact shape.

### Package Layout

Replace the single-file backend with a small package:

```text
comic_text_detector/mlx_backend/
  __init__.py
  backend.py
  configuration_textdet.py
  modeling_textdet.py
  ops.py
```

Responsibilities:

- `backend.py`: public `MlxTextDetComputeBackend`, artifact resolution, numpy in/out boundary, stream/device selection.
- `configuration_textdet.py`: config dataclass plus load/save helpers.
- `modeling_textdet.py`: MLX model classes and pure MLX forward methods.
- `ops.py`: pooling windows, upsample, activations, YOLO grid helpers, and small shared utilities.

### Boundary Split

Keep the public backend API stable:

```text
np.ndarray NCHW -> backend.forward(...) -> np.ndarray outputs
```

Inside the backend, add a pure MLX model API:

```text
mx.array NHWC -> model.forward(...) -> mx.array outputs
```

Rules for the model layer:

- no numpy calls inside model forward;
- no `np.asarray`, `np.arange`, or numpy dtype conversion inside compiled sections;
- grids are generated with `mx.arange` or loaded/cached as MLX arrays keyed by shape;
- NCHW/NHWC conversion happens only at the backend boundary.

### MLX Modules

Prefer `mlx.nn.Module` classes for the core model if they make state ownership cleaner:

```text
MlxComicTextDetector
MlxYoloBlockDetector
MlxYoloLayer
MlxSegmentationHead
MlxDbHead
```

This is not required for numerical parity, but it makes the model easier to reason about, easier to compile, and closer to normal MLX code.

### Config Shape

Replace the current audit-heavy `config.json` with a small runtime config. Move verbose conversion details into `conversion_report.json`.

Target artifact layout:

```text
mlx-comictextdetector/
  config.json
  model.safetensors
  conversion_report.json
  README.md
```

Candidate `config.json`:

```json
{
  "model_type": "comic_text_detector",
  "architectures": ["MlxComicTextDetector"],
  "format_version": 1,
  "torch_dtype": "float32",
  "input_layout": "NCHW",
  "internal_layout": "NHWC",
  "image_size": 1024,
  "num_classes": 2,
  "id2label": {"0": "eng", "1": "ja"},
  "label2id": {"eng": 0, "ja": 1},
  "weights": {
    "file": "model.safetensors",
    "format": "safetensors",
    "layout": "mlx-ohwi"
  },
  "fusion": {
    "conv_bn": "load_time",
    "trunk_bn_eps": 0.001,
    "head_bn_eps": 0.00001
  },
  "yolo": {
    "depth_multiple": 0.33,
    "width_multiple": 0.5,
    "feature_indices": [1, 3, 5, 7, 9],
    "detect_indices": [17, 20, 23],
    "anchors": [
      [[10, 13], [16, 30], [33, 23]],
      [[30, 61], [62, 45], [59, 119]],
      [[116, 90], [156, 198], [373, 326]]
    ],
    "layers": [
      {"type": "Conv", "from": -1, "repeats": 1, "out_channels": 64, "kernel": 6, "stride": 2, "padding": 2},
      {"type": "C3", "from": -1, "repeats": 3, "out_channels": 128, "shortcut": true}
    ]
  },
  "heads": {
    "segmentation": {"activation": "leaky"},
    "db": {}
  }
}
```

`conversion_report.json` should hold:

- source checkpoint path, URL, and SHA256;
- conversion command;
- package versions;
- tensor manifest;
- transform counts;
- per-tensor source/output shapes;
- fixture parity summary.

Normalize the YOLO config during conversion instead of preserving raw Torch cfg strings such as `"None"`, `"nc"`, and `"anchors"`.

### Fusion Policy

Choose one policy and encode it in config:

- conversion-time fusion: save already folded Conv+BN weights where possible;
- or load-time fusion: preserve original converted tensors, then fuse once after load.

Either is fine. Avoid recomputing Conv+BN fusion inside every forward call. The current trunk caches fused weights, but the heads still recompute.

### Compile

MLX `0.31.2` provides:

```python
mx.compile(fun, inputs=None, outputs=None, shapeless=False)
```

Use it only after the model has a pure MLX forward. First target a fixed 1024-square compiled function. Defer `shapeless=True` until it proves valid.

Add an optional compiled-forward parity test that skips when MLX or the local artifact is unavailable.

## Round 2: Wire Into Mokuro

After the idiomatic MLX cleanup, make MLX a real detector backend for mokuro without making normal installs brittle.

### Artifact Discovery

Add an MLX artifact cache path separate from the existing Torch checkpoint:

```text
cache.comic_text_detector       # existing Torch .pt
cache.comic_text_detector_mlx   # new MLX artifact dir or safetensors
```

Do not pass the Torch `.pt` path to `backend="mlx"`.

### User/API Surface

Add detector backend plumbing through mokuro:

```text
detector_backend = auto | torch | mlx | opencv
detector_artifact_path = optional local override
```

CLI shape, if exposed:

```text
--detector-backend auto|torch|mlx|opencv
--detector-artifact-path PATH
```

### Auto Selection

Keep `auto` conservative:

- ONNX models select OpenCV;
- `force_cpu` disables MLX auto-selection;
- MLX auto-selection requires Apple Silicon, importable `mlx`, and an available/downloadable MLX artifact;
- otherwise fall back to Torch with a clear log message.

Explicit `backend="mlx"` should fail clearly if the extra or artifact is missing.

### Tests

Add integration coverage:

- `backend="mlx"` reports a clear error without MLX or without an artifact;
- `TextDetector(..., backend="mlx")` can be smoke-tested with a monkeypatched MLX-like backend returning ndarrays;
- optional fixture-backed real MLX test runs when the local artifact exists;
- mokuro passes detector backend/artifact settings through to `TextDetector`;
- `force_cpu` prevents MLX auto-selection.

## Benchmarks

Benchmark after Round 1, not before. The current eager workbench path is useful for parity, but not representative of the intended MLX runtime.

Measure:

- raw compute backend forward on fixture tensors;
- `TextDetector.detect_batch(...)` including postprocess;
- full mokuro volume run.

Compare:

- Torch CPU;
- Torch MPS;
- MLX GPU fp32 eager;
- MLX GPU fp32 compiled;
- MLX bf16 if implemented.

For Apple Silicon, test detector batch sizes:

```text
1, 4, 8, 16, 32
```

Track both speed and output stability:

- raw tensor drift;
- binary mask pixel disagreement;
- NMS block count and coordinates;
- final `TextBlock` JSON drift.

## Open Questions

- Should bf16 be opt-in, or can it be the default MLX artifact after final block parity is measured?
- Should official artifacts live under mokuro, the detector fork, or a separate `comic-text-detector-mlx` model repo?
- Should conversion-time Conv+BN fusion be preferred over load-time fusion for the published artifact?
- Should MLX auto-selection ever download artifacts automatically, or should the first shipped version require an explicit artifact path?
- After compiled MLX is benchmarked, is any postprocess step worth moving out of Python/OpenCV?

## References

- MLX saving/loading arrays and safetensors: https://ml-explore.github.io/mlx/build/html/usage/saving_and_loading.html
- MLX Conv2d NHWC layout: https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.Conv2d.html
- MLX conv_transpose2d shape notes: https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.conv_transpose2d.html
- Hugging Face Hub uploads: https://huggingface.co/docs/huggingface_hub/en/guides/upload
