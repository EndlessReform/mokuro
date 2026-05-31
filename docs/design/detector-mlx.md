# MLX Text Detector Design

## Goal

Add an optional MLX compute backend for `comic-text-detector`, then expose it through mokuro as `mokuro[mlx]`.

The immediate target is the model forward pass used by bounding box detection. After detector batching, this path is large enough to matter in end-to-end runtime, and on Apple Silicon the current Torch/MPS path may still leave AMX/Metal performance on the table. The MLX backend should be a drop-in detector compute layer, not a rewrite of mokuro's page layout logic.

The boundary should stay:

```text
OpenCV/numpy preprocessing -> detector tensor -> compute backend -> ndarray outputs -> existing postprocess
```

This keeps letterboxing, inverse scaling, NMS, DB contour extraction, `group_output(...)`, `refine_mask(...)`, and `TextBlock` construction in the current Python/OpenCV code until benchmarks prove a narrower postprocess migration is worth it.

## Current Code Shape

- Mokuro initializes `TextDetector` once in `MangaPageOcr.__init__` and already chooses CUDA, then MPS, then CPU when not forced to CPU (`mokuro/manga_page_ocr.py:216`).
- Mokuro calls `process_pages(...)`, loads page layouts, calls `detect_pages(...)`, then OCRs all crop requests (`mokuro/manga_page_ocr.py:232`).
- `detect_pages(...)` already delegates to `TextDetector.detect_batch(...)` when available (`mokuro/manga_page_ocr.py:267`).
- `TextDetector.detect_batch(...)` stacks preprocessed tensors, calls `self.net(img_in)`, then loops per page for NMS, DB contour extraction, mask resize, grouping, and refinement (`comic_text_detector/inference.py:205`).
- The compute object is `TextDetBase`: `blk_det` trunk, `text_seg` U-Net style mask head, and `text_det` DB head (`comic_text_detector/basemodel.py:222`).
- `TextDetBase.forward(...)` currently returns `blks[0], mask, lines`, so the batched path depends on the forked submodule having removed that single-item assumption or having another local patch (`comic_text_detector/basemodel.py:240`).

## 1. Pyproject Throatclearing And Local Dev

The repo is in an intentionally vendored packaging state: mokuro's root package includes `comic_text_detector*` through setuptools discovery (`pyproject.toml:44`), and the detector source is tracked as a submodule. History suggests this was pragmatic rather than driven by a unique licensing requirement:

- mokuro and comic-text-detector are both GPL-3.0, so vendoring is license-compatible as long as source and notices are conveyed;
- the original setup depended on normal PyPI packages but had no installable `comic-text-detector` distribution to depend on;
- `comic_text_detector/LICENSE` was explicitly added to the pip package shortly after setup packaging;
- a uv workspace would be a good local-development shape today, but it is not itself a published PyPI dependency strategy.

So the packaging choice is less "vendoring was wrong" and more "extras need a distribution boundary." If we want `mokuro[mlx]` to pull detector MLX dependencies transitively, then `comic-text-detector` needs to become an installable dependency with its own extras. If we only care about repo-local development, a workspace or editable path source is enough.

Implemented packaged/workspace end state:

1. Make `comic_text_detector/` its own installable distribution with `project.name = "comic-text-detector"` and import package `comic_text_detector`.
2. Move detector runtime dependencies into `comic_text_detector/pyproject.toml`.
3. Add a detector extra:

```toml
[project.optional-dependencies]
mlx = [
    "mlx>=0.31",
    "safetensors>=0.5",
    "huggingface-hub>=1.0",
]
```

4. In mokuro, depend on the detector distribution instead of packaging the submodule:

```toml
[project]
dependencies = [
    "comic-text-detector",
    # existing mokuro dependencies...
]

[project.optional-dependencies]
mlx = [
    "comic-text-detector[mlx]",
]

[tool.uv.workspace]
members = ["comic_text_detector"]

[tool.uv.sources]
comic-text-detector = { workspace = true }

[tool.setuptools.packages.find]
include = ["mokuro*"]
```

The `tool.uv.workspace` and `tool.uv.sources` entries are for local development only. uv documents `project.optional-dependencies` as the published extras table, `tool.uv.workspace` as the set of local member projects, and `{ workspace = true }` as the source that tells uv to resolve `comic-text-detector` from the member package. uv installs workspace members in editable mode during `uv sync` and `uv run`. See:

- https://docs.astral.sh/uv/concepts/projects/dependencies/
- https://docs.astral.sh/uv/concepts/projects/config/#editable-mode
- https://docs.astral.sh/uv/concepts/projects/workspaces/

Local commands after the packaging split:

```bash
uv sync
uv sync --extra mlx
uv run python -c "import comic_text_detector, mokuro"
uv run --extra mlx python -c "import mlx.core as mx; import comic_text_detector"
```

Release note: for PyPI users, workspace membership is not enough by itself. mokuro releases need either a separately published `comic-text-detector` distribution or a deliberate return to vendoring for sdists/wheels. The workspace setup is the right development topology, while the published packaging story is a separate release decision.

Compatibility note: `mokuro[mlx]` should only promise "install MLX-capable detector bits." Runtime selection should still check platform and artifact availability and fall back cleanly to Torch when MLX is unavailable or unsupported.

## 2. Test Fixture Script

Before writing an MLX implementation, create a fixture generator inside `comic_text_detector` that records Torch truth as plain ndarrays. This is the DMZ between frameworks.

Implemented script:

```text
comic_text_detector/scripts/dump_detector_fixture.py
```

Default command shape:

```bash
uv run python -m comic_text_detector.scripts.dump_detector_fixture \
  --image tests/data/input/test0/vol1/000a.jpg \
  --batch-image tests/data/input/test0/vol1/001a.jpg \
  --input-size 1024 \
  --out /tmp/ctd-fixture
```

The script defaults to the same detector checkpoint mokuro uses: `${XDG_CACHE_HOME:-~/.cache}/manga-ocr/comictextdetector.pt`. If it is missing, the script downloads mokuro's default detector artifact from `beta-0.2.1/comictextdetector.pt`. `--checkpoint` remains available as an override for testing a different detector file.

Outputs:

```text
/tmp/ctd-fixture/
  manifest.json
  single.npz
  batch.npz
  single-final.json
  batch-final.json
```

`manifest.json` records:

- fixture schema, e.g. `comic_text_detector.mlx_fixture.v1`
- git commit of the detector submodule
- checkpoint path and SHA256
- image paths and image SHA256 values
- input size, activation name, dtype, device, and thresholds
- package versions for `torch`, `numpy`, `opencv-python`, and optionally `mlx`
- generated split summaries, including NMS counts, line counts, and final block counts

Each `.npz` uses ndarray values only:

```text
input.nchw.fp32             # right before compute backend; shape (N, 3, H, W)
input.nhwc.fp32             # same data transposed for MLX convenience
preprocess.dw_dh            # int32, shape (N, 2)
preprocess.resize_ratio     # float32, shape (N, 2)
trunk.feature_1             # selected YOLO feature map
trunk.feature_3
trunk.feature_5
trunk.feature_7
trunk.feature_9
head.yolo_decoded           # pre-NMS model output, shape (N, anchors, 6)
head.mask                   # raw mask tensor, shape (N, 1, H, W)
head.lines                  # DB output tensor, shape (N, 2, H, W)
post.mask_uint8             # raw uint8 mask on the fixed detector input canvas
```

Ragged outputs use fixed arrays plus counts:

```text
post.nms.values             # float32, shape (total_boxes, 6)
post.nms.counts             # int32, shape (N,)
post.lines.values           # int32, shape (total_lines, 4, 2)
post.lines.scores           # float32, shape (total_lines,)
post.lines.counts           # int32, shape (N,)
```

`single-final.json` and `batch-final.json` serialize the existing public detector result:

- resized page mask shape and checksum
- refined mask shape and checksum
- each `TextBlock.to_dict()` result

The fixture dumps intermediates with explicit hooks:

1. `preprocess_img(..., to_tensor=False)` to capture the right-before tensor as numpy, then manually produce NCHW normalized fp32.
2. `blk_det(input, detect=True)` to capture decoded YOLO output and trunk features.
3. `text_seg(*features, forward_mode=TEXTDET_INFERENCE)` to capture raw mask and seg features.
4. `text_det(*seg_features, step_eval=False)` to capture DB line maps.
5. Existing postprocess path to capture final masks and blocks.

Acceptance tests:

- Torch single and Torch batch fixtures agree for the first image at all shared boundaries within exact or near-exact tolerance.
- Fixture generation is deterministic on CPU.
- The MLX backend can load the fixture and compare every implemented layer group before plugging into mokuro.

==NOTE==: Sample for test0 is at `output/detector` (not tracked)

## 3. MLX Handoff

Keep preprocessing in numpy/OpenCV for the first pass.

Reasons:

- `preprocess_img(...)` already handles BGR/RGB conversion, letterbox geometry, padding, and resize metadata (`comic_text_detector/inference.py:75`).
- Postprocess needs per-page geometry and OpenCV-heavy work anyway (`comic_text_detector/inference.py:237`).
- Moving preprocessing to MLX would add layout conversions before we know the model forward is faster.

Implemented a small backend protocol inside `comic_text_detector/backends.py`:

```python
class TextDetComputeBackend(Protocol):
    name: str

    def forward(self, input_nchw: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return yolo_decoded, mask, lines as ndarrays."""
```

Implemented Torch backend contract:

```text
np.ndarray NCHW fp32/bf16-ish request
-> torch.from_numpy(...).to(device)
-> TextDetBase
-> detach().cpu().numpy()
```

`TorchTextDetComputeBackend.forward(...)` is now the existing detector compute path. It returns ndarray outputs at the backend boundary, and `TextDetector` feeds those arrays into the existing NMS, mask, DB line-map, grouping, and refinement postprocess.

Stubbed MLX backend:

```text
comic_text_detector/mlx_backend.py
```

The stub imports `mlx.core` lazily during `MlxTextDetComputeBackend(...)` construction, so importing `comic_text_detector`, `comic_text_detector.inference`, or `comic_text_detector.mlx_backend` does not require the optional `mlx` extra. Requesting `backend="mlx"` fails clearly when the extra is missing, and currently also fails clearly after import because model conversion and forward execution are not implemented yet.

Future MLX backend implementation:

```text
np.ndarray NCHW fp32
-> mx.array(...)
-> transpose to NHWC once
-> MLX detector model
-> transpose outputs back to NCHW-compatible ndarray shapes
```

MLX uses channels-last input for `nn.Conv2d`; the official docs describe Conv2d input as `NHWC`. MLX `conv_transpose2d` also takes `(N, H, W, C_in)` and weights in `(C_out, KH, KW, C_in)`, so weight conversion must be deliberate rather than a blind state dict load.

The public `TextDetector` wrapper can gain a `backend` argument:

```python
TextDetector(..., backend="auto")  # auto | torch | opencv | mlx
```

Implemented selection rules:

- `backend="torch"`: current behavior.
- `backend="opencv"`: current ONNX path.
- `backend="mlx"`: require `comic-text-detector[mlx]`, require an MLX artifact, and fail clearly if unavailable.
- `backend="auto"`: currently preserves existing behavior: ONNX models select OpenCV, all other models select Torch. The later artifact-aware Apple Silicon MLX auto-selection should be added only after a real MLX artifact and forward implementation exist.

Implementation notes:

- `TextDetBase.forward(...)` now returns the full decoded YOLO batch instead of `blks[0]`, matching the backend contract shape `(N, anchors, 6)`.
- `TextDetector` still keeps `backend` as the public string state for compatibility, but routes Torch/MLX compute through `self.compute_backend.forward(input_nchw)`.
- `postprocess_yolo(...)` and the batched path now accept ndarray backend outputs without wrapping them back into a Torch model object.
- Added tests for the Torch ndarray contract and the lazy MLX import/failure behavior.

Mokuro pass-through:

```python
TextDetector(
    model_path=cache.comic_text_detector,
    input_size=detector_input_size,
    device=device,
    act="leaky",
    backend=detector_backend,
)
```

Add a mokuro CLI knob later only if `auto` is not enough:

```text
--detector-backend auto|torch|mlx|opencv
```

Output contract from MLX must match the current Torch return before postprocess:

```text
yolo_decoded: np.float32, shape (N, anchors, 6)
mask:         np.float32 or np.bfloat16 converted to np.float32, shape (N, 1, H, W)
lines:        np.float32 or np.bfloat16 converted to np.float32, shape (N, 2, H, W)
```

Do not move NMS to MLX in the first pass. `non_max_suppression(...)` is already batch-aware, and the result is ragged by nature (`comic_text_detector/inference.py:119`).

## 4. Reproducible MLX Artifact Script

Created the conversion script shell in the detector repo:

```text
comic_text_detector/scripts/convert_to_mlx.py
```

Minimum viable inspection command:

```bash
uv run python -m comic_text_detector.scripts.convert_to_mlx keys \
  --out /tmp/comictextdetector-keys.txt
```

The `keys` subcommand loads the same upstream detector checkpoint source used by mokuro and the fixture script: `${XDG_CACHE_HOME:-~/.cache}/manga-ocr/comictextdetector.pt`, downloading `comictextdetector.pt` from the `zyddnys/manga-image-translator` `beta-0.2.1` release if it is not already cached. The detector repo itself does not currently contain a Hugging Face detector model id; `--hf-repo-id`, `--hf-filename`, and `--hf-revision` are explicit options only, so we can point at a real mirror later without hard-coding the OCR model id by mistake.

The key report is tab-separated text:

```text
# key  dtype  shape  elements  bytes
blk_det.weights.model.0.conv.weight  float16  (32, 3, 6, 6)  3456  6912
...

# total model size
total_tensors   670
total_elements  23453232
total_bytes     79725266
total_mib       76.032
```

Implemented local fp32 safetensors dump:

```bash
uv run --extra mlx python -m comic_text_detector.scripts.convert_to_mlx dump
```

Default local output path while modeling is under ignored `output/`:

```text
output/detector/mlx-comictextdetector/
  config.json
  model.fp32.safetensors
```

This keeps the large experimental artifact near the existing detector fixture output without making it a release asset. Pass `--out-dir` only when testing another local location.

The first dump is fp32 only. Floating tensors are converted to `torch.float32` before writing; integer buffers such as `num_batches_tracked` remain integer tensors. Based on `output/detector/comictextdetector-keys.txt`, layout transforms are:

```text
Conv2d:          torch OIHW -> MLX OHWI
ConvTranspose2d: torch IOHW -> MLX OHWI
```

Current real checkpoint conversion summary:

```text
tensor_count                    670
total_elements                  23453232
total_mib                       89.467
conv2d_oihw_to_ohwi             103
conv_transpose2d_iohw_to_ohwi   12
none                            555
```

The transposed-conv keys are classified from the detector module definitions, not just rank:

- `text_seg.upconv*.conv.1.weight`
- `text_seg.upconv6.0.weight`
- `text_det.upconv*.conv.1.weight`
- `text_det.binarize.3.weight`
- `text_det.binarize.6.weight`
- `text_det.thresh.3.weight`
- `text_det.thresh.6.weight`

`config.json` records the source checkpoint path, SHA256, upstream URL, artifact filename, dtype, tensor counts, layout transforms, per-tensor source/output shape and dtype, and the YOLOv5 `blk_det.cfg` block needed to reconstruct the trunk:

```json
{
  "schema": "comic_text_detector.mlx_conversion.v1",
  "source": {
    "checkpoint_path": ".../comictextdetector.pt",
    "checkpoint_sha256": "...",
    "checkpoint_url": "https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.2.1/comictextdetector.pt",
    "format": "comictextdetector.pt"
  },
  "artifact": {
    "file": "model.fp32.safetensors",
    "dtype": "fp32",
    "tensor_count": 670,
    "total_elements": 23453232,
    "total_bytes": 93813364,
    "total_mib": 89.467
  },
  "layouts": {
    "public_input": "NCHW",
    "mlx_internal_input": "NHWC",
    "conv2d_weight": "OHWI",
    "conv_transpose2d_weight": "OHWI"
  }
}
```

Later full conversion/upload command shape:

```bash
uv run --extra mlx python -m comic_text_detector.scripts.convert_to_mlx dump \
  --checkpoint ~/.cache/manga-ocr/comictextdetector.pt \
  --out-dir dist/mlx-comictextdetector \
  --dtype fp32 \
  --dtype bf16 \
  --fixture /tmp/ctd-fixture \
  --repo-id EndlessReform/comic-text-detector-mlx \
  --upload
```

Artifact layout:

```text
dist/mlx-comictextdetector/
  config.json
  model.fp32.safetensors
  model.bf16.safetensors
  README.md
  fixture-report.json
```

`config.json` should record:

- source checkpoint SHA256
- source detector git commit
- model architecture version
- activation mode (`leaky`)
- input layout expected by loader (`NCHW` API, internal `NHWC`)
- output names and shapes
- dtype
- conversion script version

Conversion steps:

1. Load the upstream `.pt` bundle with `torch.load(..., map_location="cpu")`.
2. Instantiate the Torch `TextDetBase` exactly as inference does.
3. Fuse Conv+BN where possible before export if the MLX graph is designed around fused convs. If unfused, export BN running stats and affine parameters.
4. Build the MLX module with the same layer graph.
5. Convert tensors:
   - Torch Conv2d: `(out, in, kh, kw)` -> MLX Conv2d: `(out, kh, kw, in)`.
   - Torch ConvTranspose2d: verify against fixture; MLX core docs specify `(C_out, KH, KW, C_in)`.
   - BatchNorm: preserve eval-mode affine/running stats or pre-fold into conv weights.
   - Buffers: anchors, strides, grids if the YOLO decode is implemented inside MLX.
6. Save `model.fp32.safetensors`.
7. Cast floating arrays to `mx.bfloat16` and save `model.bf16.safetensors`.
8. Load both artifacts back and run fixture parity.
9. Upload with `huggingface_hub.HfApi.upload_file(...)` or `hf upload`.

The MLX docs state that `mx.save_safetensors(...)` saves a dict of names to arrays, and `mx.load(...)` loads `.safetensors` into a dict. Hugging Face's hub docs show `HfApi.upload_file(...)` and `hf upload` as the normal single-file upload paths.

Parity gates:

- fp32 trunk feature max error target: start at `<= 1e-4`, relax only with measured explanation.
- fp32 final tensors: `yolo_decoded`, `mask`, `lines` should be close enough that NMS/block output is unchanged on fixtures.
- bf16 should compare against Torch autocast or a bf16 MLX baseline with looser tensor tolerances, but final block output should be tracked separately.
- Conversion fails if final block JSON changes on the canonical fixture unless `--accept-output-drift` is supplied.

## 5. Layer Breakout

### Input

Current Torch-facing input:

```text
np image BGR uint8
-> cv2.cvtColor(..., BGR2RGB)
-> letterbox(...)
-> transpose HWC to CHW
-> normalize / 255
-> shape (N, 3, 1024, 1024)
```

MLX-facing API should still accept this NCHW ndarray. The backend converts once internally:

```text
NCHW -> NHWC
```

The fixture should treat NCHW fp32 as the stable input DMZ. NHWC is a derived convenience artifact.

### Trunk

The trunk is the YOLOv5 block detector loaded by `load_yolov5_ckpt(...)` from `textdetector_dict["blk_det"]` (`comic_text_detector/basemodel.py:213`).

It returns:

- decoded YOLO boxes before NMS
- selected feature maps at indices `[1, 3, 5, 7, 9]`

The MLX implementation should start by matching only the inference graph used by the checkpoint:

- `Conv`
- `C3`
- `Bottleneck`
- `SPP/SPPF` if present in the checkpoint config
- concat/route operations from YOLO parse config
- final `Detect` decode

Do not port training-only modules unless the checkpoint config requires them.

Risk: the YOLO `Detect` layer mutates grids and anchor grids in Torch. In MLX, implement deterministic grid creation from shape, anchors, and stride and include anchors/stride in `config.json` or safetensors.

### Segmentation Head

`UnetHead` takes feature maps `(f160, f80, f40, f20, f3)` and returns:

- `mask`
- feature tuple `(f80, f40, u40)` for the DB head in inference mode

Core modules:

- `double_conv_c3`
- `double_conv_up_c3`
- `ConvTranspose2d`
- `BatchNorm2d`
- ReLU
- final sigmoid

For MLX, decide early whether to fold BN into neighboring convs. Folding simplifies runtime and reduces parity surface. Keep an unfused reference path available until the fixture passes.

### DB Head

`DBHead` takes `(f80, f40, u40)` and returns two maps in eval mode:

```text
cat(shrink_maps, threshold_maps), shape (N, 2, H, W)
```

The current postprocess only uses the first channel through `SegDetectorRepresenter.__call__`, which indexes `pred[:, 0, :, :]`. Keep both channels in the output contract anyway because it preserves the current model boundary and makes parity testing less confusing.

### Postprocess

Keep in existing code:

- `non_max_suppression(...)`
- line extraction via `SegDetectorRepresenter`
- mask conversion/resizing
- `group_output(...)`
- `refine_mask(...)`
- `refine_undetected_mask(...)`

The MLX backend should return ndarrays that can be fed into the same postprocess helpers. If helpers currently require Torch methods, update them to accept numpy arrays at the boundary rather than wrapping MLX outputs back into Torch.

## Implementation Stages

### Stage A: Packaging

- Add `comic_text_detector/pyproject.toml`.
- Move detector runtime requirements from `requirements.txt` to pyproject.
- Add `comic-text-detector[mlx]`.
- Update mokuro root `pyproject.toml` to depend on `comic-text-detector`, add `mokuro[mlx]`, add uv editable path source, and stop packaging `comic_text_detector*` directly.
- Verify `uv sync`, `uv sync --extra mlx`, and imports.

### Stage B: Fixture

- Add the fixture dump script.
- Add CPU fixture tests for single and batch.
- Commit a tiny fixture manifest or generated test output only if it is small enough for the repo; otherwise document exact generation commands.

### Stage C: MLX Skeleton

- Add `comic_text_detector/mlx_backend/`.
- Implement model classes with random weights first.
- Load safetensors and config.
- Run fixture shape checks.

### Stage D: Layer Parity

- Port trunk until selected feature maps pass.
- Port segmentation head until `mask` passes.
- Port DB head until `lines` passes.
- Only then wire `TextDetector(..., backend="mlx")`.

### Stage E: Mokuro Integration

- Add optional detector backend argument only if needed.
- Ensure `force_cpu` disables MLX auto-selection.
- Add an integration smoke test that monkeypatches an MLX-like backend returning ndarray outputs.

### Stage F: Benchmarks

Benchmark at three levels:

- raw compute backend forward on fixture tensors
- `TextDetector.detect_batch(...)` including postprocess
- full mokuro volume run with `--detector-batch-size`, `--ocr-batch-size`, and `--ocr-bf16`

Record CPU, MPS/Torch, MLX fp32, and MLX bf16 on at least one M-series machine. On M5+, include batch sizes 1, 4, 8, 16, and 32 to find the memory/performance knee.

## Open Questions

- Does MLX bf16 produce stable enough detector outputs for block grouping, or should bf16 be opt-in even inside `mokuro[mlx]`?
- Should YOLO decode live in MLX or remain numpy postprocess? Keeping decode in MLX better matches the current `TextDetBase.forward(...)` output, but returning raw head logits would make the backend boundary less drop-in.
- Should conversion fold every Conv+BN pair, including heads, or preserve BN modules for simpler state-dict traceability?
- Where should the official MLX artifact live: detector fork owner namespace, mokuro namespace, or an upstream-compatible `comic-text-detector-mlx` model repo?
- Should the default cache downloader learn about `model.fp32.safetensors` and `model.bf16.safetensors`, or should MLX artifact paths be explicit until the backend has shipped?

## References

- uv dependency fields, optional dependencies, and path/editable sources: https://docs.astral.sh/uv/concepts/projects/dependencies/
- uv editable mode: https://docs.astral.sh/uv/concepts/projects/config/#editable-mode
- uv workspaces and editable workspace dependencies: https://docs.astral.sh/uv/concepts/projects/workspaces/
- MLX saving/loading arrays and safetensors: https://ml-explore.github.io/mlx/build/html/usage/saving_and_loading.html
- MLX Conv2d NHWC layout: https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.Conv2d.html
- MLX conv_transpose2d shape notes: https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.conv_transpose2d.html
- Hugging Face Hub uploads: https://huggingface.co/docs/huggingface_hub/en/guides/upload
