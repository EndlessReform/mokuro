# OCR Inference DMZ Design

## Goal

Separate mokuro-owned business logic from hardware-specific OCR inference without turning the project into an infrastructure rewrite.

Mokuro should own:

- manga volume discovery, cache layout, and `.mokuro` generation
- page image loading
- text detection and page layout interpretation
- page-level detector batching and page-layout scheduling
- crop extraction, vertical-text normalization, chunk splitting, and chunk reassembly
- deterministic mapping from OCR results back into page/block/line output

Inference engines should own:

- model loading
- tensor preprocessing when engine-specific
- batching and scheduling
- GPU/Metal/CPU placement
- generation kernels and KV/cache behavior
- local or remote execution details

The main boundary is OCR crop inference. Text detection is intentionally left inside mokuro for now. It is lower payoff to externalize because it is one page-level detector call, materially smaller than the OCR stack, and tightly coupled to mokuro's layout/crop extraction through masks and block objects.

Update after the macOS MPS smoke test: text detection should still stay inside mokuro's public API boundary, but it should not stay welded to the per-page OCR call. On Apple Silicon, simply moving the detector from CPU to MPS made end-to-end time much better, which means detector scheduling still matters at tankobon scale. A typical volume has around 160-180 pages; even a now-smaller page-level cost is worth batching and pipelining when multiplied across the volume.

So the revised boundary is:

- Public/stable DMZ: OCR crop inference artifact and adapters.
- Internal mokuro boundary: page image batch -> page layout results.
- Internal reconstruction boundary: page layout results -> OCR crop requests -> page JSON.

The external API surface does not need to change for detector batching. We should first add an internal decoupling stage that lets mokuro batch pages through the detector/YOLOv5 path, then run the existing per-page crop extraction and OCR request collection from cached layout results.

## Why This Boundary

The current hot path is:

1. `MokuroGenerator.process_volume(...)` loops pages serially in `mokuro/mokuro_generator.py`.
2. `MangaPageOcr.__call__(...)` loads one page, runs the detector, splits lines into crops/chunks, and calls OCR once per crop in `mokuro/manga_page_ocr.py`.
3. The upstream `manga_ocr.MangaOcr` wrapper performs single-image generation.

The OCR model is still the main public DMZ candidate: roughly 111M loaded parameters, with autoregressive generation per crop. The detector should remain a mokuro-owned subsystem because its output is masks and `TextBlock`-style layout objects, but its execution should be separable from one-page-at-a-time OCR. The detector model forward is batch-capable; the current single-image constraint is in mokuro/submodule wrapper post-processing and orchestration.

The core design is therefore:

```text
mokuro page loading -> internal layout batch -> OCR batch artifact or sync batch call -> inference engine -> OCR results -> mokuro postprocess/output
```

## Native Batch Artifact

Mokuro should define its own batch artifact instead of making vLLM, SGLang, MLX, or any single engine the canonical interface.

Directory shape:

```text
batch-root/
  manifest.json
  requests.jsonl
  images/
    vol1_000a_b0_l2_c0.png
  results.jsonl
```

`manifest.json`:

```json
{
  "schema": "mokuro.ocr_batch.v1",
  "batch_id": "b123",
  "model": "kha-white/manga-ocr-base",
  "source_volume": "vol1",
  "requests": "requests.jsonl",
  "results": "results.jsonl",
  "generation": {
    "max_tokens": 300,
    "temperature": 0
  }
}
```

`requests.jsonl`:

```json
{"id":"vol1/000a:b0:l2:c0","image_uri":"images/vol1_000a_b0_l2_c0.png","page":"000a.jpg","block":0,"line":2,"chunk":0,"vertical_normalized":true}
```

`results.jsonl`:

```json
{"id":"vol1/000a:b0:l2:c0","text":"...", "error":null}
```

The `id` is the join key. Engines may return rows out of order. Mokuro must reconstruct output by sorting by `page`, `block`, `line`, and `chunk`, not by result-file order.

## Engine Adapters

The native artifact can be consumed by several adapters:

- `sync-local`: current convenience behavior, but internally batched through a backend interface.
- `export`: mokuro writes a batch artifact and stops.
- `import`: mokuro reads `results.jsonl`, reconstructs page OCR JSON, and writes `.mokuro`.
- `vllm-openai-batch`: convert native requests into OpenAI Batch JSONL and run `vllm run-batch`.
- `vllm-native`: a custom vLLM runner/plugin path for the exact manga OCR model if Chat Completions is the wrong internal abstraction.
- `mlx`, `sglang`, or custom runners: read native `requests.jsonl`, write native `results.jsonl`.

For vLLM's OpenAI Batch file format, each request line looks like:

```json
{"custom_id":"vol1/000a:b0:l2:c0","method":"POST","url":"/v1/chat/completions","body":{"model":"mokuro-ocr","messages":[{"role":"user","content":[{"type":"text","text":"Transcribe this manga text crop exactly. Return only the text."},{"type":"image_url","image_url":{"url":"file:///mnt/mokuro/batches/b123/images/vol1_000a_b0_l2_c0.png"}}]}],"temperature":0,"max_completion_tokens":300}}
```

This is useful for VLM-style OCR models that already speak OpenAI-compatible vision chat. For the exact `kha-white/manga-ocr-base` model, a vLLM-native path may be better because the model is a vision encoder plus text decoder, not a chat-tuned VLM. vLLM has encoder-decoder prompt schemas and plugin mechanisms, so this should be treated as integration work rather than impossible.

Reference docs:

- vLLM OpenAI batch format: https://docs.vllm.ai/en/stable/examples/features/openai_batch/
- vLLM `run-batch`: https://docs.vllm.ai/en/latest/cli/run-batch/
- vLLM multimodal inputs: https://docs.vllm.ai/en/v0.21.0/features/multimodal_inputs/
- vLLM encoder-decoder prompt schema: https://docs.vllm.ai/en/v0.18.2/api/vllm/inputs/llm/
- Hugging Face `VisionEncoderDecoderModel`: https://huggingface.co/docs/transformers/model_doc/vision-encoder-decoder

## Staged Plan

### Stage 0: Name the Boundary

User-facing value:

- None visible yet, but it prevents the rest of the work from becoming a rewrite.
- Gives contributors a concrete target: OCR crop inference is the DMZ; detector/layout stay in mokuro.

Implementation:

- Add a small internal OCR request/result data model.
- Document which fields are stable API and which are internal.
- Keep existing sync behavior as the only runtime path.

Acceptance criteria:

- No output changes.
- No CLI behavior changes.
- Unit tests can create OCR request/result objects without importing `manga_ocr`.

Complexity: low.

### Stage 1: Extract a Sync Batch Interface

User-facing value:

- Existing users keep the same command and get the same output.
- Developers get one place to swap OCR backends without touching page geometry code.

Implementation:

- Replace direct `self.mocr(Image.fromarray(line_crop))` calls in `MangaPageOcr.__call__`.
- Collect crop jobs first, then call `ocr_backend.recognize_batch(requests)`.
- Reassemble text lines from returned chunk results.
- Provide `MangaOcrSyncBackend` that preserves current behavior internally, even if it loops one image at a time.

Acceptance criteria:

- Existing fixture tests pass.
- A fake backend test proves block/line/chunk ordering is preserved.
- A split-line test proves chunks concatenate in `chunk` order, not result order.

Complexity: low to moderate.

### Stage 1A: Decouple Page Layout From OCR

User-facing value:

- No new CLI surface, but unlocks detector batching and clearer profiling.
- Makes it possible to run detection/layout across many pages before OCR crop inference.
- Keeps the public OCR artifact focused on crops rather than detector internals.

Implementation:

- Split `MangaPageOcr.__call__` into three internal phases:
  - load page and run text detection/layout,
  - extract OCR crop jobs from a page layout result,
  - reconstruct page JSON from OCR results.
- Introduce an internal page layout result object that holds image dimensions, detector masks needed for crop extraction, and `TextBlock` data.
- Keep this layout result internal at first; it is not the stable export/import artifact.
- Add a detector backend interface with a single-page method first and a batch method next:

```python
detect_page(image) -> PageLayout
detect_pages(images: list[np.ndarray]) -> list[PageLayout]
```

- Implement the initial backend with the current `TextDetector`.
- Then lift the current single-image wrapper into batched forward plus per-image postprocess: stack letterboxed page tensors, call `self.net(batch)`, loop each batch item through NMS, DB contour extraction, mask resize, `group_output(...)`, and `refine_mask(...)`. See "Batch Detection Internals" section for specific blockers.

### Crop Accumulation Pattern

After layout is decoupled from OCR, the orchestration layer can accumulate crops across pages before batching through the OCR model. Two patterns:

**Per-page buffer (Stage 1, simpler):**
```
for page in pages:
    layout = detect_and_layout(page)
    requests = extract_crops(layout, ...)
    results = ocr_backend.recognize_batch(requests)
    page_json = reconstruct_page(layout, requests, results)
    write_cache(page_json)
```

**Cross-page accumulate (Stage 6, higher throughput):**
```
layouts = []
for page in pages:
    layouts.append(detect_and_layout(page))

# Batch detect is already done above; now collect all crops
all_requests = []
for idx, layout in enumerate(layouts):
    all_requests.extend(extract_crops(layout, page_idx=idx, ...))

# Optional: sort by crop aspect ratio to reduce ragged generation padding
# (deferred — per-page crops are already similar enough within a bubble)

all_results = ocr_backend.recognize_batch(all_requests)  # microbatch internally

# Distribute back to pages by page_idx/blk_idx/line_idx/chunk_idx
for idx, layout in enumerate(layouts):
    page_results = [r for r in all_results if r.page_idx == idx]
    page_requests = [r for r in all_requests if r.page_idx == idx]
    page_json = reconstruct_page(layout, page_requests, page_results)
    write_cache(page_json)
```

The cross-page pattern requires `MokuroGenerator.process_volume()` to change from per-page iteration to phase-based orchestration. The per-page pattern only requires `MangaPageOcr.__call__` changes and is the recommended starting point.

Acceptance criteria:

- Existing fixture tests pass with the single-page layout backend.
- A fake detector backend can return layouts without importing `manga_ocr`.
- A batch detector smoke test proves page order is preserved even if postprocess is per-image.
- `force_cpu` still disables CUDA/MPS for both OCR and detection.

Complexity: moderate.

### Stage 2: Local Artifact Export and Import

User-facing value:

- Users can run mokuro preprocessing on one machine and OCR inference on another.
- Users can inspect and debug the exact crops sent to OCR.
- Failed inference jobs can be retried without rerunning detection and crop extraction.

Implementation:

- Add an export mode that writes `manifest.json`, `requests.jsonl`, and crop images.
- Add an import mode that reads `results.jsonl` and writes the normal `_ocr/*.json` and `.mokuro` outputs.
- Keep sync mode as the default.

Possible CLI shape:

```text
mokuro ./vol1 --ocr_mode sync
mokuro ./vol1 --ocr_mode export --ocr_batch_dir ./batches/vol1
mokuro ./vol1 --ocr_mode import --ocr_batch_dir ./batches/vol1
```

Acceptance criteria:

- Exported image count equals OCR request count.
- Import reconstructs identical `.mokuro` output when results are generated by the current backend.
- Import fails clearly when a request id has no result.
- Import fails clearly when a result id was not requested.

Complexity: moderate.

### Stage 3: Add a Simple Engine Runner

User-facing value:

- Users get a first detached workflow without learning vLLM/SGLang/MLX internals.
- The batch contract becomes executable, not just documentation.

Implementation:

- Add a runner script or module that reads native `requests.jsonl`, calls a local batched backend, and writes `results.jsonl`.
- Start with a torch/transformers implementation that batches PIL crops if practical.
- If true batching is not ready, the runner can still consume/write the artifact while looping internally; the artifact workflow remains valuable.

Possible command shape:

```text
python -m mokuro.ocr.runners.local \
  --batch-dir ./batches/vol1 \
  --model kha-white/manga-ocr-base \
  --batch-size 32
```

Acceptance criteria:

- Export -> local runner -> import produces expected test outputs.
- Runner can resume without recomputing completed result rows.
- Runner reports per-request errors without corrupting successful rows.

Complexity: moderate.

### Stage 4: vLLM/OpenAI Batch Export

User-facing value:

- Users with vLLM-compatible OCR/VLM models can use vLLM's scheduler and offline batch runner.
- Batch files can live on local disk, HTTP URLs, or S3-style storage via presigned URLs.

Implementation:

- Add a converter from native `requests.jsonl` to OpenAI Batch JSONL.
- Add a converter from vLLM output JSONL back to native `results.jsonl`.
- Support local image file URLs for trusted local runs.
- Support remote image URLs or presigned URLs for detached/cloud runs.

Possible command shape:

```text
mokuro-ocr-export-openai-batch \
  --batch-dir ./batches/vol1 \
  --output ./batches/vol1/openai_batch.jsonl \
  --model mokuro-ocr

vllm run-batch \
  -i ./batches/vol1/openai_batch.jsonl \
  -o ./batches/vol1/vllm_results.jsonl \
  --model mokuro-ocr \
  --allowed-local-media-path ./batches/vol1/images

mokuro-ocr-import-openai-batch \
  --batch-dir ./batches/vol1 \
  --input ./batches/vol1/vllm_results.jsonl
```

Acceptance criteria:

- Converter preserves `custom_id` as the native request id.
- vLLM success rows become native result rows.
- vLLM error rows become native error rows.
- The import path can complete when all errors are either absent or explicitly ignored.

Complexity: moderate.

### Stage 5: Exact-Model vLLM Integration Spike

User-facing value:

- Determines whether `kha-white/manga-ocr-base` can use vLLM scheduling directly instead of only VLM-style chat models.
- Produces a real go/no-go answer before large implementation work.

Implementation:

- Try loading the model through vLLM's Transformers backend.
- If that fails, prototype a minimal vLLM plugin/model wrapper for the manga OCR `VisionEncoderDecoderModel` shape.
- Map image crops to encoder inputs and decoder start tokens.
- Compare output parity with the current `manga_ocr` wrapper on fixed crop fixtures.

Acceptance criteria:

- A short spike note says one of:
  - works via Transformers backend,
  - works with a small plugin,
  - requires a larger model implementation,
  - not worth pursuing versus a dedicated torch/MLX runner.
- Decision is based on a working command or a concrete failure mode, not architecture speculation.

Complexity: bounded spike first, then moderate to high if plugin work is justified.

### Stage 6: Cross-Page Scheduling

User-facing value:

- Higher throughput on large volumes by giving engines more pages/crops at once.
- Better hardware utilization for offline jobs.

Implementation:

- Change orchestration so mokuro can process volumes in phases:
  - load pages,
  - batch detector/page layout,
  - collect OCR crop requests across pages,
  - run OCR inference or export artifacts,
  - import/reconstruct page JSON.
- Keep page-level cache semantics during import.
- Add progress that distinguishes detection/export, inference, and import.
- Tune detector and OCR batch sizes separately. Detector batches are page tensors around the configured detector input size; OCR batches are 224x224 crop tensors or engine-specific image requests.

Acceptance criteria:

- A volume-level batch can be exported and imported.
- A volume-level run can batch detector forward passes without changing the OCR artifact schema.
- Partial results can be imported for completed pages.
- Existing per-page cache behavior still works for sync mode.

Complexity: moderate to high.

## Explicit Non-Goals For The First Pass

- Do not externalize text detection.
- Do not make detector masks, YOLO outputs, or `TextBlock` internals part of the stable public OCR artifact in the first pass.
- Do not require vLLM for normal mokuro usage.
- Do not require S3, HTTP, or remote services.
- Do not redesign the `.mokuro` output format.
- Do not block sync mode on detached batch mode.
- Do not optimize every engine before the native artifact contract is proven.

## Decision Rules

- If a stage does not produce a user-visible capability, keep it small and internal.
- If an engine requires special behavior, put that behavior in an adapter, not in `MangaPageOcr`.
- If two engines disagree, the native mokuro artifact wins.
- If vLLM Chat Completions fits the selected model, use it.
- If exact manga-ocr needs encoder-decoder prompts or a plugin, use that behind the same artifact contract.
- If an optimization does not improve export/import/sync usability, defer it.

## Internal Data Models

### PageLayout

Internal result of running text detection + post-processing on one page. Carries everything needed for crop extraction without re-running the detector.

```python
@dataclass
class PageLayout:
    img: np.ndarray              # original BGR image (from imread)
    mask: np.ndarray             # raw binary segmentation mask (uint8, im_h x im_w)
    mask_refined: np.ndarray     # refined mask after per-block Otsu + connected components
    blk_list: list[TextBlock]    # text blocks with lines, geometry, transforms
    img_width: int               # original image width
    img_height: int              # original image height
```

This is **not** the stable export artifact. It is an internal intermediate that bridges detection and crop extraction. A `TextBlock` already exists in `comic_text_detector/utils/textblock.py` with methods `xyxy`, `vertical`, `font_size`, `lines_array()`, and `get_transformed_region(...)`.

### OcrRequest

Internal request for one OCR crop, before any batching or export.

```python
@dataclass
class OcrRequest:
    page_idx: int                # index within volume (for progress/caching)
    page_img_path: str           # relative image path (for cache key)
    blk_idx: int                 # block index within page
    line_idx: int                # line index within block
    chunk_idx: int               # chunk index within line (0 if line not split)
    crop: np.ndarray             # pre-extracted, pre-rotated crop (RGB or grayscale)
```

The `id` for the export artifact is derived: `f"{page_img_path}:b{blk_idx}:l{line_idx}:c{chunk_idx}"`.

### OcrResult

Internal result from one OCR inference call.

```python
@dataclass
class OcrResult:
    text: str                    # post-processed recognized text
    error: str | None            # non-null on failure; text is "" in that case
```

## Three-Phase Split (Stage 1A Internal Signatures)

`MangaPageOcr.__call__` at `manga_page_ocr.py:69-140` currently does everything inline. The split produces three methods:

### Phase 1: Detect and Layout

```python
def detect_and_layout(self, img_path: Path) -> PageLayout:
    """Load image, run detector, return layout result.

    Current equivalent: manga_page_ocr.py:70-79
    """
```

This wraps `self.text_detector(img, ...)` at line 79. The detector backend interface is:

```python
class DetectorBackend(Protocol):
    def detect(self, img: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[TextBlock]]: ...
    def detect_batch(self, imgs: list[np.ndarray]) -> list[tuple[np.ndarray, np.ndarray, list[TextBlock]]]: ...
```

### Phase 2: Extract Crops

```python
def extract_crops(self, layout: PageLayout, page_idx: int, img_path: str) -> list[OcrRequest]:
    """Extract rotated/split crops from a page layout.

    Current equivalent: manga_page_ocr.py:80-133 (the block/line/chunk loops
    minus the OCR call at line 112).
    """
```

This contains `split_into_chunks(...)` (line 95), vertical rotation (line 109), and crop collection. Returns a list of `OcrRequest` objects, one per chunk.

### Phase 3: Reconstruct Page JSON

```python
def reconstruct_page(
    self,
    layout: PageLayout,
    requests: list[OcrRequest],
    results: list[OcrResult],
) -> dict:
    """Reassemble OCR results into page JSON output.

    Current equivalent: manga_page_ocr.py:80-138 (the block/line loops
    with chunk text concatenation at line 133).
    """
```

Joins `results` back to `requests` by position (or by id if order was not preserved), concatenates chunks into lines, and builds the `{"blocks": [...]}` dict.

## Batch Detection Internals

The detector model forward pass (`TextDetBase.forward` at `basemodel.py:240-244`) is already batch-capable. The blockers are all in wrapper code:

### Blocker 1: Preprocessing — `inference.py:74-85`

`preprocess_img()` letterboxes one image to `(1, C, H, W)`. For batching, each page has a different aspect ratio, so letterbox padding `(dw, dh)` and resize ratio differ per image. The batched version must:

1. Letterbox each image independently (different padding per image is fine — `letterbox` handles arbitrary shapes).
2. Stack into `(N, C, H, W)` tensor.
3. Return per-image `(ratio_i, dw_i, dh_i)` for inverse mapping in post-processing.

### Blocker 2: Forward pass — `basemodel.py:244`

```python
return blks[0], mask, lines     # <-- [0] assumes batch=1
```

With batched input, `blks` is `(N, anchors, 6)`, `mask` is `(N, 1, H, W)`, `lines` is `(N, 2, H, W)`. Remove `[0]`; return full tensors. The YOLOv5 `Detect` module already reads batch size from tensor shape (`yolo.py` line 29).

### Blocker 3: NMS — `inference.py:103-104`

```python
det = non_max_suppression(det, conf_thresh, nms_thresh)[0]   # <-- [0] assumes batch=1
```

`non_max_suppression()` already returns a list of length `batch_size` (`yolov5_utils.py:151-152`). Replace `[0]` with iteration over the list, applying per-image `resize_ratio[i]`.

### Blocker 4: Post-processing — `inference.py:87-101`, `inference.py:143-180`

`postprocess_mask()` calls `img.squeeze_()` which collapses the batch dimension. `TextDetector.__call__` is structured around one image's `(im_w, im_h)`, one mask resize (line 167), one `group_output` (line 175), and one `refine_mask` (line 176). The batched version:

1. Calls `self.net(batch)` once — single fused forward across N pages.
2. Loops over batch items for NMS (per-image resize ratio), contour extraction (`seg_rep` at line 160, already per-image via `boxes_from_bitmap`), mask resize, `group_output`, and `refine_mask`.

Post-processing is **inherently per-image** (variable contour counts, OpenCV ops, per-page geometry). No point trying to batch these. The pattern is: **batched forward -> loop over batch for post-process**.

## File/Module Layout

```
mokuro/layout/
  __init__.py          # exports PageLayout, DetectorBackend
  models.py            # PageLayout dataclass
  backend.py           # DetectorBackend protocol, TextDetector wrapper implementation
  batch_runner.py      # batched preprocess -> forward -> per-image postprocess loop

mokuro/ocr/
  __init__.py          # exports OcrRequest, OcrResult, OcrBackend
  models.py            # OcrRequest, OcrResult dataclasses
  backend.py           # OcrBackend protocol with recognize_batch()
  local_backend.py     # torch/transformers batched implementation
  artifact.py          # export/import of batch directory (manifest, requests.jsonl, results.jsonl)
  reconstruct.py       # crop request -> result -> page JSON assembly

mokuro/manga_page_ocr.py
  # Refactored __call__ delegates to:
  #   self.layout_backend.detect(img)         -> PageLayout
  #   extract_crops(layout, ...)              -> list[OcrRequest]
  #   self.ocr_backend.recognize_batch(reqs)  -> list[OcrResult]
  #   reconstruct_page(layout, reqs, results) -> dict

mokuro/mokuro_generator.py
  # Cross-page batching: accumulate pages -> batch detect -> collect all crops
  # -> batch OCR -> distribute results -> write per-page cache
```

## Relevant Files And Line Numbers

| File | Lines | Current behavior | Change needed |
|------|-------|-----------------|---------------|
| `manga_page_ocr.py` | 69-140 | Single monolithic `__call__`: detect, crop, OCR, assemble | Split into three phases |
| `manga_page_ocr.py` | 112 | `self._recognize_crop(Image.fromarray(line_crop))` per chunk | Collect crops, batch through `ocr_backend.recognize_batch()` |
| `manga_page_ocr.py` | 142-154 | `_recognize_crop`: fake batch via `x[None].repeat(N)` | Replace with real batched inference |
| `mokuro_generator.py` | 53-76 | Per-page serial loop | Accumulate pages, batch detect, collect crops, batch OCR |
| `inference.py` | 74-85 | `preprocess_img`: single image letterbox + `(1,C,H,W)` tensor | Batch version: per-image letterbox, stack to `(N,C,H,W)`, return per-image params |
| `inference.py` | 103-104 | `postprocess_yolo(...)[0]` | Iterate NMS output list, apply per-image resize ratio |
| `inference.py` | 87-101 | `postprocess_mask`: `squeeze_()` assumes batch=1 | Index per batch element |
| `inference.py` | 143-180 | `TextDetector.__call__`: single image end-to-end | Batched forward -> per-image post-process loop |
| `basemodel.py` | 244 | `return blks[0], mask, lines` | Remove `[0]`, return full-batch tensors |
| `yolov5_utils.py` | 124-218 | `non_max_suppression`: already batch-aware | No change needed |
| `db_utils.py` | 40-69 | `SegDetectorRepresenter`: already batch-aware | No change needed |

## Minimal Useful End State

The first satisfying milestone is not "mokuro runs perfectly on every serving stack." It is:

```text
mokuro export -> engine writes results.jsonl -> mokuro import
```

Once that works, vLLM, SGLang, MLX, and custom inference frontends can compete behind a stable artifact boundary while mokuro remains responsible for the manga-specific pipeline.
