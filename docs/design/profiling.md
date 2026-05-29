# Application-Side Profiling Design

## Goal

Profile mokuro on both a development laptop using MPS and CUDA workstations without committing to a low-level profiler first.

The first profiling pass should answer:

- How much wall time is spent in volume orchestration, image discovery, unzip, cache reads, OCR JSON writes, and legacy HTML generation?
- For uncached pages, how much time is spent in page image decode, text detection, crop/chunk preparation, OCR model preprocessing/generation/decode, and result assembly?
- Does runtime scale with page count, detected blocks, line count, chunk count, crop pixels, or output token count?
- On MPS and CUDA, are we mostly waiting on Python/preprocessing, detector compute, OCR compute, model initialization, device synchronization, or filesystem work?
- After the OCR inference DMZ exists, can the same trace shape follow a crop request across local preprocessing, black-box OCR compute, and result import?

This is intentionally application-side instrumentation first. Nsight Systems, Xcode Instruments, and PyTorch operator traces should become easy opt-in drill-downs, not the only way to answer basic "where did the time go?" questions.

## Current Pipeline Shape

The hot path today is small and serial:

- `mokuro.run.run(...)` scans inputs, builds a `VolumeCollection`, and processes each `Volume`.
- `MokuroGenerator.process_volume(...)` creates `_ocr`, optionally converts existing `.mokuro` page data into per-page JSON cache files, gets image paths, then loops pages one at a time with `tqdm`.
- For each uncached or forced page, `MangaPageOcr.__call__(img_path)`:
  - reads the page with `imread`, which uses `cv2.imdecode(np.fromfile(...))`;
  - runs `TextDetector(img, refine_mode=1, keep_undetected_mask=True)`;
  - loops detected blocks and lines;
  - creates transformed crops with `blk.get_transformed_region(...)`;
  - splits long crops with `split_into_chunks(...)`;
  - rotates vertical crops;
  - calls upstream `MangaOcr(...)` once per crop/chunk;
  - concatenates chunk text back into lines.
- `generate_mokuro_file(...)` reads cached page JSON files and writes the volume `.mokuro`.
- If legacy HTML is enabled, `generate_legacy_html(...)` loads every page JSON and emits the old HTML overlay.

There are no profiling hooks today. Existing observability is `loguru` messages plus the page-loop `tqdm` progress bar.

## Instrumentation Principles

Use one small mokuro-owned profiling abstraction and make profiler backends optional.

The important abstraction is a nested span:

```python
with profiler.span("ocr.generate", page=page_id, block=block_idx, line=line_idx, chunk=chunk_idx):
    text = ocr_backend.recognize(crop)
```

Each span should capture:

- `trace_id`: one run-level ID.
- `span_id` and `parent_span_id`: enough to reconstruct nesting.
- `name`: stable operation name such as `page.detect_text` or `ocr.generate`.
- `start_ns`, `end_ns`, `duration_ns`: from `time.perf_counter_ns()`.
- `attributes`: JSON-safe metadata such as page path, image dimensions, block/line/chunk counts, crop dimensions, backend name, device type, cache hit/miss, and errors.
- Optional device snapshots: CUDA/MPS memory counters and explicit synchronization timing when enabled.

Keep event names low-cardinality. Put changing values in attributes. For example, use `page.process`, not `page.process.001a.jpg`.

## Proposed Library Stack

### Baseline: Custom JSONL Spans

Implement `mokuro/profiling.py` with only the standard library:

- `Profiler`: owns the run ID, span stack, output path, and enabled mode.
- `Profiler.span(name, **attrs)`: context manager.
- `Profiler.event(name, **attrs)`: instant event for cache decisions or warnings.
- `Profiler.add_attrs(**attrs)`: enrich the current span.
- `NoopProfiler`: default when profiling is disabled.

Write newline-delimited JSON to something like:

```text
<volume-parent>/.mokuro_profile/<timestamp>-<run-id>/spans.jsonl
<volume-parent>/.mokuro_profile/<timestamp>-<run-id>/summary.json
```

This keeps profiling reusable through the project and across the OCR DMZ, because every process can append or emit the same schema without importing a heavy observability SDK.

### Pyinstrument

Use `pyinstrument` as the first weekend-friendly whole-program profiler. It is a statistical profiler that samples the stack rather than tracing every call, and its docs emphasize lower overhead and wall-clock profiling, which fits mokuro's mix of Python glue, filesystem work, OpenCV, and model calls.

Suggested use:

```bash
pyinstrument -r html -o profile.html -m mokuro ./tests/data/input/test0 --disable_confirmation --no_cache
```

Use it before adding many fine-grained spans. If the flame tree says time is already dominated by one obvious call, avoid over-instrumenting.

### Scalene

Use `scalene` for CPU/native/system attribution and memory-heavy questions. It is especially useful when the question is "is this Python overhead or native library/model code?" because its reports distinguish Python, native, and system time.

Suggested use:

```bash
python -m scalene -m mokuro ./tests/data/input/test0 --disable_confirmation --no_cache
```

Treat Scalene as an investigative tool, not a dependency of mokuro's runtime profiling API.

### PyTorch Profiler

Use `torch.profiler` as the opt-in device/operator drill-down for local torch backends. It can collect CPU and accelerator activities, accepts user ranges through `torch.profiler.record_function`, and exports Chrome JSON traces.

The mokuro profiling abstraction should optionally mirror selected spans into `record_function(name)`:

```python
with profiler.span("page.detect_text"):
    with torch.profiler.record_function("page.detect_text"):
        mask, mask_refined, blk_list = self.text_detector(...)
```

Do not make `torch.profiler` always-on. It changes runtime behavior, produces large traces, and is best scoped to a small number of pages after the JSONL spans identify the interesting area.

### CUDA: NVTX

Add optional NVTX ranges when CUDA profiling is enabled. The Python `nvtx` package supports context-manager annotations, and those ranges show up in Nsight Systems.

Mokuro should not require NVTX. Instead, the profiler can opportunistically import it:

```python
with profiler.span("ocr.generate"):
    ...
```

and, if `nvtx` is installed and `--profile-nvtx` is enabled, the same span opens an NVTX range. This sets up a clean path to `nsys profile` later without scattering NVIDIA-specific calls through the codebase.

### MPS: PyTorch Signposts and Xcode Instruments

MPS profiling should stay separate from CUDA assumptions. PyTorch exposes MPS profiler signposts through environment variables such as `PYTORCH_MPS_TRACE_SIGNPOSTS`, and `torch.mps.profiler.start(...)` starts OS Signpost tracing that can be viewed in Xcode Instruments.

For the weekend version:

- keep mokuro JSONL spans as the source of truth on the laptop;
- add a CLI hint or `--profile-mps-signposts` mode that starts/stops `torch.mps.profiler` around a short run;
- avoid pretending PyTorch's CUDA-style kernel timeline gives equivalent detail on MPS.

### OpenTelemetry Later

OpenTelemetry is a good later bridge once the OCR DMZ becomes a service or separate process. Its Python API supports manual nested spans and span attributes. However, it is heavier than needed for the first local profiling pass.

Recommended sequence:

1. Implement mokuro JSONL spans.
2. Add an exporter that converts JSONL spans to OpenTelemetry or Chrome trace format.
3. When OCR runs out-of-process, propagate `trace_id`, `parent_span_id`, page ID, and crop request IDs across the DMZ manifest/API.

## CLI Shape

Add a few profiling arguments to `run(...)`:

```python
profile: bool = False
profile_dir: Optional[Union[str, Path]] = None
profile_device: bool = False
profile_torch: bool = False
profile_nvtx: bool = False
profile_mps_signposts: bool = False
profile_sync_device: bool = False
profile_sample_pages: Optional[int] = None
```

Pragmatic defaults:

- `profile=False`: use `NoopProfiler`, zero meaningful overhead.
- `profile=True`: write JSONL spans and summary only.
- `profile_device=True`: capture device metadata and memory snapshots when cheap.
- `profile_sync_device=True`: synchronize before/after selected device spans to get more honest wall timings. This is useful for measurement but should be explicitly labeled because synchronization changes runtime.
- `profile_torch=True`: wrap a small run in `torch.profiler`.
- `profile_nvtx=True`: emit NVTX ranges for CUDA drill-down.
- `profile_mps_signposts=True`: start/stop MPS signpost tracing for short MPS runs.

The first UI can stay plain:

```bash
mokuro ./vol1 --disable_confirmation --no_cache --profile
mokuro ./vol1 --disable_confirmation --no_cache --profile --profile_device
mokuro ./vol1 --disable_confirmation --no_cache --profile --profile_torch --profile_sample_pages 8
```

## Span Map

Start coarse, then add detail only where it pays off.

Run and volume:

- `run.scan_paths`
- `run.confirmation`
- `volume.process`
- `volume.unzip`
- `volume.generate_mokuro`
- `volume.generate_legacy_html`

Page loop:

- `volume.get_img_paths`
- `page.cache_lookup`
- `page.process`
- `page.write_ocr_json`

Page OCR:

- `page.imread`
- `page.detect_text`
- `page.prepare_blocks`
- `page.prepare_line`
- `line.split_chunks`
- `line.rotate_vertical_crop`
- `ocr.recognize_crop`
- `page.assemble_result`

Model lifecycle:

- `models.init`
- `models.init_text_detector`
- `models.init_ocr`

Important counters and attributes:

- `pages_total`, `pages_cached`, `pages_processed`, `pages_failed`
- `blocks_count`, `lines_count`, `chunks_count`
- `img_width`, `img_height`
- `crop_width`, `crop_height`, `crop_pixels`, `vertical`
- `detector_input_size`, `text_height`, `max_ratio`
- `ocr_model`, `detector_device`, `ocr_device`
- `cuda_name`, `mps_available`, `torch_version`
- `cache_hit`, `no_cache`, `ignore_errors`

## Summary Output

At process exit, produce `summary.json` with rollups from the span file:

```json
{
  "schema": "mokuro.profile.summary.v1",
  "run_id": "...",
  "total_duration_s": 123.4,
  "environment": {
    "platform": "Darwin",
    "python": "3.13.0",
    "torch": "2.x",
    "device": "mps"
  },
  "spans": {
    "page.detect_text": {"count": 120, "total_s": 42.1, "mean_ms": 350.8, "p50_ms": 322.0, "p95_ms": 590.0},
    "ocr.recognize_crop": {"count": 980, "total_s": 71.2, "mean_ms": 72.6, "p50_ms": 61.0, "p95_ms": 144.0}
  },
  "derived": {
    "mean_lines_per_page": 8.2,
    "mean_chunks_per_line": 1.1,
    "ocr_crop_s_per_chunk": 0.073
  }
}
```

This gives a quick answer before opening a trace UI.

## Chrome Trace Export

Add a small converter:

```bash
python -m mokuro.profiling export-chrome <profile-dir>/spans.jsonl -o trace.json
```

Chrome trace JSON is easy to inspect in Perfetto. Perfetto documents support for Chrome JSON format, so this keeps the custom span recorder useful beyond a terminal summary.

Suggested trace categories:

- `io`
- `cache`
- `preprocess`
- `detector`
- `ocr`
- `serialize`
- `html`

## OCR DMZ Compatibility

The OCR DMZ should reuse this profiling contract, not invent another one.

Add these fields to future OCR request artifacts:

```json
{
  "id": "vol1/000a:b0:l2:c0",
  "trace_id": "...",
  "parent_span_id": "...",
  "page": "000a.jpg",
  "block": 0,
  "line": 2,
  "chunk": 0
}
```

The local mokuro process creates spans like:

- `ocr.export_request`
- `ocr.wait_for_results` or `ocr.import_results`
- `ocr.reassemble_line`

The black-box OCR service creates spans like:

- `ocr_service.receive_batch`
- `ocr_service.load_image`
- `ocr_service.preprocess_batch`
- `ocr_service.generate_batch`
- `ocr_service.decode_batch`
- `ocr_service.write_results`

When a result comes back, mokuro can join by request ID and trace ID. This works whether the OCR backend is a local torch runner, MLX service, CUDA workstation job, or remote API.

## Weekend Implementation Plan

### 1. Add the local span recorder

Files:

- `mokuro/profiling.py`
- `tests/test_profiling.py`

Acceptance:

- `NoopProfiler` has near-zero behavior and is the default.
- JSONL spans preserve nesting and record exceptions.
- Summary rollup handles empty and non-empty traces.

### 2. Thread profiler through the existing pipeline

Files:

- `mokuro/run.py`
- `mokuro/mokuro_generator.py`
- `mokuro/manga_page_ocr.py`
- `mokuro/legacy/overlay_generator.py`

Acceptance:

- Existing tests pass with profiling disabled.
- Running with `--profile` writes `spans.jsonl` and `summary.json`.
- A cached run clearly shows cache hits and very little OCR work.
- A `--no_cache` run clearly separates `page.detect_text`, crop prep, and OCR.

### 3. Add optional device metadata

Acceptance:

- CUDA runs record device name and memory deltas when CUDA is available.
- MPS runs record MPS availability and cheap memory counters when available.
- `profile_sync_device` is off by default and clearly marked in summary metadata when enabled.

### 4. Add trace export and optional ranges

Acceptance:

- JSONL can be converted to Chrome trace JSON and opened in Perfetto.
- `profile_torch` creates a small PyTorch trace for a sampled run.
- `profile_nvtx` emits ranges only when `nvtx` is installed.
- `profile_mps_signposts` starts/stops signposts only on MPS-capable PyTorch.

## What Not To Do First

- Do not start with Nsight Systems for every run. It is excellent for CUDA drill-down, but it is too heavy as the main answer to "is this preprocessing, segmentation, or OCR?"
- Do not add OpenTelemetry as the first runtime dependency. Keep the core span schema small and export later.
- Do not synchronize the device around every span by default. It makes timings easier to reason about but changes the workload.
- Do not instrument every crop with huge image-specific span names. Use stable names and attributes.
- Do not refactor batching and profiling in the same patch. First measure, then optimize.

## References

- Pyinstrument docs: statistical, low-overhead, wall-clock profiling: https://pyinstrument.readthedocs.io/en/latest/how-it-works.html
- Scalene overview: sampling profiler with Python/native/system attribution and memory profiling: https://sig-rpc.github.io/profiler/python/scalene/
- PyTorch profiler API and Chrome trace export: https://docs.pytorch.org/docs/2.12/profiler.html
- NVIDIA NVTX Python annotations: https://nvidia.github.io/NVTX/python/reference.html
- PyTorch MPS profiling environment variables: https://docs.pytorch.org/docs/2.12/mps_environment_variables.html
- PyTorch MPS OS Signpost profiler: https://docs.pytorch.org/docs/2.12/generated/torch.mps.profiler.start.html
- OpenTelemetry Python manual spans: https://opentelemetry.io/docs/languages/python/instrumentation/
- Perfetto supported trace formats, including Chrome JSON: https://perfetto.dev/docs/
