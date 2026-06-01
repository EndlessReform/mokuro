# OCR Timing Profiling

## `--timings-file`

Pass `--timings-file <path>` to emit a JSONL file with one record per OCR crop (chunk):

```bash
mokuro ./my-volume.cbz --timings-file ocr-timings.jsonl --disable-confirmation
```

Each line contains:

| Field | Description |
|---|---|
| `page` | Zero-based page index within the volume |
| `blk` | Zero-based text block index on the page |
| `line` | Zero-based line index within the block |
| `chunk` | Zero-based chunk index within the line (0 if not split) |
| `crop_h` | Crop height in pixels (before rotation) |
| `crop_w` | Crop width in pixels (before rotation) |
| `area` | `crop_h * crop_w` pixel count |
| `aspect` | Width-to-height ratio (rounded to 2dp) |
| `tokens` | Number of output tokens generated for this crop (CLS/SEP special tokens subtracted). Computed by re-encoding the recognized text — slightly redundant but avoids bypassing the upstream wrapper. |
| `ocr_ms` | Wall-clock time for the OCR crop call, in milliseconds |
| `ocr_batch_size` | OCR batch size used for this crop. Normally `1`; higher only when the dev-only repeated-crop smoke flag is enabled. |

### Joining With `.mokuro` Data

The timing records can be joined against the `.mokuro` output file on `(page, blk, line)` to recover box coordinates, line polygons, verticality, font size, and recognized text:

```python
import json

# Load timings
timings = []
with open("ocr-timings.jsonl") as f:
    for line in f:
        timings.append(json.loads(line))

# Load mokuro data
with open("volume.mokuro") as f:
    mokuro = json.load(f)

# Build lookup: (page, blk, line) -> block metadata
lookup = {}
for page_idx, page in enumerate(mokuro["pages"]):
    for blk_idx, blk in enumerate(page["blocks"]):
        for line_idx in range(len(blk["lines"])):
            lookup[(page_idx, blk_idx, line_idx)] = {
                "box": blk["box"],
                "vertical": blk["vertical"],
                "font_size": blk["font_size"],
                "text": blk["lines"][line_idx],
                "lines_coords": blk["lines_coords"][line_idx],
            }

# Join
for t in timings:
    key = (t["page"], t["blk"], t["line"])
    meta = lookup.get(key, {})
    print({
        **t,
        "box": meta.get("box"),
        "text_len": len(meta.get("text", "")),
    })
```

### Use Case: Analyzing Batch Raggedness

The timing log is designed to answer questions about whether crop spatial properties correlate with OCR latency, to inform intelligent batching strategies:

- Does `area` or `aspect` ratio predict `ocr_ms`? (larger crops may have more text)
- Are there outlier crops that dominate page latency?
- Would grouping crops by `area` bucket reduce ragged-batch padding in autoregressive generation?

Example analysis with pandas:

```python
import pandas as pd

df = pd.read_json("ocr-timings.jsonl", lines=True)
print(df.groupby("page")["ocr_ms"].describe())
print(df[["area", "aspect", "ocr_ms"]].corr())
```

### Notes

- Timings are only written for pages that are actually processed (not served from cache). Use `--no-cache` to force full re-processing.
- The `crop_h` / `crop_w` dimensions reflect the crop before vertical rotation. For vertical text blocks, the effective input to the OCR model will have these swapped.
- With normal OCR, `ocr_ms` measures wall-clock time via `time.perf_counter()`, including preprocessor and tokenizer overhead inside `self.mocr()`. With `--dev-repeat-ocr-batch-size`, it includes one crop preprocess, one batched `generate(...)`, and decoding/post-processing only the first output.

## `--page-limit`

Pass `--page-limit N` to process only the first `N` pages of each volume:

```bash
mokuro ./my-volume.cbz --page-limit 3 --disable-confirmation
```

The default is no limit, which processes all pages. This is intended for quick dev loops and smoke tests.

## DEV ONLY: `--dev-repeat-ocr-batch-size`

This is a deliberately stupid smoke-test flag:

```bash
mokuro ./my-volume.cbz \
  --page-limit 3 \
  --no-cache \
  --timings-file repeated-bsz8.jsonl \
  --dev-repeat-ocr-batch-size 8 \
  --disable-confirmation
```

For every OCR crop, mokuro preprocesses the crop once, repeats the resulting image tensor to batch size `N`, calls one batched `model.generate(...)`, returns only decoded output `0`, and discards the duplicate outputs.

Use this only to ask: "if the model sees a perfectly homogeneous artificial batch, how much slower is one generate call at batch size `N` than at batch size `1`?" If batch size `N` is only modestly slower per crop, that is evidence the current path is overhead-bound enough that real batching may pay off.

What it does **not** test:

- It does not test real page or volume batching orchestration.
- It does not test mixed crop sizes, mixed token lengths, or ragged batch padding.
- It does not include the cost of preprocessing `N` separate crop images.
- It does not produce better OCR output; the extra decoded strings are thrown away.

Use `--no-cache` when comparing runs, otherwise cached pages will skip OCR and produce no timing rows.

## Detector MLX Fixture

Use the detector fixture dumper when working on the MLX detector port. It records the current Torch detector as plain ndarray artifacts plus final public detector JSON:

```bash
uv run python -m comic_text_detector.scripts.dump_detector_fixture \
  --image tests/data/input/test0/vol1/000a.jpg \
  --batch-image tests/data/input/test0/vol1/001a.jpg \
  --input-size 1024 \
  --out /tmp/ctd-fixture
```

By default, the script uses the same checkpoint mokuro uses: `${XDG_CACHE_HOME:-~/.cache}/manga-ocr/comictextdetector.pt`. If the checkpoint is missing, it downloads mokuro's default detector artifact. Pass `--checkpoint /path/to/comictextdetector.pt` only when testing another detector file.

The output directory contains:

```text
manifest.json
single.npz
batch.npz
single-final.json
batch-final.json
```

`manifest.json` records the checkpoint path and SHA256, image paths and SHA256 values, input size, activation, dtype, device, thresholds, package versions, and the generated split summaries. The `.npz` files contain fixed ndarray values for the framework boundary and postprocess checkpoints, including NCHW/NHWC inputs, resize metadata, selected trunk feature maps, decoded YOLO head output, raw mask/line heads, NMS values/counts, line values/counts, and input-canvas `post.mask_uint8`.

Use CPU for canonical fixtures. The default `--device cpu` and `--torch-threads 1` are chosen for deterministic fixture generation.

### Running Mokuro With A Local MLX Detector

After converting a local detector artifact, mokuro can use it explicitly:

```bash
uv run --extra mlx mokuro ./my-volume \
  --detector-backend mlx \
  --detector-model-path output/detector/mlx-comictextdetector
```

`--detector-compute-device cpu` uses the strict parity path. Omitting it lets MLX use its default device, which is usually faster on Apple Silicon but can have small final-coordinate drift from the GPU convolution kernels.

Use `--bf16` with the MLX detector to run both the detector and OCR in bfloat16 where the selected devices support it:

```bash
uv run --extra mlx mokuro ./my-volume \
  --detector-backend mlx \
  --detector-model-path output/detector/mlx-comictextdetector \
  --bf16
```

Use `--compile` to wrap the MLX detector's conv-heavy blocks in `mx.compile(..., shapeless=True)`, allowing those compiled functions to accept variable input shapes without recompiling for every height/width change. Shape-heavy pieces such as SPPF pooling and YOLO decode stay eager:

```bash
uv run --extra mlx mokuro ./my-volume \
  --detector-backend mlx \
  --detector-model-path output/detector/mlx-comictextdetector \
  --compile
```
