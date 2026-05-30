# Mokuro Inference Loop Findings

## Summary

The normal `python -m mokuro ./tests/data/input/test0` path is a serial page loop. For each uncached page, mokuro runs text detection on the whole page, then OCR on each detected text line or line chunk one at a time.

The current OCR path does not take advantage of batching. It calls upstream `manga_ocr.MangaOcr.__call__` once per crop, and that wrapper always constructs a batch of exactly one image before `model.generate(...)`.

The detector path also deserves first-class attention on macOS. Mokuro currently sends detection to CUDA when available, otherwise CPU; it does not choose MPS. The detector model itself is ordinary torch and its raw forward pass works with both MPS and batched tensors, so the hard part is not CUDA-specific kernels. The work is mostly exposing MPS in device selection and, for batching, lifting the single-image inference wrapper/postprocess into a batched forward plus per-image reconstruction loop.

Swapping the OCR backend is feasible and fairly local if the replacement exposes the same "image crop in, text out" shape. Swapping in a batched backend, an MLX implementation, or a remote API would be moderate work because the current code assumes synchronous per-crop calls and performs crop ordering, vertical rotation, chunk concatenation, and post-processing inline inside `MangaPageOcr.__call__`.

## Entry Point And Loop

CLI entry goes through `mokuro.__main__:main`, then `mokuro.run.run`, then `MokuroGenerator.process_volume`.

Important flow:

- `mokuro/run.py` builds one `MokuroGenerator`.
- `mokuro/mokuro_generator.py:46` iterates `for img_path_rel in tqdm(img_paths.values(), desc="Processing pages...")`.
- For every uncached page, `mokuro/mokuro_generator.py:57-58` lazily initializes models and calls `self.mpocr(volume.path_in / img_path_rel)`.
- `MangaPageOcr.__call__` reads one image, runs detection, then OCRs each detected line/chunk.

There is no page-level batch in `MokuroGenerator.process_volume`; pages are processed sequentially.

## Models Used

There are two model systems involved.

### Text Detection

`mokuro/manga_page_ocr.py:42-43` constructs:

```python
TextDetector(model_path=cache.comic_text_detector, input_size=detector_input_size, device=device, act="leaky")
```

`mokuro/cache.py` resolves `cache.comic_text_detector` to:

```text
~/.cache/manga-ocr/comictextdetector.pt
```

and downloads it from:

```text
https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.2.1/comictextdetector.pt
```

The detector implementation is torch by default for `.pt` weights, with an OpenCV DNN path only for `.onnx` weights:

- `comic_text_detector/inference.py:126-132`: `.onnx` uses `TextDetBaseDNN`; otherwise uses `TextDetBase`.
- `comic_text_detector/basemodel.py:211-220`: the `.pt` bundle contains `blk_det`, `text_seg`, and `text_det`.
- `comic_text_detector/basemodel.py:240-244`: `TextDetBase.forward` runs block detection, segmentation, and DB text-line detection.

So the detector is a bundled comic text detector made of a YOLOv5 block detector plus segmentation/DB heads, not the manga OCR transformer model.

### Text Detection Reassessment

The earlier findings underweighted detector cost on macOS. Mokuro currently only moves detection to CUDA:

```python
cuda = torch.cuda.is_available()
device = "cuda" if cuda and not force_cpu else "cpu"
```

That means Apple Silicon runs text detection on CPU even when MPS is available. OCR is different: upstream `manga_ocr.MangaOcr` already chooses CUDA, then MPS, then CPU. So on macOS the current pipeline can easily become "MPS OCR behind CPU segmentation/detection", which matches the Metal flamegraph concern.

The detector forward pass is not CUDA-specific. The submodule uses ordinary torch modules:

- YOLOv5-style conv backbone and `Detect` head.
- U-Net-like segmentation head using `Conv2d`, `BatchNorm2d`, `C3`, `ConvTranspose2d`, `ReLU`, and `Sigmoid`.
- DB text-line head using the same basic torch ops.
- Post-processing with `torchvision.ops.nms`, OpenCV contour/mask operations, pyclipper, and shapely.

There are no custom CUDA extensions, Triton kernels, cupy kernels, or handwritten device kernels in the inference path. The CUDA-specific code is mostly in training scripts or old device defaults.

The raw torch detector forward is batch-capable. A smoke test against the cached detector weights with a `(2, 3, 1024, 1024)` tensor returned:

```text
blks  (2, 64512, 7)
mask  (2, 1, 1024, 1024)
lines (2, 2, 1024, 1024)
```

The same raw forward also ran on MPS without unsupported-op failures in this environment (`torch 2.12.0`, MPS built and available).

The public `TextDetector.__call__` wrapper is still single-image:

- `preprocess_img(...)` accepts one image and creates a batch of one.
- `postprocess_yolo(...)` calls `non_max_suppression(...)[0]`, selecting one image's detections.
- line extraction later indexes `scores[0]` and `lines[0]`.
- resizing ratios, mask resize, `group_output(...)`, and `refine_mask(...)` are all for one original image.

So this is not ragged-batch hell in the model. It is a regular batched model followed by per-image post-processing and per-image page geometry. A batched detector path would need to stack preprocessed pages, call `self.net(batch)`, then loop over batch items for NMS, DB contour extraction, mask resize, grouping, and refinement.

Local timings were only smoke tests, but they make the direction clear:

```text
raw forward, warmed:
  CPU bs=1: 0.372 s/page
  CPU bs=4: 0.222 s/page
  MPS bs=1: 0.082 s/page
  MPS bs=4: 0.078 s/page

full TextDetector.__call__, warmed on tests/data/input/test0/vol1/000a.jpg:
  CPU: 0.404 s/page
  MPS: 0.108 s/page
```

The biggest near-term macOS win is therefore simply allowing detector MPS. Detection batching is still plausible and likely useful for throughput, but on MPS the forward-only per-image gain from batch size 4 was small in this quick test. The value of batching is more about amortizing Python/framework overhead and keeping the device fed across a volume than about fixing an impossible tensor shape.

### OCR

`mokuro/manga_page_ocr.py:45` constructs:

```python
MangaOcr(pretrained_model_name_or_path, force_cpu)
```

The default model name comes from both `mokuro/run.py:18` and `mokuro/manga_page_ocr.py:23`:

```text
kha-white/manga-ocr-base
```

The installed upstream wrapper is `manga-ocr 0.1.14`. In this environment it loads:

- `ViTImageProcessor.from_pretrained(...)`
- `AutoTokenizer.from_pretrained(...)`
- `VisionEncoderDecoderModel.from_pretrained(...)`, subclassed as `MangaOcrModel`

The cached config for `kha-white/manga-ocr-base` is a `VisionEncoderDecoderConfig`:

- Encoder: ViT, hidden size 768, 12 layers, 12 attention heads, 16x16 patches, 224x224 image processor input.
- Decoder: Japanese BERT configured as a decoder, hidden size 768, 2 layers, 12 attention heads, vocab size 6144.
- Tokenizer: `BertJapaneseTokenizer`.

Device selection for OCR is inside upstream `manga_ocr/ocr.py`: CUDA first, then MPS, then CPU.

### Parameter Counts On Disk

The local Hugging Face cache contains two cached revisions for `kha-white/manga-ocr-base`:

- `~/.cache/huggingface/hub/models--kha-white--manga-ocr-base/snapshots/82c77fb84369e0fce0d6e8f0ad96fa82646d0346/model.safetensors`
- `~/.cache/huggingface/hub/models--kha-white--manga-ocr-base/snapshots/aa6573bd10b0d446cbf622e29c3e084914df9741/pytorch_model.bin`

At the time of checking, `refs/main` points to the `aa6573bd10b0d446cbf622e29c3e084914df9741` revision.

Counting the active checkpoint directly gives:

- Raw active `pytorch_model.bin` tensor elements: 115,731,200.
- Actual loaded `MangaOcrModel` parameters: 111,005,952.
- Encoder parameters: 86,389,248.
- Decoder parameters: 24,616,704.

The raw `.bin` count is larger because the checkpoint stores tied decoder weights redundantly plus a small buffer. After loading with `MangaOcrModel.from_pretrained(...)`, those tied weights share storage, so the actual OCR model size is about 111M parameters.

The older cached `model.safetensors` snapshot contains 111,006,464 tensor elements:

- Encoder tensors: 86,389,248.
- Decoder tensors: 24,617,216.

For comparison, the detector bundle at `~/.cache/manga-ocr/comictextdetector.pt` contains about 23.4M estimated parameters across its three bundled components:

- `blk_det`: 7,025,023 parameters.
- `text_seg`: 12,223,616 parameters.
- `text_det`: 4,163,314 parameters.
- Total estimated detector parameters: 23,411,953.

### Layer Depth And Compile Implications

**OCR Model (`kha-white/manga-ocr-base`):**

| Component | Architecture | Layers | Hidden | Heads |
|-----------|-------------|--------|--------|-------|
| Encoder | DeiT-base ViT (`facebook/deit-base-patch16-224`) | **12** | 768 | 12 |
| Decoder | BERT-japanese-char-v2 pruned to decoder | **2** | 768 | 12 |

The encoder is standard narrow-and-deep transformer (768 hidden, 12 layers). The decoder was deliberately pruned from the original bert-base's 12 layers down to **2**, with cross-attention to the ViT features. Very shallow by design.

**Detector (`comictextdetector.pt`):**

| Component | Modules | Conv Layers | Max Channels |
|-----------|---------|-------------|--------------|
| YOLOv5s Backbone (truncated at idx 10) | 10 | **31** | 1024 |
| U-Net Segmentation Head | 7 | **34** | 512 |
| DB Text-Line Head | 8 | **18** (3 shared w/ U-Net) | 256 |
| **Total** | **25** | **~83** | |

The backbone provides the bulk of feature extraction depth (31 convs across 4 downsampling stages, C3/Bottleneck residual blocks). The U-Net head is moderately deep with 6 upsampling stages but shallow per-stage (all C3 blocks use n=1). The DB head adds relatively few layers, with 2 stages deep-copied from the U-Net during `initialize_db()`. Overall: moderate depth, moderate width. Neither extreme.

**`torch.compile()` target analysis:** The detector is the clear winner. It runs on CPU on macOS (no MPS), has ~83 small conv/BN/activation ops that compile fuses aggressively, and executes as a single `forward()` graph with no Python interleave. The OCR ViT encoder (12 layers) would benefit moderately but already runs on MPS where gains are smaller. The OCR decoder benefits negligibly — only 2 layers inside `generate()`'s autoregressive Python loop, which compile doesn't eliminate. A drop-in `self.net = torch.compile(self.net)` in `TextDetBase.__init__` is the lowest-effort, highest-impact optimization.

## How Vanilla Is The Modeling Code?

The modeling code is very vanilla. The preprocessing and page-geometry code is where most of mokuro's complexity lives.

The upstream `manga-ocr` model class is:

```python
class MangaOcrModel(VisionEncoderDecoderModel, GenerationMixin):
    pass
```

That is the important fact. There is no custom attention layer, custom decoding loop, custom image embedding scheme, special OCR-specific head, or handwritten torch module hiding in the wrapper. The actual OCR model is a Hugging Face `VisionEncoderDecoderModel`: a ViT image encoder feeding a small BERT-family Japanese text decoder through cross-attention.

From the model architecture point of view, this is ordinary encoder-decoder generation:

1. Convert a crop image into normalized pixel values.
2. Run a ViT encoder over fixed-size image patches.
3. Autoregressively generate text tokens with the decoder.
4. Decode tokens with `BertJapaneseTokenizer`.
5. Apply light text post-processing.

The wrapper's `post_process` is also not architecturally deep. It removes whitespace, normalizes repeated dot-like punctuation, and converts half-width ASCII/digits to full-width forms. A different backend would need to preserve this behavior for output parity, but it is not a model-serving obstacle.

The model's "interesting" parts are standard choices, not custom code:

- It is vision-to-text, not text-only.
- It is encoder-decoder, not decoder-only.
- The decoder is BERT configured as a decoder with cross-attention.
- Generation is autoregressive through Hugging Face `generate`.
- Inputs are images, but after `ViTImageProcessor` they are fixed-size tensors.

That means this is not tied deeply to the local `manga_ocr` Python package. The package is mostly acting as a convenience loader plus single-image inference wrapper.

### Lift-And-Shift Potential

In principle, the OCR model can be lifted out of mokuro. The clean serving boundary is:

```text
list[PIL image crops] -> list[str]
```

or, over a process/API boundary:

```text
list[encoded crop images] -> list[recognized strings]
```

Mokuro does not need to own the OCR model if an external backend can preserve:

- the same crop orientation convention,
- the same image normalization/resizing semantics or close enough behavior,
- the same tokenizer vocabulary,
- the same generation settings,
- the same post-processing.

In that world, this repo could become mostly the page-analysis/frontend/orchestration layer: detect text regions, prepare crops, send them to an OCR backend, receive strings, write `.mokuro` output.

But "vanilla architecture" does not automatically mean "drop it into any LLM serving rail." Many serving stacks are optimized around decoder-only text LLMs, sometimes with multimodal projector support. This model is a classic vision-encoder/text-decoder model. A backend needs to support that exact shape: image encoder outputs consumed by a separate autoregressive decoder with cross-attention.

### MLX Implications

MLX is plausible, but it is a port, not just a config switch.

The favorable part is that the architecture is plain:

- ViT encoder blocks,
- BERT-style decoder blocks,
- cross-attention,
- token embeddings and LM head,
- standard autoregressive search.

Those are all implementable in MLX. There is no mokuro-specific modeling trick that would make an MLX backend conceptually ugly.

The work would be in the usual porting details:

- converting Hugging Face weights into MLX arrays with matching parameter names/layouts,
- implementing or reusing ViT and BERT-decoder modules with cross-attention,
- matching image preprocessing,
- matching tokenizer behavior, likely still using a Python tokenizer unless replaced separately,
- implementing greedy/beam generation behavior close enough to `transformers.generate`,
- validating output parity on crops from real pages.

For a first MLX backend, the safest interface would be an external adapter that exposes `recognize_batch(images)`. Mokuro should not care whether that adapter calls torch/transformers, MLX, Core ML, or an HTTP service.

### vLLM-Style Serving Implications

This is less obviously lift-and-shift than MLX in the abstract, because the serving rail has to support this model family, not merely "a Hugging Face model."

If a serving system supports Hugging Face `VisionEncoderDecoderModel` or equivalent encoder-decoder vision-to-text generation, then the manga OCR model is a good candidate for external serving. If the serving system is mainly optimized for decoder-only text LLMs or a different multimodal architecture, then this model will not fit without a custom model executor.

So the practical answer is:

- The modeling code is vanilla enough to move out of mokuro.
- It is not vanilla in the "decoder-only LLM" sense.
- A backend should be evaluated for encoder-decoder vision-to-text support specifically.
- If that support is absent, the fallback is still straightforward: run a small dedicated OCR service using torch/transformers or an MLX port, and call it from mokuro.

The strongest separation is therefore not "put it on vLLM" specifically. It is "define an OCR backend boundary and make mokuro stop knowing about `MangaOcr`." Once that boundary exists, different serving technologies can compete behind it.

## Batching

OCR batching is not currently used.

The mokuro side is per crop:

- `mokuro/manga_page_ocr.py:57` runs text detection once for the page.
- `mokuro/manga_page_ocr.py:58-67` loops over detected blocks and lines.
- `mokuro/manga_page_ocr.py:73-81` splits long lines into chunks.
- `mokuro/manga_page_ocr.py:83-87` loops over `line_crops` and calls `self.mocr(Image.fromarray(line_crop))` for each crop.

The upstream OCR wrapper is also per image:

- `.venv/lib/python3.13/site-packages/manga_ocr/ocr.py:37-45` accepts one path or one PIL image.
- `.venv/lib/python3.13/site-packages/manga_ocr/ocr.py:47-48` preprocesses one image and calls `self.model.generate(x[None].to(self.model.device), max_length=300)`.
- `x[None]` explicitly creates batch size 1.
- `.venv/lib/python3.13/site-packages/manga_ocr/ocr.py:49` decodes one generated sequence.

The detector path also appears effectively single-page:

- `comic_text_detector/inference.py:145` preprocesses one image.
- `comic_text_detector/inference.py:148` calls `self.net(img_in)`.
- `comic_text_detector/inference.py:151` uses `postprocess_yolo(...)[0]`, selecting the first batch element.

The underlying transformer `generate` API can usually handle batched pixel tensors, but this wrapper and mokuro's calling convention do not expose that. A batched implementation would need to collect crops, preserve their destination block/line/chunk order, call a batch-aware OCR method, then stitch chunk text back into lines.

### Is The Raggedness Essential?

Some of the raggedness is real, but it is mostly not the reason batching is absent.

Mokuro is doing meaningful per-page and per-line work before OCR:

- The text detector finds irregular text blocks and text lines on a full manga page.
- Each detected line is transformed into a crop using the block geometry.
- Vertical text is rotated before OCR.
- Very long crops are split into multiple chunks so that a line with an extreme aspect ratio does not get crushed into the OCR model's fixed input size.
- After OCR, chunk outputs are concatenated back into one logical line.

That is not imaginary complexity. Manga pages really are ragged: one page can have no text, another can have many bubbles, vertical and horizontal writing can mix, and long vertical/horizontal lines can produce a variable number of chunks. The final data structure is hierarchical: page -> block -> line -> chunk. A batch-aware implementation has to preserve that hierarchy and put every generated string back in the right place.

But the actual tensor preprocessing for the OCR model is much less ragged than the page geometry makes it sound. Upstream `manga-ocr` uses `ViTImageProcessor`, and the cached model processor resizes every OCR crop to 224x224, normalizes it, and produces a regular tensor. In other words, once mokuro has made the crop and optionally rotated/split it, the OCR model input is fixed-size. A list of crops can become a normal batch tensor.

So there are two different questions:

1. Is crop discovery and result reconstruction ragged? Yes.
2. Does OCR model inference require one crop per forward/generation call? No.

The current code is closer to "not batching because the wrapper and loop are written one image at a time" than "not batching because batching is fundamentally blocked." The upstream `MangaOcr.__call__` accepts exactly one image, preprocesses exactly one image, adds `x[None]`, runs `generate`, and decodes one sequence. Mokuro mirrors that shape by calling `self.mocr(...)` inside the innermost crop loop.

The main batching complication is bookkeeping, not image tensor shape. A batch refactor would need to collect jobs like:

```python
{
    "block_idx": block_idx,
    "line_idx": line_idx,
    "chunk_idx": chunk_idx,
    "image": pil_crop,
}
```

then run OCR on `job["image"]` values in batches, then rebuild:

- chunks in `chunk_idx` order,
- lines by concatenating chunk text,
- blocks in the original detector order.

That bookkeeping is mildly fussy but straightforward. It is exactly the kind of complexity tests can pin down.

The more subtle batching issue is generation length, not image size. Transformer generation can batch inputs, but autoregressive decoding proceeds until each sample hits an end condition or `max_length`. Mixed short and long text crops may waste some work inside a batch because shorter samples finish earlier or continue padding while longer ones decode. That can reduce batching efficiency, but it does not make batching invalid. It mainly argues for a practical batch size and perhaps grouping crops by rough width/aspect ratio or expected text length if performance matters.

A reasonable expectation is:

- Batching OCR crops should improve GPU utilization versus one `generate` call per crop, especially on pages with many text lines.
- The improvement will not be as clean as batching same-sized classification images because generation is variable-length and chunk counts are page-dependent.
- The first useful implementation does not need heroic bucketing. A simple per-page crop batch, or fixed-size microbatches across the crops from one page, would likely remove a lot of avoidable overhead.
- Cross-page batching would improve throughput further, but it would require changing `MokuroGenerator.process_volume`, cache writes, error handling, and progress reporting. That is a bigger orchestration change than per-page crop batching.

## Backend Swap Difficulty

### Easy: Same Interface, Same Semantics

A drop-in OCR backend with:

```python
ocr(PIL.Image) -> str
```

could replace `self.mocr` with small changes. The only tight coupling in mokuro is the direct import and construction of `MangaOcr` in `mokuro/manga_page_ocr.py`.

For example, an adapter object could implement `__call__(img: PIL.Image) -> str` and be injected into `MangaPageOcr`. That would support a torch/transformers wrapper, an API client, or an MLX wrapper as long as the caller remains per crop.

### Moderate: Batched Local Backend

A proper batched backend is more involved but still contained mostly in `MangaPageOcr`.

Current logic interleaves:

- geometry extraction,
- vertical crop rotation,
- chunk splitting,
- OCR invocation,
- chunk concatenation,
- JSON result assembly.

To batch cleanly, mokuro would need to first build a list of OCR jobs containing crop image plus `(block_index, line_index, chunk_index)`, run `ocr.batch(images) -> list[str]`, then reconstruct each line in the original chunk order. This is not architecturally deep, but it requires reshaping the control flow and adding tests around ordering.

The upstream `manga_ocr.MangaOcr` does not provide a batch method today. A torch/transformers batch adapter could reuse its processor/tokenizer/model, but would need to avoid `_preprocess(...).squeeze()` and decode a list of generated sequences.

### Moderate: Remote API Backend

An API backend is conceptually simple because the boundary is image crop to text. The practical work is around throughput and reliability:

- image encoding and request payload format,
- rate limiting and retries,
- batching or concurrent calls,
- deterministic ordering,
- cache behavior,
- optional credentials/configuration.

No mokuro data structure requires torch tensors for OCR results, so an API backend would not fight the output format.

### Harder: Replacing Text Detection

The text detector is more coupled than OCR. `MangaPageOcr` expects detector output as:

```python
mask, mask_refined, blk_list
```

and each block must provide fields/methods such as `xyxy`, `vertical`, `font_size`, `lines_array()`, and `get_transformed_region(...)`.

A detector backend swap would therefore need either:

- to produce compatible `TextBlock` objects and masks, or
- to refactor crop extraction behind a detector-neutral page layout interface.

That is a larger project than swapping OCR.

## Recommended Parameterization Shape

The smallest useful refactor would be:

1. Add an OCR adapter/protocol with `recognize(image: Image.Image) -> str`.
2. Optionally add `recognize_batch(images: list[Image.Image]) -> list[str]`.
3. Change `MangaPageOcr` to accept an OCR backend object or backend name/config instead of constructing `MangaOcr` directly.
4. Keep detector construction separate from OCR construction.
5. Add tests for preserving block/line/chunk order, especially long lines split by `split_into_chunks`.

This would make torch/transformers, MLX, and API backends swappable without forcing the rest of mokuro to know how inference is performed.
