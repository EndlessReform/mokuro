from collections import Counter
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Sequence, Optional, Union

import fire
from loguru import logger

from mokuro_fast import MokuroGenerator
from mokuro_fast import __version__
from mokuro_fast.config import _UNSET, get_batch_detector, get_batch_ocr, get_precision
from mokuro_fast.legacy.overlay_generator import generate_legacy_html
from mokuro_fast.volume import VolumeCollection


def _coerce_optional_int(value, name):
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as e:
        raise ValueError(f"{name} must be an integer") from e


def run(
    *paths: Optional[Sequence[Union[str, Path]]],
    parent_dir: Optional[Union[str, Path]] = None,
    pretrained_model_name_or_path: str = "kha-white/manga-ocr-base",
    force_cpu: bool = False,
    disable_confirmation: bool = False,
    disable_ocr: bool = False,
    ignore_errors: bool = False,
    no_cache: bool = False,
    unzip: bool = False,
    legacy_html: bool = True,
    as_one_file: bool = True,
    version: bool = False,
    timings_file: Optional[Union[str, Path]] = None,
    ocr_summary_file: Optional[Union[str, Path]] = None,
    page_limit: Optional[int] = None,
    ocr_num_beams: int = 1,
    bf16: bool = False,
    compile: bool = False,
    ocr_bf16: bool = False,
    dev_repeat_ocr_batch_size: int = 1,
    detector_batch_size: int = -1,  # sentinel – resolved from config or default below
    detector_backend: str = "auto",
    detector_model_path: Optional[Union[str, Path]] = None,
    detector_compute_device: Optional[str] = None,
    ocr_batch_size: int = -1,  # sentinel – resolved from config or default below
    config: Optional[Union[str, Path]] = None,
    ocr_reorder_buffer_size: Optional[int] = None,
):
    """
    Process manga volumes with mokuro.

    Args:
        paths: Paths to manga volumes. Volume can be a directory, a zip file or a cbz file.
        parent_dir: Parent directory to scan for volumes. If provided, all volumes inside this directory will be processed.
        pretrained_model_name_or_path: Name or path of the manga-ocr model.
        force_cpu: Force the use of CPU even if CUDA is available.
        disable_confirmation: Disable confirmation prompt. If False, the user will be prompted to confirm the list of volumes to be processed.
        disable_ocr: Disable OCR processing. Generate mokuro/HTML files without OCR results.
        ignore_errors: Continue processing volumes even if an error occurs.
        no_cache: Do not use cached OCR results from previous runs (_ocr directories).
        unzip: Extract volumes in zip/cbz format in their original location.
        legacy_html: Enable legacy HTML output. If True, acts as if --unzip is True.
        as_one_file: Applies only to legacy HTML. If False, generate separate CSS and JS files instead of embedding them in the HTML file.
        version: Print the version of mokuro and exit.
        timings_file: Path to a JSONL file to write per-chunk OCR timing records. Each line contains page, block, line, chunk indices plus crop dimensions and OCR latency in milliseconds.
        ocr_summary_file: Path to a JSON file to write run-level OCR batch yield and token-raggedness summary statistics.
        page_limit: Process only the first N pages of each volume. If None, process all pages.
        ocr_num_beams: Number of beams for the OCR model (default 1, i.e. greedy decoding). Set > 1 for beam search.
        bf16: Enable bfloat16 for both OCR and the MLX detector. Requires an MLX detector backend.
        compile: Compile MLX detector conv blocks with variable-shape support. Requires an MLX detector backend.
        ocr_bf16: Cast the OCR model and image inputs to bfloat16 on CUDA/MPS. Ignored on CPU.
        dev_repeat_ocr_batch_size: DEV ONLY. Artificially batch each OCR crop by repeating it N times, return only the first decoded output, and discard the rest. This is a smoke-test knob for generation batching overhead, not a real batching implementation.
        detector_batch_size: Number of uncached pages to run through the text detector in one batch.
        detector_backend: Text detector compute backend: auto, torch, opencv, or mlx.
        detector_model_path: Optional detector model path. Supports local paths, ``hf://username/repo``, or plain ``username/repo`` for HuggingFace Hub models. For MLX without this flag, defaults to ``jkeisling/comictextdetector-mlx``.
        detector_compute_device: Optional detector backend compute device. For MLX, use cpu or gpu; None keeps the backend default.
        config: Path to a custom config.toml file. Overrides the default XDG location (~/.config/mokuro-fast/config.toml).
        ocr_batch_size: Number of OCR crops to run through decoder generation in one batch.
        ocr_reorder_buffer_size: Number of OCR crop requests to stage before OCR batching. Reserved for future crop reordering; current behavior preserves request order.
    """

    if version:
        print(f"{__version__}")
        return

    page_limit = _coerce_optional_int(page_limit, "page_limit")
    ocr_num_beams = _coerce_optional_int(ocr_num_beams, "ocr_num_beams")
    dev_repeat_ocr_batch_size = _coerce_optional_int(
        dev_repeat_ocr_batch_size, "dev_repeat_ocr_batch_size"
    )

    # --- Resolve batch sizes: CLI > config file > hardcoded default ---
    cfg_path = str(config) if config else None
    if detector_batch_size < 0:
        detector_batch_size = get_batch_detector(cfg_path)
    if detector_batch_size is _UNSET or (isinstance(detector_batch_size, int) and detector_batch_size < 1):
        detector_batch_size = 4
    else:
        detector_batch_size = _coerce_optional_int(detector_batch_size, "detector_batch_size")

    if ocr_batch_size < 0:
        ocr_batch_size = get_batch_ocr(cfg_path)
    if ocr_batch_size is _UNSET or (isinstance(ocr_batch_size, int) and ocr_batch_size < 1):
        ocr_batch_size = 1
    else:
        ocr_batch_size = _coerce_optional_int(ocr_batch_size, "ocr_batch_size")

    # --- Resolve precision: CLI > config file (no hardcoded default for bf16) ---
    if not bf16 and not ocr_bf16:
        precision = get_precision(cfg_path)
        if precision == "bf16":
            bf16 = True
            ocr_bf16 = True

    ocr_reorder_buffer_size = _coerce_optional_int(
        ocr_reorder_buffer_size, "ocr_reorder_buffer_size"
    )

    if page_limit is not None and page_limit < 0:
        raise ValueError("page_limit must be non-negative")

    if ocr_num_beams is not None and ocr_num_beams < 1:
        raise ValueError("ocr_num_beams must be at least 1")

    if dev_repeat_ocr_batch_size < 1:
        raise ValueError("dev_repeat_ocr_batch_size must be at least 1")

    if detector_batch_size < 1:
        raise ValueError("detector_batch_size must be at least 1")

    if ocr_batch_size < 1:
        raise ValueError("ocr_batch_size must be at least 1")

    if ocr_reorder_buffer_size is not None and ocr_reorder_buffer_size < 1:
        raise ValueError("ocr_reorder_buffer_size must be at least 1")

    if ocr_reorder_buffer_size is not None and ocr_reorder_buffer_size < ocr_batch_size:
        raise ValueError("ocr_reorder_buffer_size must be at least ocr_batch_size")

    if dev_repeat_ocr_batch_size > 1 and ocr_batch_size > 1:
        raise ValueError("dev_repeat_ocr_batch_size cannot be combined with ocr_batch_size > 1")

    if disable_ocr:
        logger.info("Running with OCR disabled")
    elif dev_repeat_ocr_batch_size > 1:
        logger.warning(
            "DEV ONLY: --dev-repeat-ocr-batch-size repeats every OCR crop inside one generate() batch "
            "and discards all but the first output. Do not use for production OCR."
        )

    if legacy_html:
        logger.warning(
            "Legacy HTML output is deprecated and will not be further developed. "
            "It's recommended to use .mokuro format and web reader instead. "
            "Legacy HTML will be disabled by default in the future. To explicitly enable it, run with option --legacy-html."
        )
        # legacy HTML works only with unzipped output
        unzip = True

    logger.info("Scanning paths...")

    paths_ = []
    for path in paths:
        path_normalized = Path(str(path)).expanduser().absolute()

        try:
            path_valid = path_normalized.exists()
        except OSError:
            path_valid = False

        if path_valid:
            paths_.append(path_normalized)
        else:
            logger.error(f"Invalid path: {path_normalized}")
            return

    paths = paths_

    if parent_dir is not None:
        for p in Path(parent_dir).expanduser().absolute().iterdir():
            if (
                p not in paths
                and (p.is_dir() and p.stem != "_ocr")
                or (p.is_file() and p.suffix.lower() in {".zip", ".cbz"})
            ):
                paths.append(p)

    vc = VolumeCollection()

    for path_in in paths:
        vc.add_path_in(path_in)

    if len(vc) == 0:
        logger.error("Found no paths to process. Did you set the paths correctly?")
        return

    for title in vc.titles.values():
        title.set_uuid()

    status_counter = Counter()

    print(f"\nFound {len(vc)} volumes:\n")

    for volume in vc:
        print(volume)
        status_counter[volume.status] += 1

    msg = "\nEach of the paths above will be treated as one volume.\n"
    print(msg)

    if not disable_confirmation:
        inp = input("\nContinue? [yes/no]")
        if inp.lower() not in ("y", "yes"):
            return

    timings_fh = None
    if timings_file is not None:
        timings_fh = open(timings_file, "w", encoding="utf-8")

    mg = MokuroGenerator(
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        force_cpu=force_cpu,
        disable_ocr=disable_ocr,
        timings_fh=timings_fh,
        ocr_num_beams=ocr_num_beams,
        bf16=bf16,
        detector_compile=compile,
        ocr_bf16=ocr_bf16,
        dev_repeat_ocr_batch_size=dev_repeat_ocr_batch_size,
        detector_batch_size=detector_batch_size,
        detector_backend=detector_backend,
        detector_model_path=detector_model_path,
        detector_compute_device=detector_compute_device,
        ocr_batch_size=ocr_batch_size,
        ocr_reorder_buffer_size=ocr_reorder_buffer_size,
    )

    try:
        with TemporaryDirectory() as tmp_dir:
            tmp_dir = Path(tmp_dir)

            # unzip == True means that zipped volumes will be unzipped in their original location
            # in that case, we don't use a temporary directory
            if unzip:
                tmp_dir = None

            num_sucessful = 0
            for i, volume in enumerate(vc):
                logger.info(f"Processing {i + 1}/{len(vc)}: {volume.path_in}")

                try:
                    volume.unzip(tmp_dir)
                    mg.process_volume(
                        volume,
                        ignore_errors=ignore_errors,
                        no_cache=no_cache,
                        page_limit=page_limit,
                    )
                    if legacy_html:
                        generate_legacy_html(volume, as_one_file=as_one_file, ignore_errors=ignore_errors)

                except Exception:
                    logger.exception(f"Error while processing {volume.path_in}")
                else:
                    num_sucessful += 1

            logger.info(f"Processed successfully: {num_sucessful}/{len(vc)}")

            if ocr_summary_file is not None:
                summary = mg.get_ocr_batch_summary()
                summary_json = json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
                if str(ocr_summary_file) == "-":
                    print(summary_json, end="")
                else:
                    ocr_summary_path = Path(ocr_summary_file).expanduser()
                    ocr_summary_path.parent.mkdir(parents=True, exist_ok=True)
                    ocr_summary_path.write_text(summary_json, encoding="utf-8")
    finally:
        if timings_fh is not None:
            timings_fh.close()


if __name__ == "__main__":
    fire.Fire(run)
