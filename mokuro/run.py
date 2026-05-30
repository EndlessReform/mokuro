from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Sequence, Optional, Union

import fire
from loguru import logger

from mokuro import MokuroGenerator
from mokuro import __version__
from mokuro.legacy.overlay_generator import generate_legacy_html
from mokuro.volume import VolumeCollection


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
    page_limit: Optional[int] = None,
    ocr_num_beams: Optional[int] = None,
    dev_repeat_ocr_batch_size: int = 1,
    detector_batch_size: int = 4,
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
        page_limit: Process only the first N pages of each volume. If None, process all pages.
        ocr_num_beams: Override the OCR model beam count passed to transformers generate(). If None, use the model generation config.
        dev_repeat_ocr_batch_size: DEV ONLY. Artificially batch each OCR crop by repeating it N times, return only the first decoded output, and discard the rest. This is a smoke-test knob for generation batching overhead, not a real batching implementation.
        detector_batch_size: Number of uncached pages to run through the text detector in one batch.
    """

    if version:
        print(f"{__version__}")
        return

    if page_limit is not None and page_limit < 0:
        raise ValueError("page_limit must be non-negative")

    if ocr_num_beams is not None and ocr_num_beams < 1:
        raise ValueError("ocr_num_beams must be at least 1")

    if dev_repeat_ocr_batch_size < 1:
        raise ValueError("dev_repeat_ocr_batch_size must be at least 1")

    if detector_batch_size < 1:
        raise ValueError("detector_batch_size must be at least 1")

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
        dev_repeat_ocr_batch_size=dev_repeat_ocr_batch_size,
        detector_batch_size=detector_batch_size,
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
    finally:
        if timings_fh is not None:
            timings_fh.close()


if __name__ == "__main__":
    fire.Fire(run)
