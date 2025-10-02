"""Extract fixed-size patches from a whole-slide image (WSI) using OpenSlide.

Example:
    python scripts/extract_patches.py \
        --wsi path/to/slide.svs \
        --out data/patches/train/non_tumor \
        --patch-size 512 \
        --stride 512 \
        --level 0 \
        --background-threshold 225
"""
import argparse
import logging
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import openslide
from openslide import OpenSlide, OpenSlideError
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract PNG patches from a WSI")
    parser.add_argument("--wsi", required=True, type=Path, help="Path to the input WSI file")
    parser.add_argument("--out", required=True, type=Path, help="Directory where patches will be stored")
    parser.add_argument(
        "--patch-size",
        "--tile",
        dest="patch_size",
        type=int,
        default=512,
        help="Patch edge length in pixels at the requested level (default: 512)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=512,
        help="Stride in pixels between patch origins at the requested level (default: 512)",
    )
    parser.add_argument(
        "--level",
        type=int,
        default=0,
        help="OpenSlide level to sample from (0 is highest resolution)",
    )
    parser.add_argument(
        "--background-threshold",
        type=float,
        default=225.0,
        help=(
            "Mean RGB value threshold (0-255) above which a patch is considered background. "
            "Set to <=0 to disable background filtering (default: 225)."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum number of patches to save; 0 means no limit",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (e.g. DEBUG, INFO, WARNING, ERROR)",
    )
    return parser.parse_args()


def resolve_log_level(level_name: str) -> int:
    level = logging.getLevelName(level_name.upper())
    if isinstance(level, int):
        return level
    return logging.INFO


def configure_logging(level_name: str) -> None:
    logging.basicConfig(
        level=resolve_log_level(level_name),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def validate_args(args: argparse.Namespace) -> bool:
    logger = logging.getLogger(__name__)
    if args.patch_size <= 0:
        logger.error("Patch size must be a positive integer")
        return False
    if args.stride <= 0:
        logger.error("Stride must be a positive integer")
        return False
    if args.limit is not None and args.limit < 0:
        logger.error("Limit must be >= 0")
        return False
    return True


def compute_mean_rgb(image: Image.Image) -> float:
    arr = np.asarray(image, dtype=np.uint8)
    return float(arr.mean())


def get_level_dimensions(slide: OpenSlide, level: int) -> Tuple[int, int]:
    return slide.level_dimensions[level]


def extract_patches(args: argparse.Namespace) -> int:
    logger = logging.getLogger(__name__)

    try:
        args.out.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.error("Failed to create output directory %s: %s", args.out, exc)
        return 1

    try:
        slide = openslide.OpenSlide(str(args.wsi))
    except FileNotFoundError:
        logger.error("WSI file not found: %s", args.wsi)
        return 1
    except OpenSlideError as exc:
        logger.error("Unable to open WSI %s: %s", args.wsi, exc)
        return 1
    except Exception:  # pragma: no cover - unexpected errors
        logger.exception("Unexpected error opening WSI %s", args.wsi)
        return 1

    threshold = args.background_threshold if args.background_threshold is not None else 0.0
    if threshold < 0:
        threshold = 0.0
    limit = args.limit if args.limit and args.limit > 0 else None

    try:
        if args.level < 0 or args.level >= slide.level_count:
            logger.error("Level %d is out of range for slide with %d levels", args.level, slide.level_count)
            return 1

        width, height = get_level_dimensions(slide, args.level)
        logger.info(
            "Slide level %d dimensions: %dx%d | patch_size=%d stride=%d",
            args.level,
            width,
            height,
            args.patch_size,
            args.stride,
        )

        if width < args.patch_size or height < args.patch_size:
            logger.warning("Patch size %d exceeds level dimensions; nothing to extract", args.patch_size)
            return 0

        downsample = float(slide.level_downsamples[args.level])

        saved_count = 0
        skipped_background = 0
        total_considered = 0
        limit_reached = False

        for y in range(0, height - args.patch_size + 1, args.stride):
            for x in range(0, width - args.patch_size + 1, args.stride):
                base_x = int(round(x * downsample))
                base_y = int(round(y * downsample))
                try:
                    region = slide.read_region((base_x, base_y), args.level, (args.patch_size, args.patch_size)).convert("RGB")
                except Exception:  # pragma: no cover - unexpected errors during read
                    logger.exception("Failed to read region at x=%d y=%d", x, y)
                    return 1

                total_considered += 1

                if threshold > 0:
                    mean_val = compute_mean_rgb(region)
                    if mean_val >= threshold:
                        skipped_background += 1
                        logger.debug(
                            "Skipping background patch at x=%d y=%d (mean=%.2f >= %.2f)",
                            x,
                            y,
                            mean_val,
                            threshold,
                        )
                        continue

                patch_name = f"patch_{saved_count:06d}_x{x}_y{y}.png"
                patch_path = args.out / patch_name
                try:
                    region.save(patch_path, format="PNG")
                except OSError as exc:
                    logger.error("Failed to save patch %s: %s", patch_path, exc)
                    return 1

                saved_count += 1
                if limit is not None and saved_count >= limit:
                    limit_reached = True
                    break
            if limit_reached:
                break

        logger.info(
            "Completed extraction: saved=%d skipped=%d considered=%d",
            saved_count,
            skipped_background,
            total_considered,
        )
        if limit_reached and limit is not None:
            logger.info("Reached user-specified patch limit (%d)", limit)
        return 0

    finally:
        slide.close()


def main() -> int:
    args = parse_args()
    configure_logging(args.log_level)
    if not validate_args(args):
        return 1
    return extract_patches(args)


if __name__ == "__main__":
    sys.exit(main())
