#!/usr/bin/env python3
"""Prepare a case-level dataset layout suitable for U-Net training."""

from __future__ import annotations

import argparse
import csv
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Set

import numpy as np
from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg"}
MASK_EXTENSIONS = {".png"}


@dataclass
class CaseSummary:
    case: str
    split: str
    pairs: int
    missing_img: int
    missing_msk: int


def parse_case_list(value: Optional[Iterable[str] | str]) -> Optional[Set[str]]:
    if value is None:
        return None
    raw_iterable: Iterable[str]
    if isinstance(value, str):
        raw_iterable = [value]
    else:
        raw_iterable = value

    items: Set[str] = set()
    for item in raw_iterable:
        for part in str(item).split(","):
            cleaned = part.strip()
            if cleaned:
                items.add(cleaned)
    return items if items else None


def canonical_image_key(path: Path) -> str:
    return path.stem.lower()


def canonical_mask_key(path: Path) -> str:
    return path.stem.lower()


def discover_cases(root: Path) -> Iterable[Path]:
    return sorted(p for p in root.iterdir() if p.is_dir())


def choose_split(case: str, train: Optional[Set[str]], val: Optional[Set[str]], test: Optional[Set[str]]) -> Optional[str]:
    memberships = [name for name, cases in (("train", train), ("val", val), ("test", test)) if cases and case in cases]
    if len(memberships) > 1:
        joined = ", ".join(memberships)
        raise ValueError(f"Case '{case}' is assigned to multiple splits: {joined}")
    if memberships:
        return memberships[0]
    if any(cases for cases in (train, val, test) if cases):
        return None
    return "train"


def ensure_directory(path: Path, dry_run: bool) -> None:
    if dry_run:
        return
    path.mkdir(parents=True, exist_ok=True)


def binarize_and_save(mask_path: Path, destination: Path, dry_run: bool) -> None:
    if dry_run:
        print(f"[DRY-RUN] Would write mask: {destination}")
        return
    with Image.open(mask_path) as img:
        data = np.asarray(img)
    if data.ndim == 3:
        data = data[..., 0]
    binary = np.where(data > 0, 255, 0).astype(np.uint8)
    Image.fromarray(binary).save(destination)


def copy_image(image_path: Path, destination: Path, dry_run: bool) -> None:
    if dry_run:
        print(f"[DRY-RUN] Would copy image: {image_path} -> {destination}")
        return
    shutil.copy2(image_path, destination)


def collect_files(case_dir: Path) -> tuple[Dict[str, Path], Dict[str, Path]]:
    images: Dict[str, Path] = {}
    masks: Dict[str, Path] = {}
    for path in sorted(case_dir.iterdir()):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix in IMAGE_EXTENSIONS:
            key = canonical_image_key(path)
            if key not in images:
                images[key] = path
            else:
                print(f"Warning: Duplicate image key '{key}' in {case_dir}")
        elif suffix in MASK_EXTENSIONS:
            key = canonical_mask_key(path)
            if key not in masks:
                masks[key] = path
            else:
                print(f"Warning: Duplicate mask key '{key}' in {case_dir}")
    return images, masks


def write_report(report_path: Path, summaries: Iterable[CaseSummary], dry_run: bool) -> None:
    if dry_run:
        print(f"[DRY-RUN] Would write report: {report_path}")
        return
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["case", "split", "pairs", "missing_img", "missing_msk"])
        for summary in summaries:
            writer.writerow([summary.case, summary.split, summary.pairs, summary.missing_img, summary.missing_msk])


def process_case(case_dir: Path, split: str, dst_root: Path, by_case: bool, dry_run: bool) -> CaseSummary:
    case_name = case_dir.name
    images, masks = collect_files(case_dir)
    keys = sorted(set(images) | set(masks))
    pairs = 0
    missing_images = 0
    missing_masks = 0

    if by_case:
        image_dst_dir = dst_root / split / case_name / "images"
        mask_dst_dir = dst_root / split / case_name / "masks"
    else:
        image_dst_dir = dst_root / split / "images"
        mask_dst_dir = dst_root / split / "masks"

    for key in keys:
        image_path = images.get(key)
        mask_path = masks.get(key)
        if image_path and mask_path:
            pairs += 1
            image_destination = image_dst_dir / image_path.name
            mask_destination = mask_dst_dir / f"{image_path.stem}.png"
            ensure_directory(image_destination.parent, dry_run)
            ensure_directory(mask_destination.parent, dry_run)
            copy_image(image_path, image_destination, dry_run)
            binarize_and_save(mask_path, mask_destination, dry_run)
        elif image_path and not mask_path:
            missing_masks += 1
            print(f"Missing mask for image '{image_path.name}' in case '{case_name}'")
        elif mask_path and not image_path:
            missing_images += 1
            print(f"Missing image for mask '{mask_path.name}' in case '{case_name}'")

    print(
        f"Case '{case_name}' -> split '{split}': pairs={pairs}, missing_img={missing_images}, missing_msk={missing_masks}"
    )

    return CaseSummary(case=case_name, split=split, pairs=pairs, missing_img=missing_images, missing_msk=missing_masks)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare dataset folders for U-Net training")
    parser.add_argument("--src", type=Path, help="Single root containing all case folders (legacy mode)")
    parser.add_argument("--src-train", type=Path, help="Root containing train split case folders")
    parser.add_argument("--src-val", type=Path, help="Root containing val split case folders")
    parser.add_argument("--src-test", type=Path, help="Root containing test split case folders")
    parser.add_argument("--dst", required=True, type=Path, help="Destination root for the prepared dataset")
    parser.add_argument(
        "--train-cases",
        nargs="+",
        help="Case IDs for the train split (space or comma separated)",
    )
    parser.add_argument(
        "--val-cases",
        nargs="+",
        help="Case IDs for the val split (space or comma separated)",
    )
    parser.add_argument(
        "--test-cases",
        nargs="+",
        help="Case IDs for the test split (space or comma separated)",
    )
    parser.add_argument(
        "--only-cases",
        nargs="+",
        help="If supplied, restrict processing to these case IDs",
    )
    parser.add_argument("--by-case", action="store_true", help="Organise output within split/case/ directories")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without writing any files")
    args = parser.parse_args()

    multi_split_mode = any([args.src_train, args.src_val, args.src_test])
    if not args.src and not multi_split_mode:
        parser.error("Provide --src or at least one of --src-train/--src-val/--src-test")
    if args.src and multi_split_mode:
        parser.error("Use either --src or the per-split --src-<split> options, not both")

    return args


def main() -> None:
    args = parse_args()

    multi_split_mode = any([args.src_train, args.src_val, args.src_test])
    only_cases = parse_case_list(args.only_cases)

    summaries: list[CaseSummary] = []
    total_pairs = 0
    total_missing_images = 0
    total_missing_masks = 0

    assigned: list[tuple[Path, str]] = []

    if multi_split_mode:
        split_roots = (
            (args.src_train, "train"),
            (args.src_val, "val"),
            (args.src_test, "test"),
        )
        for root, split_name in split_roots:
            if root is None:
                continue
            if not root.exists():
                raise FileNotFoundError(f"Source directory does not exist: {root}")
            if not root.is_dir():
                raise NotADirectoryError(f"Source path is not a directory: {root}")
            for case_dir in discover_cases(root):
                if only_cases and case_dir.name not in only_cases:
                    continue
                assigned.append((case_dir, split_name))
    else:
        if not args.src.exists():
            raise FileNotFoundError(f"Source directory does not exist: {args.src}")
        if not args.src.is_dir():
            raise NotADirectoryError(f"Source path is not a directory: {args.src}")

        train_cases = parse_case_list(args.train_cases)
        val_cases = parse_case_list(args.val_cases)
        test_cases = parse_case_list(args.test_cases)

        for first, second in ((train_cases, val_cases), (train_cases, test_cases), (val_cases, test_cases)):
            if first and second:
                overlap = first & second
                if overlap:
                    joined = ", ".join(sorted(overlap))
                    raise ValueError(f"Cases assigned to multiple splits: {joined}")

        case_dirs = list(discover_cases(args.src))
        if only_cases is not None:
            case_dirs = [case for case in case_dirs if case.name in only_cases]

        for case_dir in case_dirs:
            split = choose_split(case_dir.name, train_cases, val_cases, test_cases)
            if split is None:
                print(f"Skipping case '{case_dir.name}' (no split assigned)")
                continue
            assigned.append((case_dir, split))

    for case_dir, split in assigned:
        summary = process_case(case_dir, split, args.dst, args.by_case, args.dry_run)
        summaries.append(summary)
        total_pairs += summary.pairs
        total_missing_images += summary.missing_img
        total_missing_masks += summary.missing_msk

    report_path = args.dst / "report.csv"
    write_report(report_path, summaries, args.dry_run)

    print("--- Summary ---")
    print(f"Cases processed: {len(summaries)}")
    print(f"Total pairs: {total_pairs}")
    print(f"Total missing images: {total_missing_images}")
    print(f"Total missing masks: {total_missing_masks}")
    if args.dry_run:
        print("Dry-run mode enabled; no files were written.")


if __name__ == "__main__":
    main()
