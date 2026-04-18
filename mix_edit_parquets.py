#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_BASE_DIR = Path("/data/jinghao/data/edit_parquet")
DEFAULT_INPUTS = [
    DEFAULT_BASE_DIR / "part_of_zhaoyang_edit.parquet",
    DEFAULT_BASE_DIR / "z_image_human_main_0.parquet",
    DEFAULT_BASE_DIR / "z_image_human_main_0_part2.parquet",
]
DEFAULT_REFERENCE = DEFAULT_BASE_DIR / "z_image_human_main_0.parquet"
DEFAULT_OUTPUT = DEFAULT_BASE_DIR / "mixed_shuffled_edit.parquet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge and shuffle edit parquets while keeping only the columns "
            "defined by a reference parquet schema."
        )
    )
    parser.add_argument(
        "--inputs",
        type=Path,
        nargs="+",
        default=DEFAULT_INPUTS,
        help="Input parquet files to merge.",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=DEFAULT_REFERENCE,
        help="Reference parquet whose columns define the output schema.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output parquet path.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=43,
        help="Random seed used for shuffling.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.reference.is_file():
        raise FileNotFoundError(f"Reference parquet not found: {args.reference}")

    missing_inputs = [path for path in args.inputs if not path.is_file()]
    if missing_inputs:
        missing = ", ".join(str(path) for path in missing_inputs)
        raise FileNotFoundError(f"Missing input parquet(s): {missing}")

    reference_columns = list(pd.read_parquet(args.reference).columns)
    print(f"Reference parquet: {args.reference}")
    print(f"Keeping columns: {reference_columns}")

    frames: list[pd.DataFrame] = []
    for path in args.inputs:
        df = pd.read_parquet(path, columns=reference_columns)
        frames.append(df)
        print(f"Loaded {len(df):,} rows from {path}")

    merged = pd.concat(frames, ignore_index=True)
    shuffled = merged.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    shuffled.to_parquet(args.output, index=False)

    print(f"Merged rows: {len(merged):,}")
    print(f"Wrote shuffled parquet to: {args.output}")


if __name__ == "__main__":
    main()
