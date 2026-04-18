#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

DEFAULT_INPUT = Path(
    "/data/zhaoyang/data/edit/edit_filter_parquet_process_zhaoyang_edit_only_v4/step_10_final.parquet"
)
DEFAULT_OUTPUT = Path("/data/jinghao/data/edit_parquet/part_of_zhaoyang_edit.parquet")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract the first fraction of rows from a parquet file.",
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Source parquet path.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Destination parquet path.")
    parser.add_argument(
        "--denominator",
        type=int,
        default=5,
        help="Keep the first 1 / denominator rows. Defaults to 5.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.denominator <= 0:
        raise ValueError("--denominator must be a positive integer.")
    if not args.input.is_file():
        raise FileNotFoundError(f"Input parquet not found: {args.input}")

    df = pd.read_parquet(args.input)
    keep_rows = len(df) // args.denominator
    subset_df = df.iloc[:keep_rows].copy()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    subset_df.to_parquet(args.output, index=False)

    print(f"Input rows: {len(df)}")
    print(f"Output rows: {len(subset_df)}")
    print(f"Wrote parquet to {args.output}")


if __name__ == "__main__":
    main()