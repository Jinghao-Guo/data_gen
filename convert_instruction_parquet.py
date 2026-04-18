#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from generate_instructions import attach_image_resolutions, write_applicable_shards


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a legacy single instruction parquet into resolution-sharded format."
    )
    parser.add_argument(
        "--input-parquet",
        type=Path,
        default=Path("/data/jinghao/gen_human/human_edit_instructions.parquet"),
        help="Legacy parquet file to convert.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/data/jinghao/gen_human/human_edit_instructions"),
        help="Directory for resolution-sharded output parquet files.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.input_parquet.is_file():
        raise FileNotFoundError(f"Input parquet not found: {args.input_parquet}")

    df = pd.read_parquet(args.input_parquet)
    required = {"image_path", "instruction"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {args.input_parquet}: {sorted(missing)}")

    output_cols = ["image_path", "instruction"]
    if "category" in df.columns:
        output_cols.insert(1, "category")

    converted = df.loc[:, output_cols].copy()
    converted.insert(0, "row_idx", range(len(converted)))
    converted = attach_image_resolutions(converted)

    ordered_cols = ["row_idx", "image_path"]
    if "category" in converted.columns:
        ordered_cols.append("category")
    ordered_cols.extend(["instruction", "width", "height"])
    converted = converted[ordered_cols]

    shard_counts = write_applicable_shards(converted, args.output_dir)

    print(f"Loaded legacy parquet: {args.input_parquet}")
    print(f"Rows converted: {len(converted):,}")
    print(f"Output directory: {args.output_dir}")
    for shard_name, count in shard_counts:
        print(f"  {shard_name:28s} {count:>8,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
