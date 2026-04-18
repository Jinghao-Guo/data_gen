#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.multiprocessing as mp
from PIL import Image

from firered_model_spec import COMFYUI_LIGHTNING_MODEL_ID, resolve_model_spec
from firered_runtime import load_pipeline

DEFAULT_MODEL = COMFYUI_LIGHTNING_MODEL_ID
DEFAULT_INPUT_PARQUET = Path("/data/jinghao/gen_human/human_edit_instructions")
DEFAULT_PARQUET_BASE_DIR = Path("/data/project_gen/dataset_edit")
DEFAULT_IMAGE_ROOT = DEFAULT_PARQUET_BASE_DIR / "human_image"
DEFAULT_OUTPUT_PARQUET = Path("/data/jinghao/data/edit_parquet/z_image_human_main_0.parquet")
DEFAULT_CHECKPOINT_DIR = Path("/data/jinghao/gen_human/target_gen_checkpoints")
IMAGES_PER_FOLDER = 1000
JPEG_QUALITY = 95
CHECKPOINT_FLUSH_EVERY = 50
RESOLUTION_FILENAME_RE = re.compile(r"resolution_(\d+)x(\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate target images from FireRed instructions with multi-GPU inference.",
    )
    parser.add_argument(
        "--input-parquet",
        type=Path,
        default=DEFAULT_INPUT_PARQUET,
        help="Input parquet file or directory of resolution-sharded parquet files.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output-image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--output-parquet", type=Path, default=DEFAULT_OUTPUT_PARQUET)
    parser.add_argument("--parquet-base-dir", type=Path, default=DEFAULT_PARQUET_BASE_DIR)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--num-gpus", type=int, default=8, help="Number of visible GPUs to use.")
    parser.add_argument(
        "--processes-per-gpu",
        type=int,
        default=1,
        help="Number of worker processes to launch on each GPU.",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=0,
        help="Machine rank for multi-machine sharding.",
    )
    parser.add_argument(
        "--world-size",
        type=int,
        default=1,
        help="Number of machines participating in the job.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Max same-resolution images to generate in one pipeline call.",
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=None,
        help="Number of denoising steps. Defaults to the selected model's recommendation.",
    )
    parser.add_argument(
        "--optimized",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable the optimized FireRed pipeline path.",
    )
    parser.add_argument(
        "--disable-progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Disable diffusers progress bars in workers.",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume from per-rank checkpoint files when available.",
    )
    parser.add_argument("--seed", type=int, default=43, help="Base random seed.")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument(
        "--true-cfg-scale",
        type=float,
        default=1.0,
        help="CFG scale. Defaults to 1.0 because negative_prompt is unset by default.",
    )
    parser.add_argument("--negative-prompt", default=None, help="Optional negative prompt.")
    return parser.parse_args()


def infer_resolution_from_shard_path(path: Path) -> tuple[int, int] | None:
    match = RESOLUTION_FILENAME_RE.fullmatch(path.stem)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def read_image_resolution(path: str | Path) -> tuple[int, int]:
    with Image.open(path) as img:
        width, height = img.size
    return width, height


def list_input_parquet_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        files = sorted(child for child in path.iterdir() if child.suffix == ".parquet")
        if not files:
            raise FileNotFoundError(f"No parquet files found under: {path}")
        return files
    raise FileNotFoundError(f"Input parquet path not found: {path}")


def load_assignment_part(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    required = {"image_path", "instruction"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {path}: {sorted(missing)}")

    keep_cols = ["image_path", "instruction"]
    optional_cols = ["row_idx", "width", "height"]
    cols = keep_cols + [col for col in optional_cols if col in df.columns]
    part = df.loc[:, cols].copy()

    inferred_resolution = infer_resolution_from_shard_path(path)
    if inferred_resolution is not None:
        width, height = inferred_resolution
        if "width" not in part.columns:
            part["width"] = width
        if "height" not in part.columns:
            part["height"] = height

    return part


def attach_missing_resolutions(df: pd.DataFrame) -> pd.DataFrame:
    if "width" in df.columns and "height" in df.columns and not df[["width", "height"]].isnull().any().any():
        return df

    widths: list[int] = []
    heights: list[int] = []
    for image_path in df["image_path"].tolist():
        width, height = read_image_resolution(image_path)
        widths.append(width)
        heights.append(height)

    enriched = df.copy()
    enriched["width"] = widths
    enriched["height"] = heights
    return enriched


def load_assignments(path: Path) -> pd.DataFrame:
    parts = [load_assignment_part(fpath) for fpath in list_input_parquet_files(path)]
    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["image_path", "instruction"])
    df = attach_missing_resolutions(df)

    if "row_idx" not in df.columns:
        df.insert(0, "row_idx", range(len(df)))
    elif df["row_idx"].duplicated().any():
        raise RuntimeError("Duplicate row_idx values detected in input parquet data.")

    df = df.sort_values("row_idx").reset_index(drop=True)
    df["row_idx"] = df["row_idx"].astype(int)
    df["width"] = df["width"].astype(int)
    df["height"] = df["height"].astype(int)
    df = df.sort_values(["width", "height", "row_idx"]).reset_index(drop=True)
    return df


def get_target_relpath(row_idx: int) -> Path:
    subdir = f"{row_idx // IMAGES_PER_FOLDER:06d}"
    filename = f"{row_idx:09d}.jpg"
    return Path(subdir) / filename


def local_worker_count(args: argparse.Namespace) -> int:
    return args.num_gpus * args.processes_per_gpu


def device_index_for_worker(args: argparse.Namespace, local_worker_rank: int) -> int:
    return local_worker_rank // args.processes_per_gpu


def total_worker_count(args: argparse.Namespace) -> int:
    return args.world_size * local_worker_count(args)


def global_worker_rank(args: argparse.Namespace, local_worker_rank: int) -> int:
    return args.rank * local_worker_count(args) + local_worker_rank


def checkpoint_path(args: argparse.Namespace, local_worker_rank: int) -> Path:
    global_rank = global_worker_rank(args, local_worker_rank)
    global_workers = total_worker_count(args)
    return args.checkpoint_dir / f"done_{global_rank:04d}_of_{global_workers:04d}.jsonl"


def output_parquet_path(args: argparse.Namespace) -> Path:
    if args.world_size == 1:
        return args.output_parquet
    suffix = f".rank{args.rank:04d}-of-{args.world_size:04d}"
    return args.output_parquet.with_name(
        f"{args.output_parquet.stem}{suffix}{args.output_parquet.suffix}"
    )


def select_worker_rows(assignments: pd.DataFrame, args: argparse.Namespace, local_worker_rank: int) -> pd.DataFrame:
    global_rank = global_worker_rank(args, local_worker_rank)
    global_workers = total_worker_count(args)
    return assignments.iloc[global_rank::global_workers].copy()


def load_completed_records(path: Path, output_root: Path) -> dict[int, dict[str, Any]]:
    records: dict[int, dict[str, Any]] = {}
    if not path.exists():
        return records

    with path.open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            target_relpath = Path(record["target_image_relpath"])
            if (output_root / target_relpath).is_file():
                records[int(record["row_idx"])] = record

    return records


def append_checkpoint_records(path: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        return

    with path.open("a", encoding="utf-8") as fout:
        for record in records:
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")


def relativize_path(path: str | Path, base_dir: Path) -> str:
    return os.path.relpath(Path(path), start=base_dir)


def build_generation_inputs(
    instructions: list[str],
    images: list[Image.Image],
    device: str,
    seeds: list[int],
    num_inference_steps: int,
    true_cfg_scale: float,
    negative_prompt: str | None,
    width: int,
    height: int,
) -> dict[str, Any]:
    generators = [
        torch.Generator(device=device).manual_seed(seed)
        for seed in seeds
    ]
    image_input: Image.Image | list[Image.Image]
    prompt_input: str | list[str]
    generator_input: torch.Generator | list[torch.Generator]
    if len(images) == 1:
        image_input = images[0]
        prompt_input = instructions[0]
        generator_input = generators[0]
    else:
        image_input = images
        prompt_input = instructions
        generator_input = generators

    inputs: dict[str, Any] = {
        "image": image_input,
        "prompt": prompt_input,
        "generator": generator_input,
        "num_inference_steps": num_inference_steps,
        "true_cfg_scale": true_cfg_scale,
        "height": height,
        "width": width,
    }
    if negative_prompt is not None:
        inputs["negative_prompt"] = (
            negative_prompt if len(images) == 1 else [negative_prompt] * len(images)
        )
    return inputs


def process_batch(
    pipeline,
    device: str,
    rows: list[Any],
    args: argparse.Namespace,
    num_inference_steps: int,
) -> list[dict[str, Any]]:
    if not rows:
        return []

    width = int(rows[0].width)
    height = int(rows[0].height)
    source_paths: list[Path] = []
    source_images: list[Image.Image] = []
    target_paths: list[Path] = []
    instructions: list[str] = []
    seeds: list[int] = []
    row_indices: list[int] = []
    source_sizes: list[list[int]] = []

    for row in rows:
        if int(row.width) != width or int(row.height) != height:
            raise RuntimeError("Mixed resolutions encountered inside one inference batch.")

        source_path = Path(row.image_path)
        if not source_path.is_file():
            raise FileNotFoundError(f"Source image not found: {source_path}")

        target_relpath = get_target_relpath(int(row.row_idx))
        target_path = args.output_image_root / target_relpath
        target_path.parent.mkdir(parents=True, exist_ok=True)

        with Image.open(source_path) as img:
            source_image = img.convert("RGB")

        source_paths.append(source_path)
        source_images.append(source_image)
        target_paths.append(target_path)
        instructions.append(row.instruction)
        seeds.append(args.seed + int(row.row_idx))
        row_indices.append(int(row.row_idx))
        source_sizes.append([source_image.width, source_image.height])

    inputs = build_generation_inputs(
        instructions=instructions,
        images=source_images,
        device=device,
        seeds=seeds,
        num_inference_steps=num_inference_steps,
        true_cfg_scale=args.true_cfg_scale,
        negative_prompt=args.negative_prompt,
        width=width,
        height=height,
    )
    try:
        with torch.inference_mode():
            result = pipeline(**inputs)
    except ValueError as exc:
        if len(rows) > 1 and "only supports batch_size=1" in str(exc):
            if not getattr(process_batch, "_warned_batch_size_one_only", False):
                print(
                    "WARNING: current QwenImageEditPlusPipeline build only supports "
                    "batch_size=1; falling back to per-image inference.",
                    flush=True,
                )
                process_batch._warned_batch_size_one_only = True
            records: list[dict[str, Any]] = []
            for row in rows:
                records.extend(process_batch(pipeline, device, [row], args, num_inference_steps))
            return records
        raise

    if len(result.images) != len(rows):
        raise RuntimeError(
            f"Pipeline returned {len(result.images)} images for {len(rows)} inputs."
        )

    records: list[dict[str, Any]] = []
    for row, source_path, target_path, source_size, generated in zip(
        rows,
        source_paths,
        target_paths,
        source_sizes,
        result.images,
    ):
        target_relpath = get_target_relpath(int(row.row_idx))
        target_image = generated.convert("RGB")
        target_size = list(target_image.size)
        target_image.save(target_path, format="JPEG", quality=JPEG_QUALITY)

        records.append(
            {
                "row_idx": int(row.row_idx),
                "target_image_relpath": str(target_relpath),
                "target_image": str(target_path),
                "instruction": row.instruction,
                "source_image": [str(source_path)],
                "source_size": [source_size],
                "target_size": target_size,
            }
        )

    return records


def iter_resolution_batches(rows: pd.DataFrame, batch_size: int):
    current_batch: list[Any] = []
    current_key: tuple[int, int] | None = None

    for row in rows.itertuples(index=False):
        key = (int(row.width), int(row.height))
        if current_batch and (key != current_key or len(current_batch) >= batch_size):
            yield current_batch
            current_batch = []
        current_batch.append(row)
        current_key = key

    if current_batch:
        yield current_batch


def worker_main(local_worker_rank: int, args: argparse.Namespace) -> None:
    device_index = device_index_for_worker(args, local_worker_rank)
    device = f"cuda:{device_index}"
    global_rank = global_worker_rank(args, local_worker_rank)
    global_workers = total_worker_count(args)
    model_spec = resolve_model_spec(args.model)
    num_inference_steps = (
        args.num_inference_steps
        if args.num_inference_steps is not None
        else model_spec.recommended_num_inference_steps
    )
    assignments = load_assignments(args.input_parquet)
    worker_rows = select_worker_rows(assignments, args, local_worker_rank)

    args.output_image_root.mkdir(parents=True, exist_ok=True)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    done_path = checkpoint_path(args, local_worker_rank)
    completed = (
        load_completed_records(done_path, args.output_image_root) if args.resume else {}
    )
    pending_rows = worker_rows[~worker_rows["row_idx"].isin(completed)]

    if pending_rows.empty:
        print(
            f"[worker {local_worker_rank} gpu {device_index} global {global_rank}/{global_workers}] "
            f"nothing to do on {device} | "
            f"rows={len(worker_rows)} completed={len(completed)}"
        )
        return

    print(
        f"[worker {local_worker_rank} gpu {device_index} global {global_rank}/{global_workers}] "
        f"loading pipeline on {device} | "
        f"rows={len(worker_rows)} pending={len(pending_rows)} "
        f"completed={len(completed)} optimized={args.optimized} "
        f"steps={num_inference_steps}"
    )
    pipeline = load_pipeline(
        args.model,
        device=device,
        optimized=args.optimized,
        disable_progress=args.disable_progress,
    )

    started = time.time()
    processed = 0
    processed_batches = 0
    next_log_threshold = args.log_every
    buffered_records: list[dict[str, Any]] = []
    for batch_rows in iter_resolution_batches(pending_rows, args.batch_size):
        records = process_batch(pipeline, device, batch_rows, args, num_inference_steps)
        buffered_records.extend(records)
        processed += len(records)
        processed_batches += 1

        if len(buffered_records) >= CHECKPOINT_FLUSH_EVERY:
            append_checkpoint_records(done_path, buffered_records)
            buffered_records.clear()

        if processed >= next_log_threshold:
            elapsed = time.time() - started
            speed = processed / elapsed if elapsed > 0 else 0.0
            batch_width = int(batch_rows[0].width)
            batch_height = int(batch_rows[0].height)
            print(
                f"[worker {local_worker_rank} gpu {device_index} global {global_rank}/{global_workers}] "
                f"processed={processed} "
                f"batches={processed_batches} "
                f"last_batch={len(batch_rows)}@{batch_width}x{batch_height} "
                f"elapsed={elapsed:.1f}s speed={speed:.2f} img/s"
            )
            next_log_threshold += args.log_every

    append_checkpoint_records(done_path, buffered_records)


def merge_records(args: argparse.Namespace) -> None:
    assignments = load_assignments(args.input_parquet)
    valid_row_ids: set[int] = set()
    for local_worker_rank in range(local_worker_count(args)):
        worker_rows = select_worker_rows(assignments, args, local_worker_rank)
        valid_row_ids.update(worker_rows["row_idx"].tolist())
    merged: dict[int, dict[str, Any]] = {}

    for local_worker_rank in range(local_worker_count(args)):
        done_path = checkpoint_path(args, local_worker_rank)
        records = load_completed_records(done_path, args.output_image_root)
        for row_idx, record in records.items():
            if row_idx not in valid_row_ids:
                continue
            if row_idx in merged:
                raise RuntimeError(f"Duplicate completed row detected: {row_idx}")
            merged[row_idx] = record

    missing = sorted(valid_row_ids - merged.keys())
    if missing:
        raise RuntimeError(
            f"Missing {len(missing)} records after generation; first missing row_idx={missing[0]}"
        )

    ordered_records = [merged[row_idx] for row_idx in sorted(valid_row_ids)]
    output_df = pd.DataFrame.from_records(ordered_records)
    output_df["source_image"] = output_df["source_image"].map(
        lambda paths: [relativize_path(path, args.parquet_base_dir) for path in paths]
    )
    output_df["target_image"] = output_df["target_image"].map(
        lambda path: relativize_path(path, args.parquet_base_dir)
    )
    output_df = output_df[
        ["instruction", "source_image", "source_size", "target_image", "target_size"]
    ]

    final_output_path = output_parquet_path(args)
    final_output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_parquet(final_output_path, index=False)
    if args.world_size == 1:
        print(f"Wrote final parquet to {final_output_path}")
    else:
        print(
            f"Wrote rank-local parquet to {final_output_path} "
            f"(rank {args.rank}/{args.world_size})"
        )


def main() -> None:
    args = parse_args()

    if not args.input_parquet.exists():
        raise FileNotFoundError(f"Input parquet path not found: {args.input_parquet}")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be a positive integer.")
    if args.log_every <= 0:
        raise ValueError("--log-every must be a positive integer.")
    if args.num_gpus <= 0:
        raise ValueError("--num-gpus must be a positive integer.")
    if args.processes_per_gpu <= 0:
        raise ValueError("--processes-per-gpu must be a positive integer.")
    if args.world_size <= 0:
        raise ValueError("--world-size must be a positive integer.")
    if not 0 <= args.rank < args.world_size:
        raise ValueError("--rank must satisfy 0 <= rank < world-size.")

    available_gpus = torch.cuda.device_count()
    if available_gpus < args.num_gpus:
        raise RuntimeError(
            f"Requested {args.num_gpus} GPUs, but only {available_gpus} are available."
        )

    mp.spawn(worker_main, args=(args,), nprocs=local_worker_count(args), join=True)
    merge_records(args)


if __name__ == "__main__":
    main()
