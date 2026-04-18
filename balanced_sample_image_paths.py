#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


DEFAULT_SOURCE_DIR = Path("/data/project_gen/dataset_img/synthesis_data/z_image_human_main_0/images")
DEFAULT_OUTPUT_FILE = "balanced_sampled_image_paths_200000.txt"
DEFAULT_TOTAL_COUNT = 200000
IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".gif",
    ".webp",
    ".tif",
    ".tiff",
}


class StageProgress:
    def __init__(self, label: str, total: int, update_interval: float = 0.1) -> None:
        self.label = label
        self.total = max(total, 1)
        self.update_interval = update_interval
        self.start_time = 0.0
        self.last_render_time = 0.0
        self.completed = 0
        self.extra = ""

    def __enter__(self) -> "StageProgress":
        self.start_time = time.monotonic()
        self.render(force=True)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.finish()

    def advance(self, extra: str = "") -> None:
        self.completed += 1
        self.extra = extra
        now = time.monotonic()
        if now - self.last_render_time >= self.update_interval:
            self.render(extra=extra)
            self.last_render_time = now

    def render(self, extra: str = "", force: bool = False) -> None:
        if not force and not sys.stderr.isatty():
            return

        ratio = min(self.completed / self.total, 1.0)
        width = 28
        filled = min(int(ratio * width), width)
        bar = "#" * filled + "." * (width - filled)
        elapsed = max(time.monotonic() - self.start_time, 1e-9)
        rate = self.completed / elapsed
        message = (
            f"\r{self.label}: [{bar}] {self.completed}/{self.total} dirs, {rate:,.1f} dirs/s"
        )
        if extra:
            message += f" | {extra}"
        sys.stderr.write(message)
        sys.stderr.flush()

    def finish(self) -> None:
        if sys.stderr.isatty():
            self.render(extra=self.extra, force=True)
            sys.stderr.write("\n")
            sys.stderr.flush()


def parse_args() -> argparse.Namespace:
    cpu_count = os.cpu_count() or 4
    default_workers = max(4, min(32, cpu_count * 4))
    parser = argparse.ArgumentParser(
        description="Sample image paths evenly across immediate subdirectories."
    )
    parser.add_argument(
        "source_dir",
        nargs="?",
        type=Path,
        default=DEFAULT_SOURCE_DIR,
        help=f"Root directory whose immediate subdirectories will be sampled. Default: {DEFAULT_SOURCE_DIR}",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=DEFAULT_TOTAL_COUNT,
        help=f"Total number of image paths to sample. Default: {DEFAULT_TOTAL_COUNT}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.cwd(),
        help=f"Directory to store the output file. Default: {Path.cwd()}",
    )
    parser.add_argument(
        "--output-file",
        default=DEFAULT_OUTPUT_FILE,
        help=f"Output file name. Default: {DEFAULT_OUTPUT_FILE}",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=default_workers,
        help=f"Number of worker threads for counting and sampling. Default: {default_workers}",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducible sampling.",
    )
    parser.add_argument(
        "--allow-fewer",
        action="store_true",
        help="Write all discovered image paths when total available images are fewer than --count.",
    )
    return parser.parse_args()


def is_image_file(filename: str) -> bool:
    return os.path.splitext(filename)[1].lower() in IMAGE_EXTENSIONS


def iter_image_paths(root_dir: str):
    stack = [root_dir]
    while stack:
        current_dir = stack.pop()
        try:
            with os.scandir(current_dir) as entries:
                for entry in entries:
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            stack.append(entry.path)
                        elif entry.is_file(follow_symlinks=False) and is_image_file(entry.name):
                            yield entry.path
                    except OSError:
                        continue
        except OSError:
            continue


def list_immediate_subdirs(root_dir: str) -> list[str]:
    subdirs: list[str] = []
    with os.scandir(root_dir) as entries:
        for entry in entries:
            if entry.is_dir(follow_symlinks=False):
                subdirs.append(entry.path)
    subdirs.sort()
    return subdirs


def count_images_in_dir(root_dir: str) -> int:
    count = 0
    for _ in iter_image_paths(root_dir):
        count += 1
    return count


def compute_balanced_allocations(counts: dict[str, int], target_count: int) -> dict[str, int]:
    allocations = {subdir: 0 for subdir in counts}
    eligible = [subdir for subdir, count in sorted(counts.items()) if count > 0]
    remaining = target_count

    while eligible and remaining > 0:
        base_quota, remainder = divmod(remaining, len(eligible))
        next_eligible: list[str] = []
        progressed = False

        for index, subdir in enumerate(eligible):
            requested = base_quota + (1 if index < remainder else 0)
            available = counts[subdir] - allocations[subdir]
            taken = min(requested, available)
            allocations[subdir] += taken
            remaining -= taken
            progressed = progressed or taken > 0

            if allocations[subdir] < counts[subdir]:
                next_eligible.append(subdir)

        if not progressed:
            break
        eligible = next_eligible

    if remaining != 0:
        raise RuntimeError(f"Unable to allocate the requested sample count. Remaining: {remaining}")

    return allocations


def reservoir_sample_dir(root_dir: str, sample_size: int, seed: int | None) -> tuple[list[str], int]:
    rng = random.Random(seed)
    sample: list[str] = []
    total_seen = 0

    for total_seen, path in enumerate(iter_image_paths(root_dir), start=1):
        if total_seen <= sample_size:
            sample.append(path)
            continue

        replacement_index = rng.randint(1, total_seen)
        if replacement_index <= sample_size:
            sample[replacement_index - 1] = path

    return sample, total_seen


def write_paths(output_path: Path, paths: list[str]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(paths) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()

    if args.count <= 0:
        raise ValueError("--count must be a positive integer.")
    if args.workers <= 0:
        raise ValueError("--workers must be a positive integer.")

    source_dir = args.source_dir.expanduser().resolve()
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory does not exist: {source_dir}")
    if not source_dir.is_dir():
        raise NotADirectoryError(f"Source path is not a directory: {source_dir}")

    subdirs = list_immediate_subdirs(str(source_dir))
    if not subdirs:
        raise RuntimeError(f"No subdirectories found under: {source_dir}")

    counts: dict[str, int] = {}
    total_available = 0
    with ThreadPoolExecutor(max_workers=min(args.workers, len(subdirs))) as executor:
        futures = {executor.submit(count_images_in_dir, subdir): subdir for subdir in subdirs}
        with StageProgress("Counting", len(futures)) as progress:
            for future in as_completed(futures):
                subdir = futures[future]
                count = future.result()
                counts[subdir] = count
                total_available += count
                progress.advance(extra=f"images: {total_available:,}")

    if total_available == 0:
        raise RuntimeError(f"No images found under: {source_dir}")

    target_count = args.count
    if total_available < target_count:
        if not args.allow_fewer:
            raise RuntimeError(
                f"Only found {total_available} images under {source_dir}, fewer than requested {target_count}. "
                "Use --allow-fewer to write all discovered image paths instead."
            )
        target_count = total_available

    allocations = compute_balanced_allocations(counts, target_count)
    active_subdirs = [subdir for subdir, allocation in allocations.items() if allocation > 0]

    seed_rng = random.Random(args.seed)
    subdir_seeds = {subdir: seed_rng.randrange(1 << 63) for subdir in active_subdirs}

    sampled_paths: list[str] = []
    sampled_total = 0
    with ThreadPoolExecutor(max_workers=min(args.workers, len(active_subdirs))) as executor:
        futures = {
            executor.submit(reservoir_sample_dir, subdir, allocations[subdir], subdir_seeds[subdir]): subdir
            for subdir in active_subdirs
        }
        with StageProgress("Sampling", len(futures)) as progress:
            for future in as_completed(futures):
                subdir = futures[future]
                paths, found_count = future.result()
                expected_count = allocations[subdir]
                if found_count < expected_count:
                    raise RuntimeError(
                        f"Directory changed while sampling: {subdir} has {found_count} images, expected {expected_count}."
                    )
                sampled_paths.extend(paths)
                sampled_total += len(paths)
                progress.advance(extra=f"sampled: {sampled_total:,}")

    if sampled_total != target_count:
        raise RuntimeError(f"Expected {target_count} sampled paths, got {sampled_total}")

    shuffle_rng = random.Random(args.seed)
    shuffle_rng.shuffle(sampled_paths)

    output_dir = args.output_dir.expanduser().resolve()
    output_path = output_dir / args.output_file
    write_paths(output_path, sampled_paths)

    print(f"Source directory: {source_dir}")
    print(f"Subdirectories: {len(subdirs)}")
    print(f"Available images: {total_available}")
    print(f"Written paths: {sampled_total}")
    print(f"Output file: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())