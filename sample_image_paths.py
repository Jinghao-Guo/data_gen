#!/usr/bin/env python3

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path
from typing import Iterable


DEFAULT_OUTPUT_FILE = "sampled_image_paths_160000.txt"
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


class ProgressBar:
    def __init__(self, width: int = 24, update_interval: float = 0.1) -> None:
        self.width = width
        self.update_interval = update_interval
        self.start_time = 0.0
        self.last_render_time = 0.0
        self.spinner_index = 0
        self.spinner_frames = "|/-\\"

    def __enter__(self) -> "ProgressBar":
        self.start_time = time.monotonic()
        self.last_render_time = 0.0
        self.render(0, force=True)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.finish()

    def update(self, count: int) -> None:
        now = time.monotonic()
        if now - self.last_render_time < self.update_interval:
            return
        self.render(count)
        self.last_render_time = now

    def render(self, count: int, force: bool = False) -> None:
        if not force and not sys.stderr.isatty():
            return

        filled = count % (self.width + 1)
        if filled == self.width:
            filled = self.width - 1
        bar = "=" * filled + ">" + "." * (self.width - filled - 1)
        spinner = self.spinner_frames[self.spinner_index % len(self.spinner_frames)]
        self.spinner_index += 1
        elapsed = max(time.monotonic() - self.start_time, 1e-9)
        rate = count / elapsed
        sys.stderr.write(
            f"\r{spinner} [{bar}] scanned: {count} images, {rate:,.0f} img/s"
        )
        sys.stderr.flush()

    def finish(self) -> None:
        if sys.stderr.isatty():
            sys.stderr.write("\n")
            sys.stderr.flush()


def parse_args() -> argparse.Namespace:
    default_output_dir = Path.cwd()
    parser = argparse.ArgumentParser(
        description="Recursively sample image paths from a source directory."
    )
    parser.add_argument(
        "source_dir",
        type=Path,
        help="Directory to scan recursively for images.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=160_000,
        help="Number of image paths to sample. Default: 160000.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir,
        help=f"Directory to store the output file. Default: {default_output_dir}",
    )
    parser.add_argument(
        "--output-file",
        default=DEFAULT_OUTPUT_FILE,
        help=f"Output file name. Default: {DEFAULT_OUTPUT_FILE}",
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
        help="Write all discovered image paths when the source contains fewer than --count images.",
    )
    return parser.parse_args()


def iter_image_paths(source_dir: Path) -> Iterable[Path]:
    for path in source_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path.resolve()


def reservoir_sample(paths: Iterable[Path], sample_size: int, rng: random.Random) -> tuple[list[Path], int]:
    sample: list[Path] = []
    total_seen = 0

    with ProgressBar() as progress_bar:
        # Reservoir sampling keeps memory usage bounded even for very large datasets.
        for total_seen, path in enumerate(paths, start=1):
            if total_seen <= sample_size:
                sample.append(path)
            else:
                replacement_index = rng.randint(1, total_seen)
                if replacement_index <= sample_size:
                    sample[replacement_index - 1] = path

            progress_bar.update(total_seen)

        progress_bar.render(total_seen, force=True)

    return sample, total_seen


def write_paths(output_path: Path, paths: list[Path]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(str(path) for path in paths) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()

    if args.count <= 0:
        raise ValueError("--count must be a positive integer.")

    source_dir = args.source_dir.expanduser().resolve()
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory does not exist: {source_dir}")
    if not source_dir.is_dir():
        raise NotADirectoryError(f"Source path is not a directory: {source_dir}")

    rng = random.Random(args.seed)
    sampled_paths, total_images = reservoir_sample(iter_image_paths(source_dir), args.count, rng)

    if total_images == 0:
        raise RuntimeError(f"No images were found under: {source_dir}")

    if total_images < args.count and not args.allow_fewer:
        raise RuntimeError(
            f"Only found {total_images} images under {source_dir}, fewer than requested {args.count}. "
            "Use --allow-fewer to write all discovered image paths instead."
        )

    output_dir = args.output_dir.expanduser().resolve()
    output_path = output_dir / args.output_file
    sampled_paths.sort()
    write_paths(output_path, sampled_paths)

    print(f"Scanned images: {total_images}")
    print(f"Written paths: {len(sampled_paths)}")
    print(f"Output file: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())