#!/usr/bin/env python3
"""
Generate human-image edit instructions at scale using vLLM + Qwen VL.

Pipeline:
  1. Read image paths from a text file (200K images).
  2. Assign one target category per image based on predefined ratios.
  3. Run offline batch VL inference with vLLM (chunked, checkpointed).
  4. Parse JSON responses, validate schema, filter applicable results.
  5. Merge into a final parquet, then write applicable instructions as
     resolution-sharded parquet files.

Checkpoint / resume:
  Intermediate results are saved per batch in --checkpoint-dir.
  Re-running the script automatically skips completed batches.

Usage:
    python generate_instructions.py \
        --image-paths balanced_sampled_image_paths_200000.txt \
        --model Qwen/Qwen3-VL-30B-A3B-Instruct \
        --output-dir human_edit_instructions \
        --world-size 1 \
        --rank 0 \
        --tensor-parallel 8
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image

DEFAULT_OUTPUT_DIR = "human_edit_instructions"

# ---------------------------------------------------------------------------
# Fix nvidia-cutlass-dsl path so vLLM flash-attn can find `cutlass.cute`
# ---------------------------------------------------------------------------
_cutlass_dsl_pkgs = os.path.join(
    sys.prefix, "lib", f"python{sys.version_info.major}.{sys.version_info.minor}",
    "dist-packages", "nvidia_cutlass_dsl", "python_packages",
)
if os.path.isdir(_cutlass_dsl_pkgs):
    # Use PYTHONPATH so vLLM's subprocess (EngineCore) also picks it up
    existing = os.environ.get("PYTHONPATH", "")
    if _cutlass_dsl_pkgs not in existing:
        os.environ["PYTHONPATH"] = (
            f"{_cutlass_dsl_pkgs}:{existing}" if existing else _cutlass_dsl_pkgs
        )
    if _cutlass_dsl_pkgs not in sys.path:
        sys.path.insert(0, _cutlass_dsl_pkgs)

# ---------------------------------------------------------------------------
# Ensure HuggingFace models cache to /data/huggingface when env is not set
# ---------------------------------------------------------------------------
os.environ.setdefault("HF_HOME", "/data/huggingface")

# ---------------------------------------------------------------------------
# Categories and target sampling ratios (from ps_human_prompt_template.md)
# ---------------------------------------------------------------------------
CATEGORIES: list[tuple[str, float]] = [
    ("Beautify/General Enhancement", 0.18),
    ("Body Shape (Weight/Muscle/Figure)", 0.18),
    ("Facial Feature Editing", 0.16),
    ("Age Change", 0.10),
    ("Expression/Emotion Change", 0.10),
    ("Hair Editing", 0.09),
    ("Skin Editing", 0.07),
    ("Facial Hair", 0.05),
    ("Gender Transformation", 0.04),
    ("Makeup Editing", 0.03),
]

CATEGORY_NAMES: list[str] = [c[0] for c in CATEGORIES]
CATEGORY_WEIGHTS: list[float] = [c[1] for c in CATEGORIES]

# ---------------------------------------------------------------------------
# Prompts (verbatim from ps_human_prompt_template.md)
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
You are a high-precision instruction generator for human image editing data.

Your task is to inspect the input image and generate high-quality image editing instructions for a PRE-SPECIFIED target category. The target category is provided externally by the data pipeline. You must not switch categories.

Your output will be used to train an image editing model. Therefore, the instructions must be realistic, visually grounded, benchmark-aligned, and directly executable as user-style image edit requests.

You must strictly follow all rules below.

GENERAL GOAL
- Given one human image and one externally assigned target category, generate one natural edit instruction that belongs only to that target category.
- The instructions should resemble realistic user requests for portrait or person-photo editing.
- The instructions should improve category coverage for human image-edit benchmarks.

HARD CATEGORY CONSTRAINT
- You are given exactly one target category.
- Every generated instruction must belong to that category only.
- Do not mix categories in a single instruction.
- If the requested category is not visually applicable to the image, return applicable=false and no instructions.

SUPPORTED CATEGORIES
- Beautify/General Enhancement
- Body Shape (Weight/Muscle/Figure)
- Facial Feature Editing
- Age Change
- Expression/Emotion Change
- Hair Editing
- Skin Editing
- Facial Hair
- Gender Transformation
- Makeup Editing

CATEGORY DEFINITIONS
- Beautify/General Enhancement:
  natural attractiveness improvement, retouching, making the person look better, more handsome, more attractive, more polished, more youthful and stylish.
- Body Shape (Weight/Muscle/Figure):
  slimmer body, slightly thinner figure, stronger build, more muscular appearance, longer legs, narrower waist, healthier body proportions. Must remain realistic and non-sexualized.
- Facial Feature Editing:
  refine nose, slim face, rounder face, refine chin, shape eyebrows, enlarge eyes slightly, improve eye symmetry, remove visible acne or blemishes if clearly visible.
- Age Change:
  make the person look younger, older, more mature, or middle-aged.
- Expression/Emotion Change:
  happier, sadder, more serious, less angry, laughing, crying, teary eyes.
- Hair Editing:
  longer hair, straighter hair, curlier hair, fuller hair, different hair color.
- Skin Editing:
  smoother skin, clearer skin, brighter skin tone, slightly tanner skin, fewer wrinkles.
- Facial Hair:
  remove beard, shorten beard, lengthen beard, clean up facial hair.
- Gender Transformation:
  make the person look more feminine or more masculine while preserving identity and realism.
- Makeup Editing:
  remove lipstick, reduce eyeshadow, soften makeup, remove visible makeup.

VISUAL GROUNDING RULES
- Ground every instruction in what is actually visible in the image.
- Do not invent hidden body parts, invisible hair, unseen beard, or facial details that cannot be inferred from the image.
- Do not request edits on regions that are severely occluded or not visible enough.
- If multiple people are present, explicitly identify the target person inside each instruction.

PRESERVATION RULES
- Keep identity unchanged.
- Keep pose, camera viewpoint, framing, background, clothes, lighting, and scene unchanged unless the requested edit necessarily affects them.
- Prefer localized edits over global rewrites.
- Do not turn the request into a full image generation prompt.

STYLE RULES
- Output natural, concise, user-style edit requests.
- Use imperative phrasing.
- Each instruction should be 6 to 18 words.
- Avoid prompt-engineering style language.
- Avoid vague wording such as "improve it" or "make it nice".
- Avoid mentioning category labels inside the instruction.

APPLICABILITY RULES
- Beautify/General Enhancement: usually applicable for clear human photos.
- Body Shape (Weight/Muscle/Figure): only applicable if enough body shape is visible.
- Facial Feature Editing: only applicable if the face is clear enough.
- Age Change: only applicable if the face is visible enough.
- Expression/Emotion Change: only applicable if the face is visible enough.
- Hair Editing: only applicable if hair is visible enough.
- Skin Editing: only applicable if face or visible skin area is clear enough.
- Facial Hair: only applicable if the lower face is clear enough.
- Gender Transformation: only applicable for clearly visible adult humans.
- Makeup Editing: only applicable if the face is visible enough, ideally with visible makeup or makeup-edit plausibility.

SAFETY RULES
- Do not generate explicit sexual content, nudity, fetish content, hateful content, violent content, self-harm content, or humiliation content.
- Do not generate sexualized body-edit requests.
- If the person may be a minor, do not generate risky attractiveness, body-shape, gender-transformation, or sexualized requests. If uncertain and the category is risky, return applicable=false.
- Do not infer sensitive attributes.
- Do not make medical claims or diagnosis statements.

DIVERSITY RULES
- Generate a natural, non-generic instruction for the target category.
- Prefer one primary edit in the instruction.
- If two edits are included, they must be tightly coupled and still belong to the same category.

OUTPUT FORMAT
Return valid JSON only. No markdown. No explanation. No extra text.

Use exactly this schema when applicable:

{
  "category": "<target category>",
  "applicable": true,
  "target_person": "<explicit target person or null>",
  "instruction": "<edit instruction>"
}

Use exactly this schema when not applicable:

{
  "category": "<target category>",
  "applicable": false,
  "target_person": null,
  "instruction": null
}"""

USER_PROMPT_TEMPLATE = """\
Target category: {TARGET_CATEGORY}
Strict category control: true
Return JSON only: true

Please inspect the input image and generate one instruction only for the target category above.

Requirements:
- Do not switch to any other category.
- If multiple people are present, explicitly identify the edited person.
- If the requested category is not visually applicable, return applicable=false.
- Keep identity and non-target content unchanged.
- Prefer natural user-style editing requests.
- The instruction should be concise, realistic, and directly usable for image editing training."""


# ---------------------------------------------------------------------------
# CLI arguments
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate human-image edit instructions with vLLM + Qwen VL.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--image-paths",
        type=str,
        default="balanced_sampled_image_paths_200000.txt",
        help="Text file with one image path per line.",
    )
    p.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-VL-30B-A3B-Instruct",
        help="HuggingFace model ID or local path for the VL model.",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for applicable instruction parquet shards grouped by resolution.",
    )
    p.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Deprecated alias for --output-dir. If a .parquet path is given, its suffix "
            "is stripped and the stem is used as the shard directory."
        ),
    )
    p.add_argument(
        "--all-results-output",
        type=str,
        default="human_edit_all_results.parquet",
        help="Output parquet with ALL outcomes (for audit / rebalancing).",
    )
    p.add_argument(
        "--checkpoint-dir",
        type=str,
        default="instruction_gen_checkpoints",
        help="Directory for intermediate per-batch results.",
    )
    p.add_argument(
        "--tensor-parallel",
        type=int,
        default=8,
        help="Tensor-parallel size for vLLM.",
    )
    p.add_argument(
        "--rank",
        type=int,
        default=0,
        help="Machine rank for multi-machine sharding.",
    )
    p.add_argument(
        "--world-size",
        type=int,
        default=1,
        help="Number of machines participating in the job.",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=10000,
        help="Images per checkpoint batch.",
    )
    p.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Images per vLLM submission chunk (within a batch).",
    )
    p.add_argument(
        "--max-model-len",
        type=int,
        default=4096,
        help="Maximum model context length (tokens).",
    )
    p.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Max new tokens per response.",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="Sampling temperature (lower = more deterministic JSON).",
    )
    p.add_argument(
        "--max-pixels",
        type=int,
        default=1280 * 28 * 28,
        help="Max image pixels fed to the vision encoder.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible category assignment.",
    )
    p.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.90,
        help="Fraction of GPU memory for vLLM KV cache.",
    )
    return p.parse_args()


def resolve_output_dir(args: argparse.Namespace) -> Path:
    raw_output = args.output if args.output is not None else args.output_dir
    output_dir = Path(raw_output)
    if output_dir.suffix == ".parquet":
        output_dir = output_dir.with_suffix("")
        print(
            "WARNING: applicable instructions are now written as resolution-sharded parquet "
            f"files; using directory: {output_dir}"
        )
    return output_dir


def assigned_batches(num_batches: int, rank: int, world_size: int) -> list[int]:
    return list(range(rank, num_batches, world_size))


# ---------------------------------------------------------------------------
# Category assignment
# ---------------------------------------------------------------------------
def assign_categories(
    image_paths: list[str],
    seed: int = 42,
) -> list[tuple[str, str]]:
    """Return (image_path, category) pairs with deterministic random assignment."""
    rng = random.Random(seed)
    return [
        (p, rng.choices(CATEGORY_NAMES, weights=CATEGORY_WEIGHTS, k=1)[0])
        for p in image_paths
    ]


# ---------------------------------------------------------------------------
# Conversation builder
# ---------------------------------------------------------------------------
def build_conversation(image_path: str, category: str) -> list[dict]:
    """Build an OpenAI-compatible multimodal conversation for one image."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"file://{image_path}"},
                },
                {
                    "type": "text",
                    "text": USER_PROMPT_TEMPLATE.format(TARGET_CATEGORY=category),
                },
            ],
        },
    ]


# ---------------------------------------------------------------------------
# Robust JSON parser
# ---------------------------------------------------------------------------
_VALID_CATEGORIES = set(CATEGORY_NAMES)


def parse_model_output(raw_text: str) -> dict[str, Any] | None:
    """Try to extract a valid JSON dict from model output text.

    Returns the parsed dict or None on failure.
    """
    text = raw_text.strip()

    # Strip markdown code fences
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [ln for ln in lines if not ln.strip().startswith("```")]
        text = "\n".join(lines).strip()

    # Attempt 1: direct parse
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except (json.JSONDecodeError, ValueError):
        pass

    # Attempt 2: extract first {...} substring
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            obj = json.loads(text[start : end + 1])
            if isinstance(obj, dict):
                return obj
        except (json.JSONDecodeError, ValueError):
            pass

    return None


def validate_result(parsed: dict, assigned_category: str) -> dict[str, Any]:
    """Validate and normalise a parsed JSON response.

    Returns a record dict with status = 'applicable' | 'not_applicable' | 'invalid'.
    """
    applicable = parsed.get("applicable")

    if applicable is True:
        instruction = parsed.get("instruction")
        if not isinstance(instruction, str) or not instruction.strip():
            return {"status": "invalid", "instruction": None, "target_person": None}
        return {
            "status": "applicable",
            "instruction": instruction.strip(),
            "target_person": parsed.get("target_person"),
        }

    if applicable is False:
        return {"status": "not_applicable", "instruction": None, "target_person": None}

    # applicable field missing or unexpected value
    return {"status": "invalid", "instruction": None, "target_person": None}


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------
def get_completed_batches(checkpoint_dir: Path) -> set[int]:
    if not checkpoint_dir.is_dir():
        return set()
    completed: set[int] = set()
    for fname in checkpoint_dir.iterdir():
        if fname.name.startswith("batch_") and fname.suffix == ".parquet":
            try:
                idx = int(fname.stem.split("_")[1])
                pd.read_parquet(fname, columns=[])  # validate readable
                completed.add(idx)
            except (ValueError, IndexError):
                pass
            except Exception:
                # Corrupted checkpoint — will be overwritten on re-run
                print(f"WARNING: corrupted checkpoint {fname.name}, will reprocess")
    return completed


def save_checkpoint(records: list[dict], path: Path) -> None:
    df = pd.DataFrame(records)
    tmp = path.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=False)
    tmp.rename(path)  # atomic on POSIX


def merge_checkpoints(checkpoint_dir: Path) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    expected_cols: set[str] | None = None
    for fpath in sorted(checkpoint_dir.glob("batch_*.parquet")):
        try:
            df = pd.read_parquet(fpath)
        except Exception as exc:
            raise RuntimeError(f"Failed to read {fpath.name}: {exc}") from exc
        if expected_cols is None:
            expected_cols = set(df.columns)
        elif set(df.columns) != expected_cols:
            raise ValueError(
                f"Schema mismatch in {fpath.name}: "
                f"expected {expected_cols}, got {set(df.columns)}"
            )
        parts.append(df)
    if parts:
        return pd.concat(parts, ignore_index=True)
    return pd.DataFrame(
        columns=["image_path", "category", "status", "instruction", "target_person"]
    )


def read_image_resolution(image_path: str) -> tuple[int, int]:
    with Image.open(image_path) as img:
        width, height = img.size
    return width, height


def attach_image_resolutions(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        enriched = df.copy()
        enriched["width"] = pd.Series(dtype="int64")
        enriched["height"] = pd.Series(dtype="int64")
        return enriched

    max_workers = max(4, min(32, os.cpu_count() or 4))
    image_paths = df["image_path"].tolist()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        resolutions = list(executor.map(read_image_resolution, image_paths))

    widths = [width for width, _ in resolutions]
    heights = [height for _, height in resolutions]
    enriched = df.copy()
    enriched["width"] = widths
    enriched["height"] = heights
    return enriched


def shard_filename(width: int, height: int) -> str:
    return f"resolution_{width}x{height}.parquet"


def write_applicable_shards(applicable_df: pd.DataFrame, output_dir: Path) -> list[tuple[str, int]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale in output_dir.glob("resolution_*.parquet"):
        stale.unlink()

    shard_counts: list[tuple[str, int]] = []
    grouped = applicable_df.groupby(["width", "height"], sort=True)
    for (width, height), group in grouped:
        shard_path = output_dir / shard_filename(int(width), int(height))
        group.to_parquet(shard_path, index=False)
        shard_counts.append((shard_path.name, len(group)))
    return shard_counts


# ---------------------------------------------------------------------------
# Pretty progress
# ---------------------------------------------------------------------------
def fmt_elapsed(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    args = parse_args()
    output_dir = resolve_output_dir(args)
    if args.world_size <= 0:
        raise ValueError("--world-size must be a positive integer.")
    if not 0 <= args.rank < args.world_size:
        raise ValueError("--rank must satisfy 0 <= rank < world-size.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be a positive integer.")
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be a positive integer.")

    # ── 1. Read image paths ──────────────────────────────────────────────
    path_file = Path(args.image_paths)
    if not path_file.is_file():
        print(f"ERROR: image paths file not found: {path_file}", file=sys.stderr)
        return 1

    with open(path_file) as f:
        image_paths = [line.strip() for line in f if line.strip()]
    total_images = len(image_paths)
    print(f"Loaded {total_images:,} image paths from {path_file}")

    # Quick validation: check a sample of paths exist
    sample_size = min(100, total_images)
    rng_sample = random.Random(0)
    sample_indices = rng_sample.sample(range(total_images), sample_size)
    missing = sum(1 for i in sample_indices if not os.path.isfile(image_paths[i]))
    if missing > 0:
        est_missing = int(missing / sample_size * total_images)
        print(
            f"WARNING: ~{est_missing:,} image paths may not exist "
            f"(sampled {missing}/{sample_size} missing)"
        )

    # ── 2. Assign categories ─────────────────────────────────────────────
    assignments = assign_categories(image_paths, seed=args.seed)
    cat_counts = Counter(cat for _, cat in assignments)
    print("\nAssigned category distribution:")
    for name, weight in CATEGORIES:
        n = cat_counts.get(name, 0)
        print(f"  {name:40s}: {n:>7,} ({n / total_images * 100:5.1f}%)")

    # ── 3. Checkpoint setup ──────────────────────────────────────────────
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    batch_size = args.batch_size
    num_batches = math.ceil(total_images / batch_size)
    completed_batches = get_completed_batches(checkpoint_dir)
    local_batches = assigned_batches(num_batches, args.rank, args.world_size)
    pending_batches = [i for i in local_batches if i not in completed_batches]
    print(
        f"\nBatches: {num_batches} total, {len(completed_batches)} done globally, "
        f"{len(local_batches)} assigned to rank {args.rank}/{args.world_size}, "
        f"{len(pending_batches)} pending locally"
    )

    # ── 4. Inference ─────────────────────────────────────────────────────
    if pending_batches:
        from vllm import LLM, SamplingParams  # deferred import

        print(f"\nInitialising vLLM  model={args.model}")
        print(f"  tensor_parallel={args.tensor_parallel}  max_model_len={args.max_model_len}")
        print(f"  gpu_memory_utilization={args.gpu_memory_utilization}")
        print(f"  max_pixels={args.max_pixels}  temperature={args.temperature}")

        llm = LLM(
            model=args.model,
            tensor_parallel_size=args.tensor_parallel,
            max_model_len=args.max_model_len,
            limit_mm_per_prompt={"image": 1},
            trust_remote_code=True,
            gpu_memory_utilization=args.gpu_memory_utilization,
            mm_processor_kwargs={"max_pixels": args.max_pixels},
            allowed_local_media_path="/data",
            attention_config={"flash_attn_version": 2},
        )

        sampling_params = SamplingParams(
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )

        overall_t0 = time.time()
        for batch_idx in pending_batches:
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, total_images)
            batch = assignments[batch_start:batch_end]
            batch_len = len(batch)

            print(
                f"\n{'─' * 60}\n"
                f"Batch {batch_idx}/{num_batches - 1}  "
                f"images [{batch_start:,} – {batch_end - 1:,}]  "
                f"size={batch_len:,}"
            )
            batch_t0 = time.time()

            batch_records: list[dict[str, Any]] = []
            chunk_size = args.chunk_size

            for chunk_offset in range(0, batch_len, chunk_size):
                chunk = batch[chunk_offset : chunk_offset + chunk_size]
                chunk_len = len(chunk)

                # Build conversations
                conversations = [
                    build_conversation(img_path, cat) for img_path, cat in chunk
                ]

                # Run inference
                try:
                    outputs = llm.chat(
                        messages=conversations,
                        sampling_params=sampling_params,
                    )
                except Exception as exc:
                    print(f"  ERROR in chunk offset {chunk_offset}: {exc}")
                    # Record failures for this chunk
                    for img_path, cat in chunk:
                        batch_records.append(
                            {
                                "image_path": img_path,
                                "category": cat,
                                "status": "error",
                                "instruction": None,
                                "target_person": None,
                                "raw_output": str(exc)[:500],
                            }
                        )
                    continue

                # Guard against silent data loss from mismatched output count
                if len(outputs) != chunk_len:
                    raise RuntimeError(
                        f"vLLM returned {len(outputs)} outputs for "
                        f"{chunk_len} inputs — aborting to prevent data loss."
                    )

                # Parse outputs
                for (img_path, cat), output in zip(chunk, outputs):
                    raw_text = output.outputs[0].text
                    parsed = parse_model_output(raw_text)

                    if parsed is None:
                        record = {
                            "status": "parse_failure",
                            "instruction": None,
                            "target_person": None,
                        }
                    else:
                        record = validate_result(parsed, cat)

                    record["image_path"] = img_path
                    record["category"] = cat
                    record["raw_output"] = raw_text[:500]
                    batch_records.append(record)

                elapsed_chunk = time.time() - batch_t0
                print(
                    f"  chunk {chunk_offset // chunk_size + 1}/"
                    f"{math.ceil(batch_len / chunk_size)}  "
                    f"{chunk_offset + chunk_len}/{batch_len}  "
                    f"{elapsed_chunk:.0f}s",
                    flush=True,
                )

            # Save checkpoint
            batch_path = checkpoint_dir / f"batch_{batch_idx:04d}.parquet"
            save_checkpoint(batch_records, batch_path)

            batch_elapsed = time.time() - batch_t0
            n_applicable = sum(1 for r in batch_records if r["status"] == "applicable")
            n_not_app = sum(1 for r in batch_records if r["status"] == "not_applicable")
            n_fail = sum(
                1 for r in batch_records if r["status"] in ("parse_failure", "invalid", "error")
            )
            print(
                f"  ✓ saved  applicable={n_applicable}  not_applicable={n_not_app}  "
                f"failures={n_fail}  time={fmt_elapsed(batch_elapsed)}  "
                f"speed={batch_len / batch_elapsed:.0f} img/s"
            )

            # ETA for this rank's remaining assigned batches
            overall_elapsed = time.time() - overall_t0
            local_batches_done = pending_batches.index(batch_idx) + 1
            local_batches_remaining = len(pending_batches) - local_batches_done
            if local_batches_done > 0:
                avg_batch_time = overall_elapsed / (
                    local_batches_done
                )
                eta = avg_batch_time * local_batches_remaining
                print(
                    f"  rank progress: {local_batches_done}/{len(pending_batches)} local batches  "
                    f"elapsed={fmt_elapsed(overall_elapsed)}  "
                    f"ETA≈{fmt_elapsed(eta)}"
                )

    # ── 5. Merge all checkpoints ─────────────────────────────────────────
    refreshed_completed_batches = get_completed_batches(checkpoint_dir)
    all_batches_complete = len(refreshed_completed_batches) == num_batches

    if args.world_size > 1 and not all_batches_complete:
        print(f"\n{'═' * 60}")
        print(
            f"Rank {args.rank}/{args.world_size} finished its assigned work. "
            f"Global completion: {len(refreshed_completed_batches)}/{num_batches} batches."
        )
        if args.rank == 0:
            print("Final merge skipped for now; rerun rank 0 after all machines finish.")
        return 0

    if args.world_size > 1 and args.rank != 0:
        print(f"\n{'═' * 60}")
        print(
            f"Rank {args.rank}/{args.world_size} finished. "
            "Final merge is handled by rank 0."
        )
        return 0

    print(f"\n{'═' * 60}")
    print("Merging checkpoint files …")
    all_df = merge_checkpoints(checkpoint_dir)
    print(f"Total records: {len(all_df):,}")

    # Save all-results parquet
    all_df.to_parquet(args.all_results_output, index=False)
    print(f"All results saved to: {args.all_results_output}")

    # Filter applicable and save
    applicable_df = all_df[all_df["status"] == "applicable"].drop(
        columns=["status", "raw_output"], errors="ignore"
    ).reindex(columns=["image_path", "category", "instruction", "target_person"])
    applicable_df.insert(0, "row_idx", range(len(applicable_df)))
    applicable_df = attach_image_resolutions(applicable_df)
    applicable_df = applicable_df[
        ["row_idx", "image_path", "category", "instruction", "target_person", "width", "height"]
    ]
    shard_counts = write_applicable_shards(applicable_df, output_dir)
    print(f"Applicable instruction shards saved to: {output_dir}")
    for shard_name, shard_len in shard_counts:
        print(f"  {shard_name:28s} {shard_len:>8,}")

    # ── 6. Final statistics ──────────────────────────────────────────────
    status_counts = all_df["status"].value_counts()
    print(f"\nOutcome breakdown:")
    for status in ["applicable", "not_applicable", "parse_failure", "invalid", "error"]:
        n = status_counts.get(status, 0)
        pct = n / len(all_df) * 100 if len(all_df) else 0
        print(f"  {status:20s}: {n:>8,} ({pct:5.1f}%)")

    if len(applicable_df) > 0:
        cat_stats = applicable_df["category"].value_counts()
        print(f"\nApplicable instructions per category:")
        for name, _ in CATEGORIES:
            n = cat_stats.get(name, 0)
            pct = n / len(applicable_df) * 100
            print(f"  {name:40s}: {n:>7,} ({pct:5.1f}%)")

    print(f"\nDone. {len(applicable_df):,} applicable instructions ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
