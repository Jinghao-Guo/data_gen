from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from PIL import Image

from firered_model_spec import COMFYUI_LIGHTNING_MODEL_ID, resolve_model_spec
from firered_runtime import load_pipeline

DEFAULT_MODEL = COMFYUI_LIGHTNING_MODEL_ID
DEFAULT_INPUT_IMAGE = Path("demo_input.png")
DEFAULT_OUTPUT_IMAGE = Path("edited_image.png")
DEFAULT_PROMPT = "Turn this cat into a dog"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FireRed image-edit inference.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model path or Hugging Face model ID.")
    parser.add_argument("--input-image", type=Path, default=DEFAULT_INPUT_IMAGE, help="Input image path.")
    parser.add_argument("--output-image", type=Path, default=DEFAULT_OUTPUT_IMAGE, help="Output image path.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Editing prompt.")
    parser.add_argument("--negative-prompt", default=None, help="Optional negative prompt.")
    parser.add_argument("--seed", type=int, default=43, help="Random seed.")
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=None,
        help="Number of denoising steps. Defaults to the selected model's recommendation.",
    )
    parser.add_argument("--true-cfg-scale", type=float, default=1.0, help="True CFG scale.")
    parser.add_argument("--device", default="cuda", help='Target device, for example "cuda" or "cuda:0".')
    parser.add_argument(
        "--optimized",
        action="store_true",
        help="Enable the official fast path: int8 quantization, DiT cache, torch.compile, and warmup.",
    )
    parser.add_argument(
        "--disable-progress",
        action="store_true",
        help="Disable diffusers progress bars.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_spec = resolve_model_spec(args.model)
    num_inference_steps = (
        args.num_inference_steps
        if args.num_inference_steps is not None
        else model_spec.recommended_num_inference_steps
    )

    if not args.input_image.is_file():
        raise FileNotFoundError(f"Input image not found: {args.input_image}")

    args.output_image.parent.mkdir(parents=True, exist_ok=True)

    load_t0 = time.time()
    pipeline = load_pipeline(
        args.model,
        device=args.device,
        optimized=args.optimized,
        disable_progress=args.disable_progress,
    )
    load_elapsed = time.time() - load_t0

    image = Image.open(args.input_image).convert("RGB")
    generator = torch.Generator(device=args.device).manual_seed(args.seed)

    inputs = {
        "image": image,
        "prompt": args.prompt,
        "true_cfg_scale": args.true_cfg_scale,
        "num_inference_steps": num_inference_steps,
        "generator": generator,
        "height": image.height,
        "width": image.width,
    }
    if args.negative_prompt is not None:
        inputs["negative_prompt"] = args.negative_prompt

    infer_t0 = time.time()
    with torch.inference_mode():
        result = pipeline(**inputs)
    infer_elapsed = time.time() - infer_t0

    result.images[0].save(args.output_image)

    print(f"Saved edited image to: {args.output_image.resolve()}")
    print(f"Model: {args.model}")
    print(f"Inference steps: {num_inference_steps}")
    print(f"Load time: {load_elapsed:.2f}s")
    print(f"Inference time: {infer_elapsed:.2f}s")


if __name__ == "__main__":
    main()
