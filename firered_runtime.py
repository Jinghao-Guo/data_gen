from __future__ import annotations

import torch
from diffusers import QwenImageEditPlusPipeline

from firered_fast_pipeline import load_fast_pipeline
from firered_model_spec import resolve_model_spec


def configure_torch_backends(device: str) -> None:
    if device.startswith("cuda"):
        torch.backends.cuda.enable_cudnn_sdp(False)


def load_pipeline(
    model_path: str,
    device: str = "cuda",
    optimized: bool = False,
    disable_progress: bool = False,
) -> QwenImageEditPlusPipeline:
    model_spec = resolve_model_spec(model_path)
    configure_torch_backends(device)
    print(
        f"Resolved FireRed model on {device}: {model_spec.summary()} | "
        f"optimized={optimized}"
    )

    if optimized:
        return load_fast_pipeline(
            model_spec.requested_model_path,
            device=device,
            disable_progress=disable_progress,
        )

    pipe = QwenImageEditPlusPipeline.from_pretrained(
        model_spec.pipeline_model_path,
        torch_dtype=torch.bfloat16,
    )
    if model_spec.lora_repo is not None:
        pipe.load_lora_weights(
            model_spec.lora_repo,
            weight_name=model_spec.lora_weight_name,
        )
    pipe.to(device)
    pipe.set_progress_bar_config(disable=disable_progress)
    return pipe
