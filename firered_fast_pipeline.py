from __future__ import annotations

import torch
from PIL import Image
from diffusers import QwenImageEditPlusPipeline, QwenImageTransformer2DModel
from optimum.quanto import freeze, qint8, quantize
from transformers import Qwen2_5_VLForConditionalGeneration

import cache_dit
from cache_dit import DBCacheConfig, TaylorSeerCalibratorConfig

from firered_model_spec import FireRedModelSpec, resolve_model_spec


def _linear_forward_hook(self, x: torch.Tensor, *args, **kwargs):
    """Keep LoRA layers graph-capture friendly for torch.compile."""
    result = self.base_layer(x, *args, **kwargs)
    if not hasattr(self, "active_adapters"):
        return result

    for active_adapter in self.active_adapters:
        if active_adapter not in self.lora_A:
            continue

        lora_a = self.lora_A[active_adapter]
        lora_b = self.lora_B[active_adapter]
        dropout = self.lora_dropout[active_adapter]
        scaling = self.scaling[active_adapter]
        x_input = x.to(lora_a.weight.dtype)
        output = lora_b(lora_a(dropout(x_input))) * scaling
        result = result + output.to(result.dtype)

    return result


def _apply_compile(pipeline: QwenImageEditPlusPipeline) -> None:
    from peft.tuners.lora.layer import Linear

    if not hasattr(pipeline.transformer, "compile_repeated_blocks"):
        raise RuntimeError("Current diffusers build does not support compile_repeated_blocks().")

    for module in pipeline.transformer.modules():
        if isinstance(module, Linear):
            module.forward = _linear_forward_hook.__get__(module, Linear)

    torch._dynamo.config.recompile_limit = 1024
    pipeline.transformer.compile_repeated_blocks(mode="default", dynamic=True)
    pipeline.vae = torch.compile(pipeline.vae, mode="reduce-overhead")


def _apply_cache(pipeline: QwenImageEditPlusPipeline) -> None:
    cache_dit.enable_cache(
        pipeline,
        cache_config=DBCacheConfig(
            Fn_compute_blocks=8,
            Bn_compute_blocks=0,
            residual_diff_threshold=0.15,
            max_warmup_steps=3,
        ),
        calibrator_config=TaylorSeerCalibratorConfig(taylorseer_order=1),
    )


def load_fast_pipeline(
    model_path: str,
    device: str = "cuda",
    disable_progress: bool = False,
    warmup_steps: int | None = None,
) -> QwenImageEditPlusPipeline:
    model_spec = resolve_model_spec(model_path)
    weight_dtype = torch.bfloat16
    if warmup_steps is None:
        warmup_steps = min(4, model_spec.recommended_num_inference_steps)

    print(f"Initializing optimized pipeline from {model_spec.requested_model_path}...")
    print(f"Resolved spec: {model_spec.summary()}")
    print("[1/3] Loading and quantizing text encoder + transformer...")

    text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_spec.pipeline_model_path,
        subfolder="text_encoder",
        dtype=weight_dtype,
    ).to(device)
    quantize(text_encoder, weights=qint8)
    freeze(text_encoder)

    transformer = QwenImageTransformer2DModel.from_pretrained(
        model_spec.pipeline_model_path,
        subfolder="transformer",
        torch_dtype=weight_dtype,
    )
    quantize(transformer, weights=qint8, exclude=["proj_out"])
    freeze(transformer)

    pipeline = QwenImageEditPlusPipeline.from_pretrained(
        model_spec.pipeline_model_path,
        transformer=transformer,
        text_encoder=text_encoder,
        torch_dtype=weight_dtype,
    )

    if model_spec.lora_repo is not None:
        pipeline.load_lora_weights(
            model_spec.lora_repo,
            weight_name=model_spec.lora_weight_name,
        )

    print("[2/3] Enabling DiT cache and torch.compile...")
    _apply_cache(pipeline)
    _apply_compile(pipeline)

    pipeline.vae.enable_tiling()
    pipeline.vae.enable_slicing()
    pipeline.to(device)
    pipeline.set_progress_bar_config(disable=disable_progress)

    print("[3/3] Running warmup compile pass...")
    fake_pil = Image.new("RGB", (896, 896), (128, 128, 128))
    with torch.inference_mode():
        pipeline(
            image=[fake_pil],
            prompt="warmup session",
            num_inference_steps=warmup_steps,
            negative_prompt=" ",
        )

    print("Optimized pipeline is ready.")
    return pipeline
