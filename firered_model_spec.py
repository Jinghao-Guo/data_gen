from __future__ import annotations

from dataclasses import dataclass


COMFYUI_LIGHTNING_MODEL_ID = "FireRedTeam/FireRed-Image-Edit-1.1-ComfyUI"
COMFYUI_LIGHTNING_BASE_MODEL_ID = "FireRedTeam/FireRed-Image-Edit-1.1"
COMFYUI_LIGHTNING_WEIGHT_NAME = "FireRed-Image-Edit-1.1-Lightning-8steps-v1.2.safetensors"


@dataclass(frozen=True)
class FireRedModelSpec:
    requested_model_path: str
    pipeline_model_path: str
    lora_repo: str | None = None
    lora_weight_name: str | None = None
    recommended_num_inference_steps: int = 40

    def summary(self) -> str:
        lora = (
            f"{self.lora_repo}:{self.lora_weight_name}"
            if self.lora_repo is not None
            else "none"
        )
        return (
            f"requested={self.requested_model_path} | "
            f"base={self.pipeline_model_path} | "
            f"lora={lora} | "
            f"recommended_steps={self.recommended_num_inference_steps}"
        )


def resolve_model_spec(model_path: str) -> FireRedModelSpec:
    if model_path == COMFYUI_LIGHTNING_MODEL_ID:
        return FireRedModelSpec(
            requested_model_path=model_path,
            pipeline_model_path=COMFYUI_LIGHTNING_BASE_MODEL_ID,
            lora_repo=COMFYUI_LIGHTNING_MODEL_ID,
            lora_weight_name=COMFYUI_LIGHTNING_WEIGHT_NAME,
            recommended_num_inference_steps=8,
        )

    return FireRedModelSpec(
        requested_model_path=model_path,
        pipeline_model_path=model_path,
        recommended_num_inference_steps=40,
    )
