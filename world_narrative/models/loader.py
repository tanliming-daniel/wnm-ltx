from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .ltx2_adapter import LTX2Adapter, LTX2AdapterConfig
from .narrative_model import WorldNarrativeModel


def _align_vae_chunk_size(frames: int) -> int:
    if frames < 9:
        return 9
    remainder = (frames - 1) % 8
    return frames if remainder == 0 else frames + (8 - remainder)


def _maybe_load_state(model: torch.nn.Module, checkpoint: str | None) -> dict[str, int]:
    if not checkpoint:
        return {"loaded": 0, "missing": 0, "unexpected": 0}
    path = Path(checkpoint)
    if not path.exists():
        return {"loaded": 0, "missing": 0, "unexpected": 0}
    try:
        if path.suffix == ".safetensors":
            from safetensors.torch import load_file

            state = load_file(str(path), device="cpu")
        else:
            payload = torch.load(path, map_location="cpu")
            state = (
                payload.get("model_state_dict", payload.get("state_dict", payload.get("model", payload)))
                if isinstance(payload, dict)
                else payload
            )
        missing, unexpected = model.load_state_dict(state, strict=False)
        return {"loaded": len(state), "missing": len(missing), "unexpected": len(unexpected)}
    except Exception:
        return {"loaded": 0, "missing": 0, "unexpected": 0}


def _maybe_load_payload(checkpoint: str | None) -> dict[str, Any] | None:
    if not checkpoint:
        return None
    path = Path(checkpoint)
    if not path.exists():
        return None
    try:
        if path.suffix == ".safetensors":
            from safetensors.torch import load_file

            return load_file(str(path), device="cpu")
        payload = torch.load(path, map_location="cpu")
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None


def build_world_model(
    cfg,
    *,
    role: str = "student",
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    checkpoint: str | None = None,
) -> tuple[WorldNarrativeModel, LTX2Adapter]:
    chunk_frames = max(1, int(round(float(cfg.stage.chunk_seconds) * float(cfg.data.fps))))
    vae_chunk_frames = _align_vae_chunk_size(chunk_frames)
    teacher_steps = int(getattr(cfg.dmd, "teacher_steps", max(8, chunk_frames // 2)))
    adapter = LTX2Adapter(
        LTX2AdapterConfig(
            base_model=cfg.paths.base_model,
            vae=cfg.paths.vae,
            text_encoder=cfg.paths.text_encoder,
            use_lora=bool(cfg.stage.use_lora),
            lora_rank=int(getattr(cfg.stage, "lora_rank", 32)),
            lora_alpha=int(getattr(cfg.stage, "lora_alpha", 32)),
            precision=str(getattr(cfg.train, "precision", "bf16")),
            chunk_latents=vae_chunk_frames,
            hidden_dim=int(getattr(cfg.stage, "hidden_dim", 256)),
            max_frames=max(512, int(round(float(cfg.data.clip_seconds) * float(cfg.data.fps))) + chunk_frames),
            control_dim=int(getattr(cfg.data, "control_dim", 16)),
            use_action_control=bool(getattr(cfg.data, "use_camera", True)),
        )
    )
    model = WorldNarrativeModel(
        hidden_dim=adapter.cfg.hidden_dim,
        output_hw=(int(cfg.data.height), int(cfg.data.width)),
        decode_hw=(max(8, int(cfg.data.height) // 16), max(8, int(cfg.data.width) // 16)),
        max_frames=adapter.cfg.max_frames,
        control_dim=adapter.cfg.control_dim,
        num_layers=int(getattr(cfg.stage, "num_layers", 4)),
        num_heads=int(getattr(cfg.stage, "num_heads", 8)),
        chunk_frames=chunk_frames,
        teacher_steps=teacher_steps if role == "teacher" else int(getattr(cfg.dmd, "student_steps", 4)),
        backend=adapter,
    )
    model.set_backend(adapter)
    if device is not None or dtype is not None:
        model = model.to(device=device, dtype=dtype)
    load_checkpoint_into_model(model, checkpoint)
    payload = _maybe_load_payload(checkpoint)
    if payload is not None and hasattr(adapter, "load_state_dict"):
        adapter_state = payload.get("adapter", payload.get("adapter_state"))
        if isinstance(adapter_state, dict):
            adapter.load_state_dict(adapter_state)
    return model, adapter


def load_checkpoint_into_model(model: torch.nn.Module, checkpoint: str | None) -> dict[str, int]:
    return _maybe_load_state(model, checkpoint)


def describe_model(model: WorldNarrativeModel, adapter: LTX2Adapter) -> str:
    return (
        f"WorldNarrativeModel(hidden={model.hidden_dim}, chunk_frames={model.chunk_frames}, "
        f"frames={model.max_frames}, output={model.output_hw}, teacher_steps={model.teacher_steps}; "
        f"{adapter.describe()})"
    )
