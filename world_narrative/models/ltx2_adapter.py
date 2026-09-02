from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass
class LTX2AdapterConfig:
    base_model: str = ""
    vae: str = ""
    text_encoder: str = ""
    use_lora: bool = True
    lora_rank: int = 32
    lora_alpha: int = 32
    precision: str = "bf16"
    chunk_latents: int = 4
    hidden_dim: int = 256
    max_frames: int = 512
    control_dim: int = 16


class LTX2Adapter:
    """Adapter boundary for plugging an actual LTX-2 stack later.

    In the current scaffold it also acts as a best-effort checkpoint loader for
    the local narrative model so the repo stays runnable without LTX-2 assets.
    """

    def __init__(self, cfg: LTX2AdapterConfig) -> None:
        self.cfg = cfg

    def describe(self) -> str:
        return (
            f"LTX2Adapter(base={self.cfg.base_model}, vae={self.cfg.vae}, "
            f"text_encoder={self.cfg.text_encoder}, lora={self.cfg.use_lora})"
        )

    def load_weights(self, model: torch.nn.Module | None = None, *, checkpoint: str | None = None) -> dict[str, int]:
        path = Path(checkpoint or self.cfg.base_model)
        if model is None or not path.exists():
            return {"loaded": 0, "missing": 0, "unexpected": 0}
        try:
            if path.suffix == ".safetensors":
                from safetensors.torch import load_file

                state = load_file(str(path), device="cpu")
            else:
                payload = torch.load(path, map_location="cpu")
                state = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
            missing, unexpected = model.load_state_dict(state, strict=False)
            return {"loaded": len(state), "missing": len(missing), "unexpected": len(unexpected)}
        except Exception:
            return {"loaded": 0, "missing": 0, "unexpected": 0}

    def encode_prompt(self, prompt: str) -> Any:
        return prompt

    def encode_video(self, video: Any) -> Any:
        return video

    def decode_latents(self, latents: Any) -> Any:
        return latents

    def rollout_chunk(self, inputs: Any) -> Any:
        return inputs
