from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import torch
import torch.nn as nn

from .modules import NarrativeModuleConfig, build_narrative_modules


@dataclass
class NarrativeState:
    prompt: str = ""
    summary: str = ""
    history_chunks: list[Any] = field(default_factory=list)
    future_chunks: list[Any] = field(default_factory=list)
    control_state: dict[str, Any] = field(default_factory=dict)
    step_index: int = 0


class WorldNarrativeModel(nn.Module):
    """Stateful video world model wrapper.

    The current implementation is a lightweight scaffold that already exposes the
    real hooks needed by the training stages: prompt encoding, history memory,
    causal rollout, iterative refinement, and chunk-wise generation.
    """

    def __init__(
        self,
        *,
        hidden_dim: int = 256,
        output_hw: tuple[int, int] = (544, 960),
        decode_hw: tuple[int, int] = (32, 32),
        max_frames: int = 512,
        control_dim: int = 16,
        num_layers: int = 4,
        num_heads: int = 8,
        chunk_frames: int = 32,
        teacher_steps: int = 8,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_hw = output_hw
        self.decode_hw = decode_hw
        self.max_frames = max_frames
        self.control_dim = control_dim
        self.chunk_frames = chunk_frames
        self.teacher_steps = teacher_steps

        module_cfg = NarrativeModuleConfig(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            max_frames=max_frames,
            control_dim=control_dim,
            decode_hw=decode_hw,
            output_hw=output_hw,
        )
        (
            self.prompt_encoder,
            self.control_encoder,
            self.frame_encoder,
            self.history_encoder,
            self.core,
            self.refiner,
            self.frame_decoder,
        ) = build_narrative_modules(module_cfg)
        self.context_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.context_norm = nn.LayerNorm(hidden_dim)
        self.mask_token = nn.Parameter(torch.zeros(hidden_dim))

    def init_state(self, prompt: str, *, control: Mapping[str, Any] | None = None) -> NarrativeState:
        return NarrativeState(prompt=prompt, control_state=dict(control or {}))

    def encode_prompt(self, prompts: list[str], *, device: torch.device) -> torch.Tensor:
        return self.prompt_encoder(prompts, device)

    def encode_control(
        self,
        control: torch.Tensor | None,
        *,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        return self.control_encoder(control, batch_size=batch_size, device=device)

    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        return self.frame_encoder(video)

    def encode_history(self, frame_tokens: torch.Tensor, history_frames: int | None = None) -> torch.Tensor:
        if history_frames is not None and history_frames > 0:
            frame_tokens = frame_tokens[:, : min(history_frames, frame_tokens.size(1))]
        return self.history_encoder(frame_tokens)

    def _merge_context(
        self,
        prompt_ctx: torch.Tensor,
        history_ctx: torch.Tensor,
        control_ctx: torch.Tensor,
    ) -> torch.Tensor:
        ctx = prompt_ctx + history_ctx + control_ctx
        return self.context_norm(self.context_proj(ctx))

    def _normalise_frame_mask(self, frame_mask: torch.Tensor | None, *, batch_size: int, num_frames: int, device: torch.device) -> torch.Tensor:
        if frame_mask is None:
            return torch.ones(batch_size, num_frames, dtype=torch.bool, device=device)
        mask = frame_mask
        if mask.dim() == 4:
            mask = mask.squeeze(-1).squeeze(-1)
        if mask.dim() == 3:
            mask = mask.squeeze(-1)
        if mask.dim() != 2:
            raise ValueError(f"expected frame_mask [B,T] or [B,T,1,1], got {tuple(mask.shape)}")
        if mask.size(0) != batch_size or mask.size(1) != num_frames:
            raise ValueError(
                f"frame_mask shape {tuple(mask.shape)} does not match batch={batch_size}, frames={num_frames}"
            )
        return mask.to(device=device).bool()

    def forward(
        self,
        video: torch.Tensor,
        prompts: list[str],
        *,
        control: torch.Tensor | None = None,
        frame_mask: torch.Tensor | None = None,
        history_frames: int | None = None,
        causal: bool = False,
        refine_steps: int = 1,
    ) -> dict[str, torch.Tensor]:
        if video.dim() != 5:
            raise ValueError(f"expected video [B,T,C,H,W], got {tuple(video.shape)}")
        b, t, _, h, w = video.shape
        device = video.device
        frame_tokens = self.encode_video(video)
        visible_mask = self._normalise_frame_mask(frame_mask, batch_size=b, num_frames=t, device=device)
        frame_tokens = torch.where(
            visible_mask.unsqueeze(-1),
            frame_tokens,
            self.mask_token.view(1, 1, -1).expand(b, t, -1),
        )
        prompt_ctx = self.encode_prompt(prompts, device=device)
        hist_ctx = self.encode_history(frame_tokens, history_frames=history_frames)
        control_ctx = self.encode_control(control, batch_size=b, device=device)
        context = self._merge_context(prompt_ctx, hist_ctx, control_ctx)
        core_tokens, context_token = self.core(frame_tokens, context, causal=causal)
        for _ in range(max(1, int(refine_steps))):
            core_tokens = self.refiner(core_tokens, context_token)
        pred_video = self.frame_decoder(core_tokens)
        return {
            "video": pred_video,
            "tokens": core_tokens,
            "context": context,
            "context_token": context_token,
            "prompt_context": prompt_ctx,
            "history_context": hist_ctx,
            "control_context": control_ctx,
        }

    def predict_future(
        self,
        history_video: torch.Tensor,
        prompts: list[str],
        *,
        future_frames: int,
        control: torch.Tensor | None = None,
        refine_steps: int = 1,
    ) -> dict[str, torch.Tensor]:
        if history_video.dim() != 5:
            raise ValueError(f"expected history video [B,T,C,H,W], got {tuple(history_video.shape)}")
        b, history_t, c, h, w = history_video.shape
        device = history_video.device
        future_pad = torch.zeros(b, future_frames, c, h, w, device=device, dtype=history_video.dtype)
        video = torch.cat([history_video, future_pad], dim=1)
        mask = torch.zeros(b, history_t + future_frames, dtype=torch.bool, device=device)
        mask[:, :history_t] = True
        out = self.forward(
            video,
            prompts,
            control=control,
            frame_mask=mask,
            history_frames=history_t,
            causal=True,
            refine_steps=refine_steps,
        )
        out["future_video"] = out["video"][:, -future_frames:]
        return out

    def rollout(
        self,
        history_video: torch.Tensor,
        prompts: list[str],
        *,
        rounds: int,
        future_frames: int | None = None,
        control: torch.Tensor | None = None,
        refine_steps: int = 1,
    ) -> tuple[list[torch.Tensor], torch.Tensor, list[dict[str, torch.Tensor]]]:
        future_frames = int(future_frames or self.chunk_frames)
        current = history_video
        chunks: list[torch.Tensor] = []
        traces: list[dict[str, torch.Tensor]] = []
        for _ in range(int(rounds)):
            out = self.predict_future(
                current,
                prompts,
                future_frames=future_frames,
                control=control,
                refine_steps=refine_steps,
            )
            chunk = out["future_video"]
            chunks.append(chunk)
            traces.append(out)
            current = torch.cat([current, chunk], dim=1)
        return chunks, current, traces

    def update_state(self, state: NarrativeState, *, new_summary: str, generated_chunk: Any) -> NarrativeState:
        state.summary = new_summary
        state.history_chunks.append(generated_chunk)
        state.step_index += 1
        return state
