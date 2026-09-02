from __future__ import annotations

import random
from typing import Any

from .base import BaseTrainer, StageSummary


class MemoryPretrainTrainer(BaseTrainer):
    stage_name = "memory_pretrain"

    def stage_summary(self) -> StageSummary:
        return StageSummary(
            kind="memory_pretrain",
            title="History-future reasoning pretrain",
            objective="Train memory tokens with masked history and future prediction",
            losses=("masked_history", "future_prediction", "temporal_consistency"),
            notes="This stage teaches the model to remember a prefix and continue into the future.",
        )

    def compute_step(
        self,
        batch: dict[str, Any],
        *,
        train: bool,
    ) -> tuple[torch.Tensor, dict[str, Any], dict[str, torch.Tensor]]:
        import torch

        from world_narrative.models.losses import masked_mse_loss, temporal_consistency_loss

        assert self.model is not None
        video = batch["video"].to(device=self.device, dtype=self.dtype)
        history_video = batch["history_video"].to(device=self.device, dtype=self.dtype)
        future_video = batch["future_video"].to(device=self.device, dtype=self.dtype)
        control = batch["control"].to(device=self.device, dtype=self.dtype)
        prompts = batch["prompt"]

        b, t, _, _, _ = video.shape
        history_len = history_video.size(1)
        future_len = future_video.size(1)

        omega_len = max(1, min(history_len, self.model.chunk_frames // 2))
        if train and history_len > omega_len:
            omega_start = random.randint(0, history_len - omega_len)
        else:
            omega_start = max(0, (history_len - omega_len) // 2)

        frame_mask = torch.zeros(b, t, dtype=torch.bool, device=self.device)
        frame_mask[:, omega_start : omega_start + omega_len] = True

        refine_steps = int(self.cfg.validation.refine_steps if not train else 1)
        out = self.model(
            video,
            prompts,
            control=control,
            frame_mask=frame_mask,
            history_frames=history_len,
            causal=True,
            refine_steps=refine_steps,
        )
        pred = out["video"]
        history_loss = masked_mse_loss(pred[:, :history_len], video[:, :history_len], frame_mask[:, :history_len])
        future_loss = torch.nn.functional.mse_loss(pred[:, history_len:], future_video)
        temporal = temporal_consistency_loss(pred)
        total = history_loss + future_loss + 0.05 * temporal
        logs = {
            "omega_start": omega_start,
            "omega_len": omega_len,
            "history_loss": float(history_loss.item()),
            "future_loss": float(future_loss.item()),
            "temporal": float(temporal.item()),
        }
        return total, logs, out
