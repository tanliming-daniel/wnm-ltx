from __future__ import annotations

from typing import Any

from .base import BaseTrainer, StageSummary


class AutoregressiveTrainer(BaseTrainer):
    stage_name = "autoregressive"

    def stage_summary(self) -> StageSummary:
        return StageSummary(
            kind="autoregressive",
            title="Autoregressive rollout training",
            objective="Align the model with chunked causal rollout at inference time",
            losses=("causal_rollout", "prompt_following", "temporal_consistency"),
            notes="The model only sees the history prefix before predicting the future block.",
        )

    def compute_step(
        self,
        batch: dict[str, Any],
        *,
        train: bool,
    ) -> tuple[torch.Tensor, dict[str, Any], dict[str, torch.Tensor]]:
        import torch

        from world_narrative.models.losses import temporal_consistency_loss

        assert self.model is not None
        history_video = batch["history_video"].to(device=self.device, dtype=self.dtype)
        future_video = batch["future_video"].to(device=self.device, dtype=self.dtype)
        control = batch["control"].to(device=self.device, dtype=self.dtype)
        prompts = batch["prompt"]

        future_len = future_video.size(1)
        refine_steps = int(self.cfg.validation.refine_steps if not train else 1)
        out = self.model.predict_future(
            history_video,
            prompts,
            future_frames=future_len,
            control=control,
            refine_steps=refine_steps,
        )
        pred_future = out["future_video"]
        future_loss = torch.nn.functional.mse_loss(pred_future, future_video)
        rollout_loss = torch.nn.functional.mse_loss(out["video"][:, -future_len:], future_video)
        temporal = temporal_consistency_loss(out["video"])
        total = future_loss + 0.5 * rollout_loss + 0.05 * temporal
        logs = {
            "future_len": future_len,
            "future_loss": float(future_loss.item()),
            "rollout_loss": float(rollout_loss.item()),
            "temporal": float(temporal.item()),
        }
        return total, logs, out
