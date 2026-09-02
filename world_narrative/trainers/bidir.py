from __future__ import annotations

import random
from typing import Any

from .base import BaseTrainer, StageSummary


class BidirTrainer(BaseTrainer):
    stage_name = "bidir"

    def stage_summary(self) -> StageSummary:
        return StageSummary(
            kind="bidir",
            title="Bidirectional pretrain",
            objective="Learn a base long-horizon prior from full clips",
            losses=("reconstruction", "temporal_consistency"),
            notes="Supports i2v / v2v / t2v style masking over the whole clip.",
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
        control = batch["control"].to(device=self.device, dtype=self.dtype)
        prompts = batch["prompt"]
        b, t, _, _, _ = video.shape

        if train:
            r = random.random()
            if r < 0.55:
                cond_frames = 1
                mode = "i2v"
            elif r < 0.85:
                ratio = random.uniform(0.2, 0.6)
                cond_frames = max(1, int(round(t * ratio)))
                mode = "v2v"
            else:
                cond_frames = 0
                mode = "t2v"
        else:
            cond_frames = max(1, min(t // 8 or 1, self.model.chunk_frames))
            mode = "eval"

        frame_mask = torch.zeros(b, t, dtype=torch.bool, device=self.device)
        if cond_frames > 0:
            frame_mask[:, :cond_frames] = True

        refine_steps = int(self.cfg.validation.refine_steps if not train else 1)
        out = self.model(
            video,
            prompts,
            control=control,
            frame_mask=frame_mask,
            history_frames=cond_frames if cond_frames > 0 else None,
            causal=False,
            refine_steps=refine_steps,
        )
        pred = out["video"]
        target_mask = ~frame_mask
        recon = masked_mse_loss(pred, video, target_mask)
        temporal = temporal_consistency_loss(pred)
        total = recon + 0.05 * temporal
        logs = {
            "mode": mode,
            "cond_frames": cond_frames,
            "recon": float(recon.item()),
            "temporal": float(temporal.item()),
        }
        return total, logs, out
