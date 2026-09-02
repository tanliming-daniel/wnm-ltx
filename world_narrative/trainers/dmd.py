from __future__ import annotations

from typing import Any

from .base import BaseTrainer, StageSummary


class DmdTrainer(BaseTrainer):
    stage_name = "dmd"

    def stage_summary(self) -> StageSummary:
        return StageSummary(
            kind="dmd",
            title="Few-step DMD distillation",
            objective="Distill the autoregressive teacher into a low-latency student",
            losses=("distillation", "consistency"),
            notes=f"student_steps={self.cfg.dmd.student_steps}, teacher_steps={self.cfg.dmd.teacher_steps}",
        )

    def compute_step(
        self,
        batch: dict[str, Any],
        *,
        train: bool,
    ) -> tuple[torch.Tensor, dict[str, Any], dict[str, torch.Tensor]]:
        import torch

        from world_narrative.models.losses import (
            distillation_loss,
            summary_consistency_loss,
            temporal_consistency_loss,
        )

        assert self.model is not None
        assert self.teacher_model is not None
        history_video = batch["history_video"].to(device=self.device, dtype=self.dtype)
        future_video = batch["future_video"].to(device=self.device, dtype=self.dtype)
        control = batch["control"].to(device=self.device, dtype=self.dtype)
        prompts = batch["prompt"]

        student_steps = int(self.cfg.dmd.student_steps)
        teacher_steps = int(self.cfg.dmd.teacher_steps)
        student_out = self.model.predict_future(
            history_video,
            prompts,
            future_frames=future_video.size(1),
            control=control,
            refine_steps=student_steps if train else int(self.cfg.validation.refine_steps),
        )
        with torch.no_grad():
            teacher_out = self.teacher_model.predict_future(
                history_video,
                prompts,
                future_frames=future_video.size(1),
                control=control,
                refine_steps=teacher_steps,
            )

        student_future = student_out["future_video"]
        teacher_future = teacher_out["future_video"]
        distill = distillation_loss(student_future, teacher_future, temperature=float(getattr(self.cfg.validation, "temperature", 1.0)))
        consistency = summary_consistency_loss(student_out["context"], teacher_out["context"])
        temporal = temporal_consistency_loss(student_out["video"])
        supervised = torch.nn.functional.mse_loss(student_future, future_video)
        total = (
            float(self.cfg.dmd.distill_weight) * distill
            + float(self.cfg.dmd.consistency_weight) * consistency
            + 0.25 * supervised
            + 0.05 * temporal
        )
        logs = {
            "student_steps": student_steps,
            "teacher_steps": teacher_steps,
            "distill": float(distill.item()),
            "consistency": float(consistency.item()),
            "supervised": float(supervised.item()),
            "temporal": float(temporal.item()),
        }
        return total, logs, student_out
