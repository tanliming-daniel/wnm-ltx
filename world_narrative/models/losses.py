from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_mse_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    if mask is None:
        return F.mse_loss(pred, target)
    if mask.dim() == pred.dim() - 1:
        mask = mask.unsqueeze(-1)
    if mask.dim() == pred.dim() - 2:
        mask = mask.unsqueeze(-1).unsqueeze(-1)
    weight = mask.to(dtype=pred.dtype)
    numerator = ((pred - target) ** 2 * weight).sum()
    denominator = weight.sum().clamp_min(1.0)
    return numerator / denominator


def temporal_consistency_loss(video: torch.Tensor) -> torch.Tensor:
    if video.size(1) < 3:
        return torch.zeros((), device=video.device, dtype=video.dtype)
    first = video[:, 1:] - video[:, :-1]
    second = first[:, 1:] - first[:, :-1]
    return second.abs().mean()


def summary_consistency_loss(student_summary: torch.Tensor, teacher_summary: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(student_summary, teacher_summary)


def distillation_loss(student: torch.Tensor, teacher: torch.Tensor, *, temperature: float = 1.0) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    return F.mse_loss(student / temperature, teacher / temperature)


def clip_level_loss(pred: torch.Tensor, target: torch.Tensor, *, mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    recon = masked_mse_loss(pred, target, mask)
    temporal = temporal_consistency_loss(pred)
    return {
        "recon": recon,
        "temporal": temporal,
        "total": recon + 0.1 * temporal,
    }
