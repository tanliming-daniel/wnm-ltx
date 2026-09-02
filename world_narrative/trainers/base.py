from __future__ import annotations

import json
import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class StageSummary:
    kind: str
    title: str
    objective: str
    losses: tuple[str, ...] = ()
    notes: str = ""


class BaseTrainer(ABC):
    stage_name: str = "base"

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.output_dir = Path(cfg.run.output_dir)
        self.log_dir = Path(cfg.run.log_dir)
        self.device = None
        self.dtype = None

        self.model = None
        self.teacher_model = None
        self.adapter = None
        self.optimizer = None
        self.scheduler = None
        self.train_loader = None
        self.val_loader = None
        self.global_step = 0
        self.best_val: float | None = None

    @abstractmethod
    def stage_summary(self) -> StageSummary:
        raise NotImplementedError

    @abstractmethod
    def compute_step(self, batch: dict[str, Any], *, train: bool) -> tuple[torch.Tensor, dict[str, Any], dict[str, torch.Tensor]]:
        raise NotImplementedError

    def describe(self) -> None:
        summary = self.stage_summary()
        print(f"[Describe] stage={summary.kind} title={summary.title}")
        print(f"[Describe] objective={summary.objective}")
        if summary.losses:
            print(f"[Describe] losses={', '.join(summary.losses)}")
        if summary.notes:
            print(f"[Describe] notes={summary.notes}")

    def setup(self) -> None:
        import torch

        from world_narrative.data import build_train_dataloader
        from world_narrative.models.loader import build_world_model, describe_model

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        precision = str(getattr(self.cfg.train, "precision", "bf16")).lower()
        self.dtype = torch.bfloat16 if precision == "bf16" and self.device.type == "cuda" else torch.float32

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        checkpoints = self.output_dir / "checkpoints"
        checkpoints.mkdir(parents=True, exist_ok=True)

        self.model, self.adapter = build_world_model(
            self.cfg,
            role="student",
            device=self.device,
            dtype=self.dtype,
            checkpoint=self.cfg.paths.resume_checkpoint,
        )

        if self.stage_name == "dmd":
            teacher_path = self.cfg.dmd.teacher_checkpoint or self.cfg.paths.teacher_checkpoint
            self.teacher_model, _ = build_world_model(
                self.cfg,
                role="teacher",
                device=self.device,
                dtype=self.dtype,
                checkpoint=teacher_path,
            )
            if teacher_path:
                print(f"[Setup] teacher={teacher_path} loaded")
            self.teacher_model.eval()
            for param in self.teacher_model.parameters():
                param.requires_grad_(False)

        self.train_loader = build_train_dataloader(self.cfg, split="train")
        self.val_loader = build_train_dataloader(self.cfg, split="val")

        params = [p for p in self.model.parameters() if p.requires_grad]
        if not params:
            raise RuntimeError("no trainable parameters were found")
        self.optimizer = torch.optim.AdamW(
            params,
            lr=float(self.cfg.train.lr),
            weight_decay=float(self.cfg.train.weight_decay),
        )
        total_steps = self.cfg.train.max_steps
        if total_steps is None:
            total_steps = max(1, len(self.train_loader) * int(self.cfg.train.epochs))
        warmup_steps = max(1, int(min(self.cfg.train.validate_every, total_steps // 10 or 1)))

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return float(step + 1) / float(warmup_steps)
            remaining = max(1, total_steps - warmup_steps)
            progress = min(1.0, float(step - warmup_steps) / float(remaining))
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambda)
        print(f"[Setup] {describe_model(self.model, self.adapter)}")

    def train(self) -> None:
        import torch
        from torch.nn.utils import clip_grad_norm_

        self.setup()
        if self.cfg.validation.enabled and self.cfg.validation.before_train:
            self.validate(self.global_step)

        assert self.train_loader is not None
        assert self.model is not None
        assert self.optimizer is not None
        assert self.scheduler is not None

        grad_accum = max(1, int(getattr(self.cfg.train, "grad_accum_steps", 1)))
        max_steps = self.cfg.train.max_steps
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        for epoch in range(int(self.cfg.train.epochs)):
            dataset = getattr(self.train_loader, "dataset", None)
            if hasattr(dataset, "set_epoch"):
                dataset.set_epoch(epoch)

            for batch_idx, batch in enumerate(self.train_loader):
                step_start = time.time()
                loss, logs, outputs = self.compute_step(batch, train=True)
                loss = loss / grad_accum
                loss.backward()

                if (batch_idx + 1) % grad_accum != 0:
                    continue

                grad_norm = clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    float(self.cfg.train.grad_clip),
                )
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)

                self.global_step += 1
                if self.global_step % int(self.cfg.train.log_every) == 0:
                    lr = self.scheduler.get_last_lr()[0]
                    metrics = self._format_metrics(logs)
                    print(
                        f"[Train] step={self.global_step} epoch={epoch} "
                        f"{metrics} loss={float(loss.item()) * grad_accum:.6f} "
                        f"grad={float(grad_norm):.4f} lr={lr:.2e} "
                        f"time={time.time() - step_start:.2f}s",
                        flush=True,
                    )

                if self.global_step % int(self.cfg.train.checkpoint_every) == 0:
                    self.save_checkpoint(self.global_step, tag="step")

                if self.cfg.validation.enabled and self.global_step % int(self.cfg.validation.interval) == 0:
                    val_metric = self.validate(self.global_step)
                    if self.best_val is None or val_metric < self.best_val:
                        self.best_val = val_metric
                        self.save_checkpoint(self.global_step, tag="best")

                if max_steps is not None and self.global_step >= int(max_steps):
                    if bool(getattr(self.cfg.train, "save_last", True)):
                        self.save_checkpoint(self.global_step, tag="last")
                    return

        if bool(getattr(self.cfg.train, "save_last", True)):
            self.save_checkpoint(self.global_step, tag="last")

    def validate(self, step: int) -> float:
        import torch

        assert self.model is not None
        assert self.val_loader is not None
        self.model.eval()
        if self.teacher_model is not None:
            self.teacher_model.eval()

        losses: list[float] = []
        aggregate: dict[str, list[float]] = {}
        preview_written = False
        with torch.no_grad():
            for idx, batch in enumerate(self.val_loader):
                loss, logs, outputs = self.compute_step(batch, train=False)
                losses.append(float(loss.item()))
                for key, value in logs.items():
                    if isinstance(value, (int, float)):
                        aggregate.setdefault(key, []).append(float(value))
                if not preview_written and bool(getattr(self.cfg.validation, "save_preview", True)):
                    self._save_preview(step, batch, outputs)
                    preview_written = True
                if idx + 1 >= int(self.cfg.validation.max_samples):
                    break

        mean_loss = float(sum(losses) / len(losses)) if losses else 0.0
        summary = {key: float(sum(vals) / len(vals)) for key, vals in aggregate.items()}
        payload = {
            "step": int(step),
            "loss": mean_loss,
            "metrics": summary,
        }
        out_dir = self.output_dir / "validation"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"step-{step:06d}.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"[Validate] step={step} loss={mean_loss:.6f} metrics={summary}")
        self.model.train()
        if self.teacher_model is not None:
            self.teacher_model.eval()
        return mean_loss

    def save_checkpoint(self, step: int, *, tag: str) -> None:
        import torch

        assert self.model is not None
        assert self.optimizer is not None
        assert self.scheduler is not None
        ckpt_dir = self.output_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "step": int(step),
            "tag": tag,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "cfg": self.cfg.run.name,
            "stage": self.stage_name,
        }
        path = ckpt_dir / f"{tag}.pt"
        torch.save(payload, path)
        print(f"[Checkpoint] saved {path}")

    def _save_preview(self, step: int, batch: dict[str, Any], outputs: dict[str, torch.Tensor]) -> None:
        import imageio.v2 as imageio
        import numpy as np
        import torch

        video = batch["video"][0].detach().cpu().clamp(0, 1)
        pred = outputs["video"][0].detach().cpu().clamp(0, 1)
        n = min(6, video.size(0), pred.size(0))
        if n <= 0:
            return
        gt = video[:n]
        pr = pred[:n]
        preview = torch.cat([gt, pr], dim=3)
        arr = (preview.permute(0, 2, 3, 1).numpy() * 255.0).astype(np.uint8)
        out_dir = self.output_dir / "validation" / f"step-{step:06d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        imageio.imwrite(out_dir / "preview.png", arr[0])

    @staticmethod
    def _format_metrics(logs: dict[str, Any]) -> str:
        items = []
        for key, value in logs.items():
            if isinstance(value, float):
                items.append(f"{key}={value:.4f}")
            elif isinstance(value, int):
                items.append(f"{key}={value}")
        return " ".join(items)
