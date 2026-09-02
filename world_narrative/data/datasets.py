from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


@dataclass
class NarrativeSample:
    clip_id: str
    scene_id: str = ""
    video_path: str = ""
    prompt: str = ""
    camera_path: str | None = None
    control_path: str | None = None
    future_prompt: str | None = None
    split: str = "train"
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "NarrativeSample":
        metadata = dict(raw.get("metadata") or {})
        for key in (
            "clip_id",
            "scene_id",
            "video_path",
            "prompt",
            "camera_path",
            "control_path",
            "future_prompt",
            "split",
        ):
            if key in raw and key not in metadata:
                metadata[key] = raw[key]
        return cls(
            clip_id=str(raw.get("clip_id", "")),
            scene_id=str(raw.get("scene_id", raw.get("clip_id", ""))),
            video_path=str(raw.get("video_path", "")),
            prompt=str(raw.get("prompt", "")),
            camera_path=raw.get("camera_path"),
            control_path=raw.get("control_path"),
            future_prompt=raw.get("future_prompt"),
            split=str(raw.get("split", "train")),
            metadata=metadata,
        )


@dataclass
class WindowSpec:
    history_frames: int
    future_frames: int
    total_frames: int
    chunk_frames: int
    clip_seconds: float
    history_seconds: float
    future_seconds: float


def read_manifest(path: str | Path) -> Iterator[NarrativeSample]:
    path = Path(path)
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield NarrativeSample.from_mapping(json.loads(line))


def build_window_spec(
    *,
    fps: float,
    clip_seconds: float,
    history_seconds: float,
    future_seconds: float,
    chunk_seconds: float,
) -> WindowSpec:
    history_frames = max(0, int(round(history_seconds * fps)))
    future_frames = max(1, int(round(future_seconds * fps)))
    total_frames = max(1, int(round(clip_seconds * fps)))
    chunk_frames = max(1, int(round(chunk_seconds * fps)))
    return WindowSpec(
        history_frames=history_frames,
        future_frames=future_frames,
        total_frames=total_frames,
        chunk_frames=chunk_frames,
        clip_seconds=clip_seconds,
        history_seconds=history_seconds,
        future_seconds=future_seconds,
    )


def _resize_video(video: torch.Tensor, *, height: int, width: int) -> torch.Tensor:
    if video.dim() != 4:
        raise ValueError(f"expected [T,C,H,W], got {tuple(video.shape)}")
    if video.size(1) not in (1, 3, 4):
        raise ValueError(f"expected 1/3/4 channels, got {video.size(1)}")
    if video.size(1) == 4:
        video = video[:, :3]
    if video.size(1) == 1:
        video = video.repeat(1, 3, 1, 1)
    if tuple(video.shape[-2:]) == (height, width):
        return video
    return F.interpolate(video, size=(height, width), mode="bilinear", align_corners=False)


def _synthetic_video(sample: NarrativeSample, *, total_frames: int, height: int, width: int) -> torch.Tensor:
    seed = abs(hash((sample.clip_id, sample.scene_id, sample.video_path))) % (2**32)
    g = torch.Generator(device="cpu").manual_seed(seed)
    base = torch.rand(total_frames, 3, height, width, generator=g)
    time = torch.linspace(0, 1, total_frames).view(total_frames, 1, 1, 1)
    x = torch.linspace(0, 1, width).view(1, 1, 1, width)
    y = torch.linspace(0, 1, height).view(1, 1, height, 1)
    wave = torch.sin(2 * math.pi * (time * 0.8 + x * 0.5 + y * 0.3))
    video = 0.75 * base + 0.25 * wave.abs().expand_as(base)
    return video.clamp(0.0, 1.0)


def _read_video_file(path: Path) -> torch.Tensor | None:
    if not path.exists():
        return None
    try:
        import imageio.v2 as imageio

        reader = imageio.get_reader(str(path))
        frames = [np.asarray(frame) for frame in reader]
        reader.close()
        if not frames:
            return None
        arr = np.stack(frames, axis=0)
        tensor = torch.from_numpy(arr).float() / 255.0
        if tensor.dim() != 4:
            return None
        if tensor.size(-1) in (1, 3, 4):
            tensor = tensor.permute(0, 3, 1, 2).contiguous()
        elif tensor.size(1) in (1, 3, 4):
            tensor = tensor.contiguous()
        else:
            return None
        return tensor
    except Exception:
        return None


def _load_camera_control(sample: NarrativeSample, *, control_dim: int = 16) -> torch.Tensor | None:
    if not sample.camera_path:
        return None
    path = Path(sample.camera_path)
    if not path.exists():
        return None
    try:
        payload = torch.load(path, map_location="cpu")
    except Exception:
        return None
    values: list[float] = []
    if isinstance(payload, dict):
        for key in ("cam_c2w", "c2w", "camera", "pose", "intrinsics"):
            if key in payload:
                tensor = torch.as_tensor(payload[key], dtype=torch.float32).flatten()
                values.extend(tensor.tolist())
    else:
        values.extend(torch.as_tensor(payload, dtype=torch.float32).flatten().tolist())
    if not values:
        return None
    arr = torch.tensor(values[:control_dim], dtype=torch.float32)
    if arr.numel() < control_dim:
        arr = F.pad(arr, (0, control_dim - arr.numel()))
    return arr


def _load_json_control(sample: NarrativeSample, *, control_dim: int = 16) -> torch.Tensor | None:
    if not sample.control_path:
        return None
    path = Path(sample.control_path)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    values: list[float] = []

    def _collect(x: Any) -> None:
        if isinstance(x, dict):
            for value in x.values():
                _collect(value)
        elif isinstance(x, (list, tuple)):
            for value in x:
                _collect(value)
        elif isinstance(x, (int, float)):
            values.append(float(x))

    _collect(payload)
    if not values:
        return None
    arr = torch.tensor(values[:control_dim], dtype=torch.float32)
    if arr.numel() < control_dim:
        arr = F.pad(arr, (0, control_dim - arr.numel()))
    return arr


class NarrativeWindowDataset(Dataset):
    def __init__(
        self,
        cfg,
        *,
        split: str = "train",
        synthetic_ok: bool = True,
        max_samples: int | None = None,
    ) -> None:
        self.cfg = cfg
        self.split = split
        self.synthetic_ok = synthetic_ok
        self.window = build_window_spec(
            fps=float(cfg.data.fps),
            clip_seconds=float(cfg.data.clip_seconds),
            history_seconds=float(cfg.data.history_seconds),
            future_seconds=float(cfg.data.future_seconds),
            chunk_seconds=float(cfg.stage.chunk_seconds),
        )
        manifest_path = Path(cfg.data.manifest)
        samples = [item for item in read_manifest(manifest_path) if item.split == split or not item.split]
        if max_samples is not None:
            samples = samples[:max_samples]
        if not samples and synthetic_ok:
            samples = [
                NarrativeSample(
                    clip_id=f"synthetic_{i:04d}",
                    scene_id=f"synthetic_{i:04d}",
                    video_path="",
                    prompt=f"synthetic sample {i}",
                    split=split,
                )
                for i in range(8)
            ]
        self.samples = samples
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.samples)

    def _load_video(self, sample: NarrativeSample) -> torch.Tensor:
        total_frames = self.window.total_frames
        height = int(self.cfg.data.height)
        width = int(self.cfg.data.width)
        path = Path(sample.video_path)
        video = _read_video_file(path) if path.name else None
        if video is None:
            if not self.synthetic_ok:
                raise FileNotFoundError(f"missing video file for sample={sample.clip_id}: {path}")
            return _synthetic_video(sample, total_frames=total_frames, height=height, width=width)
        video = _resize_video(video, height=height, width=width)
        if video.size(0) >= total_frames:
            seed = abs(hash((sample.clip_id, self.epoch))) % max(1, video.size(0) - total_frames + 1)
            start = seed if video.size(0) > total_frames else 0
            video = video[start : start + total_frames]
        else:
            pad = total_frames - video.size(0)
            tail = video[-1:].repeat(pad, 1, 1, 1)
            video = torch.cat([video, tail], dim=0)
        return video.clamp(0.0, 1.0)

    def _load_control(self, sample: NarrativeSample) -> torch.Tensor:
        control = _load_camera_control(sample, control_dim=int(getattr(self.cfg.data, "control_dim", 16)))
        if control is None:
            control = _load_json_control(sample, control_dim=int(getattr(self.cfg.data, "control_dim", 16)))
        if control is None:
            control = torch.zeros(int(getattr(self.cfg.data, "control_dim", 16)), dtype=torch.float32)
        return control

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index % len(self.samples)]
        video = self._load_video(sample)
        history_frames = min(self.window.history_frames, video.size(0))
        future_frames = min(self.window.future_frames, max(1, video.size(0) - history_frames))
        history_video = video[:history_frames].contiguous()
        future_video = video[history_frames : history_frames + future_frames].contiguous()
        if future_video.size(0) < future_frames:
            tail = video[-1:].repeat(future_frames - future_video.size(0), 1, 1, 1)
            future_video = torch.cat([future_video, tail], dim=0)
        control = self._load_control(sample)
        metadata = {
            **sample.metadata,
            "clip_id": sample.clip_id,
            "scene_id": sample.scene_id,
            "split": sample.split,
            "history_frames": history_frames,
            "future_frames": future_frames,
            "total_frames": int(video.size(0)),
        }
        return {
            "video": video,
            "history_video": history_video,
            "future_video": future_video,
            "prompt": sample.prompt,
            "future_prompt": sample.future_prompt or sample.prompt,
            "control": control,
            "sample": sample,
            "metadata": metadata,
        }


def collate_narrative_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    video = torch.stack([item["video"] for item in batch], dim=0)
    history = torch.stack([item["history_video"] for item in batch], dim=0)
    future = torch.stack([item["future_video"] for item in batch], dim=0)
    control = torch.stack([item["control"] for item in batch], dim=0)
    return {
        "video": video,
        "history_video": history,
        "future_video": future,
        "prompt": [item["prompt"] for item in batch],
        "future_prompt": [item["future_prompt"] for item in batch],
        "control": control,
        "sample": [item["sample"] for item in batch],
        "metadata": [item["metadata"] for item in batch],
    }


def build_train_dataloader(cfg, *, split: str = "train") -> DataLoader:
    dataset = NarrativeWindowDataset(
        cfg,
        split=split,
        synthetic_ok=bool(getattr(cfg.data, "synthetic_ok", True)),
        max_samples=getattr(cfg.data, "max_samples_per_source", None),
    )
    return DataLoader(
        dataset,
        batch_size=int(cfg.train.batch_size),
        shuffle=split == "train",
        num_workers=int(cfg.train.num_workers),
        pin_memory=False,
        drop_last=split == "train",
        collate_fn=collate_narrative_batch,
    )
