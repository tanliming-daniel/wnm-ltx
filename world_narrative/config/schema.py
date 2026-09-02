from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, Mapping


def _merge_dataclass(instance: Any, values: Mapping[str, Any]) -> Any:
    for item in fields(instance):
        if item.name not in values:
            continue
        value = values[item.name]
        current = getattr(instance, item.name)
        if is_dataclass(current) and isinstance(value, Mapping):
            _merge_dataclass(current, value)
        else:
            setattr(instance, item.name, value)
    return instance


@dataclass
class RunConfig:
    name: str = "world_narrative"
    seed: int = 42
    output_dir: str = "./outputs"
    log_dir: str = "./logs"


@dataclass
class PathsConfig:
    base_model: str = ""
    resume_checkpoint: str | None = None
    teacher_checkpoint: str | None = None
    student_checkpoint: str | None = None
    vae: str = ""
    text_encoder: str = ""
    cache_dir: str = "./cache"


@dataclass
class DataConfig:
    manifest: str = ""
    video_root: str = ""
    prompt_root: str = ""
    annotation_root: str = ""
    camera_root: str = ""
    fps: float = 24.0
    height: int = 544
    width: int = 960
    temporal_stride: int = 8
    clip_seconds: float = 20.0
    history_seconds: float = 0.0
    future_seconds: float = 20.0
    use_camera: bool = True
    use_long_horizon: bool = True
    control_dim: int = 16
    synthetic_ok: bool = True


@dataclass
class StageConfig:
    kind: str = "bidir"
    name: str = "bidirectional_pretrain"
    objective: str = "bidirectional pretraining"
    chunk_seconds: float = 1.33
    history_chunks: int = 0
    future_chunks: int = 4
    use_lora: bool = True
    hidden_dim: int = 256
    num_layers: int = 4
    num_heads: int = 8
    lora_rank: int = 32
    lora_alpha: int = 32


@dataclass
class TrainingConfig:
    batch_size: int = 1
    lr: float = 1.0e-5
    epochs: int = 1
    max_steps: int | None = None
    weight_decay: float = 0.001
    grad_clip: float = 1.0
    precision: str = "bf16"
    num_workers: int = 4
    log_every: int = 10
    validate_every: int = 500
    checkpoint_every: int = 2000
    gradient_checkpointing: bool = True
    grad_accum_steps: int = 1
    save_last: bool = True


@dataclass
class DmdConfig:
    enabled: bool = False
    teacher_checkpoint: str | None = None
    student_checkpoint: str | None = None
    student_steps: int = 4
    teacher_steps: int = 8
    sigma_list: list[float] = field(default_factory=lambda: [1.0, 0.75, 0.5, 0.25])
    distill_weight: float = 1.0
    consistency_weight: float = 0.0
    ema_decay: float = 0.99
    use_gan: bool = False


@dataclass
class ValidationConfig:
    enabled: bool = True
    before_train: bool = True
    interval: int = 500
    max_samples: int = 2
    save_preview: bool = True
    refine_steps: int = 1
    temperature: float = 1.0


@dataclass
class InferenceConfig:
    mode: str = "rollout"
    input_prefix: str | None = None
    output_dir: str = "./outputs/infer"
    rounds: int = 45
    chunk_seconds: float = 1.33
    history_chunks: int = 8
    prompt_swap_every: int | None = None
    preserve_memory: bool = True
    ttc: bool = False


@dataclass
class WorldNarrativeConfig:
    run: RunConfig = field(default_factory=RunConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    data: DataConfig = field(default_factory=DataConfig)
    stage: StageConfig = field(default_factory=StageConfig)
    train: TrainingConfig = field(default_factory=TrainingConfig)
    dmd: DmdConfig = field(default_factory=DmdConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "WorldNarrativeConfig":
        if not isinstance(raw, Mapping):
            raise TypeError("config must be a mapping")
        cfg = cls()
        _merge_dataclass(cfg, raw)
        return cfg
