from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


def _byte_tensor_from_prompts(prompts: list[str], *, max_len: int, device: torch.device) -> torch.Tensor:
    if not prompts:
        return torch.zeros(1, max_len, dtype=torch.long, device=device)
    out = torch.full((len(prompts), max_len), 256, dtype=torch.long, device=device)
    for row, prompt in enumerate(prompts):
        raw = prompt.encode("utf-8", errors="ignore")[:max_len]
        if raw:
            out[row, : len(raw)] = torch.tensor(list(raw), dtype=torch.long, device=device)
    return out


class PromptEncoder(nn.Module):
    def __init__(self, hidden_dim: int, *, max_len: int = 128) -> None:
        super().__init__()
        self.max_len = max_len
        self.embed = nn.Embedding(257, hidden_dim)
        self.mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(self, prompts: list[str], device: torch.device) -> torch.Tensor:
        byte_ids = _byte_tensor_from_prompts(prompts, max_len=self.max_len, device=device)
        x = self.embed(byte_ids)
        x = x.mean(dim=1)
        return self.mlp(x)


class ControlEncoder(nn.Module):
    def __init__(self, hidden_dim: int, control_dim: int = 16) -> None:
        super().__init__()
        self.control_dim = control_dim
        self.net = nn.Sequential(
            nn.Linear(control_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, control: torch.Tensor | None, *, batch_size: int, device: torch.device) -> torch.Tensor:
        if control is None:
            control = torch.zeros(batch_size, self.control_dim, device=device)
        if control.dim() > 2:
            control = control.flatten(start_dim=1)
        if control.size(1) < self.control_dim:
            control = F.pad(control, (0, self.control_dim - control.size(1)))
        elif control.size(1) > self.control_dim:
            control = control[:, : self.control_dim]
        return self.net(control.to(device=device, dtype=torch.float32))


class FrameEncoder(nn.Module):
    def __init__(self, hidden_dim: int, *, input_channels: int = 3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, hidden_dim // 2, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv2d(hidden_dim // 2, hidden_dim, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        if video.dim() != 5:
            raise ValueError(f"expected video [B,T,C,H,W], got {tuple(video.shape)}")
        b, t, c, h, w = video.shape
        x = video.reshape(b * t, c, h, w)
        x = self.net(x).flatten(1)
        return x.view(b, t, -1)


class HistoryEncoder(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, history_tokens: torch.Tensor | None) -> torch.Tensor:
        if history_tokens is None or history_tokens.numel() == 0:
            return torch.zeros(1, self.gru.hidden_size, device=self.gru.weight_ih_l0.device)
        _, h = self.gru(history_tokens)
        return self.norm(h[-1])


class NarrativeCore(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        *,
        num_layers: int = 4,
        num_heads: int = 8,
        max_frames: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_frames = max_frames
        self.pos_embed = nn.Parameter(torch.randn(max_frames + 1, hidden_dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, frame_tokens: torch.Tensor, context: torch.Tensor, *, causal: bool) -> tuple[torch.Tensor, torch.Tensor]:
        if frame_tokens.dim() != 3:
            raise ValueError(f"expected frame tokens [B,T,D], got {tuple(frame_tokens.shape)}")
        b, t, d = frame_tokens.shape
        if t > self.max_frames:
            raise ValueError(f"sequence length {t} exceeds max_frames={self.max_frames}")
        pos = self.pos_embed[: t + 1]
        ctx = context.unsqueeze(1) + pos[:1]
        seq = frame_tokens + pos[1 : t + 1]
        seq = torch.cat([ctx, seq], dim=1)
        attn_mask = None
        if causal:
            attn_mask = torch.full((t + 1, t + 1), float("-inf"), device=seq.device)
            attn_mask = torch.triu(attn_mask, diagonal=1)
            attn_mask[0].zero_()
        out = self.encoder(seq, mask=attn_mask)
        out = self.norm(out)
        return out[:, 1:], out[:, 0]


class RefinementBlock(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(self, tokens: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        ctx = context.unsqueeze(1).expand(-1, tokens.size(1), -1)
        delta = self.net(torch.cat([tokens, ctx], dim=-1))
        return tokens + delta


class FrameDecoder(nn.Module):
    def __init__(self, hidden_dim: int, *, output_hw: tuple[int, int], decode_hw: tuple[int, int]) -> None:
        super().__init__()
        self.output_hw = output_hw
        self.decode_hw = decode_hw
        out_h, out_w = decode_hw
        self.proj = nn.Linear(hidden_dim, 3 * out_h * out_w)
        self.post = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(3, 3, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, frame_tokens: torch.Tensor) -> torch.Tensor:
        if frame_tokens.dim() != 3:
            raise ValueError(f"expected frame tokens [B,T,D], got {tuple(frame_tokens.shape)}")
        b, t, d = frame_tokens.shape
        out_h, out_w = self.decode_hw
        x = self.proj(frame_tokens).view(b * t, 3, out_h, out_w)
        x = F.interpolate(x, size=self.output_hw, mode="bilinear", align_corners=False)
        x = self.post(x)
        return x.view(b, t, 3, *self.output_hw)


@dataclass
class NarrativeModuleConfig:
    hidden_dim: int = 256
    num_layers: int = 4
    num_heads: int = 8
    max_frames: int = 512
    control_dim: int = 16
    decode_hw: tuple[int, int] = (32, 32)
    output_hw: tuple[int, int] = (544, 960)


def build_narrative_modules(cfg: NarrativeModuleConfig) -> tuple[PromptEncoder, ControlEncoder, FrameEncoder, HistoryEncoder, NarrativeCore, RefinementBlock, FrameDecoder]:
    prompt_encoder = PromptEncoder(cfg.hidden_dim)
    control_encoder = ControlEncoder(cfg.hidden_dim, control_dim=cfg.control_dim)
    frame_encoder = FrameEncoder(cfg.hidden_dim)
    history_encoder = HistoryEncoder(cfg.hidden_dim)
    core = NarrativeCore(
        cfg.hidden_dim,
        num_layers=cfg.num_layers,
        num_heads=cfg.num_heads,
        max_frames=cfg.max_frames,
    )
    refiner = RefinementBlock(cfg.hidden_dim)
    decoder = FrameDecoder(cfg.hidden_dim, output_hw=cfg.output_hw, decode_hw=cfg.decode_hw)
    return prompt_encoder, control_encoder, frame_encoder, history_encoder, core, refiner, decoder
