from __future__ import annotations

import inspect
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .lora import LoRAForwardManager


def _rank0() -> bool:
    return os.environ.get("RANK", "0") == "0"


def _is_valid_vae_chunk_size(chunk_size: int) -> bool:
    return chunk_size >= 9 and (chunk_size - 1) % 8 == 0


def _align_vae_chunk_size(chunk_size: int) -> int:
    if _is_valid_vae_chunk_size(chunk_size):
        return chunk_size
    if chunk_size < 9:
        return 9
    remainder = (chunk_size - 1) % 8
    return chunk_size + (8 - remainder)


def _to_bcthw(video: torch.Tensor) -> torch.Tensor:
    if video.dim() != 5:
        raise ValueError(f"expected 5D video tensor, got {tuple(video.shape)}")
    if video.shape[1] == 3:
        return video
    if video.shape[2] == 3:
        return video.permute(0, 2, 1, 3, 4).contiguous()
    raise ValueError(f"expected video layout [B,C,T,H,W] or [B,T,C,H,W], got {tuple(video.shape)}")


@dataclass
class LTX2AdapterConfig:
    base_model: str = ""
    vae: str = ""
    text_encoder: str = ""
    use_lora: bool = True
    lora_rank: int = 32
    lora_alpha: int = 32
    precision: str = "bf16"
    chunk_latents: int = 33
    hidden_dim: int = 256
    max_frames: int = 512
    control_dim: int = 16
    use_action_control: bool = True
    lora_targets: list[str] = field(
        default_factory=lambda: [
            "blocks.",
            "caption_projection",
            "patchify_proj",
            "adaln_single",
            "prompt_adaln_single",
            "action_adaln_projection",
        ]
    )


class LTX2Adapter:
    """Best-effort bridge to the real LTX-2 stack."""

    def __init__(
        self,
        cfg: LTX2AdapterConfig,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        self.cfg = cfg
        self.device = device or torch.device("cpu")
        precision = str(cfg.precision).lower()
        self.dtype = dtype or (torch.bfloat16 if precision == "bf16" else torch.float32)

        self.transformer: nn.Module | None = None
        self.vae_encoder = None
        self.vae_decoder: nn.Module | None = None
        self.text_encoder: nn.Module | None = None
        self.encode_text = None
        self.lora_manager: LoRAForwardManager | None = None
        self.prompt_projection: nn.Module | None = None
        self.video_projection: nn.Module | None = None
        self.real_stack_ready = False
        self.load_report: dict[str, Any] = {}

        self._load_real_stack()

    @property
    def has_real_transformer(self) -> bool:
        return self.transformer is not None

    @property
    def has_real_vae(self) -> bool:
        return self.vae_encoder is not None and self.vae_decoder is not None

    @property
    def has_real_text_encoder(self) -> bool:
        return self.text_encoder is not None and self.encode_text is not None

    def describe(self) -> str:
        parts = [
            f"base={self.cfg.base_model or 'none'}",
            f"vae={self.cfg.vae or 'none'}",
            f"text={self.cfg.text_encoder or 'none'}",
            f"lora={'on' if self.lora_manager is not None else 'off'}",
            f"transformer={'ready' if self.has_real_transformer else 'fallback'}",
            f"vae={'ready' if self.has_real_vae else 'fallback'}",
            f"text_encoder={'ready' if self.has_real_text_encoder else 'fallback'}",
        ]
        return "LTX2Adapter(" + ", ".join(parts) + ")"

    def get_trainable_parameters(self) -> list[nn.Parameter]:
        params: list[nn.Parameter] = []
        if self.lora_manager is not None:
            params.extend(self.lora_manager.get_trainable_parameters())
        for module in (self.prompt_projection, self.video_projection):
            if module is None:
                continue
            params.extend([p for p in module.parameters() if p.requires_grad])
        return params

    def state_dict(self) -> dict[str, Any]:
        state: dict[str, Any] = {}
        if self.lora_manager is not None:
            state["lora"] = self.lora_manager.state_dict()
        if self.prompt_projection is not None:
            state["prompt_projection"] = self.prompt_projection.state_dict()
        if self.video_projection is not None:
            state["video_projection"] = self.video_projection.state_dict()
        return state

    def load_state_dict(self, state: dict[str, Any]) -> dict[str, int]:
        report = {"lora": 0, "prompt_projection": 0, "video_projection": 0}
        if not isinstance(state, dict):
            return report
        if self.lora_manager is not None and isinstance(state.get("lora"), dict):
            report["lora"] = self.lora_manager.load_state_dict(state["lora"])
        if self.prompt_projection is not None and isinstance(state.get("prompt_projection"), dict):
            missing, unexpected = self.prompt_projection.load_state_dict(state["prompt_projection"], strict=False)
            report["prompt_projection"] = len(state["prompt_projection"]) - len(missing) - len(unexpected)
        if self.video_projection is not None and isinstance(state.get("video_projection"), dict):
            missing, unexpected = self.video_projection.load_state_dict(state["video_projection"], strict=False)
            report["video_projection"] = len(state["video_projection"]) - len(missing) - len(unexpected)
        return report

    def load_weights(self, model: torch.nn.Module | None = None, *, checkpoint: str | None = None) -> dict[str, int]:
        if model is not None:
            return self._load_checkpoint_into_module(model, checkpoint or self.cfg.base_model)
        self._load_real_stack()
        return {
            "loaded": int(self.real_stack_ready),
            "missing": 0,
            "unexpected": 0,
        }

    def encode_prompt(self, prompts: list[str], *, device: torch.device | None = None) -> torch.Tensor:
        if not self.has_real_text_encoder:
            raise RuntimeError("real text encoder is not available")

        device = device or self.device
        encoded = self.encode_text(self.text_encoder, prompts)
        pooled = []
        for item in encoded:
            if isinstance(item, dict):
                tensor = item["prompt_embeds"]
            else:
                tensor = item
            if tensor.dim() == 3:
                tensor = tensor.squeeze(0)
            pooled.append(tensor.mean(dim=0))
        batch = torch.stack(pooled, dim=0).to(device=device, dtype=self.dtype)
        if self.prompt_projection is not None:
            batch = self.prompt_projection(batch)
        return batch

    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        if not self.has_real_vae:
            raise RuntimeError("real VAE is not available")

        video_bcthw = _to_bcthw(video).to(device=self.device, dtype=self.dtype)
        raw_frames = int(video_bcthw.shape[2])
        if (raw_frames - 1) % 8 != 0:
            target_frames = _align_vae_chunk_size(raw_frames)
            pad_frames = target_frames - raw_frames
            tail = video_bcthw[:, :, -1:].expand(-1, -1, pad_frames, -1, -1)
            video_bcthw = torch.cat([video_bcthw, tail], dim=2)

        with torch.no_grad():
            latents = self.vae_encoder.encode(
                video_bcthw,
                chunk_size=_align_vae_chunk_size(int(self.cfg.chunk_latents)),
                verbose=False,
            )
        pooled = latents.mean(dim=(-1, -2)).transpose(1, 2).contiguous()
        if pooled.shape[1] != raw_frames:
            pooled = F.interpolate(
                pooled.transpose(1, 2),
                size=raw_frames,
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)
        if self.video_projection is not None:
            pooled = self.video_projection(pooled)
        return pooled

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        if not self.has_real_vae:
            return latents
        if latents.dim() != 5:
            return latents
        latents_bcthw = latents
        latent_channels = self._infer_vae_latent_channels()
        if latents.shape[1] != latent_channels:
            if latents.shape[2] == latent_channels:
                latents_bcthw = latents.permute(0, 2, 1, 3, 4).contiguous()
            else:
                return latents
        with torch.no_grad():
            video = self.vae_decoder(latents_bcthw.to(device=self.device, dtype=self.dtype))
        return video

    def rollout_chunk(self, inputs: Any) -> Any:
        if not self.has_real_transformer or not isinstance(inputs, dict):
            return inputs
        latents = inputs.get("latents")
        context = inputs.get("context")
        t = inputs.get("t")
        if t is None:
            t = inputs.get("timesteps")
        seq_len = inputs.get("seq_len")
        if latents is None or context is None or t is None:
            return inputs
        seq_len = int(seq_len or (latents.shape[2] * latents.shape[3] * latents.shape[4]))
        with torch.no_grad():
            output = self.transformer(
                latents.to(device=self.device, dtype=self.dtype),
                t.to(device=self.device, dtype=self.dtype),
                context.to(device=self.device, dtype=self.dtype),
                seq_len=seq_len,
            )
        payload = dict(inputs)
        payload["latents"] = output
        payload["video"] = self.decode_latents(output)
        return payload

    def _load_real_stack(self) -> None:
        self.load_report = {
            "transformer": "missing",
            "vae": "missing",
            "text_encoder": "missing",
            "lora": "off",
        }
        self.transformer = None
        self.vae_encoder = None
        self.vae_decoder = None
        self.text_encoder = None
        self.encode_text = None
        self.lora_manager = None
        self.prompt_projection = None
        self.video_projection = None
        self.real_stack_ready = False

        shared_state = None
        base_path = Path(self.cfg.base_model).expanduser()
        vae_path = Path(self.cfg.vae).expanduser()
        text_path = Path(self.cfg.text_encoder).expanduser()

        if base_path.exists():
            try:
                from safetensors.torch import load_file

                shared_state = load_file(str(base_path), device="cpu")
            except Exception as exc:
                self.load_report["transformer_error"] = str(exc)

        if base_path.exists():
            try:
                self.transformer = self._load_transformer(base_path, shared_state)
                self.load_report["transformer"] = "ready"
            except Exception as exc:
                self.load_report["transformer_error"] = str(exc)

        if self.transformer is not None and self.cfg.use_lora:
            try:
                self.lora_manager = LoRAForwardManager(trainable=True)
                count = self.lora_manager.init_for_training(
                    self.transformer,
                    target_keywords=list(self.cfg.lora_targets),
                    rank=int(self.cfg.lora_rank),
                    alpha=float(self.cfg.lora_alpha),
                    dtype=self.dtype,
                    device=self.device,
                )
                self.lora_manager.register_hooks(self.transformer)
                self.lora_manager.enable()
                self.load_report["lora"] = f"ready:{count}"
            except Exception as exc:
                self.load_report["lora_error"] = str(exc)

        if vae_path.exists():
            try:
                self.vae_encoder, self.vae_decoder = self._load_vae(vae_path, shared_state)
                self.load_report["vae"] = "ready"
            except Exception as exc:
                self.load_report["vae_error"] = str(exc)

        if text_path.exists() and base_path.exists():
            try:
                self.text_encoder, self.encode_text = self._load_text_encoder(base_path, text_path, shared_state)
                self.load_report["text_encoder"] = "ready"
            except Exception as exc:
                self.load_report["text_encoder_error"] = str(exc)

        if self.has_real_text_encoder and self.text_encoder is not None:
            text_dim = int(getattr(self.text_encoder.embeddings_connector, "inner_dim", self.cfg.hidden_dim))
            self.prompt_projection = nn.Sequential(
                nn.LayerNorm(text_dim),
                nn.Linear(text_dim, self.cfg.hidden_dim, bias=True),
            ).to(device=self.device, dtype=self.dtype)
            if not self.cfg.use_lora:
                for param in self.prompt_projection.parameters():
                    param.requires_grad_(False)
            self.load_report["prompt_projection"] = f"{text_dim}->{self.cfg.hidden_dim}"

        if self.has_real_vae and self.vae_encoder is not None:
            latent_channels = self._infer_vae_latent_channels()
            self.video_projection = nn.Sequential(
                nn.LayerNorm(latent_channels),
                nn.Linear(latent_channels, self.cfg.hidden_dim, bias=True),
            ).to(device=self.device, dtype=self.dtype)
            if not self.cfg.use_lora:
                for param in self.video_projection.parameters():
                    param.requires_grad_(False)
            self.load_report["video_projection"] = f"{latent_channels}->{self.cfg.hidden_dim}"

        if self.has_real_transformer or self.has_real_vae or self.has_real_text_encoder:
            self.real_stack_ready = True

    def _runtime_transformer_overrides(self) -> dict[str, Any]:
        from ltx2.modules.attention import AttentionFunction
        from ltx2.modules.rope import LTXRopeType

        return {
            "attention_type": AttentionFunction.FLASH_ATTENTION_3,
            "rope_type": LTXRopeType.SPLIT,
            "normalize_time_by_fps": True,
            "normalize_rope_positions": True,
            "positional_embedding_max_pos": [20, 2048, 2048],
            "apply_gated_attention": True,
            "cross_attention_adaln": True,
            "caption_proj_before_connector": True,
            "compact_spatial_tokens": False,
            "enable_action_control": bool(self.cfg.use_action_control),
        }

    def _read_transformer_config(self, checkpoint_path: Path) -> dict[str, Any]:
        if checkpoint_path.suffix != ".safetensors" or not checkpoint_path.exists():
            return {}
        try:
            from safetensors import safe_open

            with safe_open(str(checkpoint_path), framework="pt") as handle:
                metadata = handle.metadata() or {}
            return json.loads(metadata.get("config", "{}")).get("transformer", {})
        except Exception:
            return {}

    def _load_transformer(self, checkpoint_path: Path, state_dict: dict[str, torch.Tensor] | None) -> nn.Module:
        from safetensors.torch import load_file
        from ltx2.modules.model_ltx_2_3 import LTX23Model

        config = self._read_transformer_config(checkpoint_path)
        config.update(self._runtime_transformer_overrides())
        valid = set(inspect.signature(LTX23Model.__init__).parameters.keys())
        filtered = {key: value for key, value in config.items() if key in valid}

        model = LTX23Model(**filtered)
        if state_dict is None:
            state_dict = load_file(str(checkpoint_path), device="cpu")
        model_keys = {name for name, _ in model.named_parameters()}
        model_keys.update(name for name, _ in model.named_buffers())
        converted = self._convert_transformer_state_dict(state_dict, model_keys)
        model.load_state_dict(converted, strict=False)
        model.to(device=self.device, dtype=self.dtype).eval()
        for param in model.parameters():
            param.requires_grad_(False)
        if _rank0():
            print(
                f"[LTX2] transformer loaded from {checkpoint_path} "
                f"(matched={len(converted)}, params={sum(p.numel() for p in model.parameters()) / 1e9:.2f}B)"
            )
        return model

    def _load_vae(
        self,
        checkpoint_path: Path,
        state_dict: dict[str, torch.Tensor] | None,
    ) -> tuple[Any, nn.Module]:
        from safetensors.torch import load_file
        from ltx2.modules.vae import create_video_decoder, create_video_encoder
        from ltx2.utils.ltx2_streaming_vae import StreamingVAEEncoder

        config = self._read_vae_config(checkpoint_path)
        encoder = create_video_encoder(config)
        decoder = create_video_decoder(config)
        if state_dict is None:
            state_dict = load_file(str(checkpoint_path), device="cpu")

        enc_state: dict[str, torch.Tensor] = {}
        dec_state: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            if key.startswith("vae.encoder."):
                enc_state[key.removeprefix("vae.encoder.")] = value
            elif key.startswith("vae.decoder."):
                dec_state[key.removeprefix("vae.decoder.")] = value
            elif key.startswith("vae.per_channel_statistics."):
                short = key.removeprefix("vae.")
                enc_state[short] = value
                dec_state[short] = value
        if enc_state:
            encoder.load_state_dict(enc_state, strict=False)
        if dec_state:
            decoder.load_state_dict(dec_state, strict=False)

        encoder.to(device=self.device, dtype=self.dtype).eval()
        decoder.to(device=self.device, dtype=self.dtype).eval()
        for module in (encoder, decoder):
            for param in module.parameters():
                param.requires_grad_(False)
        if _rank0():
            print(f"[LTX2] VAE loaded from {checkpoint_path}")
        return StreamingVAEEncoder(encoder, device=self.device, dtype=self.dtype), decoder

    def _load_text_encoder(
        self,
        checkpoint_path: Path,
        text_root: Path,
        state_dict: dict[str, torch.Tensor] | None,
    ) -> tuple[nn.Module, Any]:
        from ltx2.modules.text_encoder import (
            AVGemmaTextEncoderModel,
            Embeddings1DConnector,
            GemmaFeaturesExtractorProjLinear,
            LTXVGemmaTokenizer,
        )
        from ltx2.modules.rope import LTXRopeType
        from transformers import Gemma3ForConditionalGeneration

        config = self._read_transformer_config(checkpoint_path)
        caption_proj_before_connector = bool(config.get("caption_proj_before_connector", True))
        if caption_proj_before_connector:
            video_inner_dim = int(config.get("num_attention_heads", 32)) * int(config.get("attention_head_dim", 128))
            feature_extractor = GemmaFeaturesExtractorProjLinear(out_dim=video_inner_dim, bias=True, use_video_key=True)
        else:
            feature_extractor = GemmaFeaturesExtractorProjLinear()

        connector = Embeddings1DConnector(
            attention_head_dim=int(config.get("connector_attention_head_dim", 128)),
            num_attention_heads=int(config.get("connector_num_attention_heads", 32)),
            num_layers=int(config.get("connector_num_layers", 8)),
            positional_embedding_max_pos=config.get("connector_positional_embedding_max_pos", [1]),
            rope_type=LTXRopeType(config.get("rope_type", "split")),
            apply_gated_attention=bool(config.get("connector_apply_gated_attention", True)),
        )

        tokenizer = LTXVGemmaTokenizer(str(text_root))
        gemma = Gemma3ForConditionalGeneration.from_pretrained(
            str(text_root),
            local_files_only=True,
            torch_dtype=self.dtype,
        ).to(self.device).eval()
        text_encoder = AVGemmaTextEncoderModel(
            feature_extractor,
            connector,
            None,
            tokenizer=tokenizer,
            model=gemma,
            dtype=self.dtype,
            use_v2_norm=caption_proj_before_connector,
            gemma_embedding_dim=3840,
        )

        if state_dict is None:
            state_dict = {}

        fe = {
            key.removeprefix("text_embedding_projection."): value
            for key, value in state_dict.items()
            if key.startswith("text_embedding_projection.")
        }
        if fe:
            text_encoder.feature_extractor_linear.load_state_dict(fe, strict=False)

        ec = {
            key.replace("model.diffusion_model.video_embeddings_connector.", ""): value
            for key, value in state_dict.items()
            if "video_embeddings_connector" in key
        }
        if ec:
            text_encoder.embeddings_connector.load_state_dict(ec, strict=False)

        text_encoder.to(self.device).eval()
        for param in text_encoder.parameters():
            param.requires_grad_(False)

        def encode_text(encoder: nn.Module, prompts: list[str]) -> list[torch.Tensor]:
            outputs = []
            with torch.no_grad():
                for prompt in prompts:
                    outputs.append(encoder(prompt).video_encoding)
            return outputs

        if _rank0():
            print(f"[LTX2] text encoder loaded from {text_root}")
        return text_encoder, encode_text

    def _read_vae_config(self, checkpoint_path: Path) -> dict[str, Any]:
        if checkpoint_path.suffix != ".safetensors" or not checkpoint_path.exists():
            return {}
        try:
            from safetensors import safe_open

            with safe_open(str(checkpoint_path), framework="pt") as handle:
                metadata = handle.metadata() or {}
            return json.loads(metadata.get("config", "{}"))
        except Exception:
            return {}

    def _infer_vae_latent_channels(self) -> int:
        encoder = getattr(self.vae_encoder, "encoder", None)
        if encoder is None:
            return self.cfg.hidden_dim
        stats = getattr(encoder, "per_channel_statistics", None)
        if stats is not None and hasattr(stats, "_buffers"):
            buf = stats._buffers.get("mean-of-means")
            if buf is not None:
                return int(buf.shape[0])
        conv_out = getattr(encoder, "conv_out", None)
        if conv_out is not None and hasattr(conv_out, "out_channels"):
            return int(max(1, conv_out.out_channels - 1))
        return self.cfg.hidden_dim

    def _convert_transformer_state_dict(self, state_dict: dict[str, torch.Tensor], model_keys: set[str]) -> dict[str, torch.Tensor]:
        converted: dict[str, torch.Tensor] = {}
        skip_prefixes = (
            "audio_",
            "av_ca_",
            "_a2v_",
            "_v2a_",
            "vae.",
            "vocoder.",
            "text_embedding_projection.",
            "model.diffusion_model.video_embeddings_connector.",
            "model.diffusion_model.audio_",
            "model.diffusion_model.av_ca_",
        )
        for raw_key, value in state_dict.items():
            if any(raw_key.startswith(prefix) for prefix in skip_prefixes):
                continue
            candidates = []
            if raw_key.startswith("model.diffusion_model."):
                candidates.append(
                    raw_key.removeprefix("model.diffusion_model.").replace("transformer_blocks.", "blocks.")
                )
            cleaned = raw_key.replace("_fsdp_wrapped_module.", "").replace("_checkpoint_wrapped_module.", "")
            cleaned = cleaned.replace("transformer_blocks.", "blocks.")
            candidates.extend([cleaned, raw_key])
            for key in candidates:
                if key in model_keys:
                    converted[key] = value
                    break
        return converted

    def _load_checkpoint_into_module(self, model: torch.nn.Module, checkpoint_path: str | None) -> dict[str, int]:
        if not checkpoint_path:
            return {"loaded": 0, "missing": 0, "unexpected": 0}
        path = Path(checkpoint_path).expanduser()
        if not path.exists():
            return {"loaded": 0, "missing": 0, "unexpected": 0}
        try:
            if path.suffix == ".safetensors":
                from safetensors.torch import load_file

                state = load_file(str(path), device="cpu")
            else:
                payload = torch.load(path, map_location="cpu")
                state = payload.get("model_state_dict", payload.get("state_dict", payload)) if isinstance(payload, dict) else payload
            missing, unexpected = model.load_state_dict(state, strict=False)
            return {"loaded": len(state), "missing": len(missing), "unexpected": len(unexpected)}
        except Exception:
            return {"loaded": 0, "missing": 0, "unexpected": 0}
