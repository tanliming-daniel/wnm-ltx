# World Narrative LTX2

`world_narrative_ltx2` 是一个面向长时序仿真视频的训练与推理仓库。项目以 LTX-2 作为底座，提供双向预训练、历史-未来推演预训练、自回归训练和 few-step DMD 蒸馏四个阶段，以及统一的长视频滚动推理入口。

## 目录

- `world_narrative/` 训练、推理、模型与数据代码
- `ltx2/` 本地 vendored 的 LTX-2 运行时代码
- `fastvideo/` 兼容层
- `configs/` 各阶段配置文件
- `scripts/` 训练与推理启动脚本
- `docs/architecture.md` 架构说明

## 组件

- Transformer: 真实 LTX-2 transformer
- VAE: 真实 video VAE encoder / decoder
- Text Encoder: Gemma 系列文本编码器
- LoRA: 基于 forward hook 的适配器
- Fallback: 当外部权重未配置时，保留本地 scaffold 路径

## 训练阶段

1. `stage1_bidir`：双向预训练
2. `stage2_memory_pretrain`：历史帧与未来帧推演预训练
3. `stage3_autoregressive`：自回归训练
4. `stage4_dmd`：few-step DMD 蒸馏

## 配置说明

常用路径字段：

- `paths.base_model`
- `paths.vae`
- `paths.text_encoder`
- `paths.resume_checkpoint`
- `paths.teacher_checkpoint`

常用训练字段：

- `stage.kind`
- `stage.use_lora`
- `stage.chunk_seconds`
- `data.manifest`
- `data.video_root`
- `data.prompt_root`
- `data.camera_root`

## 数据格式

推荐使用 `jsonl` manifest，每行一个样本：

```json
{
  "clip_id": "scene_001",
  "scene_id": "scene_001",
  "video_path": "data/videos/scene_001.mp4",
  "prompt": "a long simulation preview of a street scene",
  "camera_path": "data/cameras/scene_001.pt",
  "control_path": "data/control/scene_001.json",
  "split": "train"
}
```

## 运行

安装依赖：

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

训练：

```bash
CONFIG_PATH=configs/stage1_bidir.yaml bash scripts/train.sh
CONFIG_PATH=configs/stage2_memory_pretrain.yaml bash scripts/train.sh
CONFIG_PATH=configs/stage3_autoregressive.yaml bash scripts/train.sh
CONFIG_PATH=configs/stage4_dmd.yaml bash scripts/train.sh
```

推理：

```bash
CONFIG_PATH=configs/infer_rollout.yaml INPUT_PREFIX=playground/case1 bash scripts/infer.sh
```

## 说明

- `README` 侧重仓库入口和使用方式
- `docs/architecture.md` 侧重模块关系和训练阶段说明
- 当 `paths.base_model`、`paths.vae`、`paths.text_encoder` 可用时，仓库会优先加载真实 LTX-2 组件
