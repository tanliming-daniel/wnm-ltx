# World Narrative LTX2

这是一个基于 LTX-2 的 long-horizon world narrative 训练/推理仓库骨架。
目标是把仿真里的长时序预演视频，转成可持续滚动、可逐段控制的视频生成流程。

## 你要的主线

1. 双向预训练
2. 历史帧和未来帧推演预训练
3. 自回归训练
4. few-step DMD 蒸馏

这个仓库按“一个 launcher + 多个 stage config”的方式组织，和 AlayaWorld 的用法接近，但这里是我们自己的框架版。

## 核心想法

- 输入不是单张图，而是仿真产生的长视频、轨迹、prompt 和控制信号
- 模型不是一次性吐完整视频，而是按 chunk 滚动生成
- 每个 chunk 结束时更新 narrative state，再继续往后推
- 基座用 LTX-2，训练时优先做 LoRA / memory / rollout 适配，最后再蒸馏到 few-step student

## 仓库结构

```text
world_narrative/
  config/        配置加载和 schema
  data/          manifest 解析、窗口切片、长视频样本定义
  models/        LTX-2 适配层 + narrative state
  trainers/      bidir / memory pretrain / autoregressive / dmd
  inference/     长视频分块滚动推理
configs/         各阶段 yaml
scripts/         统一训练/推理入口
docs/            架构说明
```

## 启动方式

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

## 数据格式

推荐用 `jsonl` 做 manifest，每行一个长视频样本：

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

训练时建议把样本切成：

- history window
- target future window
- optional control window
- optional prompt swap boundary

## 备注

这个版本先把工程骨架和阶段划分定下来，真正的 LTX-2 微调逻辑放在 `models/` 和 `trainers/` 里后续补齐。
