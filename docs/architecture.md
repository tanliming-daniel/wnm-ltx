# Architecture

This repo is organized around one idea: keep a persistent narrative state while rolling out long videos in chunks.

## Data flow

`simulation preview video -> clip manifest -> window sampler -> latent encoder -> narrative state -> chunk generator -> state update`

The latent encoder path now resolves to the real LTX-2 VAE/text encoder stack when the checkpoint paths are present; otherwise it falls back to the local scaffold.

## Training stages

### Stage 1: Bidirectional pretrain

- use full clips or long spans
- predict in both temporal directions
- focus on base video prior and global consistency

### Stage 2: History/future reasoning pretrain

- feed masked history
- probe future continuation
- train the memory path and a light adapter

### Stage 3: Autoregressive training

- causal chunk rollout
- teacher forcing over long sequences
- align the model with the inference-time rollout loop

### Stage 4: Few-step DMD

- compress the rollout into a small number of steps
- distill from the autoregressive teacher
- keep long-horizon consistency while lowering latency

## Inference

Inference should expose two modes:

1. long rollout for simulation preview videos
2. few-step student for low-latency generation

Both modes should reuse the same narrative state interface.
