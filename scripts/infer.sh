#!/usr/bin/env bash
set -euo pipefail

: "${CONFIG_PATH:?CONFIG_PATH=configs/infer_rollout.yaml is required}"
: "${INPUT_PREFIX:?INPUT_PREFIX is required}"
cd "$(dirname "$0")/.."

python -m world_narrative.infer --config "$CONFIG_PATH" --input "$INPUT_PREFIX" "$@"
