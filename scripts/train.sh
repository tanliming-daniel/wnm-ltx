#!/usr/bin/env bash
set -euo pipefail

: "${CONFIG_PATH:?CONFIG_PATH=configs/stage1_bidir.yaml is required}"
cd "$(dirname "$0")/.."

ARGS=()
if [ "${VALIDATE_ONLY:-0}" = "1" ]; then
  ARGS+=(--validate-only)
fi
if [ "${DESCRIBE:-0}" = "1" ]; then
  ARGS+=(--describe)
fi

python -m world_narrative.train --config "$CONFIG_PATH" "${ARGS[@]}" "$@"
