from __future__ import annotations

import argparse

from world_narrative.config.loader import load_config
from world_narrative.inference import NarrativeRolloutRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="World Narrative LTX2 inference")
    parser.add_argument("--config", required=True)
    parser.add_argument("--input", required=True, help="input prefix for the simulation preview case")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--mode", default=None, choices=["rollout", "dmd"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    cfg.inference.input_prefix = args.input
    if args.output_dir is not None:
        cfg.inference.output_dir = args.output_dir
    if args.rounds is not None:
        cfg.inference.rounds = args.rounds
    if args.mode is not None:
        cfg.inference.mode = args.mode
    runner = NarrativeRolloutRunner(cfg)
    runner.run()


if __name__ == "__main__":
    main()
