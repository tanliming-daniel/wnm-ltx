from __future__ import annotations

import argparse

from world_narrative.config.loader import load_config
from world_narrative.trainers import build_trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="World Narrative LTX2 training")
    parser.add_argument("--config", required=True)
    parser.add_argument("--describe", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    trainer = build_trainer(cfg)
    if args.describe:
        trainer.describe()
        return
    if args.validate_only:
        trainer.setup()
        trainer.validate(0)
        return
    trainer.train()


if __name__ == "__main__":
    main()
