from __future__ import annotations


def build_trainer(cfg):
    kind = str(cfg.stage.kind).lower()
    if kind in {"bidir", "bidirectional"}:
        from .bidir import BidirTrainer

        return BidirTrainer(cfg)
    if kind in {"memory_pretrain", "history_pretrain"}:
        from .memory_pretrain import MemoryPretrainTrainer

        return MemoryPretrainTrainer(cfg)
    if kind in {"autoregressive", "ar"}:
        from .autoregressive import AutoregressiveTrainer

        return AutoregressiveTrainer(cfg)
    if kind == "dmd":
        from .dmd import DmdTrainer

        return DmdTrainer(cfg)
    raise ValueError("unknown stage.kind={!r}; expected bidir, memory_pretrain, autoregressive, or dmd".format(cfg.stage.kind))


__all__ = ["build_trainer"]
