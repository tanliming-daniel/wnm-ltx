from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class RolloutPlan:
    input_prefix: str
    mode: str
    rounds: int
    chunk_seconds: float
    history_chunks: int
    student_steps: int
    ttc: bool
    output_dir: str


class NarrativeRolloutRunner:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        from world_narrative.models.loader import build_world_model

        self.model, self.adapter = build_world_model(cfg, role="student")

    def plan(self) -> RolloutPlan:
        return RolloutPlan(
            input_prefix=str(self.cfg.inference.input_prefix or ""),
            mode=str(self.cfg.inference.mode),
            rounds=int(self.cfg.inference.rounds),
            chunk_seconds=float(self.cfg.inference.chunk_seconds),
            history_chunks=int(self.cfg.inference.history_chunks),
            student_steps=int(self.cfg.dmd.student_steps),
            ttc=bool(self.cfg.inference.ttc),
            output_dir=str(self.cfg.inference.output_dir),
        )

    def describe(self) -> None:
        plan = self.plan()
        print(f"[Infer] mode={plan.mode} input={plan.input_prefix}")
        print(
            f"[Infer] rounds={plan.rounds} chunk_seconds={plan.chunk_seconds} "
            f"history_chunks={plan.history_chunks} student_steps={plan.student_steps} ttc={plan.ttc}"
        )
        print(f"[Infer] adapter={self.adapter.describe()}")

    def run(self) -> dict[str, Any]:
        self.cfg.inference.output_dir = str(self.cfg.inference.output_dir)
        out_dir = Path(self.cfg.inference.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        plan = self.plan()
        self.describe()
        state = self.model.init_state(prompt="")
        payload = {
            "plan": asdict(plan),
            "state": {
                "prompt": state.prompt,
                "summary": state.summary,
                "step_index": state.step_index,
            },
        }
        (out_dir / "rollout_plan.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[Infer] scaffold only: wrote {out_dir / 'rollout_plan.json'}")
        return payload
