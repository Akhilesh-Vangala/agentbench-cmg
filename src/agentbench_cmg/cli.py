from __future__ import annotations

import argparse
import json
from pathlib import Path

from agentbench_cmg.harness import run_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AgentBench-CMG ablations")
    parser.add_argument(
        "--tasks",
        default=str(Path(__file__).resolve().parents[2] / "benchmarks" / "tasks" / "seed_tasks.json"),
    )
    parser.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parents[2] / "reports" / "latest.json"),
    )
    args = parser.parse_args()
    report = run_benchmark(Path(args.tasks))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    print(json.dumps(report.summary, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
