from __future__ import annotations

import argparse

from agentbench_cmg.analyze import main as analyze


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze cmg-deep-claude-agent ablation runs")
    parser.add_argument("reports", nargs="+", help="eval-*.json files written by cmg-eval-agent")
    args = parser.parse_args()
    analyze(args.reports)


if __name__ == "__main__":
    main()
