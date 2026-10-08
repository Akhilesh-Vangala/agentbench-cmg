"""Analyze ablation runs produced by cmg-deep-claude-agent's ``cmg-eval-agent``.

Reads one or more eval report JSONs, merges rows by config, and writes:
  - results/summary.json   per-config metrics with bootstrap 95% CIs
  - results/paired.json    paired per-task differences between configs
  - reports/REPORT.md      tables, failure taxonomy, cost/quality tradeoffs
  - reports/*.png          charts
"""

from __future__ import annotations

import json
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]

CONFIG_LABELS = {
    "BASELINE": "Fixed pipeline (no LLM)",
    "A_closed_book": "A. Claude, no tools",
    "B_tools_only": "B. Claude + MCP tools",
    "C_tools_skills": "C. Claude + MCP tools + skills",
    "D_tools_skills_sonnet": "D. C on Sonnet",
    "E_open_weight": "E. Qwen 2.5 7B + same tools and skills",
}
ORDER = list(CONFIG_LABELS)


def _label(cfg: str) -> str:
    return CONFIG_LABELS.get(cfg, cfg)


def load_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows = []
    for p in paths:
        data = json.loads(p.read_text())
        for r in data["rows"]:
            if r["config"].startswith("E_open_weight"):
                r["config"] = "E_open_weight"
            rows.append(r)
    return rows


def boot(values: list[float], n: int = 4000, seed: int = 11) -> tuple[float, float, float]:
    rng = random.Random(seed)
    means = sorted(statistics.fmean(rng.choices(values, k=len(values))) for _ in range(n))
    return statistics.fmean(values), means[int(0.025 * n)], means[int(0.975 * n) - 1]


def per_task_pass(rows: list[dict[str, Any]]) -> dict[str, float]:
    by: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        by[r["task_id"]].append(1.0 if r["passed"] else 0.0)
    return {k: statistics.fmean(v) for k, v in by.items()}


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out = {}
    configs = sorted({r["config"] for r in rows}, key=lambda c: ORDER.index(c) if c in ORDER else 99)
    for cfg in configs:
        rs = [r for r in rows if r["config"] == cfg]
        tp = per_task_pass(rs)
        mean, lo, hi = boot(list(tp.values()))
        prec = [r["claim_precision"] for r in rs if r.get("claim_precision") is not None]
        cost = [r["cost_usd_list_price"] for r in rs if r.get("cost_usd_list_price") is not None]
        tools = [r["tool_calls"] for r in rs if isinstance(r.get("tool_calls"), int)]
        reps = Counter(r["task_id"] for r in rs)
        consistency = [1.0 if v in (0.0, 1.0) else 0.0 for k, v in tp.items() if reps[k] > 1]
        cats: dict[str, list[float]] = defaultdict(list)
        for r in rs:
            cats[r.get("category") or "?"].append(1.0 if r["passed"] else 0.0)
        out[cfg] = {
            "label": _label(cfg),
            "n_runs": len(rs),
            "n_tasks": len(tp),
            "task_pass_rate": round(mean, 3),
            "ci95": [round(lo, 3), round(hi, 3)],
            "claim_precision": round(statistics.fmean(prec), 3) if prec else None,
            "runs_with_invented_nct": sum(1 for r in rs if r.get("hallucinated_ncts")),
            "repeat_consistency": round(statistics.fmean(consistency), 3) if consistency else None,
            "latency_s_median": round(statistics.median(r["latency_s"] for r in rs), 1),
            "cost_usd_per_task": round(statistics.fmean(cost), 4) if cost else None,
            "mcp_tool_calls_mean": round(statistics.fmean(tools), 1) if tools else None,
            "by_category": {k: round(statistics.fmean(v), 2) for k, v in sorted(cats.items())},
            "failed_checks": dict(Counter(k for r in rs for k, ok in r["checks"].items() if not ok).most_common()),
            "claim_failure_reasons": dict(Counter(x for r in rs for x in r.get("failed_claim_reasons") or []).most_common()),
            "errors": sum(1 for r in rs if r.get("error")),
        }
    return out


def paired(rows: list[dict[str, Any]], a: str, b: str) -> dict[str, Any] | None:
    pa = per_task_pass([r for r in rows if r["config"] == a])
    pb = per_task_pass([r for r in rows if r["config"] == b])
    common = sorted(set(pa) & set(pb))
    if not common:
        return None
    diffs = [pb[t] - pa[t] for t in common]
    mean, lo, hi = boot(diffs)
    return {
        "from": a, "to": b, "n_tasks": len(common),
        "pass_rate_diff": round(mean, 3), "ci95": [round(lo, 3), round(hi, 3)],
        "tasks_improved": [t for t in common if pb[t] > pa[t]],
        "tasks_regressed": [t for t in common if pb[t] < pa[t]],
    }


def charts(summary: dict[str, Any], out_dir: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(summary)
    labels = [summary[c]["label"] for c in names]
    rates = [summary[c]["task_pass_rate"] for c in names]
    err = [[r - summary[c]["ci95"][0] for c, r in zip(names, rates)], [summary[c]["ci95"][1] - r for c, r in zip(names, rates)]]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.barh(labels, rates, xerr=err, color="#4C72B0", capsize=3)
    ax.set_xlim(0, 1)
    ax.invert_yaxis()
    ax.set_xlabel("Golden-task pass rate (bootstrap 95% CI over tasks)")
    for i, r in enumerate(rates):
        ax.text(0.015, i, f"{r:.0%}", va="center", fontsize=9, color="white", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "pass_rate.png", dpi=160)
    plt.close(fig)

    pts = [(summary[c]["cost_usd_per_task"], summary[c]["task_pass_rate"], summary[c]["label"]) for c in names
           if summary[c]["cost_usd_per_task"] is not None]
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for x, y, lab in pts:
        ax.scatter(x, y, s=40, color="#DD8452")
        ax.annotate(lab, (x, y), textcoords="offset points", xytext=(5, 4), fontsize=8)
    ax.set_xlabel("API list-price cost per task (USD)")
    ax.set_ylabel("Pass rate")
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(out_dir / "cost_vs_quality.png", dpi=160)
    plt.close(fig)
    return ["pass_rate.png", "cost_vs_quality.png"]


def main(paths: list[str]) -> None:
    rows = load_rows([Path(p) for p in paths])
    summary = summarize(rows)
    pairs = [p for p in (
        paired(rows, "BASELINE", "C_tools_skills"),
        paired(rows, "A_closed_book", "B_tools_only"),
        paired(rows, "B_tools_only", "C_tools_skills"),
        paired(rows, "C_tools_skills", "D_tools_skills_sonnet"),
        paired(rows, "E_open_weight", "C_tools_skills"),
    ) if p]
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "results/summary.json").write_text(json.dumps(summary, indent=2))
    (ROOT / "results/paired.json").write_text(json.dumps(pairs, indent=2))
    charts(summary, ROOT / "reports")
    print(json.dumps({"summary": {k: {kk: v[kk] for kk in ("task_pass_rate", "ci95", "claim_precision", "cost_usd_per_task", "latency_s_median")} for k, v in summary.items()}, "paired": pairs}, indent=2))


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
