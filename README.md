# AgentBench-CMG

Reproducible **agent reliability** framework for CMG healthcare workflows.

Compares:

| Config | Description |
| --- | --- |
| A | No tools (single-prompt style) |
| B | MCP tools, no specialized skills |
| C | MCP tools + reusable skills |
| D | Skills + controlled short memory |

## Research question

> When do reusable agent skills improve task completion enough to justify additional cost (tokens / latency)?

## Metrics

- Task completion
- Citation accuracy
- Tool reliability
- Latency (median / p95)
- Estimated tokens (cost proxy)
- Escalations / unsupported claims
- Bootstrap 95% CIs on completion

## Quick start

```bash
# from sibling checkout of cmg-deep-claude-agent
cd ../cmg-deep-claude-agent && pip install -e .
cd ../agentbench-cmg && pip install -e .
agentbench-cmg
```

Report: `reports/latest.json`

## Expanding to 50–100 tasks

Start from `benchmarks/tasks/seed_tasks.json`. Add held-out, manually checked items for:

- missing data
- conflicting sources
- misleading tool results
- tool failures

## Fit

Transfers NYU research habits (controlled experiments, calibration mindset, honest baselines) into LLM-agent evaluation.
