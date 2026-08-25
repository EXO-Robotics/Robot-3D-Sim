# EXO Agent v0.1 qualification evidence

`spec.json` freezes the first Agent qualification: the qualified Foundation v1
receipt, `ScriptedBaselineAgent`, the three public contracts,
AGENT-001@1.0.0 scenarios 0–4, the paused simulated-time scheduler, exact
same-runtime repeatability, and strict v2 replay integrity.

Generate the canonical receipt only from a clean implementation commit:

```bash
uv run --frozen exo-bench agent qualify \
  --output benchmark/qualification/agent-v0.1/qualification.json \
  --evidence-dir benchmark/qualification/evidence
```

Require `qualified: true`, validate the representative archive with
`exo-bench replay --require-archive-receipt`, copy that same archive to
`public/replays/AGENT-001-seed-0.exorun` for the Pages viewer, and commit only
the new receipt/replay evidence in the following evidence commit. The receipt
binds the preceding clean implementation commit; it cannot truthfully contain
the hash of the commit that contains itself.
