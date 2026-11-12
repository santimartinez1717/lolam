"""Corrige un JSONL de generaciones y calcula métricas agregadas."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from lolam.eval.grading import grade
from lolam.eval.passk import mean_pass_at_k, valid_ks


def load_records(path: Path) -> list[dict]:
    recs = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    recs.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return recs


def score(task: str, records: list[dict]) -> dict:
    by_id: dict[str, list[bool]] = defaultdict(list)
    truncated, finished_correct, finished = 0, 0, 0
    lengths = []
    for r in records:
        ok, _ = grade(task, r["output"], r["gold"])
        by_id[r["id"]].append(ok)
        lengths.append(r["n_tokens"])
        if r["finish_reason"] == "length":
            truncated += 1
        else:
            finished += 1
            finished_correct += ok

    n_min = min(len(v) for v in by_id.values())
    counts = [(len(v), sum(v)) for v in by_id.values()]
    # Bootstrap por problema para el intervalo de la precisión.
    per_problem = np.array([np.mean(v) for v in by_id.values()])
    rng = np.random.default_rng(0)
    boots = rng.choice(per_problem, size=(2000, len(per_problem)), replace=True).mean(1)

    return {
        "task": task,
        "n_problems": len(by_id),
        "n_samples": len(records),
        "accuracy": float(per_problem.mean()),
        "accuracy_ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
        "truncation_rate": truncated / len(records),
        # Ojo: está sesgada (los problemas que terminan suelen ser más fáciles). Es descriptiva.
        "accuracy_if_finished": finished_correct / finished if finished else None,
        "mean_tokens": float(np.mean(lengths)),
        "median_tokens": float(np.median(lengths)),
        "pass_at_k": {str(k): mean_pass_at_k(counts, k) for k in valid_ks(n_min)},
    }
