"""Carga de benchmarks en un formato común: [{"id", "question", "gold"}]."""

from __future__ import annotations

import random

MATH_INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."


def load_task(task: str, limit: int | None = None, seed: int = 0) -> list[dict]:
    from datasets import load_dataset

    if task == "math500":
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        items = [
            {"id": r["unique_id"], "question": r["problem"], "gold": r["answer"], "level": r["level"]}
            for r in ds
        ]
    elif task == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split="test")
        items = [
            {"id": f"gsm8k/{i}", "question": r["question"], "gold": r["answer"].split("####")[-1].strip()}
            for i, r in enumerate(ds)
        ]
    else:
        raise ValueError(f"tarea desconocida: {task}")

    if limit is not None and limit < len(items):
        # Subconjunto fijo y reproducible: mismo seed => mismos problemas para todos los modelos.
        rng = random.Random(seed)
        items = rng.sample(items, limit)
        items.sort(key=lambda r: r["id"])
    return items


def build_messages(item: dict) -> list[dict]:
    return [{"role": "user", "content": f"{item['question']}\n\n{MATH_INSTRUCTION}"}]
