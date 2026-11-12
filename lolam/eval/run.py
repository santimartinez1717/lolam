"""CLI: genera (reanudable) y corrige.

Ejemplo:
    lolam-eval --model Qwen/Qwen3.5-0.8B --task math500 --limit 200 --n 4 \
        --out runs/base-think/math500
    lolam-eval ... --score-only   # solo corrige lo que haya
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lolam.eval.data import load_task
from lolam.eval.score import load_records, score


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--task", required=True, choices=["math500", "gsm8k"])
    p.add_argument("--out", required=True, type=Path, help="directorio de la ejecución")
    p.add_argument("--limit", type=int, default=None, help="subconjunto fijo de N problemas")
    p.add_argument("--subset-seed", type=int, default=0)
    p.add_argument("--n", type=int, default=4, help="muestras por problema")
    p.add_argument("--no-thinking", action="store_true")
    p.add_argument("--max-tokens", type=int, default=32768)
    p.add_argument("--max-model-len", type=int, default=34816)
    p.add_argument("--gpu-mem", type=float, default=0.85)
    p.add_argument("--chunk-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--score-only", action="store_true")
    a = p.parse_args(argv)

    gens = a.out / "generations.jsonl"
    items = load_task(a.task, a.limit, a.subset_seed)

    if not a.score_only:
        from lolam.eval.generate import GenConfig, generate, write_config

        cfg = GenConfig(
            model=a.model,
            thinking=not a.no_thinking,
            n=a.n,
            max_tokens=a.max_tokens,
            seed=a.seed,
            max_model_len=a.max_model_len,
            gpu_memory_utilization=a.gpu_mem,
            chunk_size=a.chunk_size,
        )
        write_config(cfg, {"task": a.task, "limit": a.limit, "subset_seed": a.subset_seed}, a.out / "config.json")
        generate(items, cfg, gens)

    wanted = {it["id"] for it in items}
    records = [r for r in load_records(gens) if r["id"] in wanted]
    if not records:
        raise SystemExit(f"no hay generaciones en {gens}")
    metrics = score(a.task, records)
    (a.out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
