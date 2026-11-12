"""Mide el rendimiento real de generación en la porción de GPU.

Genera con `ignore_eos` para fijar la longitud, a varios tamaños de lote.
El resultado sirve para recalcular el presupuesto de cómputo del PLAN (§5).

    lolam-throughput --model Qwen/Qwen3.5-0.8B --out reports/throughput.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[8, 32, 128, 256])
    p.add_argument("--gen-tokens", type=int, default=2048)
    p.add_argument("--max-model-len", type=int, default=34816)
    p.add_argument("--gpu-mem", type=float, default=0.85)
    p.add_argument("--out", type=Path, default=Path("reports/throughput.json"))
    a = p.parse_args(argv)

    import torch
    from vllm import LLM, SamplingParams

    llm = LLM(model=a.model, dtype="bfloat16", max_model_len=a.max_model_len,
              gpu_memory_utilization=a.gpu_mem, limit_mm_per_prompt={"image": 0, "video": 0})
    prompt = "Prove that there are infinitely many primes. Think carefully, step by step."
    params = SamplingParams(max_tokens=a.gen_tokens, temperature=0.6, ignore_eos=True)

    llm.generate([prompt] * 4, SamplingParams(max_tokens=64), use_tqdm=False)  # calentamiento
    results = []
    for bs in a.batch_sizes:
        t0 = time.time()
        outs = llm.generate([prompt] * bs, params, use_tqdm=False)
        dt = time.time() - t0
        ntok = sum(len(o.outputs[0].token_ids) for o in outs)
        r = {"batch_size": bs, "gen_tokens": a.gen_tokens, "seconds": dt, "tok_per_s": ntok / dt}
        results.append(r)
        print(f"lote {bs:4d}: {r['tok_per_s']:,.0f} tok/s ({dt:.1f} s)", flush=True)

    info = {
        "model": a.model,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_mem_total_gb": torch.cuda.get_device_properties(0).total_memory / 2**30,
        "results": results,
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
