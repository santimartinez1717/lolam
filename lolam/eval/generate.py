"""Generación con vLLM, reanudable.

Cada muestra se escribe en JSONL en cuanto termina su bloque. Si el job de Slurm muere
(límite de 8 h), al relanzar se saltan las (id, muestra) ya hechas.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from lolam.eval.data import build_messages


@dataclass
class GenConfig:
    model: str
    thinking: bool = True
    n: int = 4
    max_tokens: int = 32768
    # Ajustes del paper de WSP (y recomendados por Qwen para thinking).
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20
    presence_penalty: float = 1.5
    seed: int = 0
    max_model_len: int = 34816
    gpu_memory_utilization: float = 0.85
    chunk_size: int = 64  # prompts por bloque antes de escribir a disco


def sample_seed(base: int, item_id: str, sample: int) -> int:
    # hash() de Python cambia entre procesos; sha256 no, así que reanudar es determinista.
    h = hashlib.sha256(f"{base}|{item_id}|{sample}".encode()).digest()
    return int.from_bytes(h[:4], "little") & 0x7FFFFFFF


def done_keys(path: Path) -> set[tuple[str, int]]:
    if not path.exists():
        return set()
    keys = set()
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue  # línea a medias de un job matado: se ignora
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            keys.add((r["id"], r["sample"]))
    return keys


def build_llm(cfg: GenConfig):
    from vllm import LLM

    return LLM(
        model=cfg.model,
        dtype="bfloat16",
        max_model_len=cfg.max_model_len,
        gpu_memory_utilization=cfg.gpu_memory_utilization,
        seed=cfg.seed,
        enable_prefix_caching=True,
        # Solo texto: no reservar memoria para el encoder de visión.
        limit_mm_per_prompt={"image": 0, "video": 0},
    )


def render_prompt(tokenizer, item: dict, thinking: bool) -> str:
    return tokenizer.apply_chat_template(
        build_messages(item),
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=thinking,
    )


def generate(items: list[dict], cfg: GenConfig, out_path: Path) -> None:
    from vllm import SamplingParams

    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = done_keys(out_path)
    # Cada (problema, muestra) es una petición con su propia semilla: así reanudar no
    # cambia las muestras ya hechas ni las que faltan.
    todo = [(it, s) for it in items for s in range(cfg.n) if (it["id"], s) not in done]
    print(f"[generate] {len(done)} hechas, {len(todo)} pendientes -> {out_path}", flush=True)
    if not todo:
        return

    llm = build_llm(cfg)
    tok = llm.get_tokenizer()

    with out_path.open("a") as f:
        for start in range(0, len(todo), cfg.chunk_size):
            chunk = todo[start : start + cfg.chunk_size]
            prompts = [render_prompt(tok, it, cfg.thinking) for it, _ in chunk]
            params = [
                SamplingParams(
                    n=1,
                    max_tokens=cfg.max_tokens,
                    temperature=cfg.temperature,
                    top_p=cfg.top_p,
                    top_k=cfg.top_k,
                    presence_penalty=cfg.presence_penalty,
                    seed=sample_seed(cfg.seed, it["id"], s),
                )
                for it, s in chunk
            ]
            t0 = time.time()
            outs = llm.generate(prompts, params, use_tqdm=False)
            dt = time.time() - t0
            ntok = 0
            for (it, s), o in zip(chunk, outs):
                c = o.outputs[0]
                ntok += len(c.token_ids)
                rec = {
                    "id": it["id"],
                    "sample": s,
                    "gold": it["gold"],
                    "output": c.text,
                    "n_tokens": len(c.token_ids),
                    "finish_reason": c.finish_reason,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            print(
                f"[generate] {start + len(chunk)}/{len(todo)}  {ntok / dt:,.0f} tok/s",
                flush=True,
            )


def write_config(cfg: GenConfig, extra: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({**asdict(cfg), **extra}, indent=2))
