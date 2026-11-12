"""Compara dos checkpoints tensor a tensor y analiza la geometría del cambio.

Lee los safetensors con mmap y sin torch (cabe en 16 GB de RAM y funciona en el Mac).
Para cada tensor que difiere: norma del cambio, norma relativa, nº de entradas cambiadas,
rango numérico y valores singulares. Guarda los vectores singulares principales del cambio
(`directions.npz`): con ellos se construye la "dirección de bucles" que se usa en la Fase 4.

    lolam-wsp-diff Qwen/Qwen3.5-0.8B RikiyaT/Qwen3.5-0.8B-weight-space-programming \
        --out reports/wsp_diff
"""

from __future__ import annotations

import argparse
import json
import mmap
import struct
from pathlib import Path

import numpy as np

_DTYPES = {"F32": np.float32, "F16": np.float16, "BF16": np.uint16, "I64": np.int64, "I32": np.int32,
           "U8": np.uint8, "I8": np.int8, "BOOL": np.bool_}


class SafetensorsFile:
    """Lector mínimo de safetensors (formato: u64 tamaño de cabecera + JSON + datos)."""

    def __init__(self, path: Path):
        self._f = path.open("rb")
        self._mm = mmap.mmap(self._f.fileno(), 0, access=mmap.ACCESS_READ)
        (n,) = struct.unpack("<Q", self._mm[:8])
        self.header = {k: v for k, v in json.loads(self._mm[8 : 8 + n]).items() if k != "__metadata__"}
        self._base = 8 + n

    def raw(self, name: str) -> bytes:
        a, b = self.header[name]["data_offsets"]
        return self._mm[self._base + a : self._base + b]

    def tensor(self, name: str) -> np.ndarray:
        """Siempre devuelve float32 (bf16 se expande a mano)."""
        meta = self.header[name]
        arr = np.frombuffer(self.raw(name), dtype=_DTYPES[meta["dtype"]]).reshape(meta["shape"])
        if meta["dtype"] == "BF16":
            return (arr.astype(np.uint32) << 16).view(np.float32)
        return arr.astype(np.float32)


def open_checkpoint(repo_or_dir: str) -> dict[str, SafetensorsFile]:
    """Devuelve {nombre_tensor: fichero} para todos los shards."""
    path = Path(repo_or_dir)
    if not path.exists():
        from huggingface_hub import snapshot_download

        path = Path(snapshot_download(repo_or_dir, allow_patterns=["*.safetensors", "*.json"]))
    index: dict[str, SafetensorsFile] = {}
    for shard in sorted(path.glob("*.safetensors")):
        sf = SafetensorsFile(shard)
        for name in sf.header:
            index[name] = sf
    if not index:
        raise FileNotFoundError(f"no hay .safetensors en {path}")
    return index


def analyze_delta(w0: np.ndarray, w1: np.ndarray, k: int = 8) -> tuple[dict, dict]:
    d = (w1 - w0).astype(np.float64)
    info = {
        "shape": list(d.shape),
        "n_changed": int(np.count_nonzero(d)),
        "frac_changed": float(np.count_nonzero(d) / d.size),
        "delta_fro": float(np.linalg.norm(d)),
        "rel_delta_fro": float(np.linalg.norm(d) / max(np.linalg.norm(w0), 1e-12)),
    }
    vecs = {}
    if d.ndim == 2:
        # Columnas y filas tocadas: dicen qué neuronas de entrada/salida cambió la edición.
        info["changed_cols"] = np.flatnonzero(np.abs(d).sum(0)).tolist()
        info["changed_rows_count"] = int(np.count_nonzero(np.abs(d).sum(1)))
        u, s, vt = np.linalg.svd(d, full_matrices=False)
        tol = s[0] * max(d.shape) * np.finfo(np.float32).eps if s.size else 0.0
        info["rank"] = int((s > tol).sum())
        info["top_singular_values"] = s[:k].tolist()
        info["energy_top1"] = float(s[0] ** 2 / (s**2).sum()) if s.size else 0.0
        vecs = {"U": u[:, :k].astype(np.float32), "S": s[:k].astype(np.float32), "Vt": vt[:k].astype(np.float32)}
    return info, vecs


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("base")
    p.add_argument("edited")
    p.add_argument("--out", type=Path, default=Path("reports/wsp_diff"))
    p.add_argument("--k", type=int, default=8, help="vectores singulares que se guardan")
    a = p.parse_args(argv)

    base, edit = open_checkpoint(a.base), open_checkpoint(a.edited)
    only_base, only_edit = sorted(set(base) - set(edit)), sorted(set(edit) - set(base))
    report: dict = {"base": a.base, "edited": a.edited, "only_in_base": only_base,
                    "only_in_edited": only_edit, "changed": {}}
    arrays = {}
    for name in sorted(set(base) & set(edit)):
        if base[name].raw(name) == edit[name].raw(name):
            continue
        info, vecs = analyze_delta(base[name].tensor(name), edit[name].tensor(name), a.k)
        report["changed"][name] = info
        for key, v in vecs.items():
            arrays[f"{name}::{key}"] = v
        print(f"{name}: {info['n_changed']:,} entradas, rango {info.get('rank')}, "
              f"|Δ|/|W| = {info['rel_delta_fro']:.2e}", flush=True)

    total = sum(i["n_changed"] for i in report["changed"].values())
    report["total_changed_entries"] = total
    print(f"\n{len(report['changed'])} tensores distintos, {total:,} pesos cambiados")
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "diff.json").write_text(json.dumps(report, indent=2))
    if arrays:
        np.savez(a.out / "directions.npz", **arrays)


if __name__ == "__main__":
    main()
