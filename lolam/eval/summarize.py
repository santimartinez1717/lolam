"""Tabla markdown comparando ejecuciones: lolam-summary runs/*/math500 runs/*/gsm8k"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+", type=Path)
    a = p.parse_args(argv)

    rows = ["| ejecución | tarea | problemas × n | precisión (IC 95 %) | truncado | tokens medios |",
            "|---|---|---|---|---|---|"]
    for run in a.runs:
        mp = run / "metrics.json"
        if not mp.exists():
            continue
        m = json.loads(mp.read_text())
        lo, hi = m["accuracy_ci95"]
        n = m["n_samples"] // m["n_problems"]
        rows.append(
            f"| {run.parent.name} | {m['task']} | {m['n_problems']} × {n} | "
            f"{100 * m['accuracy']:.1f} ({100 * lo:.1f}–{100 * hi:.1f}) | "
            f"{100 * m['truncation_rate']:.0f} % | {m['mean_tokens']:,.0f} |"
        )
    print("\n".join(rows))


if __name__ == "__main__":
    main()
