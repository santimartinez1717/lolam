"""Estimador insesgado de pass@k (Chen et al., 2021)."""

from __future__ import annotations

import numpy as np


def pass_at_k(n: int, c: int, k: int) -> float:
    """Probabilidad de que al menos una de k muestras sea correcta.

    n: muestras generadas para el problema, c: correctas, k <= n.
    """
    if k > n:
        raise ValueError(f"k={k} > n={n}")
    if n - c < k:
        return 1.0
    # 1 - C(n-c, k) / C(n, k), en forma de producto para evitar desbordes.
    return float(1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))


def mean_pass_at_k(counts: list[tuple[int, int]], k: int) -> float:
    """Media de pass@k sobre problemas; counts = [(n, c), ...]."""
    vals = [pass_at_k(n, c, k) for n, c in counts if n >= k]
    if not vals:
        raise ValueError(f"ningún problema tiene n >= {k}")
    return float(np.mean(vals))


def valid_ks(n: int) -> list[int]:
    """Potencias de dos hasta n (más n si no es potencia de dos)."""
    ks = [2**i for i in range(n.bit_length()) if 2**i <= n]
    if ks[-1] != n:
        ks.append(n)
    return ks
