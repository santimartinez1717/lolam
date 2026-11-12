"""Extracción y comparación de respuestas.

Reglas:
- Si la salida tiene bloque de thinking, solo se mira lo que va después de `</think>`.
  Si el thinking no se cerró (salida truncada), no hay respuesta: cuenta como fallo.
- MATH: último `\\boxed{...}`, comparado con math-verify (equivalencia simbólica),
  con respaldo de comparación normalizada de cadenas.
- GSM8K: último número de `\\boxed{}` o, si no hay, último número del texto.
"""

from __future__ import annotations

import re

THINK_END = "</think>"


def strip_thinking(text: str) -> str | None:
    """Devuelve la parte final de la respuesta, o None si el thinking no se cerró."""
    if THINK_END in text:
        return text.rsplit(THINK_END, 1)[1]
    if "<think>" in text:
        return None
    return text


def last_boxed(text: str) -> str | None:
    """Contenido del último \\boxed{...} (o \\fbox{...}), respetando llaves anidadas."""
    idx = max(text.rfind("\\boxed"), text.rfind("\\fbox"))
    if idx < 0:
        return None
    i = text.find("{", idx)
    if i < 0:
        return None
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1 : j].strip()
    return None


def _normalize(s: str) -> str:
    s = s.strip().strip("$").strip()
    s = re.sub(r"\\text\{(.*?)\}", r"\1", s)
    s = re.sub(r"\\(left|right|!|,|;|:|\s)", "", s)
    s = s.replace("dfrac", "frac").replace("tfrac", "frac")
    s = s.replace("^{\\circ}", "").replace("^\\circ", "")
    s = re.sub(r"\s+", "", s)
    s = s.rstrip(".")
    return s


def math_equal(pred: str, gold: str) -> bool:
    if _normalize(pred) == _normalize(gold):
        return True
    try:
        from math_verify import parse, verify

        return bool(verify(parse(f"${gold}$"), parse(f"${pred}$")))
    except Exception:
        return False


_NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def _to_number(s: str) -> float | None:
    m = _NUM.findall(s)
    if not m:
        return None
    try:
        return float(m[-1].replace(",", ""))
    except ValueError:
        return None


def grade(task: str, output: str, gold: str) -> tuple[bool, str | None]:
    """Devuelve (correcta, respuesta_extraída)."""
    answer_part = strip_thinking(output)
    if answer_part is None:
        return False, None
    if task == "math500":
        pred = last_boxed(answer_part)
        return (pred is not None and math_equal(pred, gold)), pred
    if task == "gsm8k":
        boxed = last_boxed(answer_part)
        pred = boxed if boxed is not None else answer_part
        x, y = _to_number(pred), _to_number(gold)
        ok = x is not None and y is not None and abs(x - y) < 1e-6
        return ok, None if x is None else str(x)
    raise ValueError(f"tarea desconocida: {task}")
