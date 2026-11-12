import json

import numpy as np
import pytest

from lolam.eval.generate import done_keys, sample_seed
from lolam.eval.grading import grade, last_boxed, math_equal, strip_thinking
from lolam.eval.passk import mean_pass_at_k, pass_at_k, valid_ks
from lolam.eval.score import score


# --- pass@k ---------------------------------------------------------------

def test_pass_at_k_edges():
    assert pass_at_k(4, 0, 1) == 0.0
    assert pass_at_k(4, 4, 1) == 1.0
    assert pass_at_k(4, 1, 1) == pytest.approx(0.25)
    assert pass_at_k(4, 1, 4) == 1.0


def test_pass_at_k_matches_combinatorics():
    from math import comb

    for n, c, k in [(10, 3, 2), (16, 5, 8), (8, 1, 3)]:
        assert pass_at_k(n, c, k) == pytest.approx(1 - comb(n - c, k) / comb(n, k))


def test_mean_pass_and_ks():
    assert mean_pass_at_k([(4, 0), (4, 4)], 1) == pytest.approx(0.5)
    assert valid_ks(4) == [1, 2, 4]
    assert valid_ks(6) == [1, 2, 4, 6]


# --- extracción y comparación --------------------------------------------

def test_last_boxed_nested():
    assert last_boxed(r"so \boxed{1} no wait \boxed{\frac{1}{2}}") == r"\frac{1}{2}"
    assert last_boxed("sin respuesta") is None
    assert last_boxed(r"\boxed{\frac{1}{2}") is None  # sin cerrar


def test_strip_thinking():
    assert strip_thinking("<think>blah</think>\n\nThe answer") == "\n\nThe answer"
    assert strip_thinking("<think>blah blah sin cerrar") is None
    assert strip_thinking("respuesta directa") == "respuesta directa"


@pytest.mark.parametrize("pred,gold", [
    (r"\frac{1}{2}", r"\dfrac{1}{2}"),
    ("0.5", r"\frac12"),
    (r"2\sqrt{2}", r"\sqrt{8}"),
    ("10", "10"),
    (r"\text{(C)}", "(C)"),
])
def test_math_equal_true(pred, gold):
    assert math_equal(pred, gold)


@pytest.mark.parametrize("pred,gold", [("3", "4"), (r"\frac{1}{3}", r"\frac{1}{2}")])
def test_math_equal_false(pred, gold):
    assert not math_equal(pred, gold)


def test_grade_truncated_thinking_is_wrong():
    # Aunque haya un \boxed correcto dentro del thinking, si no se cerró no cuenta.
    ok, pred = grade("math500", r"<think>creo que es \boxed{4} pero", "4")
    assert not ok and pred is None


def test_grade_gsm8k():
    assert grade("gsm8k", "<think>x</think> The total is $1,234.", "1234")[0]
    assert grade("gsm8k", r"<think>x</think> \boxed{18}", "18")[0]
    assert not grade("gsm8k", "<think>x</think> It is 17.", "18")[0]


# --- reanudación y métricas ----------------------------------------------

def test_sample_seed_deterministic():
    assert sample_seed(0, "a", 1) == sample_seed(0, "a", 1)
    assert sample_seed(0, "a", 1) != sample_seed(0, "a", 2)


def test_done_keys_ignores_partial_line(tmp_path):
    p = tmp_path / "g.jsonl"
    p.write_text(json.dumps({"id": "x", "sample": 0}) + "\n" + '{"id": "y", "sam')
    assert done_keys(p) == {("x", 0)}


def test_score_metrics():
    recs = [
        {"id": "a", "sample": 0, "gold": "4", "output": r"<think>.</think>\boxed{4}", "n_tokens": 10, "finish_reason": "stop"},
        {"id": "a", "sample": 1, "gold": "4", "output": "<think>loop loop", "n_tokens": 100, "finish_reason": "length"},
        {"id": "b", "sample": 0, "gold": "1", "output": r"<think>.</think>\boxed{2}", "n_tokens": 10, "finish_reason": "stop"},
        {"id": "b", "sample": 1, "gold": "1", "output": r"<think>.</think>\boxed{2}", "n_tokens": 10, "finish_reason": "stop"},
    ]
    m = score("math500", recs)
    assert m["accuracy"] == pytest.approx(0.25)
    assert m["truncation_rate"] == pytest.approx(0.25)
    assert m["pass_at_k"]["2"] == pytest.approx(0.5)
    assert m["accuracy_if_finished"] == pytest.approx(1 / 3)
