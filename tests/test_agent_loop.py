"""Quick Phase 3 test: verifies the loop-control logic in
src.agent_loop.run_rewrite_loop -- best-score retention, early stop on
target hit, max-iteration cap, and truthfulness-rejection handling.

Mocks generate_rewrite, check_truthfulness, and score_resume directly so
this only exercises the loop's control flow, independent of the real
spaCy-based scorer/truthfulness pipeline (which has its own dedicated
tests in test_scorer.py and test_rewriter.py). No LLM calls are made."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import src.agent_loop as agent_loop
from src.rewriter import TruthfulnessResult
from src.scorer import ScoreResult

ORIGINAL = "original resume text"
JD = "jd text"


class Patcher:
    def __init__(self):
        self._saved = []

    def set(self, obj, name, value):
        self._saved.append((obj, name, getattr(obj, name)))
        setattr(obj, name, value)

    def undo(self):
        for obj, name, value in reversed(self._saved):
            setattr(obj, name, value)


def _fake_score(value):
    return ScoreResult(score=value, matched_keywords=[], missing_keywords=["m"] if value < 100 else [])


def _draft_generator(texts):
    it = iter(texts)
    return lambda *a, **k: next(it)


def _truthfulness_checker(flags):
    it = iter(flags)

    def fn(draft, original):
        ok = next(it)
        return TruthfulnessResult(passed=ok, unverified_terms=[] if ok else ["fake_bad_term"])

    return fn


def _score_lookup(initial_score, draft_scores):
    def fn(text, jd):
        if text == ORIGINAL:
            return _fake_score(initial_score)
        return _fake_score(draft_scores[text])

    return fn


def test_stops_early_once_target_hit(p):
    p.set(agent_loop, "generate_rewrite", _draft_generator(["draft-1"]))
    p.set(agent_loop, "check_truthfulness", _truthfulness_checker([True]))
    p.set(agent_loop, "score_resume", _score_lookup(30.0, {"draft-1": 90.0}))

    result = agent_loop.run_rewrite_loop(ORIGINAL, JD, target_score=80.0, max_iterations=5)
    print(f"[INFO] iterations_run={result.iterations_run} best_score={result.best_score} hit_target={result.hit_target}")
    assert result.hit_target
    assert result.iterations_run == 1, "should stop after the first successful iteration"
    assert result.best_score == 90.0
    print("[PASS] loop stops early once target score is hit")


def test_keeps_best_score_even_if_later_iteration_regresses(p):
    p.set(agent_loop, "generate_rewrite", _draft_generator(["draft-A", "draft-B"]))
    p.set(agent_loop, "check_truthfulness", _truthfulness_checker([True, True]))
    p.set(agent_loop, "score_resume", _score_lookup(30.0, {"draft-A": 95.0, "draft-B": 60.0}))

    result = agent_loop.run_rewrite_loop(ORIGINAL, JD, target_score=101.0, max_iterations=2)
    print(f"[INFO] best_score={result.best_score} best_text={result.best_resume_text!r}")
    assert result.best_score == 95.0, "best_score must be the max across all iterations, not the last"
    assert result.best_resume_text == "draft-A", "must keep draft-A's text, not the regressed draft-B"
    assert not result.hit_target
    print("[PASS] loop retains the best-scoring version even after a later regression")


def test_max_iterations_cap_respected(p):
    p.set(agent_loop, "generate_rewrite", _draft_generator(["d1", "d2", "d3"]))
    p.set(agent_loop, "check_truthfulness", _truthfulness_checker([True, True, True]))
    p.set(agent_loop, "score_resume", _score_lookup(30.0, {"d1": 40.0, "d2": 45.0, "d3": 50.0}))

    result = agent_loop.run_rewrite_loop(ORIGINAL, JD, target_score=101.0, max_iterations=3)
    print(f"[INFO] iterations_run={result.iterations_run} hit_target={result.hit_target}")
    assert not result.hit_target
    assert result.iterations_run == 3
    print("[PASS] loop respects max_iterations and reports hit_target=False")


def test_truthfulness_rejection_does_not_crash_the_loop(p):
    # Iteration 1: both truthfulness-retry attempts fail. Iteration 2: passes and hits target.
    p.set(agent_loop, "generate_rewrite", _draft_generator(["bad-1a", "bad-1b", "good-2"]))
    p.set(agent_loop, "check_truthfulness", _truthfulness_checker([False, False, True]))
    p.set(agent_loop, "score_resume", _score_lookup(30.0, {"good-2": 95.0}))

    result = agent_loop.run_rewrite_loop(ORIGINAL, JD, target_score=90.0, max_iterations=2)
    print(f"[INFO] history: {[(h.iteration, h.accepted, h.rejection_reason) for h in result.history]}")
    assert result.history[1].iteration == 1 and result.history[1].accepted is False
    assert result.history[1].rejection_reason is not None
    assert result.history[2].iteration == 2 and result.history[2].accepted is True
    assert result.hit_target and result.best_score == 95.0
    print("[PASS] truthfulness rejection recorded without crashing the loop")


if __name__ == "__main__":
    for test_fn in [
        test_stops_early_once_target_hit,
        test_keeps_best_score_even_if_later_iteration_regresses,
        test_max_iterations_cap_respected,
        test_truthfulness_rejection_does_not_crash_the_loop,
    ]:
        p = Patcher()
        try:
            test_fn(p)
        finally:
            p.undo()

    print("\nPhase 3 agent_loop tests completed.")
