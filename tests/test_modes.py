"""Quick Phase 4 test: employer mode ranking + graceful per-resume error
handling (generates synthetic PDFs on the fly), and candidate mode's
wiring (parse -> score -> rewrite loop) verified with a mocked rewrite
loop so no LLM calls are made here."""

import os
import sys
import tempfile

import pymupdf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import src.modes as modes
from src.agent_loop import LoopResult
from src.modes import candidate_mode, employer_mode

JD_TEXT = "Looking for a candidate with Python, Docker, and Flask experience."


def _make_pdf(path: str, text: str) -> None:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    doc.save(path)
    doc.close()


def test_employer_mode_ranks_by_score_and_reports_gaps():
    with tempfile.TemporaryDirectory() as tmp:
        strong_path = os.path.join(tmp, "strong.pdf")
        weak_path = os.path.join(tmp, "weak.pdf")
        _make_pdf(strong_path, "Built services with Python, Docker, and Flask.")
        _make_pdf(weak_path, "Marketing coordinator with social media experience.")

        rankings = employer_mode([weak_path, strong_path], JD_TEXT)  # weak listed first on purpose
        print(f"[INFO] ranking: {[(r.resume_path, r.score) for r in rankings]}")

        assert rankings[0].resume_path == strong_path, "higher-scoring resume must rank first"
        assert rankings[0].score > rankings[1].score
        assert "python" in rankings[1].missing_keywords or "docker" in rankings[1].missing_keywords
        print("[PASS] employer mode ranks resumes by score and flags gaps for the weaker one")


def test_employer_mode_handles_unparseable_resume_without_crashing():
    with tempfile.TemporaryDirectory() as tmp:
        good_path = os.path.join(tmp, "good.pdf")
        bad_path = os.path.join(tmp, "not_a_pdf.txt")
        _make_pdf(good_path, "Built services with Python, Docker, and Flask.")
        with open(bad_path, "w") as f:
            f.write("this is not a pdf")

        rankings = employer_mode([good_path, bad_path], JD_TEXT)
        by_path = {r.resume_path: r for r in rankings}

        assert by_path[bad_path].error is not None
        assert by_path[bad_path].score == 0.0
        assert by_path[good_path].error is None
        print("[PASS] employer mode reports a per-resume error instead of crashing the whole batch")


def test_candidate_mode_wires_parse_score_and_loop(monkeypatch=None):
    with tempfile.TemporaryDirectory() as tmp:
        resume_path = os.path.join(tmp, "resume.pdf")
        _make_pdf(resume_path, "Built services with Python, Docker, and Flask.")

        fake_loop_result = LoopResult(
            best_resume_text="rewritten resume text",
            best_score=95.0,
            iterations_run=1,
            hit_target=True,
            history=[],
        )
        original = modes.run_rewrite_loop
        modes.run_rewrite_loop = lambda *a, **k: fake_loop_result
        try:
            result = candidate_mode(resume_path, JD_TEXT, target_score=90.0, max_iterations=1)
        finally:
            modes.run_rewrite_loop = original

        print(f"[INFO] initial_score={result.initial_score} final_score={result.final_score}")
        assert result.initial_score > 0  # parsed and scored the real synthetic resume
        assert result.final_score == 95.0
        assert result.final_resume_text == "rewritten resume text"
        assert result.hit_target
        print("[PASS] candidate mode correctly wires parse -> score -> rewrite loop")


if __name__ == "__main__":
    test_employer_mode_ranks_by_score_and_reports_gaps()
    test_employer_mode_handles_unparseable_resume_without_crashing()
    test_candidate_mode_wires_parse_score_and_loop()
    print("\nPhase 4 modes tests completed.")
