"""Quick Phase 4 test: main.py CLI wiring, via subprocess so it exercises
the real argparse setup and imports exactly as a user invoking the CLI
would. Uses employer mode (no LLM calls) against synthetic PDFs."""

import os
import subprocess
import sys
import tempfile

import pymupdf

ROOT = os.path.join(os.path.dirname(__file__), "..")
PYTHON = os.path.join(ROOT, "venv", "Scripts", "python.exe")
MAIN = os.path.join(ROOT, "main.py")


def _make_pdf(path: str, text: str) -> None:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    doc.save(path)
    doc.close()


def test_help_runs_without_error():
    result = subprocess.run(
        [PYTHON, MAIN, "--help"], capture_output=True, text=True, timeout=30
    )
    print(f"[INFO] --help exit code: {result.returncode}")
    assert result.returncode == 0
    assert "candidate" in result.stdout and "employer" in result.stdout
    print("[PASS] main.py --help runs and lists both subcommands")


def test_employer_subcommand_end_to_end():
    with tempfile.TemporaryDirectory() as tmp:
        strong_path = os.path.join(tmp, "strong.pdf")
        weak_path = os.path.join(tmp, "weak.pdf")
        _make_pdf(strong_path, "Built services with Python, Docker, and Flask.")
        _make_pdf(weak_path, "Marketing coordinator with social media experience.")

        result = subprocess.run(
            [
                PYTHON, MAIN, "employer",
                "--resumes", strong_path, weak_path,
                "--jd", "Looking for a candidate with Python, Docker, and Flask experience.",
            ],
            capture_output=True, text=True, timeout=60,
        )
        print(f"[INFO] employer subcommand exit code: {result.returncode}")
        print(result.stdout)
        assert result.returncode == 0, result.stderr
        assert "Gap summaries" in result.stdout
        # strong.pdf should be ranked above weak.pdf
        strong_pos = result.stdout.find(os.path.basename(strong_path))
        weak_pos = result.stdout.find(os.path.basename(weak_path))
        assert 0 <= strong_pos < weak_pos
        print("[PASS] `main.py employer` runs end-to-end and ranks correctly")


if __name__ == "__main__":
    test_help_runs_without_error()
    test_employer_subcommand_end_to_end()
    print("\nPhase 4 CLI tests completed.")
