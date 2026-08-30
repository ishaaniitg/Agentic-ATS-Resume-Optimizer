"""Quick Phase 3-output test: generates a PDF from plain resume text and
verifies it's a valid, readable PDF -- round-tripping it back through
PyMuPDF (already a project dependency) to confirm the content survived,
rather than just checking "some bytes came out"."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pymupdf

from src.pdf_generator import generate_resume_pdf

SAMPLE_RESUME = """Ambesh Dixit
+91-9335528194
d.ambesh@iitg.ac.in

Experience
Software Developer Intern
- Built REST APIs using Flask and Python
- Deployed with Docker on AWS SageMaker

Skills
Python, Flask, Docker, React, MongoDB
"""


def _extract_text(pdf_bytes: bytes) -> str:
    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
        return "\n".join(page.get_text() for page in doc)


def test_generates_valid_pdf_bytes():
    pdf_bytes = generate_resume_pdf(SAMPLE_RESUME)
    print(f"[INFO] generated {len(pdf_bytes)} bytes")
    assert pdf_bytes.startswith(b"%PDF"), "output should be a real PDF file"
    assert len(pdf_bytes) > 500
    print("[PASS] generate_resume_pdf produces valid PDF bytes")


def test_content_survives_round_trip():
    pdf_bytes = generate_resume_pdf(SAMPLE_RESUME)
    extracted = _extract_text(pdf_bytes)
    print(f"[INFO] extracted text: {extracted!r}")

    for expected in ["Ambesh Dixit", "Software Developer Intern", "Flask", "Docker", "MongoDB"]:
        assert expected in extracted, f"expected {expected!r} to survive the PDF round-trip"
    print("[PASS] resume content survives the text -> PDF -> text round-trip")


def test_section_headers_and_bullets_rendered():
    pdf_bytes = generate_resume_pdf(SAMPLE_RESUME)
    extracted = _extract_text(pdf_bytes)
    assert "Experience" in extracted and "Skills" in extracted
    assert "•" in extracted, "bullet lines should render with a bullet marker"
    print("[PASS] section headers and bullet points render correctly")


def test_empty_resume_does_not_crash():
    pdf_bytes = generate_resume_pdf("")
    assert pdf_bytes.startswith(b"%PDF")
    print("[PASS] empty resume text handled gracefully, still produces a valid PDF")


def test_special_characters_do_not_break_rendering():
    """Resume/LLM text could contain '<', '>', '&' -- these must be
    escaped, not break reportlab's Paragraph markup parser."""
    tricky = "R&D Engineer\nBuilt systems for A<B and C>D comparisons.\n"
    pdf_bytes = generate_resume_pdf(tricky)
    extracted = _extract_text(pdf_bytes)
    print(f"[INFO] extracted: {extracted!r}")
    assert "R&D Engineer" in extracted
    assert "A<B" in extracted and "C>D" in extracted
    print("[PASS] special XML characters (&, <, >) render literally, don't break the PDF")


if __name__ == "__main__":
    test_generates_valid_pdf_bytes()
    test_content_survives_round_trip()
    test_section_headers_and_bullets_rendered()
    test_empty_resume_does_not_crash()
    test_special_characters_do_not_break_rendering()
    print("\nPDF generator tests completed.")
