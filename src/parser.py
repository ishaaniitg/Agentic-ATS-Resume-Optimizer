"""Resume PDF parsing: PyMuPDF text extraction with an OCR fallback for
scanned/image-only resumes (pdf2image + pytesseract)."""

from __future__ import annotations

import pymupdf
import pytesseract
from pdf2image import convert_from_path

# Below this average characters-per-page, treat the PDF as having no usable
# text layer (i.e. a scanned image) and fall back to OCR.
MIN_CHARS_PER_PAGE = 40


class ParseError(Exception):
    """Raised when a resume PDF cannot be parsed into usable text."""


def _extract_via_ocr(file_path: str) -> str:
    try:
        images = convert_from_path(file_path)
    except Exception as exc:
        raise ParseError(
            "OCR fallback requires Poppler to be installed and on PATH "
            f"(pdf2image dependency). Original error: {exc}"
        ) from exc

    pages = []
    for image in images:
        try:
            pages.append(pytesseract.image_to_string(image))
        except Exception as exc:
            raise ParseError(
                "OCR fallback requires the Tesseract binary to be installed "
                f"and on PATH (pytesseract dependency). Original error: {exc}"
            ) from exc
    return "\n".join(pages)


def _normalize(text: str) -> str:
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    return "\n".join(lines)


def parse_resume(file_path: str) -> str:
    """Extract normalized plain text from a resume PDF.

    Tries the embedded text layer first (PyMuPDF). If the extracted text is
    empty or near-empty relative to the page count (a scanned/image-only
    resume), falls back to OCR via pdf2image + pytesseract.
    """
    if not file_path.lower().endswith(".pdf"):
        raise ValueError("Only PDF files are supported.")

    with pymupdf.open(file_path) as doc:
        text = "\n".join(page.get_text() for page in doc)
        page_count = doc.page_count or 1

    if len(text.strip()) < MIN_CHARS_PER_PAGE * page_count:
        text = _extract_via_ocr(file_path)

    normalized = _normalize(text)
    if not normalized:
        raise ParseError(
            "No text could be extracted from this PDF, even after OCR fallback."
        )

    return normalized
