"""Converts the Phase 3 rewrite loop's plain-text output into a clean,
single-column, ATS-friendly PDF for download. Presentation only -- a new
capability, not a modification of any Phase 1-4 pipeline logic."""

from __future__ import annotations

import io
from xml.sax.saxutils import escape

from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

# Single-column, no tables/graphics/multi-column layout, standard fonts --
# the layout choices ATS parsers actually need, not just a nice-looking PDF.
_SECTION_HEADERS = {
    "summary", "objective", "skills", "experience", "education",
    "projects", "certifications",
}
_BULLET_PREFIXES = ("-", "*", "•", "–", "—")


def _build_styles() -> dict[str, ParagraphStyle]:
    return {
        "name": ParagraphStyle(
            "ATSName", fontName="Helvetica-Bold", fontSize=15, leading=18,
            spaceAfter=6,
        ),
        "heading": ParagraphStyle(
            "ATSHeading", fontName="Helvetica-Bold", fontSize=12, leading=15,
            spaceBefore=10, spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "ATSBody", fontName="Helvetica", fontSize=10, leading=13,
            spaceAfter=2,
        ),
        "bullet": ParagraphStyle(
            "ATSBullet", fontName="Helvetica", fontSize=10, leading=13,
            spaceAfter=2, leftIndent=14,
        ),
    }


def _classify_line(stripped: str, is_first_content_line: bool) -> tuple[str, str]:
    """Returns (style_key, text_to_render)."""
    if is_first_content_line:
        return "name", stripped
    if stripped.lower().rstrip(":") in _SECTION_HEADERS:
        return "heading", stripped
    if stripped.startswith(_BULLET_PREFIXES):
        return "bullet", "• " + stripped.lstrip("".join(_BULLET_PREFIXES)).strip()
    return "body", stripped


def generate_resume_pdf(resume_text: str) -> bytes:
    """Render plain resume text as a clean, single-column ATS-friendly PDF.
    Returns the PDF file content as bytes, ready for st.download_button."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=LETTER,
        leftMargin=0.75 * inch, rightMargin=0.75 * inch,
        topMargin=0.6 * inch, bottomMargin=0.6 * inch,
        title="Tailored Resume",
    )
    styles = _build_styles()

    flowables = []
    is_first_content_line = True
    for raw_line in resume_text.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            flowables.append(Spacer(1, 6))
            continue

        style_key, text = _classify_line(stripped, is_first_content_line)
        # Escape for reportlab's mini-XML markup in Paragraph text (and
        # incidentally guards against any literal '<'/'>'/'&' in the
        # resume/LLM output breaking the parser).
        flowables.append(Paragraph(escape(text), styles[style_key]))
        is_first_content_line = False

    if not flowables:
        flowables.append(Paragraph("(empty resume)", styles["body"]))

    doc.build(flowables)
    return buffer.getvalue()
