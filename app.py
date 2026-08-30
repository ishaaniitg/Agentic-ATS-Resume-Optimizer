"""Streamlit UI: a thin manual-testing surface over the existing Phase 1-4
pipeline. Every button below calls the existing src/* functions directly --
no parsing, scoring, rewriting, or ranking logic is reimplemented here.
Everything in this file is presentation only (layout, CSS, diff rendering).

Entry points used (see PLAN.md phases 1-4):
  - src.parser.parse_resume          (Phase 1)
  - src.scorer.score_resume          (Phase 2)
  - src.agent_loop.run_rewrite_loop  (Phase 3 -- used directly, not via
    src.modes.candidate_mode, because the UI needs the per-iteration
    history/changed_sections that candidate_mode discards)
  - src.modes.employer_mode          (Phase 4, employer mode wrapper)
  - src.pdf_generator.generate_resume_pdf  (tailored-resume PDF download)
"""

import difflib
import html
import os
import tempfile

import plotly.graph_objects as go
import streamlit as st
from google.genai import errors as genai_errors

from src.agent_loop import DEFAULT_MAX_ITERATIONS, DEFAULT_TARGET_SCORE, run_rewrite_loop
from src.modes import employer_mode
from src.parser import ParseError, parse_resume
from src.pdf_generator import generate_resume_pdf
from src.rewriter import get_active_model_status
from src.scorer import score_resume

st.set_page_config(
    page_title="Resume Tailor -- AI ATS Optimizer",
    page_icon="🎯",
    layout="wide",
)

# ---------------------------------------------------------------- styling --
# All color/spacing/radius tokens and component classes live in
# assets/theme.css -- this is the single place to change the look.

_THEME_CSS_PATH = os.path.join(os.path.dirname(__file__), "assets", "theme.css")
with open(_THEME_CSS_PATH, encoding="utf-8") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# --------------------------------------------------------------- helpers --


def _save_bytes_to_temp(name: str, data: bytes) -> str:
    """parse_resume() takes a file path, not bytes -- save to a temp file
    so we can call it unmodified."""
    suffix = os.path.splitext(name)[1] or ".pdf"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(data)
    tmp.close()
    return tmp.name


def sync_single_upload(widget_value, results: dict, results_key: str) -> None:
    """st.file_uploader's return value is lost across reruns where the
    widget itself isn't rendered (e.g. while the other mode is active) --
    even with a stable `key`. Persist the raw bytes into our own results
    dict the moment a file arrives, so uploads survive mode switches."""
    if widget_value is not None:
        results[results_key] = {"name": widget_value.name, "bytes": widget_value.getvalue()}


def sync_multi_upload(widget_value, results: dict, results_key: str) -> None:
    if widget_value:
        results[results_key] = [{"name": f.name, "bytes": f.getvalue()} for f in widget_value]


def _score_class(score: float) -> str:
    if score >= 80:
        return "score-good"
    if score >= 50:
        return "score-mid"
    return "score-low"


def render_score_card(label: str, score: float) -> None:
    st.markdown(
        f"""
        <div class="score-card {_score_class(score)}">
          <div class="score-label">{html.escape(label)}</div>
          <div class="score-value">{score:.1f}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.progress(min(max(score / 100, 0.0), 1.0))


def render_score_gauge(label: str, score: float, target_score: float) -> None:
    """Radial gauge for the ATS match score -- red/amber/green zones with
    the target score marked as a threshold line, replacing a plain number."""
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number={"suffix": "%", "font": {"size": 34, "color": "#1e293b"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#94a3b8"},
                "bar": {"color": "#4f46e5", "thickness": 0.28},
                "bgcolor": "white",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 50], "color": "#fee2e2"},
                    {"range": [50, 80], "color": "#fef3c7"},
                    {"range": [80, 100], "color": "#dcfce7"},
                ],
                "threshold": {
                    "line": {"color": "#1e293b", "width": 3},
                    "thickness": 0.9,
                    "value": target_score,
                },
            },
        )
    )
    fig.update_layout(
        height=200,
        margin=dict(l=25, r=25, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter, sans-serif"},
    )
    st.markdown(f'<div class="section-label" style="margin:0 0 -.5rem 0">{html.escape(label)}</div>', unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.caption(f"Target marked at {target_score:.0f}%")


def render_subscore_bar(label: str, value: float) -> None:
    st.markdown(
        f'<div class="subscore-row"><span>{html.escape(label)}</span>'
        f'<span class="subscore-value">{value:.1f}%</span></div>',
        unsafe_allow_html=True,
    )
    st.progress(min(max(value / 100, 0.0), 1.0))


def render_stat_card(label: str, value: str, pill: str | None = None) -> None:
    cls = f"pill-{pill}" if pill else ""
    st.markdown(
        f"""
        <div class="stat-card">
          <div class="stat-label">{html.escape(label)}</div>
          <div class="stat-value {cls}">{html.escape(str(value))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_model_status_card() -> None:
    """Compact sidebar status card for the Gemini model/key the rewrite
    loop would currently use -- reflects key rotation and model fallback
    state (src/rewriter.py)."""
    try:
        status = get_active_model_status()
    except RuntimeError as exc:
        st.sidebar.markdown(
            f"""
            <div class="sidebar-status-card status-error">
              <div class="status-title">⚡ Model status</div>
              <div class="status-model">⚠️ Unavailable</div>
              <div class="status-detail">{html.escape(str(exc))}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    detail = f"key {status['active_key_index']}/{status['total_keys']}"
    if status["exhausted_pairs"]:
        detail += f" · {status['exhausted_pairs']}/{status['total_pairs']} quota slots used today"
    st.sidebar.markdown(
        f"""
        <div class="sidebar-status-card">
          <div class="status-title">⚡ Model status</div>
          <div class="status-model">{html.escape(status["active_model"])}</div>
          <div class="status-detail">{html.escape(detail)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def stepper_html(steps: list[str], current_index: int) -> str:
    """Horizontal progress stepper markup. Steps before current_index
    render as done (checkmark), the step at current_index as active, the
    rest as pending -- purely a display of where session state currently
    is. Returns HTML rather than rendering directly: callers fill an
    st.empty() placeholder with this *after* running that mode's button
    handlers, so the stepper reflects state changes made during the same
    script run (e.g. clicking Analyze) instead of the state from before
    that run's handlers executed."""
    parts = ['<div class="stepper">']
    for i, label in enumerate(steps):
        if i < current_index:
            cls, marker = "done", "✓"
        elif i == current_index:
            cls, marker = "active", str(i + 1)
        else:
            cls, marker = "", str(i + 1)
        parts.append(
            f'<div class="stepper-step {cls}">'
            f'<div class="step-circle">{marker}</div>'
            f'<div class="step-label">{html.escape(label)}</div>'
            f"</div>"
        )
        if i < len(steps) - 1:
            line_cls = "done" if i < current_index else ""
            parts.append(f'<div class="stepper-line {line_cls}"></div>')
    parts.append("</div>")
    return "".join(parts)


def render_parse_error(exc: Exception) -> None:
    """Human-readable parse/OCR failure state -- distinguishes the specific
    ParseError/ValueError causes raised by src/parser.py rather than
    surfacing a raw exception string with no context."""
    msg = str(exc)
    if isinstance(exc, ValueError):
        st.error("⚠️ **Unsupported file type.** Only PDF resumes are supported -- please upload a `.pdf` file.")
    elif "Poppler" in msg or "Tesseract" in msg:
        st.error(
            "⚠️ **This looks like a scanned resume.** Reading it needs OCR, but a required "
            f"OCR tool isn't installed on this machine.\n\nDetails: {msg}"
        )
    elif "No text could be extracted" in msg:
        st.error(
            "⚠️ **Couldn't read any text from this PDF**, even with OCR. The file may be "
            "corrupted, password-protected, or blank."
        )
    else:
        st.error(f"⚠️ **Couldn't parse this resume.** {msg}")


def render_upload_chip(name: str, size_bytes: int, key: str) -> bool:
    """Renders a filename + size chip for a persisted (already-uploaded)
    file, with a remove button. Returns True if the user clicked remove."""
    chip_col, remove_col = st.columns([6, 1])
    with chip_col:
        st.markdown(
            f'<div class="upload-chip">📄 <b>{html.escape(name)}</b>'
            f'<span class="upload-chip-size">{size_bytes / 1024:.1f} KB</span></div>',
            unsafe_allow_html=True,
        )
    with remove_col:
        return st.button("✕", key=key, help=f"Remove {name}")


def render_badges(keywords: list[str], kind: str) -> None:
    if not keywords:
        st.caption("(none)")
        return
    cls = "badge-matched" if kind == "matched" else "badge-missing"
    st.markdown(
        "".join(f'<span class="badge {cls}">{html.escape(k)}</span>' for k in keywords),
        unsafe_allow_html=True,
    )


def render_line_diff(before: str, after: str) -> None:
    """Line-level diff, purely for display -- computed from the already-
    generated before/after text, not part of the pipeline."""
    before_lines, after_lines = before.splitlines(), after.splitlines()
    matcher = difflib.SequenceMatcher(None, before_lines, after_lines)
    parts = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for line in before_lines[i1:i2]:
                parts.append(f'<div class="diff-line diff-eq">{html.escape(line)}</div>')
        else:
            for line in before_lines[i1:i2]:
                parts.append(f'<div class="diff-line diff-del">- {html.escape(line)}</div>')
            for line in after_lines[j1:j2]:
                parts.append(f'<div class="diff-line diff-add">+ {html.escape(line)}</div>')
    st.markdown(f'<div class="diff-box">{"".join(parts)}</div>', unsafe_allow_html=True)


def section(icon: str, title: str):
    """Returns a bordered container with an icon + header already rendered
    inside it -- the standard wrapper for every phase section below."""
    container = st.container(border=True)
    with container:
        st.markdown(
            f'<div class="section-header">{icon} {html.escape(title)}</div>',
            unsafe_allow_html=True,
        )
    return container


# ------------------------------------------------------------------ sidebar --

with st.sidebar:
    st.markdown('<div class="sidebar-title">🎯 Resume Tailor</div>', unsafe_allow_html=True)
    st.caption("AI ATS Optimizer")
    mode = st.segmented_control(
        "Mode", ["Candidate", "Employer"], default="Candidate", key="active_mode"
    ) or "Candidate"
    st.divider()
    render_model_status_card()

# Single source of truth for computed results, namespaced by mode so a
# mode switch (or Streamlit rerun) never loses Phase 2/3 output. Input
# widgets below are bound via stable `key=` params for the same reason --
# without a key, a widget's value resets whenever it isn't rendered on a
# given run (e.g. while the other mode is active).
if "analysis_results" not in st.session_state:
    st.session_state["analysis_results"] = {"candidate": {}, "employer": {}}
candidate_results = st.session_state["analysis_results"]["candidate"]
employer_results = st.session_state["analysis_results"]["employer"]

# Centered, max-width main content column -- consistent padding on wide
# screens instead of content stretching edge-to-edge.
_pad_l, main_col, _pad_r = st.columns([1, 8, 1])

with main_col:
    # ------------------------------------------------------------------ hero --

    st.markdown(
        """
        <div class="app-hero">
          <h1>🎯 Resume Tailor -- AI ATS Optimizer</h1>
          <p>Upload a resume and a job description to see a deterministic ATS-style match score,
          then let an AI rewrite loop tailor the resume toward your target score -- with a
          truthfulness check that blocks any rewrite introducing skills you don't actually have.
          Switch to <b>Employer</b> mode in the sidebar to rank several candidates against one
          job description.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ------------------------------------------------------------- candidate --

    if mode == "Candidate":
        stepper_slot = st.empty()

        with section("📤", "Upload & Target"):
            upl_col, jd_col = st.columns([1, 1.4])
            with upl_col:
                resume_file = st.file_uploader("Resume (PDF)", type=["pdf"], key="candidate_pdf")
                sync_single_upload(resume_file, candidate_results, "uploaded_pdf")
                has_resume = "uploaded_pdf" in candidate_results
                if resume_file is None and has_resume:
                    st.caption(f"📎 Using previously uploaded **{candidate_results['uploaded_pdf']['name']}** -- upload a new file to replace it.")
                if has_resume and "resume_text" in candidate_results:
                    st.markdown(
                        f'<div class="upload-success">✅ Parsed successfully -- '
                        f'{len(candidate_results["resume_text"]):,} characters extracted</div>',
                        unsafe_allow_html=True,
                    )
                c1, c2 = st.columns(2)
                with c1:
                    target_score = st.number_input(
                        "Target score", value=DEFAULT_TARGET_SCORE, min_value=0.0, max_value=100.0,
                        key="candidate_target_score",
                    )
                with c2:
                    max_iterations = st.number_input(
                        "Max iterations", value=DEFAULT_MAX_ITERATIONS, min_value=1, step=1,
                        key="candidate_max_iterations",
                    )
            with jd_col:
                jd_text = st.text_area("Job description", height=190, key="shared_jd")
                _word_count = len(jd_text.split())
                st.caption(f"{len(jd_text):,} characters · {_word_count:,} words")

            if not has_resume and not jd_text:
                analyze_help = "Upload a resume and paste a job description to enable analysis."
            elif not has_resume:
                analyze_help = "Upload a resume (PDF) to enable analysis."
            elif not jd_text:
                analyze_help = "Paste a job description to enable analysis."
            else:
                analyze_help = "Parse the resume and score it against the job description."

            if st.button("🔍  Analyze", type="primary", disabled=not (has_resume and jd_text), help=analyze_help):
                try:
                    upload = candidate_results["uploaded_pdf"]
                    resume_path = _save_bytes_to_temp(upload["name"], upload["bytes"])
                    resume_text = parse_resume(resume_path)
                    candidate_results["resume_text"] = resume_text
                    candidate_results["jd_text"] = jd_text
                    candidate_results["score_result"] = score_resume(resume_text, jd_text)
                    candidate_results.pop("loop_result", None)  # stale after re-analyze
                except (ParseError, ValueError) as exc:
                    render_parse_error(exc)

        if "resume_text" in candidate_results:
            with section("📄", "Parsed Resume"):
                st.text_area("Extracted text", candidate_results["resume_text"], height=220, label_visibility="collapsed")

            with section("🎯", "Match Score"):
                result = candidate_results["score_result"]
                score_col, kw_col = st.columns([1, 2])
                with score_col:
                    render_score_gauge("ATS match score", result.score, target_score)
                    render_subscore_bar("Semantic match", result.semantic_score)
                    render_subscore_bar("Keyword match", result.keyword_score)
                with kw_col:
                    mcol, xcol = st.columns(2)
                    with mcol:
                        st.markdown("**✅ Matched keywords**")
                        render_badges(result.matched_keywords, "matched")
                    with xcol:
                        st.markdown("**⚠️ Missing keywords**")
                        render_badges(result.missing_keywords, "missing")
                    if result.conceptual_gaps:
                        with st.expander("🧠 Conceptual skill gaps"):
                            st.markdown(
                                '<span class="info-tooltip" title="Skills the job conceptually '
                                'needs but that don\'t appear anywhere in the resume, even '
                                'phrased differently -- distinct from missing keywords, which '
                                'are exact terms.">ⓘ What does this mean?</span>',
                                unsafe_allow_html=True,
                            )
                            render_badges(result.conceptual_gaps, "missing")

            with section("🛠️", "Rewrite + Truthfulness Loop"):
                if st.button("✍️  Rewrite", type="primary"):
                    try:
                        with st.spinner("Running generator -> truthfulness check -> judge loop..."):
                            candidate_results["loop_result"] = run_rewrite_loop(
                                candidate_results["resume_text"],
                                candidate_results["jd_text"],
                                target_score=target_score,
                                max_iterations=int(max_iterations),
                            )
                    except genai_errors.ClientError as exc:
                        st.error(f"Gemini API error (not a pipeline bug): {exc}")
                    except genai_errors.ServerError as exc:
                        st.error(f"Gemini API unavailable after retries (not a pipeline bug): {exc}")

                if "loop_result" in candidate_results:
                    loop_result = candidate_results["loop_result"]

                    st.markdown("**Per-iteration log**")
                    prev_score = loop_result.history[0].score if loop_result.history else 0.0
                    for log in loop_result.history:
                        if log.iteration == 0:
                            st.markdown(f"🏁 &nbsp;**Iteration 0** (initial parse) -- score **{log.score:.1f}%**", unsafe_allow_html=True)
                            continue

                        if log.accepted:
                            delta = log.score - prev_score
                            prev_score = log.score
                            delta_badge = f'<span class="delta-badge {"delta-up" if delta >= 0 else "delta-down"}">{delta:+.1f}%</span>'
                            title = f"Iteration {log.iteration} · score {log.score:.1f}% · ✅ Passed"
                        else:
                            delta_badge = '<span class="delta-badge delta-rejected">⛔ rejected</span>'
                            title = f"Iteration {log.iteration} · ⛔ Rejected by truthfulness check"

                        with st.expander(title):
                            st.markdown(delta_badge, unsafe_allow_html=True)
                            if log.accepted and log.changed_sections:
                                for section_name, (before, after) in log.changed_sections.items():
                                    st.markdown(f"**Section: {section_name}**")
                                    render_line_diff(before, after)
                            elif log.accepted:
                                st.caption("Accepted, but the section splitter found no per-section diff.")
                            else:
                                st.error(
                                    f"Rejected -- {log.rejection_reason}. This draft was discarded and never scored."
                                )

            if "loop_result" in candidate_results:
                loop_result = candidate_results["loop_result"]
                with section("🏆", "Final Result"):
                    f1, f2, f3 = st.columns(3)
                    with f1:
                        render_score_gauge("Final score", loop_result.best_score, target_score)
                    with f2:
                        render_stat_card("Iterations run", loop_result.iterations_run)
                    with f3:
                        render_stat_card("Hit target", "Yes" if loop_result.hit_target else "No",
                                          pill="yes" if loop_result.hit_target else "no")

                    accepted_scores = {log.iteration: log.score for log in loop_result.history if log.accepted}
                    best_iteration = max(accepted_scores, key=accepted_scores.get) if accepted_scores else 0
                    if best_iteration != loop_result.iterations_run:
                        st.markdown(
                            f'<div class="info-banner">Best score ({loop_result.best_score:.1f}%) came from '
                            f'iteration {best_iteration}, not the last iteration run ({loop_result.iterations_run}). '
                            "The text below is that best-scoring version -- confirming the loop kept it rather "
                            "than defaulting to the final draft.</div>",
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            f'<div class="info-banner">Best score ({loop_result.best_score:.1f}%) is from '
                            f"iteration {best_iteration}, which is also the last iteration run.</div>",
                            unsafe_allow_html=True,
                        )

                    st.markdown("**Best-scoring resume text**")
                    st.text_area("Final resume", loop_result.best_resume_text, height=280, label_visibility="collapsed")

                    try:
                        pdf_bytes = generate_resume_pdf(loop_result.best_resume_text)
                        st.download_button(
                            "⬇️  Download tailored resume (PDF)",
                            data=pdf_bytes,
                            file_name="tailored_resume.pdf",
                            mime="application/pdf",
                            type="primary",
                        )
                    except Exception as exc:
                        st.error(f"Could not generate the PDF (not a scoring/rewrite bug): {exc}")

        if "loop_result" in candidate_results:
            step_index = 2
        elif "resume_text" in candidate_results:
            step_index = 1
        else:
            step_index = 0
        stepper_slot.markdown(stepper_html(["Upload", "Score", "Rewrite"], step_index), unsafe_allow_html=True)

    # --------------------------------------------------------------- employer --

    else:
        stepper_slot = st.empty()

        with section("📤", "Upload Candidates & Job Description"):
            upl_col, jd_col = st.columns([1, 1.4])
            with upl_col:
                resume_files = st.file_uploader(
                    "Resumes (PDF)", type=["pdf"], accept_multiple_files=True, key="employer_pdfs"
                )
                sync_multi_upload(resume_files, employer_results, "uploaded_pdfs")
                has_resumes = bool(employer_results.get("uploaded_pdfs"))
                if not resume_files and has_resumes:
                    st.caption("📎 Using previously uploaded resumes -- remove any you don't want, or upload new files to replace them.")
                    remove_idx = None
                    for i, u in enumerate(employer_results["uploaded_pdfs"]):
                        if render_upload_chip(u["name"], len(u["bytes"]), key=f"remove_pdf_{i}"):
                            remove_idx = i
                    if remove_idx is not None:
                        employer_results["uploaded_pdfs"].pop(remove_idx)
                        has_resumes = bool(employer_results["uploaded_pdfs"])
            with jd_col:
                jd_text = st.text_area("Job description", height=150, key="shared_jd")
                _word_count = len(jd_text.split())
                st.caption(f"{len(jd_text):,} characters · {_word_count:,} words")

            if not has_resumes and not jd_text:
                rank_help = "Upload one or more resumes and paste a job description to enable ranking."
            elif not has_resumes:
                rank_help = "Upload one or more resumes (PDF) to enable ranking."
            elif not jd_text:
                rank_help = "Paste a job description to enable ranking."
            else:
                rank_help = "Rank all uploaded resumes against this job description."

            if st.button("📊  Rank", type="primary", disabled=not (has_resumes and jd_text), help=rank_help):
                resume_paths = []
                name_by_path = {}
                for upload in employer_results["uploaded_pdfs"]:
                    path = _save_bytes_to_temp(upload["name"], upload["bytes"])
                    resume_paths.append(path)
                    name_by_path[path] = upload["name"]

                employer_results["rankings"] = employer_mode(resume_paths, jd_text)
                employer_results["name_by_path"] = name_by_path

        if "rankings" in employer_results:
            rankings = employer_results["rankings"]
            name_by_path = employer_results["name_by_path"]

            with section("🏆", "Ranked Candidates"):
                sort_col, filter_col = st.columns(2)
                with sort_col:
                    sort_choice = st.selectbox(
                        "Sort by",
                        ["Score (high to low)", "Score (low to high)", "Name (A-Z)"],
                        key="employer_sort",
                    )
                with filter_col:
                    min_score = st.slider("Minimum score", 0, 100, 0, key="employer_min_score")

                visible = [r for r in rankings if r.score >= min_score]
                if sort_choice == "Score (high to low)":
                    visible.sort(key=lambda r: r.score, reverse=True)
                elif sort_choice == "Score (low to high)":
                    visible.sort(key=lambda r: r.score)
                else:
                    visible.sort(key=lambda r: name_by_path.get(r.resume_path, r.resume_path).lower())

                if not visible:
                    st.caption(f"No candidates scored {min_score}% or higher.")

                for i, r in enumerate(visible, start=1):
                    label = name_by_path.get(r.resume_path, r.resume_path)
                    with st.container(border=True):
                        rank_col, name_col, score_col, status_col = st.columns([0.6, 2.4, 3, 1.3])
                        with rank_col:
                            st.markdown(f'<div class="rank-badge">{i}</div>', unsafe_allow_html=True)
                        with name_col:
                            st.markdown(f'<div class="candidate-name">{html.escape(label)}</div>', unsafe_allow_html=True)
                        with score_col:
                            st.progress(min(max(r.score / 100, 0.0), 1.0))
                            st.caption(f"{r.score:.1f}%")
                        with status_col:
                            status_cls = "badge-missing" if r.error else "badge-matched"
                            status_text = "⚠️ Parse error" if r.error else "✅ OK"
                            st.markdown(f'<span class="badge {status_cls}">{status_text}</span>', unsafe_allow_html=True)

                        with st.expander("View gap summary"):
                            if r.error:
                                st.error(r.error)
                            else:
                                st.caption(f"= 50% semantic ({r.semantic_score:.1f}%) + 50% keyword ({r.keyword_score:.1f}%)")
                                mcol, xcol = st.columns(2)
                                with mcol:
                                    st.markdown("**✅ Matched**")
                                    render_badges(r.matched_keywords, "matched")
                                with xcol:
                                    st.markdown("**⚠️ Missing**")
                                    render_badges(r.missing_keywords, "missing")
                                if r.conceptual_gaps:
                                    st.markdown("**🧠 Conceptual skill gaps**")
                                    render_badges(r.conceptual_gaps, "missing")

        step_index = 1 if "rankings" in employer_results else 0
        stepper_slot.markdown(stepper_html(["Upload", "Rank"], step_index), unsafe_allow_html=True)

    st.markdown(
        '<div class="footnote">Thin debugging/demo surface over src/parser.py, src/scorer.py, '
        "src/agent_loop.py and src/modes.py -- no pipeline logic lives in this file.</div>",
        unsafe_allow_html=True,
    )
