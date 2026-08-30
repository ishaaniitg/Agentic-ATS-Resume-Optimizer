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


def render_model_badge() -> None:
    """Shows which Gemini model/key the rewrite loop would currently use --
    reflects key rotation and model fallback state (src/rewriter.py)."""
    try:
        status = get_active_model_status()
    except RuntimeError as exc:
        st.markdown(
            f'<div class="model-badge model-badge-error">⚠️ {html.escape(str(exc))}</div>',
            unsafe_allow_html=True,
        )
        return

    detail = f"key {status['active_key_index']}/{status['total_keys']}"
    if status["exhausted_pairs"]:
        detail += f" · {status['exhausted_pairs']}/{status['total_pairs']} quota slots used today"
    st.markdown(
        f'<div class="model-badge">⚡ Active Model: {html.escape(status["active_model"])}'
        f'<span class="model-badge-detail">({html.escape(detail)})</span></div>',
        unsafe_allow_html=True,
    )


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


# ------------------------------------------------------------------ hero --

st.markdown(
    """
    <div class="app-hero">
      <h1>🎯 Resume Tailor -- AI ATS Optimizer</h1>
      <p>Upload a resume and a job description to see a deterministic ATS-style match score,
      then let an AI rewrite loop tailor the resume toward your target score -- with a
      truthfulness check that blocks any rewrite introducing skills you don't actually have.
      Switch to <b>Employer</b> mode to rank several candidates against one job description.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

render_model_badge()

mode = st.segmented_control("Mode", ["Candidate", "Employer"], default="Candidate", key="active_mode") or "Candidate"

# Single source of truth for computed results, namespaced by mode so a
# mode switch (or Streamlit rerun) never loses Phase 2/3 output. Input
# widgets below are bound via stable `key=` params for the same reason --
# without a key, a widget's value resets whenever it isn't rendered on a
# given run (e.g. while the other mode is active).
if "analysis_results" not in st.session_state:
    st.session_state["analysis_results"] = {"candidate": {}, "employer": {}}
candidate_results = st.session_state["analysis_results"]["candidate"]
employer_results = st.session_state["analysis_results"]["employer"]

# ------------------------------------------------------------- candidate --

if mode == "Candidate":
    st.markdown('<div class="section-label">1. Upload &amp; target</div>', unsafe_allow_html=True)

    upl_col, jd_col = st.columns([1, 1.4])
    with upl_col:
        resume_file = st.file_uploader("Resume (PDF)", type=["pdf"], key="candidate_pdf")
        sync_single_upload(resume_file, candidate_results, "uploaded_pdf")
        has_resume = "uploaded_pdf" in candidate_results
        if resume_file is None and has_resume:
            st.caption(f"📎 Using previously uploaded **{candidate_results['uploaded_pdf']['name']}** -- upload a new file to replace it.")
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

    if st.button("🔍  Analyze", type="primary", disabled=not (has_resume and jd_text)):
        try:
            upload = candidate_results["uploaded_pdf"]
            resume_path = _save_bytes_to_temp(upload["name"], upload["bytes"])
            resume_text = parse_resume(resume_path)
            candidate_results["resume_text"] = resume_text
            candidate_results["jd_text"] = jd_text
            candidate_results["score_result"] = score_resume(resume_text, jd_text)
            candidate_results.pop("loop_result", None)  # stale after re-analyze
        except (ParseError, ValueError) as exc:
            st.error(f"Parsing failed: {exc}")

    if "resume_text" in candidate_results:
        st.markdown('<div class="section-label">2. Parsed resume (Phase 1)</div>', unsafe_allow_html=True)
        st.text_area("Extracted text", candidate_results["resume_text"], height=220, label_visibility="collapsed")

        st.markdown('<div class="section-label">3. Match score (Phase 2)</div>', unsafe_allow_html=True)
        result = candidate_results["score_result"]
        score_col, kw_col = st.columns([1, 2])
        with score_col:
            render_score_card("ATS match score", result.score)
            st.caption(f"= 50% semantic ({result.semantic_score:.1f}%) + 50% keyword ({result.keyword_score:.1f}%)")
        with kw_col:
            mcol, xcol = st.columns(2)
            with mcol:
                st.markdown("**✅ Matched keywords**")
                render_badges(result.matched_keywords, "matched")
            with xcol:
                st.markdown("**⚠️ Missing keywords**")
                render_badges(result.missing_keywords, "missing")
            if result.conceptual_gaps:
                st.markdown("**🧠 Conceptual skill gaps** (not present anywhere, even under different wording)")
                render_badges(result.conceptual_gaps, "missing")

        st.markdown('<div class="section-label">4. Rewrite + truthfulness loop (Phase 3)</div>', unsafe_allow_html=True)
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
            for log in loop_result.history:
                if log.iteration == 0:
                    st.markdown(f"🏁 &nbsp;**Iteration 0** (initial parse) -- score **{log.score:.1f}%**", unsafe_allow_html=True)
                    continue

                status_label = "✅ Passed" if log.accepted else "❌ Rejected"
                with st.expander(f"Iteration {log.iteration} · score {log.score:.1f}% · {status_label}"):
                    if log.accepted and log.changed_sections:
                        for section, (before, after) in log.changed_sections.items():
                            st.markdown(f"**Section: {section}**")
                            render_line_diff(before, after)
                    elif log.accepted:
                        st.caption("Accepted, but the section splitter found no per-section diff.")
                    else:
                        st.error(
                            f"Rejected -- {log.rejection_reason}. This draft was discarded and never scored."
                        )

            st.markdown('<div class="section-label">5. Final result</div>', unsafe_allow_html=True)
            f1, f2, f3 = st.columns(3)
            with f1:
                render_score_card("Final score", loop_result.best_score)
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

# --------------------------------------------------------------- employer --

else:
    st.markdown('<div class="section-label">1. Upload candidates &amp; job description</div>', unsafe_allow_html=True)

    upl_col, jd_col = st.columns([1, 1.4])
    with upl_col:
        resume_files = st.file_uploader(
            "Resumes (PDF)", type=["pdf"], accept_multiple_files=True, key="employer_pdfs"
        )
        sync_multi_upload(resume_files, employer_results, "uploaded_pdfs")
        has_resumes = bool(employer_results.get("uploaded_pdfs"))
        if not resume_files and has_resumes:
            names = ", ".join(u["name"] for u in employer_results["uploaded_pdfs"])
            st.caption(f"📎 Using previously uploaded: **{names}** -- upload new files to replace them.")
    with jd_col:
        jd_text = st.text_area("Job description", height=150, key="shared_jd")

    if st.button("📊  Rank", type="primary", disabled=not (has_resumes and jd_text)):
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

        st.markdown('<div class="section-label">2. Ranked candidates</div>', unsafe_allow_html=True)
        st.dataframe(
            [
                {
                    "Rank": i,
                    "Candidate": name_by_path.get(r.resume_path, r.resume_path),
                    "Score": r.score,
                    "Status": "⚠️ Parse error" if r.error else "✅ OK",
                }
                for i, r in enumerate(rankings, start=1)
            ],
            column_config={
                "Rank": st.column_config.NumberColumn("Rank", width="small"),
                "Score": st.column_config.ProgressColumn(
                    "Score", min_value=0, max_value=100, format="%.1f%%"
                ),
            },
            hide_index=True,
            use_container_width=True,
        )

        st.markdown('<div class="section-label">3. Gap summaries</div>', unsafe_allow_html=True)
        for r in rankings:
            label = name_by_path.get(r.resume_path, r.resume_path)
            status = "⚠️" if r.error else "✅"
            with st.expander(f"{status} {label} -- score {r.score:.1f}%"):
                if r.error:
                    st.error(r.error)
                    continue
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

st.markdown(
    '<div class="footnote">Thin debugging/demo surface over src/parser.py, src/scorer.py, '
    "src/agent_loop.py and src/modes.py -- no pipeline logic lives in this file.</div>",
    unsafe_allow_html=True,
)
