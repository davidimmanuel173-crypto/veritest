"""VeriTest Streamlit application."""

from itertools import combinations
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

from db import (
    get_all_exams,
    get_audit_logs,
    get_exam_responses,
    get_individual_analyses,
    get_pair_analyses,
    log_audit_event,
    save_exam_dataset,
    save_individual_analyses,
    save_pair_analyses,
)

DB_PATH = str(Path(__file__).with_name("veritest.db"))
REQUIRED_RESPONSE_COLUMNS = {"student_id", "question_id", "answer"}


@st.cache_data(show_spinner=False)
def read_csv(uploaded_file) -> pd.DataFrame:
    """Read an uploaded CSV into a cached DataFrame."""
    return pd.read_csv(uploaded_file)


def clean_optional_upload(uploaded_file) -> Optional[pd.DataFrame]:
    if uploaded_file is None:
        return None
    return read_csv(uploaded_file)


def calculate_analyses(responses: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate transparent pair and individual indicators from response rows."""
    working = responses.copy()
    if "is_correct" not in working.columns:
        raise ValueError("Responses must include is_correct for analysis.")
    working["is_correct"] = pd.to_numeric(working["is_correct"], errors="coerce").fillna(0).astype(int)

    scores = working.groupby("student_id", as_index=False)["is_correct"].agg(raw_score="sum", total_questions="count")
    scores["percentage"] = scores["raw_score"] / scores["total_questions"].replace(0, np.nan) * 100
    scores["percentage"] = scores["percentage"].fillna(0)
    mean = scores["percentage"].mean()
    std = scores["percentage"].std(ddof=0)
    scores["anomaly_z"] = 0.0 if not std else (scores["percentage"] - mean) / std
    scores["is_anomaly"] = (scores["anomaly_z"].abs() >= 2).astype(int)
    scores["irt_ability"] = (scores["percentage"] / 100).clip(0.001, 0.999).apply(lambda value: np.log(value / (1 - value)))
    scores["predicted_score"] = scores["percentage"]
    scores["notes"] = np.where(scores["is_anomaly"], "Score is at least two standard deviations from the exam mean.", "")
    individual = scores[["student_id", "raw_score", "total_questions", "percentage", "irt_ability", "predicted_score", "anomaly_z", "is_anomaly", "notes"]]

    answer_matrix = working.pivot_table(index="student_id", columns="question_id", values="answer", aggfunc="first")
    correct_matrix = working.pivot_table(index="student_id", columns="question_id", values="is_correct", aggfunc="first").fillna(0)
    pairs = []
    for student_a, student_b in combinations(correct_matrix.index, 2):
        a = correct_matrix.loc[student_a]
        b = correct_matrix.loc[student_b]
        comparable = a.index.intersection(b.index)
        if len(comparable) == 0:
            continue
        a_answers = answer_matrix.loc[student_a, comparable].astype(str)
        b_answers = answer_matrix.loc[student_b, comparable].astype(str)
        both_wrong = (a.loc[comparable] == 0) & (b.loc[comparable] == 0)
        wrong_agreement = int((both_wrong & (a_answers == b_answers)).sum())
        total_wrong_both = int(both_wrong.sum())
        agreement_rate = wrong_agreement / total_wrong_both if total_wrong_both else 0.0
        pairs.append({
            "student_a": student_a,
            "student_b": student_b,
            "wrong_agreement": wrong_agreement,
            "total_wrong_both": total_wrong_both,
            "total_questions": len(comparable),
            "wesolowsky_z": agreement_rate,
            "p_value": 1 - agreement_rate,
            "composite_score": agreement_rate,
            "risk_level": "High" if agreement_rate >= 0.75 and total_wrong_both >= 2 else "Review" if agreement_rate >= 0.5 else "Low",
            "confidence": "Standard" if total_wrong_both >= 3 else "Limited",
            "flags_json": {"method": "wrong-answer agreement"},
        })
    return pd.DataFrame(pairs), individual


def render_ingest() -> None:
    st.subheader("Ingest an exam")
    with st.form("ingest_form"):
        title = st.text_input("Exam title", placeholder="Midterm 1")
        course = st.text_input("Course", placeholder="BIO-201")
        responses_file = st.file_uploader("Responses CSV", type="csv", help="Required columns: student_id, question_id, answer, is_correct")
        answer_key_file = st.file_uploader("Answer key CSV (optional)", type="csv")
        baselines_file = st.file_uploader("Baselines CSV (optional)", type="csv")
        submitted = st.form_submit_button("Import exam", type="primary")

    if not submitted:
        return
    if not title.strip() or responses_file is None:
        st.error("Provide an exam title and responses CSV.")
        return
    responses = read_csv(responses_file)
    missing = REQUIRED_RESPONSE_COLUMNS - set(responses.columns)
    if missing:
        st.error(f"Responses CSV is missing: {', '.join(sorted(missing))}.")
        return
    if "is_correct" not in responses.columns:
        st.warning("No is_correct column found. The exam will be stored, but analysis requires it.")
    exam_id = save_exam_dataset(title.strip(), course.strip(), responses, clean_optional_upload(answer_key_file), clean_optional_upload(baselines_file), db_path=DB_PATH)
    log_audit_event("instructor", "import_exam", "exam", str(exam_id), f"Imported {len(responses)} response rows.", db_path=DB_PATH)
    st.success(f"Imported exam #{exam_id}.")


def render_dashboard(exam_id: int) -> None:
    responses = get_exam_responses(exam_id, db_path=DB_PATH)
    if responses.empty:
        st.info("This exam has no responses yet.")
        return
    first, second, third = st.columns(3)
    first.metric("Students", responses["student_id"].nunique())
    second.metric("Questions", responses["question_id"].nunique())
    if "is_correct" in responses:
        third.metric("Responses", len(responses))
        st.dataframe(responses.head(100), use_container_width=True, hide_index=True)
    else:
        third.metric("Responses", len(responses))


def render_analysis(exam_id: int) -> None:
    responses = get_exam_responses(exam_id, db_path=DB_PATH)
    if "is_correct" not in responses.columns or responses.empty:
        st.info("Analysis requires response data with an is_correct column.")
        return
    if st.button("Run analysis", type="primary"):
        pairs, individuals = calculate_analyses(responses)
        save_pair_analyses(exam_id, pairs, db_path=DB_PATH) if not pairs.empty else None
        save_individual_analyses(exam_id, individuals, db_path=DB_PATH)
        log_audit_event("instructor", "run_analysis", "exam", str(exam_id), db_path=DB_PATH)
        st.success("Analysis saved.")
    pair_results = get_pair_analyses(exam_id, db_path=DB_PATH)
    individual_results = get_individual_analyses(exam_id, db_path=DB_PATH)
    st.write("Pair indicators")
    st.dataframe(pair_results, use_container_width=True, hide_index=True)
    st.write("Individual indicators")
    st.dataframe(individual_results, use_container_width=True, hide_index=True)


def main() -> None:
    st.set_page_config(page_title="VeriTest", page_icon="V", layout="wide")
    st.title("VeriTest")
    st.caption("Exam integrity review workspace")
    exams = get_all_exams(db_path=DB_PATH)
    with st.sidebar:
        st.header("Workspace")
        page = st.radio("View", ["Dashboard", "Import", "Audit log"], label_visibility="collapsed")
        exam_id = st.selectbox("Exam", [exam["id"] for exam in exams], format_func=lambda value: next(exam["title"] for exam in exams if exam["id"] == value)) if exams else None

    if page == "Import":
        render_ingest()
    elif page == "Audit log":
        st.subheader("Audit log")
        st.dataframe(get_audit_logs(db_path=DB_PATH), use_container_width=True, hide_index=True)
    elif exam_id is None:
        st.info("Import an exam to begin.")
    else:
        render_dashboard(exam_id)
        st.divider()
        render_analysis(exam_id)


if __name__ == "__main__":
    main()
