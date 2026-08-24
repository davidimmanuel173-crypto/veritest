import pandas as pd

from db import (
    get_audit_logs,
    get_exam_responses,
    get_individual_analyses,
    get_pair_analyses,
    log_audit_event,
    save_exam_dataset,
    save_individual_analyses,
    save_pair_analyses,
)


def test_exam_and_analysis_round_trip(tmp_path):
    db_path = str(tmp_path / "veritest.db")
    responses = pd.DataFrame(
        [
            {"student_id": "S1", "question_id": "Q1", "answer": "A", "is_correct": 1},
            {"student_id": "S1", "question_id": "Q2", "answer": "B", "is_correct": 0},
            {"student_id": "S2", "question_id": "Q1", "answer": "A", "is_correct": 1},
            {"student_id": "S2", "question_id": "Q2", "answer": "B", "is_correct": 0},
        ]
    )

    exam_id = save_exam_dataset("Midterm", "BIO-201", responses, db_path=db_path)
    stored_responses = get_exam_responses(exam_id, db_path=db_path)
    assert stored_responses[responses.columns].equals(responses)

    pair_results = pd.DataFrame([{"student_a": "S1", "student_b": "S2", "composite_score": 0.9}])
    individual_results = pd.DataFrame([{"student_id": "S1", "raw_score": 1, "total_questions": 2}])
    save_pair_analyses(exam_id, pair_results, db_path=db_path)
    save_individual_analyses(exam_id, individual_results, db_path=db_path)
    log_audit_event("instructor", "run_analysis", "exam", str(exam_id), db_path=db_path)

    assert len(get_pair_analyses(exam_id, db_path=db_path)) == 1
    assert len(get_individual_analyses(exam_id, db_path=db_path)) == 1
    assert get_audit_logs(db_path=db_path).iloc[0]["action"] == "run_analysis"
