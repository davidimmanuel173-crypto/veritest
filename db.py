"""Database management and SQLite models for VeriTest.
Provides persistence for exams, raw student responses, answer keys,
historical baselines, seating coordinates, statistical analysis results,
and compliance audit logs.
"""

import sqlite3
import json
from datetime import datetime
from typing import Optional, List, Dict, Any
import pandas as pd

DB_PATH = "veritest.db"


def get_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    """Returns a SQLite connection with Row factory enabled."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def validate_response_data(responses_df: pd.DataFrame) -> None:
    """Rejects response data that cannot be analyzed or safely stored."""
    required = {"student_id", "question_id", "answer"}
    missing = required - set(responses_df.columns)
    if missing:
        raise ValueError(f"Missing required response columns: {', '.join(sorted(missing))}")
    if responses_df.empty:
        raise ValueError("The responses file is empty.")
    if responses_df[list(required)].isna().any().any():
        raise ValueError("student_id, question_id, and answer cannot be blank.")
    if responses_df["student_id"].astype(str).str.strip().eq("").any() or responses_df["question_id"].astype(str).str.strip().eq("").any():
        raise ValueError("student_id and question_id cannot be blank.")
    duplicate_rows = responses_df.duplicated(["student_id", "question_id"], keep=False)
    if duplicate_rows.any():
        raise ValueError("Each student_id/question_id pair must appear only once.")
    if "is_correct" in responses_df.columns:
        values = pd.to_numeric(responses_df["is_correct"], errors="coerce")
        if values.isna().any() or not values.isin([0, 1]).all():
            raise ValueError("is_correct must contain only 0 or 1 values.")


def init_db(db_path: str = DB_PATH) -> None:
    """Initializes the SQLite database schema if not already present."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()

        # Exams table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS exams (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                course_name TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                num_questions INTEGER DEFAULT 0,
                num_students INTEGER DEFAULT 0
            )
        """)

        # Student responses table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS responses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                exam_id INTEGER NOT NULL,
                student_id TEXT NOT NULL,
                question_id TEXT NOT NULL,
                answer TEXT,
                is_correct INTEGER,
                response_time_sec REAL,
                seat_id TEXT,
                timestamp TEXT,
                FOREIGN KEY (exam_id) REFERENCES exams (id) ON DELETE CASCADE
            )
        """)

        # Answer key table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS answer_keys (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                exam_id INTEGER NOT NULL,
                question_id TEXT NOT NULL,
                correct_answer TEXT NOT NULL,
                distractors TEXT,
                FOREIGN KEY (exam_id) REFERENCES exams (id) ON DELETE CASCADE
            )
        """)

        # Student historical baselines table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS student_baselines (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                exam_id INTEGER NOT NULL,
                student_id TEXT NOT NULL,
                gpa REAL,
                prior_score REAL,
                FOREIGN KEY (exam_id) REFERENCES exams (id) ON DELETE CASCADE
            )
        """)

        # Seating coordinates table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS seat_maps (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                exam_id INTEGER NOT NULL,
                seat_id TEXT NOT NULL,
                row_num INTEGER,
                col_num INTEGER,
                x_pos REAL,
                y_pos REAL,
                FOREIGN KEY (exam_id) REFERENCES exams (id) ON DELETE CASCADE
            )
        """)

        # Computed pair analyses table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS pair_analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                exam_id INTEGER NOT NULL,
                student_a TEXT NOT NULL,
                student_b TEXT NOT NULL,
                wrong_agreement INTEGER DEFAULT 0,
                total_wrong_both INTEGER DEFAULT 0,
                total_questions INTEGER DEFAULT 0,
                wesolowsky_z REAL DEFAULT 0.0,
                p_value REAL DEFAULT 1.0,
                timing_corr REAL DEFAULT 0.0,
                timing_p_value REAL DEFAULT 1.0,
                seat_distance REAL DEFAULT -1.0,
                seat_weight REAL DEFAULT 1.0,
                composite_score REAL DEFAULT 0.0,
                risk_level TEXT DEFAULT 'Low',
                confidence TEXT DEFAULT 'Standard',
                flags_json TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (exam_id) REFERENCES exams (id) ON DELETE CASCADE
            )
        """)

        # Computed individual IRT / anomaly analyses table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS individual_analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                exam_id INTEGER NOT NULL,
                student_id TEXT NOT NULL,
                raw_score INTEGER DEFAULT 0,
                total_questions INTEGER DEFAULT 0,
                percentage REAL DEFAULT 0.0,
                irt_ability REAL DEFAULT 0.0,
                predicted_score REAL DEFAULT 0.0,
                anomaly_z REAL DEFAULT 0.0,
                is_anomaly INTEGER DEFAULT 0,
                gpa REAL,
                prior_score REAL,
                notes TEXT,
                FOREIGN KEY (exam_id) REFERENCES exams (id) ON DELETE CASCADE
            )
        """)

        # Audit logs table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS audit_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                user_role TEXT NOT NULL,
                action TEXT NOT NULL,
                target_type TEXT,
                target_id TEXT,
                details TEXT
            )
        """)

        conn.commit()


def save_exam_dataset(
    title: str,
    course_name: str,
    responses_df: pd.DataFrame,
    answer_key_df: Optional[pd.DataFrame] = None,
    baselines_df: Optional[pd.DataFrame] = None,
    seat_map_df: Optional[pd.DataFrame] = None,
    db_path: str = DB_PATH,
) -> int:
    """Saves a newly ingested exam and its associated data tables into SQLite."""
    validate_response_data(responses_df)
    init_db(db_path)
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        num_students = int(responses_df["student_id"].nunique()) if "student_id" in responses_df else 0
        num_questions = int(responses_df["question_id"].nunique()) if "question_id" in responses_df else 0

        cursor.execute(
            "INSERT INTO exams (title, course_name, num_questions, num_students) VALUES (?, ?, ?, ?)",
            (title, course_name, num_questions, num_students),
        )
        exam_id = cursor.lastrowid

        # Insert responses
        for _, row in responses_df.iterrows():
            cursor.execute(
                """
                INSERT INTO responses (
                    exam_id, student_id, question_id, answer, is_correct, response_time_sec, seat_id, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    exam_id,
                    str(row.get("student_id", "")),
                    str(row.get("question_id", "")),
                    str(row.get("answer", "")),
                    int(row.get("is_correct", 0)) if pd.notna(row.get("is_correct")) else None,
                    float(row.get("response_time_sec", 0.0)) if pd.notna(row.get("response_time_sec")) else None,
                    str(row.get("seat_id", "")) if pd.notna(row.get("seat_id")) else None,
                    str(row.get("timestamp", "")) if pd.notna(row.get("timestamp")) else None,
                ),
            )

        # Insert answer keys if provided
        if answer_key_df is not None and not answer_key_df.empty:
            for _, row in answer_key_df.iterrows():
                distractors = row.get("distractors", "")
                if isinstance(distractors, list):
                    distractors = json.dumps(distractors)
                cursor.execute(
                    """
                    INSERT INTO answer_keys (exam_id, question_id, correct_answer, distractors)
                    VALUES (?, ?, ?, ?)
                    """,
                    (exam_id, str(row.get("question_id", "")), str(row.get("correct_answer", "")), str(distractors)),
                )

        # Insert student baselines if provided
        if baselines_df is not None and not baselines_df.empty:
            for _, row in baselines_df.iterrows():
                cursor.execute(
                    """
                    INSERT INTO student_baselines (exam_id, student_id, gpa, prior_score)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        exam_id,
                        str(row.get("student_id", "")),
                        float(row.get("gpa", 0.0)) if pd.notna(row.get("gpa")) else None,
                        float(row.get("prior_score", 0.0)) if pd.notna(row.get("prior_score")) else None,
                    ),
                )

        # Insert seat maps if provided
        if seat_map_df is not None and not seat_map_df.empty:
            for _, row in seat_map_df.iterrows():
                cursor.execute(
                    """
                    INSERT INTO seat_maps (exam_id, seat_id, row_num, col_num, x_pos, y_pos)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        exam_id,
                        str(row.get("seat_id", "")),
                        int(row.get("row_num", 0)) if pd.notna(row.get("row_num")) else None,
                        int(row.get("col_num", 0)) if pd.notna(row.get("col_num")) else None,
                        float(row.get("x_pos", 0.0)) if pd.notna(row.get("x_pos")) else None,
                        float(row.get("y_pos", 0.0)) if pd.notna(row.get("y_pos")) else None,
                    ),
                )

        conn.commit()
        return exam_id


def get_all_exams(db_path: str = DB_PATH) -> List[Dict[str, Any]]:
    """Retrieves metadata list for all stored exams."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM exams ORDER BY id DESC")
        rows = cursor.fetchall()
        return [dict(r) for r in rows]


def get_latest_exam_id(db_path: str = DB_PATH) -> Optional[int]:
    """Retrieves the ID of the most recently created exam."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM exams ORDER BY id DESC LIMIT 1")
        row = cursor.fetchone()
        return row["id"] if row else None


def get_exam_responses(exam_id: int, db_path: str = DB_PATH) -> pd.DataFrame:
    """Retrieves full student response DataFrame for an exam."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        return pd.read_sql_query(
            "SELECT student_id, question_id, answer, is_correct, response_time_sec, seat_id, timestamp "
            "FROM responses WHERE exam_id = ? ORDER BY student_id, question_id",
            conn,
            params=(exam_id,),
        )


def get_exam_answer_key(exam_id: int, db_path: str = DB_PATH) -> pd.DataFrame:
    """Retrieves answer key DataFrame for an exam."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        return pd.read_sql_query(
            "SELECT question_id, correct_answer, distractors FROM answer_keys WHERE exam_id = ?",
            conn,
            params=(exam_id,),
        )


def get_exam_baselines(exam_id: int, db_path: str = DB_PATH) -> pd.DataFrame:
    """Retrieves student baseline performance data."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        return pd.read_sql_query(
            "SELECT student_id, gpa, prior_score FROM student_baselines WHERE exam_id = ?",
            conn,
            params=(exam_id,),
        )


def get_exam_seat_map(exam_id: int, db_path: str = DB_PATH) -> pd.DataFrame:
    """Retrieves seating coordinates for an exam."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        return pd.read_sql_query(
            "SELECT seat_id, row_num, col_num, x_pos, y_pos FROM seat_maps WHERE exam_id = ?",
            conn,
            params=(exam_id,),
        )


def save_pair_analyses(exam_id: int, pairs_df: pd.DataFrame, db_path: str = DB_PATH) -> None:
    """Saves computed pair-wise statistical detection results."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM pair_analyses WHERE exam_id = ?", (exam_id,))
        for _, row in pairs_df.iterrows():
            flags = row.get("flags_json", {})
            if isinstance(flags, (dict, list)):
                flags = json.dumps(flags)
            cursor.execute(
                """
                INSERT INTO pair_analyses (
                    exam_id, student_a, student_b, wrong_agreement, total_wrong_both, total_questions,
                    wesolowsky_z, p_value, timing_corr, timing_p_value, seat_distance, seat_weight,
                    composite_score, risk_level, confidence, flags_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    exam_id,
                    str(row["student_a"]),
                    str(row["student_b"]),
                    int(row.get("wrong_agreement", 0)),
                    int(row.get("total_wrong_both", 0)),
                    int(row.get("total_questions", 0)),
                    float(row.get("wesolowsky_z", 0.0)),
                    float(row.get("p_value", 1.0)),
                    float(row.get("timing_corr", 0.0)),
                    float(row.get("timing_p_value", 1.0)),
                    float(row.get("seat_distance", -1.0)),
                    float(row.get("seat_weight", 1.0)),
                    float(row.get("composite_score", 0.0)),
                    str(row.get("risk_level", "Low")),
                    str(row.get("confidence", "Standard")),
                    str(flags),
                ),
            )
        conn.commit()


def get_pair_analyses(exam_id: int, db_path: str = DB_PATH) -> pd.DataFrame:
    """Retrieves computed pair analysis results for an exam."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        return pd.read_sql_query(
            "SELECT student_a, student_b, wrong_agreement, total_wrong_both, total_questions, "
            "wesolowsky_z, p_value, timing_corr, timing_p_value, seat_distance, seat_weight, "
            "composite_score, risk_level, confidence, flags_json, created_at "
            "FROM pair_analyses WHERE exam_id = ? ORDER BY composite_score DESC, p_value ASC",
            conn,
            params=(exam_id,),
        )


def save_individual_analyses(exam_id: int, ind_df: pd.DataFrame, db_path: str = DB_PATH) -> None:
    """Saves computed individual student IRT and anomaly results."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM individual_analyses WHERE exam_id = ?", (exam_id,))
        for _, row in ind_df.iterrows():
            cursor.execute(
                """
                INSERT INTO individual_analyses (
                    exam_id, student_id, raw_score, total_questions, percentage,
                    irt_ability, predicted_score, anomaly_z, is_anomaly, gpa, prior_score, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    exam_id,
                    str(row["student_id"]),
                    int(row.get("raw_score", 0)),
                    int(row.get("total_questions", 0)),
                    float(row.get("percentage", 0.0)),
                    float(row.get("irt_ability", 0.0)),
                    float(row.get("predicted_score", 0.0)),
                    float(row.get("anomaly_z", 0.0)),
                    int(row.get("is_anomaly", 0)),
                    float(row.get("gpa", 0.0)) if pd.notna(row.get("gpa")) else None,
                    float(row.get("prior_score", 0.0)) if pd.notna(row.get("prior_score")) else None,
                    str(row.get("notes", "")),
                ),
            )
        conn.commit()


def get_individual_analyses(exam_id: int, db_path: str = DB_PATH) -> pd.DataFrame:
    """Retrieves computed individual analysis results for an exam."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        return pd.read_sql_query(
            "SELECT student_id, raw_score, total_questions, percentage, irt_ability, "
            "predicted_score, anomaly_z, is_anomaly, gpa, prior_score, notes "
            "FROM individual_analyses WHERE exam_id = ? ORDER BY anomaly_z DESC",
            conn,
            params=(exam_id,),
        )


def log_audit_event(
    user_role: str,
    action: str,
    target_type: Optional[str] = None,
    target_id: Optional[str] = None,
    details: Optional[str] = None,
    db_path: str = DB_PATH,
) -> None:
    """Writes an immutable compliance audit record to SQLite."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO audit_logs (user_role, action, target_type, target_id, details)
            VALUES (?, ?, ?, ?, ?)
            """,
            (user_role, action, target_type, target_id, details),
        )
        conn.commit()


def get_audit_logs(limit: int = 100, db_path: str = DB_PATH) -> pd.DataFrame:
    """Retrieves recent audit logs for review and compliance oversight."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        return pd.read_sql_query(
            "SELECT id, timestamp, user_role, action, target_type, target_id, details "
            "FROM audit_logs ORDER BY id DESC LIMIT ?",
            conn,
            params=(limit,),
        )


def purge_all_data(db_path: str = DB_PATH) -> None:
    """Compliance purge: permanently deletes all exam, student, response, analysis, and audit records."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM pair_analyses")
        cursor.execute("DELETE FROM individual_analyses")
        cursor.execute("DELETE FROM responses")
        cursor.execute("DELETE FROM answer_keys")
        cursor.execute("DELETE FROM student_baselines")
        cursor.execute("DELETE FROM seat_maps")
        cursor.execute("DELETE FROM exams")
        cursor.execute("DELETE FROM audit_logs")
        conn.commit()
