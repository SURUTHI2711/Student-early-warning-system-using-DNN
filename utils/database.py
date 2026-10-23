"""
Database Access Layer
Project: At-Risk Student Early Warning System Using Deep Neural Networks
Provides query abstractions, CRUD operations, and analytical aggregations.
"""

import sqlite3
import json
from datetime import datetime
import config

def get_db_connection():
    """Returns a SQLite connection with Row factory enabled."""
    conn = sqlite3.connect(config.DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn

# ----------------- User Queries -----------------

def get_user_by_username(username):
    conn = get_db_connection()
    user = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
    conn.close()
    return user

def get_user_by_id(user_id):
    conn = get_db_connection()
    user = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    conn.close()
    return user

# ----------------- Student Queries -----------------

def get_all_students(search=None, department=None, risk_level=None):
    """
    Retrieves all students joined with their latest prediction and latest academic record.
    Supports filtering by keyword search, department, and risk level.
    """
    conn = get_db_connection()
    
    query = """
    SELECT s.*, 
           p.risk_level, p.risk_code, p.confidence, p.low_probability, p.medium_probability, 
           p.high_probability, p.prediction_date, p.id as prediction_id,
           a.attendance_percentage, a.internal_marks, a.assignment_score, a.previous_gpa, 
           a.backlogs, a.study_hours_weekly, a.engagement_score
    FROM students s
    LEFT JOIN (
        SELECT * FROM predictions WHERE id IN (
            SELECT MAX(id) FROM predictions GROUP BY student_id
        )
    ) p ON s.student_id = p.student_id
    LEFT JOIN (
        SELECT * FROM academic_records WHERE id IN (
            SELECT MAX(id) FROM academic_records GROUP BY student_id
        )
    ) a ON s.student_id = a.student_id
    WHERE 1=1
    """
    params = []
    
    if search:
        query += " AND (s.name LIKE ? OR s.student_id LIKE ? OR s.email LIKE ?)"
        term = f"%{search}%"
        params.extend([term, term, term])
        
    if department and department != 'All':
        query += " AND s.department = ?"
        params.append(department)
        
    if risk_level and risk_level != 'All':
        query += " AND p.risk_level = ?"
        params.append(risk_level)
        
    query += " ORDER BY s.id DESC"
    
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return rows

def get_student_by_id(student_id):
    """Retrieves single student demographic record."""
    conn = get_db_connection()
    student = conn.execute("SELECT * FROM students WHERE student_id = ?", (student_id,)).fetchone()
    conn.close()
    return student

def get_student_full_profile(student_id):
    """Retrieves student along with latest academic record and latest prediction."""
    conn = get_db_connection()
    student = conn.execute("SELECT * FROM students WHERE student_id = ?", (student_id,)).fetchone()
    if not student:
        conn.close()
        return None, None, None
        
    latest_acad = conn.execute(
        "SELECT * FROM academic_records WHERE student_id = ? ORDER BY id DESC LIMIT 1",
        (student_id,)
    ).fetchone()
    
    latest_pred = conn.execute(
        "SELECT * FROM predictions WHERE student_id = ? ORDER BY id DESC LIMIT 1",
        (student_id,)
    ).fetchone()
    
    conn.close()
    return student, latest_acad, latest_pred

def get_student_academic_history(student_id):
    conn = get_db_connection()
    records = conn.execute(
        "SELECT * FROM academic_records WHERE student_id = ? ORDER BY id DESC",
        (student_id,)
    ).fetchall()
    conn.close()
    return records

def get_student_prediction_history(student_id):
    conn = get_db_connection()
    preds = conn.execute(
        "SELECT * FROM predictions WHERE student_id = ? ORDER BY id DESC",
        (student_id,)
    ).fetchall()
    conn.close()
    return preds

def create_student(student_id, name, email, department, year, semester, contact_number=""):
    conn = get_db_connection()
    try:
        conn.execute(
            """INSERT INTO students (student_id, name, email, department, year, semester, contact_number)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (student_id, name, email, department, int(year), int(semester), contact_number)
        )
        conn.commit()
        return True, "Student registered successfully."
    except sqlite3.IntegrityError as e:
        return False, f"Error: Student ID or Email already exists ({e})."
    finally:
        conn.close()

# ----------------- Academic & Prediction Records -----------------

def add_academic_record(student_id, acad_dict):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        """INSERT INTO academic_records 
           (student_id, attendance_percentage, internal_marks, assignment_score, previous_gpa,
            study_hours_weekly, backlogs, assignment_submission_rate, engagement_score,
            previous_failures, exam_score, semester_term)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (student_id, acad_dict['attendance_percentage'], acad_dict['internal_marks'],
         acad_dict['assignment_score'], acad_dict['previous_gpa'], acad_dict['study_hours_weekly'],
         int(acad_dict['backlogs']), acad_dict['assignment_submission_rate'], acad_dict['engagement_score'],
         int(acad_dict['previous_failures']), acad_dict['exam_score'],
         acad_dict.get('semester_term', 'Current Semester'))
    )
    rec_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return rec_id

def save_prediction(student_id, pred_res, academic_record_id=None, predicted_by="Faculty Advisor"):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        """INSERT INTO predictions
           (student_id, risk_level, risk_code, confidence, low_probability, medium_probability,
            high_probability, predicted_by, recommendations, risk_factors, explanation, academic_record_id)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (student_id, pred_res['risk_level'], pred_res['risk_code'], pred_res['confidence'],
         pred_res['probabilities']['low'], pred_res['probabilities']['medium'],
         pred_res['probabilities']['high'], predicted_by,
         json.dumps(pred_res['recommendations']), json.dumps(pred_res['risk_factors']),
         pred_res['explanation'], academic_record_id)
    )
    pred_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return pred_id

def get_prediction_by_id(prediction_id):
    conn = get_db_connection()
    query = """
    SELECT p.*, s.name, s.email, s.department, s.year, s.semester, s.contact_number,
           a.attendance_percentage, a.internal_marks, a.assignment_score, a.previous_gpa,
           a.study_hours_weekly, a.backlogs, a.assignment_submission_rate, a.engagement_score,
           a.previous_failures, a.exam_score, a.semester_term
    FROM predictions p
    JOIN students s ON p.student_id = s.student_id
    LEFT JOIN academic_records a ON p.academic_record_id = a.id
    WHERE p.id = ?
    """
    row = conn.execute(query, (prediction_id,)).fetchone()
    conn.close()
    return row

def get_all_predictions(limit=100, risk_level=None):
    conn = get_db_connection()
    query = """
    SELECT p.*, s.name, s.department, a.attendance_percentage, a.internal_marks, a.backlogs
    FROM predictions p
    JOIN students s ON p.student_id = s.student_id
    LEFT JOIN academic_records a ON p.academic_record_id = a.id
    WHERE 1=1
    """
    params = []
    if risk_level and risk_level != 'All':
        query += " AND p.risk_level = ?"
        params.append(risk_level)
        
    query += " ORDER BY p.id DESC LIMIT ?"
    params.append(limit)
    
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return rows

# ----------------- Dashboard & Analytics Aggregations -----------------

def get_dashboard_stats():
    """Computes summary KPI metrics for the faculty/admin dashboard."""
    conn = get_db_connection()
    
    total_students = conn.execute("SELECT COUNT(*) FROM students").fetchone()[0]
    
    # Latest predictions counts
    risk_counts = conn.execute("""
        SELECT p.risk_level, COUNT(*) as cnt
        FROM (
            SELECT student_id, risk_level FROM predictions 
            WHERE id IN (SELECT MAX(id) FROM predictions GROUP BY student_id)
        ) p
        GROUP BY p.risk_level
    """).fetchall()
    
    risk_dict = {'LOW RISK': 0, 'MEDIUM RISK': 0, 'HIGH RISK': 0}
    for row in risk_counts:
        risk_dict[row['risk_level']] = row['cnt']
        
    # Average Academic KPIs from latest records
    acad_averages = conn.execute("""
        SELECT 
            AVG(a.attendance_percentage) as avg_att,
            AVG(a.internal_marks) as avg_marks,
            AVG(a.previous_gpa) as avg_gpa,
            SUM(a.backlogs) as total_backlogs
        FROM academic_records a
        WHERE a.id IN (SELECT MAX(id) FROM academic_records GROUP BY student_id)
    """).fetchone()
    
    # Recent Predictions (last 6)
    recent_preds = conn.execute("""
        SELECT p.*, s.name, s.department, a.attendance_percentage, a.internal_marks
        FROM predictions p
        JOIN students s ON p.student_id = s.student_id
        LEFT JOIN academic_records a ON p.academic_record_id = a.id
        ORDER BY p.id DESC LIMIT 6
    """).fetchall()
    
    # High Risk Students Requiring Attention
    attention_students = conn.execute("""
        SELECT s.*, p.confidence, p.high_probability, a.attendance_percentage, a.internal_marks, a.backlogs
        FROM students s
        JOIN predictions p ON s.student_id = p.student_id
        LEFT JOIN academic_records a ON p.academic_record_id = a.id
        WHERE p.id IN (SELECT MAX(id) FROM predictions GROUP BY student_id)
          AND p.risk_level = 'HIGH RISK'
        ORDER BY p.high_probability DESC LIMIT 5
    """).fetchall()
    
    conn.close()
    
    return {
        'total_students': total_students,
        'low_risk_count': risk_dict['LOW RISK'],
        'medium_risk_count': risk_dict['MEDIUM RISK'],
        'high_risk_count': risk_dict['HIGH RISK'],
        'avg_attendance': round(acad_averages['avg_att'] or 0.0, 1),
        'avg_marks': round(acad_averages['avg_marks'] or 0.0, 1),
        'avg_gpa': round(acad_averages['avg_gpa'] or 0.0, 2),
        'total_backlogs': int(acad_averages['total_backlogs'] or 0),
        'recent_predictions': recent_preds,
        'attention_students': attention_students
    }

def get_analytics_data():
    """Prepares structured analytics payload for interactive Chart.js charts."""
    conn = get_db_connection()
    
    # 1. Department Risk Breakdown
    dept_rows = conn.execute("""
        SELECT s.department, p.risk_level, COUNT(*) as cnt
        FROM students s
        JOIN (
            SELECT * FROM predictions WHERE id IN (SELECT MAX(id) FROM predictions GROUP BY student_id)
        ) p ON s.student_id = p.student_id
        GROUP BY s.department, p.risk_level
    """).fetchall()
    
    departments = sorted(list(set([r['department'] for r in dept_rows])))
    dept_data = {
        'labels': departments,
        'low': [0] * len(departments),
        'medium': [0] * len(departments),
        'high': [0] * len(departments)
    }
    
    for r in dept_rows:
        d_idx = departments.index(r['department'])
        if r['risk_level'] == 'LOW RISK':
            dept_data['low'][d_idx] = r['cnt']
        elif r['risk_level'] == 'MEDIUM RISK':
            dept_data['medium'][d_idx] = r['cnt']
        elif r['risk_level'] == 'HIGH RISK':
            dept_data['high'][d_idx] = r['cnt']
            
    # 2. Scatter / Distribution of Attendance vs Marks
    points = conn.execute("""
        SELECT s.name, s.student_id, a.attendance_percentage, a.internal_marks, p.risk_level
        FROM students s
        JOIN academic_records a ON s.student_id = a.student_id AND a.id IN (SELECT MAX(id) FROM academic_records GROUP BY student_id)
        JOIN predictions p ON s.student_id = p.student_id AND p.id IN (SELECT MAX(id) FROM predictions GROUP BY student_id)
    """).fetchall()
    
    scatter_data = []
    for p in points:
        scatter_data.append({
            'x': round(p['attendance_percentage'], 1),
            'y': round(p['internal_marks'], 1),
            'name': p['name'],
            'id': p['student_id'],
            'risk': p['risk_level']
        })
        
    conn.close()
    
    return {
        'departments': dept_data,
        'scatter': scatter_data
    }
