-- Schema for At-Risk Student Early Warning System
-- Database: students.db

CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL CHECK(role IN ('admin', 'faculty', 'student')),
    student_ref_id TEXT,
    full_name TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS students (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id TEXT UNIQUE NOT NULL,
    name TEXT NOT NULL,
    email TEXT UNIQUE NOT NULL,
    department TEXT NOT NULL,
    year INTEGER NOT NULL CHECK(year BETWEEN 1 AND 4),
    semester INTEGER NOT NULL CHECK(semester BETWEEN 1 AND 8),
    contact_number TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS academic_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id TEXT NOT NULL,
    attendance_percentage REAL NOT NULL,
    internal_marks REAL NOT NULL,
    assignment_score REAL NOT NULL,
    previous_gpa REAL NOT NULL,
    study_hours_weekly REAL NOT NULL,
    backlogs INTEGER NOT NULL DEFAULT 0,
    assignment_submission_rate REAL NOT NULL,
    engagement_score REAL NOT NULL,
    previous_failures INTEGER NOT NULL DEFAULT 0,
    exam_score REAL NOT NULL,
    semester_term TEXT DEFAULT 'Current Semester',
    recorded_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (student_id) REFERENCES students(student_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id TEXT NOT NULL,
    risk_level TEXT NOT NULL,
    risk_code INTEGER NOT NULL,
    confidence REAL NOT NULL,
    low_probability REAL NOT NULL,
    medium_probability REAL NOT NULL,
    high_probability REAL NOT NULL,
    predicted_by TEXT DEFAULT 'Faculty',
    recommendations TEXT,
    risk_factors TEXT,
    explanation TEXT,
    academic_record_id INTEGER,
    prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (student_id) REFERENCES students(student_id) ON DELETE CASCADE,
    FOREIGN KEY (academic_record_id) REFERENCES academic_records(id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_student_id ON students(student_id);
CREATE INDEX IF NOT EXISTS idx_academic_student_id ON academic_records(student_id);
CREATE INDEX IF NOT EXISTS idx_predictions_student_id ON predictions(student_id);
