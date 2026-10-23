import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
DATABASE_DIR = os.path.join(BASE_DIR, 'database')
STATIC_DIR = os.path.join(BASE_DIR, 'static')
TEMPLATES_DIR = os.path.join(BASE_DIR, 'templates')
PLOTS_DIR = os.path.join(STATIC_DIR, 'plots')

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATABASE_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

DATASET_PATH = os.path.join(DATA_DIR, 'student_data.csv')
DATABASE_PATH = os.path.join(DATABASE_DIR, 'students.db')
MODEL_PATH = os.path.join(MODELS_DIR, 'student_risk_model.keras')
SCALER_PATH = os.path.join(MODELS_DIR, 'scaler.pkl')
METRICS_PATH = os.path.join(MODELS_DIR, 'metrics.json')
BASELINE_MODELS_PATH = os.path.join(MODELS_DIR, 'baseline_models.pkl')

SECRET_KEY = os.environ.get('SECRET_KEY', 'at-risk-student-early-warning-dnn-secret-key-cse2026')
DEBUG = True

FEATURE_COLUMNS = [
    'attendance_percentage',
    'internal_marks',
    'assignment_score',
    'previous_gpa',
    'study_hours_weekly',
    'backlogs',
    'assignment_submission_rate',
    'engagement_score',
    'previous_failures',
    'exam_score'
]

TARGET_COLUMN = 'risk_level'

RISK_LABELS = {
    0: 'LOW RISK',
    1: 'MEDIUM RISK',
    2: 'HIGH RISK'
}

RISK_COLORS = {
    'LOW RISK': '#10b981',      # Emerald Green
    'MEDIUM RISK': '#f59e0b',   # Amber / Orange
    'HIGH RISK': '#ef4444'      # Crimson Red
}

DISCLAIMER_NOTICE = (
    "Educational Early Warning Notice: This system uses Deep Neural Network predictive "
    "modeling as a supportive academic indicator for mentoring and timely intervention. "
    "Predictions are not definitive evaluations, disciplinary judgments, or psychological diagnoses."
)
