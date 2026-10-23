"""
Main Flask Application
Project: At-Risk Student Early Warning System Using Deep Neural Networks
"""

import os
import json
from functools import wraps
from flask import (
    Flask, render_template, request, redirect, url_for,
    flash, session, jsonify, abort
)
from werkzeug.security import check_password_hash

import config
from utils.database import (
    get_user_by_username, get_user_by_id, get_all_students,
    get_student_by_id, get_student_full_profile, get_student_academic_history,
    get_student_prediction_history, create_student, add_academic_record,
    save_prediction, get_prediction_by_id, get_all_predictions,
    get_dashboard_stats, get_analytics_data
)
from utils.preprocessing import FEATURE_METADATA
from utils.prediction import predict_student_risk
from utils.report_generator import format_student_report_data

app = Flask(__name__)
app.config['SECRET_KEY'] = config.SECRET_KEY
app.config['DEBUG'] = config.DEBUG

# ----------------- Auth Decorators -----------------

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash("Please log in to access this page.", "warning")
            return redirect(url_for('login', next=request.url))
        return f(*args, **kwargs)
    return decorated_function

def role_required(*allowed_roles):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if 'user_id' not in session:
                flash("Please log in to continue.", "warning")
                return redirect(url_for('login'))
            user_role = session.get('role')
            if user_role not in allowed_roles:
                flash("Access denied: You do not have permissions for this section.", "danger")
                if user_role == 'student':
                    return redirect(url_for('student_dashboard'))
                return redirect(url_for('dashboard'))
            return f(*args, **kwargs)
        return decorated_function
    return decorator

@app.context_processor
def inject_globals():
    """Makes common variables available to all Jinja2 templates."""
    return {
        'current_user': {
            'id': session.get('user_id'),
            'username': session.get('username'),
            'role': session.get('role'),
            'full_name': session.get('full_name'),
            'student_ref_id': session.get('student_ref_id')
        },
        'risk_colors': config.RISK_COLORS,
        'disclaimer': config.DISCLAIMER_NOTICE,
        'feature_metadata': FEATURE_METADATA
    }

# ----------------- Authentication Routes -----------------

@app.route('/')
def index():
    if 'user_id' in session:
        if session.get('role') == 'student':
            return redirect(url_for('student_dashboard'))
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'user_id' in session:
        if session.get('role') == 'student':
            return redirect(url_for('student_dashboard'))
        return redirect(url_for('dashboard'))
        
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        
        user = get_user_by_username(username)
        if user and check_password_hash(user['password_hash'], password):
            session['user_id'] = user['id']
            session['username'] = user['username']
            session['role'] = user['role']
            session['full_name'] = user['full_name']
            session['student_ref_id'] = user['student_ref_id']
            
            flash(f"Welcome back, {user['full_name']}!", "success")
            next_url = request.args.get('next')
            if next_url and next_url.startswith('/'):
                return redirect(next_url)
            if user['role'] == 'student':
                return redirect(url_for('student_dashboard'))
            return redirect(url_for('dashboard'))
        else:
            flash("Invalid username or password. Please try again.", "danger")
            
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash("You have been securely logged out.", "info")
    return redirect(url_for('login'))

# ----------------- Dashboard Routes -----------------

@app.route('/dashboard')
@login_required
@role_required('admin', 'faculty')
def dashboard():
    stats = get_dashboard_stats()
    return render_template('dashboard.html', stats=stats)

@app.route('/student-dashboard')
@login_required
def student_dashboard():
    # If faculty, can view if requested or redirected
    student_id = session.get('student_ref_id')
    if not student_id and session.get('role') in ['admin', 'faculty']:
        # If faculty visits student dashboard, default to first student
        student_id = 'CSE2024001'
        
    if not student_id:
        flash("No student profile linked to this account.", "warning")
        return redirect(url_for('dashboard'))
        
    student, latest_acad, latest_pred = get_student_full_profile(student_id)
    acad_history = get_student_academic_history(student_id)
    pred_history = get_student_prediction_history(student_id)
    
    recs = []
    risk_factors = []
    if latest_pred and latest_pred['recommendations']:
        try:
            recs = json.loads(latest_pred['recommendations'])
        except Exception:
            recs = [latest_pred['recommendations']]
            
    if latest_pred and latest_pred['risk_factors']:
        try:
            risk_factors = json.loads(latest_pred['risk_factors'])
        except Exception:
            risk_factors = [latest_pred['risk_factors']]
            
    return render_template(
        'student_dashboard.html',
        student=student,
        academic=latest_acad,
        prediction=latest_pred,
        recommendations=recs,
        risk_factors=risk_factors,
        acad_history=acad_history,
        pred_history=pred_history
    )

# ----------------- Prediction System -----------------

@app.route('/predict', methods=['GET', 'POST'])
@login_required
@role_required('admin', 'faculty')
def predict():
    students_list = get_all_students()
    prefill_student = None
    prefill_student_id = request.args.get('student_id')
    prefill_acad = None
    
    if prefill_student_id:
        prefill_student, prefill_acad, _ = get_student_full_profile(prefill_student_id)
        
    if request.method == 'POST':
        student_id = request.form.get('student_id', '').strip()
        student_name = request.form.get('student_name', '').strip()
        department = request.form.get('department', 'Computer Science & Engineering')
        year = request.form.get('year', 3)
        semester = request.form.get('semester', 6)
        email = request.form.get('email', f"{student_id.lower()}@college.edu")
        
        # Ensure student exists in DB or create dynamically
        existing_student = get_student_by_id(student_id)
        if not existing_student and student_id and student_name:
            create_student(student_id, student_name, email, department, year, semester)
            
        # Parse 10 Academic Features
        acad_data = {}
        for col in config.FEATURE_COLUMNS:
            raw_val = request.form.get(col)
            meta = FEATURE_METADATA.get(col, {})
            try:
                acad_data[col] = float(raw_val) if raw_val not in (None, '') else float(meta.get('default', 0.0))
            except ValueError:
                acad_data[col] = float(meta.get('default', 0.0))
                
        # Run DNN Prediction
        pred_result = predict_student_risk(acad_data)
        
        # Record into SQLite
        acad_rec_id = add_academic_record(student_id, acad_data)
        predictor_name = session.get('full_name', 'Faculty Advisor')
        pred_id = save_prediction(student_id, pred_result, acad_rec_id, predicted_by=predictor_name)
        
        flash(f"Prediction computed successfully for student {student_id}!", "success")
        return redirect(url_for('result', prediction_id=pred_id))
        
    return render_template(
        'prediction.html',
        students=students_list,
        prefill_student=prefill_student,
        prefill_acad=prefill_acad
    )

@app.route('/result/<int:prediction_id>')
@login_required
def result(prediction_id):
    report_data = format_student_report_data(prediction_id)
    if not report_data:
        flash("Prediction record not found.", "danger")
        return redirect(url_for('dashboard'))
    return render_template('result.html', data=report_data)

# ----------------- Student Management -----------------

@app.route('/students')
@login_required
@role_required('admin', 'faculty')
def students():
    search = request.args.get('search', '').strip()
    department = request.args.get('department', 'All')
    risk = request.args.get('risk_level', 'All')
    
    student_list = get_all_students(search=search, department=department, risk_level=risk)
    return render_template(
        'students.html',
        students=student_list,
        search=search,
        selected_dept=department,
        selected_risk=risk
    )

@app.route('/student/<student_id>')
@login_required
def student_detail(student_id):
    student, latest_acad, latest_pred = get_student_full_profile(student_id)
    if not student:
        flash("Student not found.", "danger")
        return redirect(url_for('students'))
        
    # Check authorization if logged in as student
    if session.get('role') == 'student' and session.get('student_ref_id') != student_id:
        flash("You are not authorized to view other students' profiles.", "danger")
        return redirect(url_for('student_dashboard'))
        
    acad_history = get_student_academic_history(student_id)
    pred_history = get_student_prediction_history(student_id)
    
    recs = []
    risk_factors = []
    if latest_pred and latest_pred['recommendations']:
        try:
            recs = json.loads(latest_pred['recommendations'])
        except Exception:
            recs = [latest_pred['recommendations']]
    if latest_pred and latest_pred['risk_factors']:
        try:
            risk_factors = json.loads(latest_pred['risk_factors'])
        except Exception:
            risk_factors = [latest_pred['risk_factors']]
            
    return render_template(
        'student_detail.html',
        student=student,
        academic=latest_acad,
        prediction=latest_pred,
        recommendations=recs,
        risk_factors=risk_factors,
        acad_history=acad_history,
        pred_history=pred_history
    )

@app.route('/student/add', methods=['GET', 'POST'])
@login_required
@role_required('admin', 'faculty')
def add_student():
    if request.method == 'POST':
        student_id = request.form.get('student_id', '').strip()
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip()
        department = request.form.get('department', 'Computer Science & Engineering')
        year = request.form.get('year', 1)
        semester = request.form.get('semester', 1)
        contact = request.form.get('contact', '').strip()
        
        if not student_id or not name or not email:
            flash("Student ID, Name, and Email are mandatory fields.", "danger")
            return render_template('add_student.html')
            
        success, msg = create_student(student_id, name, email, department, year, semester, contact)
        if success:
            flash(f"Student {name} ({student_id}) enrolled successfully!", "success")
            return redirect(url_for('predict', student_id=student_id))
        else:
            flash(msg, "danger")
            
    return render_template('add_student.html')

# ----------------- Analytics & Performance -----------------

@app.route('/analytics')
@login_required
@role_required('admin', 'faculty')
def analytics():
    stats = get_dashboard_stats()
    chart_data = get_analytics_data()
    return render_template('analytics.html', stats=stats, chart_data=chart_data)

@app.route('/model-performance')
@login_required
@role_required('admin', 'faculty')
def model_performance():
    metrics = {}
    if os.path.exists(config.METRICS_PATH):
        with open(config.METRICS_PATH, 'r') as f:
            metrics = json.load(f)
            
    return render_template('model_performance.html', metrics=metrics)

@app.route('/history')
@login_required
@role_required('admin', 'faculty')
def history():
    risk = request.args.get('risk_level', 'All')
    predictions_list = get_all_predictions(limit=150, risk_level=risk)
    return render_template('history.html', predictions=predictions_list, selected_risk=risk)

@app.route('/reports')
@login_required
@role_required('admin', 'faculty')
def reports():
    predictions_list = get_all_predictions(limit=100)
    return render_template('reports.html', predictions=predictions_list)

@app.route('/report/<int:prediction_id>')
@login_required
def report_view(prediction_id):
    report_data = format_student_report_data(prediction_id)
    if not report_data:
        flash("Report not found.", "danger")
        return redirect(url_for('reports'))
        
    # If logged in as student, verify ownership
    if session.get('role') == 'student' and session.get('student_ref_id') != report_data['student']['id']:
        flash("Unauthorized access to report.", "danger")
        return redirect(url_for('student_dashboard'))
        
    return render_template('report_view.html', data=report_data)

@app.route('/about')
def about():
    return render_template('about.html')

# ----------------- RESTful API Endpoints -----------------

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """
    REST API endpoint for programmatic inference.
    Accepts JSON body with student academic features.
    """
    if not request.is_json:
        return jsonify({'status': 'error', 'message': 'Request body must be JSON'}), 400
        
    payload = request.get_json()
    try:
        result_dict = predict_student_risk(payload)
        return jsonify({
            'status': 'success',
            'data': result_dict
        }), 200
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/analytics', methods=['GET'])
@login_required
def api_analytics():
    """REST API endpoint returning structured analytics for front-end charts."""
    data = get_analytics_data()
    return jsonify(data), 200

if __name__ == '__main__':
    print("=" * 65)
    print("  AT-RISK STUDENT EARLY WARNING SYSTEM - PRODUCTION SERVER")
    print("  Subject: Deep Neural Networks (3rd-Year CSE/AIML Project)")
    print("  Local URL: http://127.0.0.1:5000")
    print("  Admin Credentials  : admin / admin123")
    print("  Student Credentials: student1 / student123")
    print("=" * 65)
    app.run(host='127.0.0.1', port=5000, debug=True)
