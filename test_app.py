"""
Comprehensive Automated Test Suite
Project: At-Risk Student Early Warning System Using Deep Neural Networks
Verifies all Flask routes, database queries, authentication flows,
prediction pipelines, and REST API endpoints.
"""

import os
import json
from app import app

def run_tests():
    print("=" * 65)
    print("  RUNNING AUTOMATED TEST SUITE")
    print("=" * 65)
    
    app.config['TESTING'] = True
    client = app.test_client()
    
    passed = 0
    failed = 0
    
    def test_case(name, cond, details=""):
        nonlocal passed, failed
        if cond:
            print(f"  [PASS] {name}")
            passed += 1
        else:
            print(f"  [FAIL] {name} - {details}")
            failed += 1

    # 1. Unauthenticated root & login
    res = client.get('/')
    test_case("Root redirect to /login", res.status_code == 302 and '/login' in res.headers['Location'])
    
    res = client.get('/login')
    test_case("GET /login renders properly", res.status_code == 200 and b'Sign In to Portal' in res.data)
    
    # 2. Authentication with Admin Credentials
    res = client.post('/login', data={'username': 'admin', 'password': 'wrongpassword'}, follow_redirects=True)
    test_case("Invalid login rejection", b'Invalid username or password' in res.data)
    
    res = client.post('/login', data={'username': 'admin', 'password': 'admin123'}, follow_redirects=True)
    test_case("Valid Admin Login", res.status_code == 200 and b'Faculty Dashboard' in res.data)
    
    # 3. Authenticated Faculty Routes
    res = client.get('/dashboard')
    test_case("GET /dashboard", res.status_code == 200 and b'Institutional Academic Overview' in res.data)
    
    res = client.get('/students')
    test_case("GET /students", res.status_code == 200 and b'Student Roster' in res.data)
    
    res = client.get('/predict')
    test_case("GET /predict", res.status_code == 200 and b'Student Academic Risk Prediction' in res.data)
    
    # 4. Form Prediction Submission
    predict_payload = {
        'student_id': 'TEST2026099',
        'student_name': 'Test Student Evaluation',
        'department': 'Computer Science & Engineering',
        'year': '3',
        'semester': '6',
        'attendance_percentage': '55.0',
        'internal_marks': '32.0',
        'assignment_score': '40.0',
        'previous_gpa': '4.8',
        'study_hours_weekly': '4.0',
        'backlogs': '3',
        'assignment_submission_rate': '45.0',
        'engagement_score': '3.0',
        'previous_failures': '2',
        'exam_score': '35.0'
    }
    res = client.post('/predict', data=predict_payload, follow_redirects=True)
    test_case("POST /predict generates inference", res.status_code == 200 and b'Prediction Result' in res.data or b'Inference Verdict' in res.data)
    
    # 5. Result Page
    res = client.get('/result/1')
    test_case("GET /result/1", res.status_code == 200 and b'Softmax Output Probability Distribution' in res.data)
    
    # 6. Analytics Page
    res = client.get('/analytics')
    test_case("GET /analytics", res.status_code == 200 and b'Institutional Academic Performance Analytics' in res.data)
    
    # 7. Model Performance Page
    res = client.get('/model-performance')
    test_case("GET /model-performance", res.status_code == 200 and b'Comparative Performance Evaluation' in res.data and b'Deep Neural Network' in res.data)
    
    # 8. Prediction History Page
    res = client.get('/history')
    test_case("GET /history", res.status_code == 200 and b'Student Risk Prediction Audit Log' in res.data)
    
    # 9. Reports Center & Printable Report Dossier
    res = client.get('/reports')
    test_case("GET /reports", res.status_code == 200 and b'Student Reports Center' in res.data)
    
    res = client.get('/report/1')
    test_case("GET /report/1 printable dossier", res.status_code == 200 and b'Print / Save as PDF' in res.data)
    
    # 10. About Page
    res = client.get('/about')
    test_case("GET /about", res.status_code == 200 and b'Deep Neural Network Mathematical Formulation' in res.data)
    
    # 11. REST API Inference Endpoint
    api_payload = {
        'attendance_percentage': 92.0,
        'internal_marks': 85.0,
        'assignment_score': 88.0,
        'previous_gpa': 8.7,
        'study_hours_weekly': 20.0,
        'backlogs': 0,
        'assignment_submission_rate': 95.0,
        'engagement_score': 8.5,
        'previous_failures': 0,
        'exam_score': 84.0
    }
    res = client.post('/api/predict', data=json.dumps(api_payload), content_type='application/json')
    test_case("POST /api/predict JSON endpoint", res.status_code == 200)
    if res.status_code == 200:
        json_resp = res.get_json()
        test_case("API returns valid prediction data", 'data' in json_resp and json_resp['data']['risk_level'] == 'LOW RISK')
        
    # 12. Student Login & Student Dashboard Access
    client.get('/logout')
    res = client.post('/login', data={'username': 'student1', 'password': 'student123'}, follow_redirects=True)
    test_case("Student Login & Student Dashboard", res.status_code == 200 and b'Student Academic Progress Portal' in res.data)
    
    print("=" * 65)
    print(f"  TEST RESULTS: {passed} PASSED, {failed} FAILED")
    print("=" * 65)
    return failed == 0

if __name__ == '__main__':
    success = run_tests()
    exit(0 if success else 1)
