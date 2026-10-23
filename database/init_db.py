"""
Database Initialization and Realistic Seeding Script
Project: At-Risk Student Early Warning System Using Deep Neural Networks
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import sqlite3
import json
from werkzeug.security import generate_password_hash
import config
from utils.prediction import predict_student_risk

def init_database():
    print("=" * 65)
    print("  INITIALIZING AT-RISK STUDENT SYSTEM DATABASE")
    print("=" * 65)
    
    schema_path = os.path.join(config.DATABASE_DIR, 'schema.sql')
    db_path = config.DATABASE_PATH
    
    # Remove existing db if re-initializing
    if os.path.exists(db_path):
        try:
            os.remove(db_path)
            print(f"[*] Removed existing database: {db_path}")
        except Exception as e:
            print(f"[!] Warning removing existing db: {e}")
            
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    with open(schema_path, 'r', encoding='utf-8') as f:
        cursor.executescript(f.read())
        
    print("[+] Database schema successfully created.")
    
    # 1. Seed Users
    users = [
        ('admin', generate_password_hash('admin123'), 'admin', None, 'Dr. Ramesh Sharma (Dean / HOD)'),
        ('faculty1', generate_password_hash('faculty123'), 'faculty', None, 'Prof. Ananya Sen (AIML Advisor)'),
        ('faculty2', generate_password_hash('faculty123'), 'faculty', None, 'Dr. Vikram Joshi (CSE Coordinator)'),
        ('student1', generate_password_hash('student123'), 'student', 'CSE2024001', 'Suruthi S'),
        ('student2', generate_password_hash('student123'), 'student', 'AIML2024015', 'Surya'),
        ('student3', generate_password_hash('student123'), 'student', 'CSE2024042', 'Priya'),
        ('student4', generate_password_hash('student123'), 'student', 'IT2024008', 'Sneha Iyer'),
        ('student5', generate_password_hash('student123'), 'student', 'DS2024019', 'Kabir Verma'),
    ]
    
    cursor.executemany(
        '''INSERT INTO users (username, password_hash, role, student_ref_id, full_name)
           VALUES (?, ?, ?, ?, ?)''',
        users
    )
    print(f"[+] Seeded {len(users)} user accounts (admin, faculty, students).")
    
    # 2. Seed Realistic Students & Academic Profiles
    student_seeds = [
        # --- LOW RISK PROFILES ---
        {
            'student_id': 'CSE2024001',
            'name': 'Suruthi S',
            'email': 'suruthi.s@college.edu',
            'department': 'Computer Science & Engineering',
            'year': 3, 'semester': 6, 'contact': '+91 98765 43210',
            'acad': {
                'attendance_percentage': 92.5, 'internal_marks': 86.0, 'assignment_score': 90.0,
                'previous_gpa': 8.9, 'study_hours_weekly': 20.0, 'backlogs': 0,
                'assignment_submission_rate': 96.0, 'engagement_score': 8.8,
                'previous_failures': 0, 'exam_score': 84.0
            }
        },
        {
            'student_id': 'AIML2024002',
            'name': 'Diya Nair',
            'email': 'diya.nair@college.edu',
            'department': 'Artificial Intelligence & Machine Learning',
            'year': 3, 'semester': 5, 'contact': '+91 98765 43211',
            'acad': {
                'attendance_percentage': 89.0, 'internal_marks': 82.0, 'assignment_score': 85.0,
                'previous_gpa': 8.5, 'study_hours_weekly': 18.5, 'backlogs': 0,
                'assignment_submission_rate': 92.0, 'engagement_score': 8.2,
                'previous_failures': 0, 'exam_score': 81.0
            }
        },
        {
            'student_id': 'IT2024003',
            'name': 'Aditya Kulkarni',
            'email': 'aditya.k@college.edu',
            'department': 'Information Technology',
            'year': 2, 'semester': 4, 'contact': '+91 98765 43212',
            'acad': {
                'attendance_percentage': 94.0, 'internal_marks': 91.0, 'assignment_score': 94.0,
                'previous_gpa': 9.2, 'study_hours_weekly': 22.0, 'backlogs': 0,
                'assignment_submission_rate': 98.0, 'engagement_score': 9.0,
                'previous_failures': 0, 'exam_score': 89.0
            }
        },
        {
            'student_id': 'DS2024004',
            'name': 'Meera Sundaram',
            'email': 'meera.s@college.edu',
            'department': 'Data Science',
            'year': 3, 'semester': 6, 'contact': '+91 98765 43213',
            'acad': {
                'attendance_percentage': 88.0, 'internal_marks': 79.0, 'assignment_score': 82.0,
                'previous_gpa': 8.1, 'study_hours_weekly': 17.0, 'backlogs': 0,
                'assignment_submission_rate': 90.0, 'engagement_score': 7.8,
                'previous_failures': 0, 'exam_score': 78.0
            }
        },
        {
            'student_id': 'CSE2024005',
            'name': 'Karthik Rao',
            'email': 'karthik.rao@college.edu',
            'department': 'Computer Science & Engineering',
            'year': 4, 'semester': 7, 'contact': '+91 98765 43214',
            'acad': {
                'attendance_percentage': 90.5, 'internal_marks': 84.5, 'assignment_score': 88.0,
                'previous_gpa': 8.6, 'study_hours_weekly': 19.0, 'backlogs': 0,
                'assignment_submission_rate': 95.0, 'engagement_score': 8.4,
                'previous_failures': 0, 'exam_score': 83.0
            }
        },
        
        # --- MEDIUM RISK PROFILES ---
        {
            'student_id': 'AIML2024015',
            'name': 'Priya Patel',
            'email': 'priya.patel@college.edu',
            'department': 'Artificial Intelligence & Machine Learning',
            'year': 3, 'semester': 5, 'contact': '+91 98765 43215',
            'acad': {
                'attendance_percentage': 76.5, 'internal_marks': 58.0, 'assignment_score': 62.0,
                'previous_gpa': 6.8, 'study_hours_weekly': 12.0, 'backlogs': 1,
                'assignment_submission_rate': 74.0, 'engagement_score': 6.0,
                'previous_failures': 0, 'exam_score': 59.0
            }
        },
        {
            'student_id': 'IT2024008',
            'name': 'Sneha Iyer',
            'email': 'sneha.iyer@college.edu',
            'department': 'Information Technology',
            'year': 2, 'semester': 3, 'contact': '+91 98765 43216',
            'acad': {
                'attendance_percentage': 73.0, 'internal_marks': 54.0, 'assignment_score': 65.0,
                'previous_gpa': 6.5, 'study_hours_weekly': 11.5, 'backlogs': 1,
                'assignment_submission_rate': 70.0, 'engagement_score': 5.8,
                'previous_failures': 1, 'exam_score': 56.0
            }
        },
        {
            'student_id': 'DS2024019',
            'name': 'Kabir Verma',
            'email': 'kabir.verma@college.edu',
            'department': 'Data Science',
            'year': 3, 'semester': 5, 'contact': '+91 98765 43217',
            'acad': {
                'attendance_percentage': 78.0, 'internal_marks': 52.0, 'assignment_score': 60.0,
                'previous_gpa': 6.4, 'study_hours_weekly': 10.0, 'backlogs': 1,
                'assignment_submission_rate': 75.0, 'engagement_score': 5.5,
                'previous_failures': 0, 'exam_score': 54.0
            }
        },
        {
            'student_id': 'CSE2024018',
            'name': 'Ankit Reddy',
            'email': 'ankit.reddy@college.edu',
            'department': 'Computer Science & Engineering',
            'year': 3, 'semester': 6, 'contact': '+91 98765 43218',
            'acad': {
                'attendance_percentage': 74.0, 'internal_marks': 56.0, 'assignment_score': 58.0,
                'previous_gpa': 6.7, 'study_hours_weekly': 11.0, 'backlogs': 1,
                'assignment_submission_rate': 72.0, 'engagement_score': 6.2,
                'previous_failures': 0, 'exam_score': 57.0
            }
        },
        {
            'student_id': 'AIML2024021',
            'name': 'Simran Kaur',
            'email': 'simran.kaur@college.edu',
            'department': 'Artificial Intelligence & Machine Learning',
            'year': 2, 'semester': 4, 'contact': '+91 98765 43219',
            'acad': {
                'attendance_percentage': 79.0, 'internal_marks': 61.0, 'assignment_score': 68.0,
                'previous_gpa': 7.0, 'study_hours_weekly': 13.0, 'backlogs': 1,
                'assignment_submission_rate': 78.0, 'engagement_score': 6.4,
                'previous_failures': 1, 'exam_score': 62.0
            }
        },
        
        # --- HIGH RISK PROFILES ---
        {
            'student_id': 'CSE2024042',
            'name': 'Rohan Gupta',
            'email': 'rohan.gupta@college.edu',
            'department': 'Computer Science & Engineering',
            'year': 3, 'semester': 6, 'contact': '+91 98765 43220',
            'acad': {
                'attendance_percentage': 56.0, 'internal_marks': 34.0, 'assignment_score': 42.0,
                'previous_gpa': 4.9, 'study_hours_weekly': 4.5, 'backlogs': 3,
                'assignment_submission_rate': 48.0, 'engagement_score': 3.1,
                'previous_failures': 2, 'exam_score': 32.0
            }
        },
        {
            'student_id': 'IT2024035',
            'name': 'Vikram Choudhury',
            'email': 'vikram.c@college.edu',
            'department': 'Information Technology',
            'year': 3, 'semester': 5, 'contact': '+91 98765 43221',
            'acad': {
                'attendance_percentage': 52.0, 'internal_marks': 28.0, 'assignment_score': 38.0,
                'previous_gpa': 4.5, 'study_hours_weekly': 3.0, 'backlogs': 4,
                'assignment_submission_rate': 40.0, 'engagement_score': 2.8,
                'previous_failures': 3, 'exam_score': 29.0
            }
        },
        {
            'student_id': 'DS2024031',
            'name': 'Neha Deshmukh',
            'email': 'neha.d@college.edu',
            'department': 'Data Science',
            'year': 2, 'semester': 4, 'contact': '+91 98765 43222',
            'acad': {
                'attendance_percentage': 61.0, 'internal_marks': 38.0, 'assignment_score': 45.0,
                'previous_gpa': 5.2, 'study_hours_weekly': 5.0, 'backlogs': 2,
                'assignment_submission_rate': 55.0, 'engagement_score': 3.5,
                'previous_failures': 2, 'exam_score': 36.0
            }
        },
        {
            'student_id': 'CSE2024058',
            'name': 'Tanmay Bhatt',
            'email': 'tanmay.bhatt@college.edu',
            'department': 'Computer Science & Engineering',
            'year': 4, 'semester': 7, 'contact': '+91 98765 43223',
            'acad': {
                'attendance_percentage': 58.5, 'internal_marks': 31.0, 'assignment_score': 40.0,
                'previous_gpa': 4.7, 'study_hours_weekly': 4.0, 'backlogs': 3,
                'assignment_submission_rate': 45.0, 'engagement_score': 3.0,
                'previous_failures': 2, 'exam_score': 33.0
            }
        },
        {
            'student_id': 'AIML2024040',
            'name': 'Farhan Khan',
            'email': 'farhan.k@college.edu',
            'department': 'Artificial Intelligence & Machine Learning',
            'year': 3, 'semester': 5, 'contact': '+91 98765 43224',
            'acad': {
                'attendance_percentage': 63.0, 'internal_marks': 39.0, 'assignment_score': 46.0,
                'previous_gpa': 5.1, 'study_hours_weekly': 6.0, 'backlogs': 2,
                'assignment_submission_rate': 52.0, 'engagement_score': 3.6,
                'previous_failures': 1, 'exam_score': 38.0
            }
        },
        {
            'student_id': 'IT2024045',
            'name': 'Pooja Hegde',
            'email': 'pooja.h@college.edu',
            'department': 'Information Technology',
            'year': 2, 'semester': 3, 'contact': '+91 98765 43225',
            'acad': {
                'attendance_percentage': 84.0, 'internal_marks': 74.0, 'assignment_score': 77.0,
                'previous_gpa': 7.6, 'study_hours_weekly': 15.0, 'backlogs': 0,
                'assignment_submission_rate': 86.0, 'engagement_score': 7.2,
                'previous_failures': 0, 'exam_score': 72.0
            }
        },
        {
            'student_id': 'CSE2024065',
            'name': 'Manish Tiwari',
            'email': 'manish.t@college.edu',
            'department': 'Computer Science & Engineering',
            'year': 3, 'semester': 5, 'contact': '+91 98765 43226',
            'acad': {
                'attendance_percentage': 71.5, 'internal_marks': 50.0, 'assignment_score': 55.0,
                'previous_gpa': 6.1, 'study_hours_weekly': 9.5, 'backlogs': 1,
                'assignment_submission_rate': 68.0, 'engagement_score': 5.2,
                'previous_failures': 1, 'exam_score': 51.0
            }
        },
        {
            'student_id': 'DS2024025',
            'name': 'Shruti Nambiar',
            'email': 'shruti.n@college.edu',
            'department': 'Data Science',
            'year': 2, 'semester': 4, 'contact': '+91 98765 43227',
            'acad': {
                'attendance_percentage': 91.0, 'internal_marks': 85.0, 'assignment_score': 89.0,
                'previous_gpa': 8.8, 'study_hours_weekly': 19.5, 'backlogs': 0,
                'assignment_submission_rate': 94.0, 'engagement_score': 8.6,
                'previous_failures': 0, 'exam_score': 86.0
            }
        }
    ]
    
    for item in student_seeds:
        cursor.execute(
            '''INSERT INTO students (student_id, name, email, department, year, semester, contact_number)
               VALUES (?, ?, ?, ?, ?, ?, ?)''',
            (item['student_id'], item['name'], item['email'], item['department'],
             item['year'], item['semester'], item['contact'])
        )
        
        acad = item['acad']
        cursor.execute(
            '''INSERT INTO academic_records 
               (student_id, attendance_percentage, internal_marks, assignment_score, previous_gpa,
                study_hours_weekly, backlogs, assignment_submission_rate, engagement_score,
                previous_failures, exam_score)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
            (item['student_id'], acad['attendance_percentage'], acad['internal_marks'],
             acad['assignment_score'], acad['previous_gpa'], acad['study_hours_weekly'],
             acad['backlogs'], acad['assignment_submission_rate'], acad['engagement_score'],
             acad['previous_failures'], acad['exam_score'])
        )
        acad_rec_id = cursor.lastrowid
        
        # Run DNN prediction on the seeded academic record
        pred_result = predict_student_risk(acad)
        
        cursor.execute(
            '''INSERT INTO predictions 
               (student_id, risk_level, risk_code, confidence, low_probability, medium_probability,
                high_probability, predicted_by, recommendations, risk_factors, explanation, academic_record_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
            (item['student_id'], pred_result['risk_level'], pred_result['risk_code'],
             pred_result['confidence'], pred_result['probabilities']['low'],
             pred_result['probabilities']['medium'], pred_result['probabilities']['high'],
             'System Initializer (DNN v1.0)',
             json.dumps(pred_result['recommendations']),
             json.dumps(pred_result['risk_factors']),
             pred_result['explanation'],
             acad_rec_id)
        )
        
    conn.commit()
    conn.close()
    print(f"[+] Successfully seeded {len(student_seeds)} students, academic records, and DNN predictions.")
    print(f"[OK] Database ready at:\n    {db_path}")
    print("=" * 65)

if __name__ == '__main__':
    init_database()
