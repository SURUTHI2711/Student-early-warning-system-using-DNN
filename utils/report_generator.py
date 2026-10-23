"""
Report Generator Utility
Project: At-Risk Student Early Warning System Using Deep Neural Networks
Formats comprehensive academic intervention reports.
"""

import json
from datetime import datetime
import config
from utils.database import get_prediction_by_id, get_student_full_profile

def format_student_report_data(prediction_id):
    """
    Assembles a complete, structured data dictionary for a formal
    Student Academic Risk Assessment & Mentoring Report.
    """
    pred_row = get_prediction_by_id(prediction_id)
    if not pred_row:
        return None
        
    # Parse JSON fields safely
    recommendations = []
    if pred_row['recommendations']:
        try:
            recommendations = json.loads(pred_row['recommendations'])
        except Exception:
            recommendations = [pred_row['recommendations']]
            
    risk_factors = []
    if pred_row['risk_factors']:
        try:
            risk_factors = json.loads(pred_row['risk_factors'])
        except Exception:
            risk_factors = [pred_row['risk_factors']]
            
    formatted_date = pred_row['prediction_date']
    try:
        dt = datetime.strptime(pred_row['prediction_date'], '%Y-%m-%d %H:%M:%S')
        formatted_date = dt.strftime('%B %d, %Y at %I:%M %p')
    except Exception:
        pass
        
    report = {
        'report_id': f"EWS-REP-{pred_row['id']:05d}",
        'generated_date': formatted_date,
        'student': {
            'id': pred_row['student_id'],
            'name': pred_row['name'],
            'email': pred_row['email'],
            'department': pred_row['department'],
            'year': pred_row['year'],
            'semester': pred_row['semester'],
            'contact': pred_row['contact_number']
        },
        'assessment': {
            'risk_level': pred_row['risk_level'],
            'risk_color': config.RISK_COLORS.get(pred_row['risk_level'], '#3b82f6'),
            'confidence': pred_row['confidence'],
            'probabilities': {
                'low': pred_row['low_probability'],
                'medium': pred_row['medium_probability'],
                'high': pred_row['high_probability']
            },
            'predicted_by': pred_row['predicted_by'],
            'explanation': pred_row['explanation']
        },
        'academic_metrics': {
            'attendance': pred_row['attendance_percentage'],
            'internal_marks': pred_row['internal_marks'],
            'assignment_score': pred_row['assignment_score'],
            'gpa': pred_row['previous_gpa'],
            'study_hours': pred_row['study_hours_weekly'],
            'backlogs': pred_row['backlogs'],
            'submission_rate': pred_row['assignment_submission_rate'],
            'engagement_score': pred_row['engagement_score'],
            'previous_failures': pred_row['previous_failures'],
            'exam_score': pred_row['exam_score']
        },
        'risk_factors': risk_factors,
        'recommendations': recommendations,
        'disclaimer': config.DISCLAIMER_NOTICE
    }
    return report
