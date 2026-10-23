"""
Prediction and Explainability Engine
Project: At-Risk Student Early Warning System Using Deep Neural Networks
Loads trained Keras DNN model, computes multi-class softmax probabilities,
and synthesizes personalized academic intervention recommendations.
"""

import os
import numpy as np
import tensorflow as tf
import config
from utils.preprocessing import preprocess_single_input, load_scaler, FEATURE_METADATA

_cached_model = None

def load_dnn_model(model_path=None):
    """
    Loads and caches the compiled Keras DNN model.
    """
    global _cached_model
    if _cached_model is not None:
        return _cached_model
        
    path = model_path or config.MODEL_PATH
    if os.path.exists(path):
        _cached_model = tf.keras.models.load_model(path)
        return _cached_model
    return None

def predict_student_risk(input_data):
    """
    Predicts the academic risk level for a student.
    
    Args:
      input_data: dict of academic indicators or numpy feature array
      
    Returns:
      dict containing:
        - risk_level (str): 'LOW RISK' | 'MEDIUM RISK' | 'HIGH RISK'
        - risk_code (int): 0 | 1 | 2
        - risk_color (str): hex color code
        - confidence (float): confidence percentage of predicted class
        - probabilities (dict): { 'low': float, 'medium': float, 'high': float }
        - raw_features (dict): sanitized input features
        - risk_factors (list): key identified academic concerns
        - positive_factors (list): areas of strength
        - recommendations (list): actionable mentoring steps
        - explanation (str): comprehensive analytical summary
        - disclaimer (str): ethical compliance notice
    """
    model = load_dnn_model()
    scaler = load_scaler()
    
    if model is None:
        raise FileNotFoundError(f"Trained DNN model file not found at {config.MODEL_PATH}. Run train_model.py first.")
        
    scaled_vector, raw_features, errors = preprocess_single_input(input_data, scaler)
    
    # Forward pass through Deep Neural Network
    probs = model.predict(scaled_vector, verbose=0)[0]
    
    low_p = float(probs[0]) * 100.0
    med_p = float(probs[1]) * 100.0
    high_p = float(probs[2]) * 100.0
    
    pred_code = int(np.argmax(probs))
    risk_label = config.RISK_LABELS[pred_code]
    risk_color = config.RISK_COLORS[risk_label]
    confidence = float(probs[pred_code]) * 100.0
    
    # Generate Explainability & Recommendations
    risk_factors, positive_factors, recommendations, explanation = generate_recommendations(
        raw_features, risk_label, confidence, low_p, med_p, high_p
    )
    
    return {
        'risk_level': risk_label,
        'risk_code': pred_code,
        'risk_color': risk_color,
        'confidence': round(confidence, 1),
        'probabilities': {
            'low': round(low_p, 1),
            'medium': round(med_p, 1),
            'high': round(high_p, 1)
        },
        'raw_features': raw_features,
        'risk_factors': risk_factors,
        'positive_factors': positive_factors,
        'recommendations': recommendations,
        'explanation': explanation,
        'disclaimer': config.DISCLAIMER_NOTICE,
        'errors': errors
    }

def generate_recommendations(features, risk_label, confidence, low_p, med_p, high_p):
    """
    Generates tailored, actionable academic recommendations and explainability
    insights based on individual parameter evaluations.
    """
    risk_factors = []
    positive_factors = []
    recommendations = []
    
    att = features.get('attendance_percentage', 75.0)
    internal = features.get('internal_marks', 60.0)
    assignments = features.get('assignment_score', 65.0)
    gpa = features.get('previous_gpa', 7.0)
    study = features.get('study_hours_weekly', 15.0)
    backlogs = int(features.get('backlogs', 0))
    sub_rate = features.get('assignment_submission_rate', 80.0)
    engagement = features.get('engagement_score', 6.5)
    failures = int(features.get('previous_failures', 0))
    exam = features.get('exam_score', 60.0)
    
    # 1. Attendance Analysis
    if att < 65.0:
        risk_factors.append(f"Critical attendance deficit: Currently at {att:.1f}%, significantly below university statutory minimum (75%). High risk of semester exam debarment.")
        recommendations.append("Immediate attendance recovery plan: Attend all remaining scheduled lectures and supplementary lab sessions to improve attendance percentage above 75%.")
    elif att < 75.0:
        risk_factors.append(f"Borderline attendance: {att:.1f}% is below the recommended 75% threshold, placing the student in conditional eligibility status.")
        recommendations.append("Submit medical documentation if absences were health-related and ensure consistent presence over upcoming instructional weeks.")
    else:
        positive_factors.append(f"Strong attendance record: {att:.1f}% satisfies institutional lecture criteria.")

    # 2. Internal Assessment & Exam Analysis
    if internal < 40.0:
        risk_factors.append(f"Failing internal marks: {internal:.1f}/100 indicates substantial concept comprehension gaps in current semester coursework.")
        recommendations.append("Schedule 1-on-1 subject mentor remedial sessions and retake diagnostic review quizzes to elevate continuous evaluation marks.")
    elif internal < 55.0:
        risk_factors.append(f"Sub-optimal internal marks: {internal:.1f}/100 is close to the minimum pass boundary.")
        recommendations.append("Participate in peer study groups and faculty doubt-clearing clinics before end-semester examinations.")
    else:
        positive_factors.append(f"Solid internal assessment scores: {internal:.1f}/100 demonstrates adequate syllabus grasp.")

    # 3. Backlogs & Failure History
    if backlogs >= 3:
        risk_factors.append(f"Heavy arrear accumulation: {backlogs} active backlogs create severe academic burden and potential year-down risk.")
        recommendations.append("Prioritize clearing pending backlogs: Meet faculty advisor to draft an arrear clearance study calendar.")
    elif backlogs >= 1:
        risk_factors.append(f"Existing backlogs: {backlogs} uncleared subject(s) require active remediation alongside ongoing semester subjects.")
        recommendations.append("Allocate dedicated weekend revision hours exclusively to uncleared syllabus modules.")
    else:
        positive_factors.append("Clean backlog record: Zero uncleared courses.")

    # 4. Study Hours & Homework Submission
    if study < 8.0:
        risk_factors.append(f"Low self-study commitment: Only {study:.1f} hours/week outside lectures. Industry/curriculum guidelines recommend 15+ hours.")
        recommendations.append("Establish a structured daily study timetable allocating at least 2.5 to 3 hours per evening for core technical subjects.")
    elif study >= 16.0:
        positive_factors.append(f"Commendable self-study discipline: {study:.1f} hours/week of independent study.")

    if sub_rate < 70.0 or assignments < 50.0:
        risk_factors.append(f"Incomplete assignment coursework: Submission rate is {sub_rate:.1f}% with average score {assignments:.1f}/100.")
        recommendations.append("Complete and submit all overdue lab assignments and mini-projects to secure essential practical internal credits.")

    # 5. Engagement
    if engagement < 4.5:
        risk_factors.append(f"Low classroom/LMS participation score ({engagement:.1f}/10).")
        recommendations.append("Increase LMS portal activity, review digital courseware materials, and contribute to classroom seminar discussions.")

    # General Synthesis Explanation
    if risk_label == 'HIGH RISK':
        explanation = (
            f"The Deep Neural Network classifies this student profile as HIGH RISK with {confidence:.1f}% model confidence "
            f"(Low Risk: {low_p:.1f}%, Medium Risk: {med_p:.1f}%, High Risk: {high_p:.1f}%). "
            f"The primary vulnerability stems from {len(risk_factors)} identified academic stress factors, "
            f"most notably {'attendance shortage' if att < 75 else 'low assessment performance' if internal < 50 else 'arrear accumulation'}. "
            "Timely intervention by the department academic advisory committee is strongly recommended."
        )
    elif risk_label == 'MEDIUM RISK':
        explanation = (
            f"The Deep Neural Network classifies this student profile as MEDIUM RISK with {confidence:.1f}% model confidence "
            f"(Low Risk: {low_p:.1f}%, Medium Risk: {med_p:.1f}%, High Risk: {high_p:.1f}%). "
            "The student displays borderline performance across select metrics while maintaining resilience in others. "
            "Early supportive mentoring and periodic progress tracking can prevent escalation into high risk."
        )
    else:
        explanation = (
            f"The Deep Neural Network classifies this student profile as LOW RISK with {confidence:.1f}% model confidence "
            f"(Low Risk: {low_p:.1f}%, Medium Risk: {med_p:.1f}%, High Risk: {high_p:.1f}%). "
            "Academic indicators reflect stable attendance, satisfactory continuous evaluation, and manageable workload. "
            "Continued encouragement and participation in co-curricular/competitive activities is advised."
        )

    if not recommendations:
        recommendations.append("Maintain current academic discipline, sustain regular attendance, and explore advanced honors/elective subjects.")

    return risk_factors, positive_factors, recommendations, explanation
