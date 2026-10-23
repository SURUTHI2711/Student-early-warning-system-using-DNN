"""
Preprocessing Pipeline Module
Handles data cleaning, normalization, scaling, and input validation.
"""

import os
import joblib
import numpy as np
import pandas as pd
import config

FEATURE_METADATA = {
    'attendance_percentage': {
        'label': 'Attendance Percentage',
        'unit': '%',
        'min': 0.0,
        'max': 100.0,
        'default': 75.0,
        'description': 'Classroom and laboratory physical attendance'
    },
    'internal_marks': {
        'label': 'Internal Assessment Marks',
        'unit': '/100',
        'min': 0.0,
        'max': 100.0,
        'default': 60.0,
        'description': 'Aggregated continuous internal assessment scores'
    },
    'assignment_score': {
        'label': 'Assignment Performance',
        'unit': '/100',
        'min': 0.0,
        'max': 100.0,
        'default': 65.0,
        'description': 'Average score across submitted homework and coursework'
    },
    'previous_gpa': {
        'label': 'Previous Cumulative GPA',
        'unit': '/10.0',
        'min': 0.0,
        'max': 10.0,
        'default': 7.0,
        'description': 'Cumulative grade point average from earlier semesters'
    },
    'study_hours_weekly': {
        'label': 'Weekly Study Hours',
        'unit': 'hrs/week',
        'min': 0.0,
        'max': 50.0,
        'default': 15.0,
        'description': 'Dedicated self-study hours outside regular lecture timetable'
    },
    'backlogs': {
        'label': 'Active Backlogs',
        'unit': 'courses',
        'min': 0,
        'max': 12,
        'default': 0,
        'description': 'Count of currently uncleared arrear / failed subjects'
    },
    'assignment_submission_rate': {
        'label': 'Assignment Submission Rate',
        'unit': '%',
        'min': 0.0,
        'max': 100.0,
        'default': 80.0,
        'description': 'Ratio of assignments submitted on or before deadline'
    },
    'engagement_score': {
        'label': 'Classroom & LMS Engagement',
        'unit': '/10',
        'min': 1.0,
        'max': 10.0,
        'default': 6.5,
        'description': 'Interactive participation rating in discussions & LMS portals'
    },
    'previous_failures': {
        'label': 'Historical Course Failures',
        'unit': 'courses',
        'min': 0,
        'max': 8,
        'default': 0,
        'description': 'Total prior examination failures across collegiate record'
    },
    'exam_score': {
        'label': 'Midterm / Mid-Semester Exam Score',
        'unit': '/100',
        'min': 0.0,
        'max': 100.0,
        'default': 60.0,
        'description': 'Score in recent midterm or prerequisite diagnostic examination'
    }
}

_cached_scaler = None

def load_scaler(scaler_path=None):
    """
    Loads and caches the fitted StandardScaler instance.
    """
    global _cached_scaler
    if _cached_scaler is not None:
        return _cached_scaler
        
    path = scaler_path or config.SCALER_PATH
    if os.path.exists(path):
        _cached_scaler = joblib.load(path)
        return _cached_scaler
    return None

def preprocess_single_input(raw_dict, scaler=None):
    """
    Validates and transforms a single student input record into a scaled numpy array
    suitable for DNN inference.
    
    Returns:
      (scaled_features_array, raw_features_dict, validation_errors_list)
    """
    errors = []
    parsed_values = {}
    
    if scaler is None:
        scaler = load_scaler()
        
    for feature in config.FEATURE_COLUMNS:
        meta = FEATURE_METADATA.get(feature, {})
        val = raw_dict.get(feature)
        
        if val is None or val == '':
            val = meta.get('default', 0.0)
            parsed_values[feature] = float(val)
        else:
            try:
                num_val = float(val)
                min_v = meta.get('min', -1e9)
                max_v = meta.get('max', 1e9)
                
                if num_val < min_v or num_val > max_v:
                    errors.append(
                        f"{meta.get('label', feature)} must be between {min_v} and {max_v} (received {num_val})."
                    )
                parsed_values[feature] = num_val
            except (ValueError, TypeError):
                errors.append(f"Invalid numeric value for {meta.get('label', feature)}: '{val}'")
                parsed_values[feature] = float(meta.get('default', 0.0))
                
    # Build 2D array in exact canonical order
    feature_vector = np.array([[parsed_values[col] for col in config.FEATURE_COLUMNS]], dtype=np.float32)
    
    if scaler is not None:
        scaled_vector = scaler.transform(feature_vector)
    else:
        scaled_vector = feature_vector
        
    return scaled_vector, parsed_values, errors
