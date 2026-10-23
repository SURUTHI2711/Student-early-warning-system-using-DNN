"""
Dataset Preparation and Acquisition Pipeline
Project: At-Risk Student Early Warning System Using Deep Neural Networks
"""

import os
import sys
import io
import zipfile
import urllib.request
import pandas as pd
import numpy as np
import config

def try_download_uci_dataset():
    """
    Attempts to download the authentic UCI Machine Learning Repository
    Student Performance Dataset (Cortez and Silva, 2008).
    URL: https://archive.ics.uci.edu/static/public/320/student+performance.zip
    """
    url = "https://archive.ics.uci.edu/static/public/320/student+performance.zip"
    print(f"[*] Attempting to download UCI Student Performance dataset from:\n    {url}")
    
    try:
        req = urllib.request.Request(
            url, 
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        )
        with urllib.request.urlopen(req, timeout=12) as response:
            zip_bytes = response.read()
            with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
                if 'student-mat.csv' in zf.namelist():
                    with zf.open('student-mat.csv') as f:
                        df_mat = pd.read_csv(f, sep=';')
                    df_por = pd.DataFrame()
                    if 'student-por.csv' in zf.namelist():
                        with zf.open('student-por.csv') as f:
                            df_por = pd.read_csv(f, sep=';')
                    df_raw = pd.concat([df_mat, df_por], ignore_index=True).drop_duplicates()
                    print(f"[+] Successfully fetched UCI Student Performance dataset ({len(df_raw)} records)!")
                    return map_uci_to_standard_schema(df_raw)
    except Exception as e:
        print(f"[-] Could not download UCI dataset directly ({e}).")
        print("[*] Falling back to robust realistic empirical academic dataset generation.")
        return None

def map_uci_to_standard_schema(df_uci):
    df = pd.DataFrame()
    df['attendance_percentage'] = np.clip(100 - (df_uci['absences'] * 1.6), 35.0, 100.0).round(1)
    df['internal_marks'] = np.clip((df_uci['G1'] / 20.0 * 100.0) + np.random.normal(0, 3, len(df_uci)), 20.0, 100.0).round(1)
    df['assignment_score'] = np.clip((df_uci['G2'] / 20.0 * 100.0) + np.random.normal(0, 4, len(df_uci)), 25.0, 100.0).round(1)
    
    base_grade = (df_uci['G1'] + df_uci['G2']) / 40.0 * 10.0
    df['previous_gpa'] = np.clip(base_grade + np.random.normal(0, 0.4, len(df_uci)), 3.0, 10.0).round(2)
    
    study_map = {1: 4.0, 2: 9.0, 3: 16.0, 4: 25.0}
    df['study_hours_weekly'] = df_uci['studytime'].map(study_map) + np.random.uniform(-1.5, 2.5, len(df_uci))
    df['study_hours_weekly'] = np.clip(df['study_hours_weekly'], 1.0, 38.0).round(1)
    
    df['previous_failures'] = df_uci['failures'].astype(int)
    df['backlogs'] = np.clip(df_uci['failures'] + np.random.choice([0, 1], p=[0.75, 0.25], size=len(df_uci)), 0, 8)
    df['assignment_submission_rate'] = np.clip(df['assignment_score'] * 0.9 + np.random.uniform(5, 12, len(df_uci)), 30.0, 100.0).round(1)
    df['engagement_score'] = np.clip((df['attendance_percentage'] / 15.0) + (df['study_hours_weekly'] / 6.0) + np.random.normal(0, 0.8, len(df_uci)), 1.0, 10.0).round(1)
    df['exam_score'] = np.clip((df_uci['G3'] / 20.0 * 100.0) + np.random.normal(0, 3, len(df_uci)), 15.0, 100.0).round(1)
    df['risk_level'] = compute_academic_risk_level(df)
    return df

def generate_realistic_academic_dataset(n_samples=1800, random_seed=42):
    """
    Generates a realistic, calibrated 1,800+ record dataset based on
    engineering/AIML student performance patterns with realistic academic correlations.
    """
    np.random.seed(random_seed)
    print(f"[*] Synthesizing realistic academic dataset with {n_samples} student profiles...")
    
    # Latent academic aptitude factor Z ~ N(0, 1)
    z = np.random.normal(0, 1.0, n_samples)
    
    # 1. Attendance Percentage (Mean 77%, std 13)
    attendance = 77.0 + (12.5 * z) + np.random.normal(0, 6.0, n_samples)
    attendance = np.clip(attendance, 35.0, 100.0).round(1)
    
    # 2. Internal Marks (0 - 100)
    internal_marks = 62.0 + (17.0 * z) + (0.2 * (attendance - 75.0)) + np.random.normal(0, 5.0, n_samples)
    internal_marks = np.clip(internal_marks, 20.0, 98.0).round(1)
    
    # 3. Assignment Score (0 - 100)
    assignment_score = 66.0 + (15.0 * z) + (0.15 * (attendance - 75.0)) + np.random.normal(0, 6.0, n_samples)
    assignment_score = np.clip(assignment_score, 25.0, 100.0).round(1)
    
    # 4. Previous GPA (0.0 - 10.0)
    previous_gpa = 7.2 + (1.2 * z) + np.random.normal(0, 0.4, n_samples)
    previous_gpa = np.clip(previous_gpa, 3.5, 9.9).round(2)
    
    # 5. Weekly Study Hours (1 to 35 hours)
    study_hours = 15.0 + (5.0 * z) + np.random.normal(0, 3.0, n_samples)
    study_hours = np.clip(study_hours, 1.5, 36.0).round(1)
    
    # 6. Backlogs (0 to 6)
    backlog_logits = 0.5 - (1.2 * z) - (0.03 * (internal_marks - 60.0))
    backlog_probs = 1.0 / (1.0 + np.exp(-backlog_logits))
    backlogs = np.random.binomial(n=4, p=np.clip(backlog_probs * 0.4, 0.01, 0.85), size=n_samples)
    
    # 7. Previous Failures
    previous_failures = np.clip(backlogs - np.random.choice([0, 1], p=[0.7, 0.3], size=n_samples), 0, 4)
    
    # 8. Assignment Submission Rate (0 - 100%)
    sub_rate = 76.0 + (13.0 * z) + np.random.normal(0, 6.5, n_samples)
    sub_rate = np.clip(sub_rate, 30.0, 100.0).round(1)
    
    # 9. Engagement Score (1.0 - 10.0)
    engagement = 6.6 + (1.5 * z) + (0.02 * (attendance - 75.0)) + np.random.normal(0, 0.6, n_samples)
    engagement = np.clip(engagement, 1.2, 10.0).round(1)
    
    # 10. Exam Score (0 - 100)
    exam_score = 63.0 + (18.0 * z) + (0.2 * (internal_marks - 60.0)) + np.random.normal(0, 6.0, n_samples)
    exam_score = np.clip(exam_score, 18.0, 99.0).round(1)
    
    df = pd.DataFrame({
        'attendance_percentage': attendance,
        'internal_marks': internal_marks,
        'assignment_score': assignment_score,
        'previous_gpa': previous_gpa,
        'study_hours_weekly': study_hours,
        'backlogs': backlogs,
        'assignment_submission_rate': sub_rate,
        'engagement_score': engagement,
        'previous_failures': previous_failures,
        'exam_score': exam_score
    })
    
    df['risk_level'] = compute_academic_risk_level(df)
    return df

def compute_academic_risk_level(df):
    """
    Computes a realistic 3-class academic risk level:
      0 = LOW RISK (Good academic standing)
      1 = MEDIUM RISK (Moderate vulnerability)
      2 = HIGH RISK (Severe danger of failure/detention)
    """
    risk_score = np.zeros(len(df))
    
    # Attendance penalty (<75% standard university cutoff)
    risk_score += np.where(df['attendance_percentage'] < 65, 28, 
                  np.where(df['attendance_percentage'] < 75, 16, 
                  np.where(df['attendance_percentage'] < 82, 5, 0)))
    
    # Internal marks penalty (<40% passing cutoff)
    risk_score += np.where(df['internal_marks'] < 40, 24, 
                  np.where(df['internal_marks'] < 55, 12, 
                  np.where(df['internal_marks'] < 68, 4, 0)))
    
    # Backlogs penalty
    risk_score += (df['backlogs'] * 12)
    
    # Failures penalty
    risk_score += (df['previous_failures'] * 6)
    
    # GPA penalty
    risk_score += np.where(df['previous_gpa'] < 5.5, 20,
                  np.where(df['previous_gpa'] < 6.8, 8, 0))
    
    # Study hours deficit
    risk_score += np.where(df['study_hours_weekly'] < 7, 12,
                  np.where(df['study_hours_weekly'] < 12, 5, 0))
    
    # Submission rate deficit
    risk_score += np.where(df['assignment_submission_rate'] < 60, 10,
                  np.where(df['assignment_submission_rate'] < 75, 4, 0))
    
    # Engagement score penalty
    risk_score += np.where(df['engagement_score'] < 4.0, 10,
                  np.where(df['engagement_score'] < 6.0, 4, 0))
    
    # Realistic noise
    noise = np.random.normal(0, 4.0, len(df))
    final_score = risk_score + noise
    
    # Balanced 3-class thresholds
    risk_labels = np.zeros(len(df), dtype=int)
    risk_labels[final_score >= 42] = 2                      # High Risk (~28%)
    risk_labels[(final_score >= 18) & (final_score < 42)] = 1 # Medium Risk (~34%)
    risk_labels[final_score < 18] = 0                       # Low Risk (~38%)
    
    return risk_labels

def prepare_dataset(force_download=False):
    print("=" * 65)
    print("  AT-RISK STUDENT EARLY WARNING SYSTEM - DATASET PIPELINE")
    print("=" * 65)
    
    df = None
    if force_download:
        df = try_download_uci_dataset()
    
    if df is None:
        df = generate_realistic_academic_dataset(n_samples=1800, random_seed=42)
    
    # Drop any accidental duplicates
    df = df.drop_duplicates().reset_index(drop=True)
    
    print("\n[+] Dataset Validation Summary:")
    print(f"    Total Student Records : {len(df):,}")
    print(f"    Total Feature Columns : {len(config.FEATURE_COLUMNS)}")
    print(f"    Missing / Null Values : {df.isnull().sum().sum()}")
    
    risk_counts = df['risk_level'].value_counts().sort_index()
    print("\n[+] Risk Level Distribution:")
    for code, name in config.RISK_LABELS.items():
        count = risk_counts.get(code, 0)
        pct = (count / len(df)) * 100
        print(f"    [{code}] {name:<12} : {count:>5} students ({pct:>5.1f}%)")
    
    df.to_csv(config.DATASET_PATH, index=False)
    print(f"\n[OK] Clean dataset saved successfully to:\n    {config.DATASET_PATH}")
    print("=" * 65)
    return df

if __name__ == '__main__':
    force_dl = '--download' in sys.argv
    prepare_dataset(force_download=force_dl)
