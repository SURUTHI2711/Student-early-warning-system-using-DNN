# At-Risk Student Early Warning System Using Deep Neural Networks

An AI-powered academic intervention and early warning web platform built using **TensorFlow/Keras Deep Neural Networks (DNN)**, **Python Flask**, **Scikit-learn**, and **SQLite**, designed as a complete 3rd-Year CSE/AIML course project.

---

## 1. Abstract
Academic underperformance, attendance shortages, and course backlogs in higher engineering education frequently lead to semester failures and student attrition. Conventional institutional audits typically detect academic deficiencies retroactively following end-semester examinations, precluding timely remedial counseling. 

This project presents an **At-Risk Student Early Warning System** driven by a multi-layer Deep Neural Network (DNN). By analyzing continuous assessment marks, physical attendance records, assignment submission rates, self-study commitment, and backlog patterns, the system classifies students into three calibrated tiers: **LOW RISK**, **MEDIUM RISK**, and **HIGH RISK**. Crucially, the platform provides explainable academic risk factor attribution and actionable mentoring recommendations for faculty advisors and students.

---

## 2. Problem Statement
Engineering programs enforce strict eligibility requirements (such as 75% minimum lecture attendance and continuous internal assessment passing thresholds). Students facing academic distress often accumulate arrear backlogs and fail to maintain coursework submissions. Existing campus ERP platforms merely record raw marks without predictive foresight or risk stratification. There is an acute need for a proactive, intelligent early warning system that:
1. Identifies academically vulnerable students weeks before final examinations.
2. Accurately stratifies student vulnerability with probabilistic confidence.
3. Translates multi-variate diagnostic indicators into empathetic, actionable counseling guidance.

---

## 3. Key Features
- **Deep Neural Network (DNN) AI Core:** Multi-layer feedforward architecture with Batch Normalization, Dropout regularization, and Softmax output.
- **Multi-Class Risk Stratification:** Computes exact posterior class probabilities for **LOW RISK**, **MEDIUM RISK**, and **HIGH RISK**.
- **Actionable Explainability Engine:** Pinpoints underlying risk triggers (e.g., attendance shortage < 75%, failing internal marks < 40%, backlog accumulation) and recommends personalized remedial actions.
- **Faculty / Admin Dashboard:** Overview with KPI stat cards, risk distribution donut charts, departmental risk comparisons, priority attention roster, and recent evaluations.
- **Dedicated Student Portal:** Secure login allowing students to track their academic standing, review personalized recommendations, and inspect competency trends.
- **Searchable Student Directory:** Filter by department, risk level, and keywords.
- **Institutional Analytics:** Interactive Chart.js cohort visualizations (Attendance vs. Marks scatter correlations, departmental distributions).
- **DNN Model Performance Suite:** Live display of training/validation loss & accuracy curves, confusion matrix heatmap, factor importance, and comparative ML baseline benchmarks.
- **Official Report Generation:** Generates and prints official Student Academic Early Warning & Mentoring Dossiers with institutional formatting and signature blocks.
- **RESTful API:** `POST /api/predict` and `GET /api/analytics` endpoints for external integration.

---

## 4. Technology Stack
| Layer | Technologies |
|---|---|
| **Programming Language** | Python 3.11 |
| **Deep Learning Framework** | TensorFlow / Keras 2.x (Sequential DNN) |
| **Machine Learning & Preprocessing** | Scikit-learn, StandardScaler, Joblib |
| **Data Manipulation & Analysis** | Pandas, NumPy |
| **Visualizations & Plots** | Matplotlib, Seaborn, Chart.js |
| **Web Framework** | Flask 3.x (Jinja2, Werkzeug Security) |
| **Database** | SQLite3 (`database/students.db`) |
| **Styling & UI** | Custom Responsive CSS (Academic Glassmorphism, Dark/Light accents) |

---

## 5. System Architecture & Flow

```
[Student / Faculty Input] ──> [Input Validation & Sanitization]
                                      │
                                      ▼
                        [StandardScaler Normalization (Z-Score)]
                                      │
                                      ▼
                        [Deep Neural Network Inference]
                          - Input Layer (10 Features)
                          - Dense(64, ReLU) + BatchNorm + Dropout(0.3)
                          - Dense(32, ReLU) + Dropout(0.2)
                          - Dense(16, ReLU)
                          - Dense(3, Softmax Output)
                                      │
                                      ▼
                    [Probabilities: Low %, Medium %, High %]
                                      │
                                      ▼
                      [Explainability & Mentoring Engine]
                                      │
               ┌──────────────────────┴──────────────────────┐
               ▼                                             ▼
     [Interactive Web UI]                         [SQLite Database Log]
  (Dashboards, Reports, Charts)                (Predictions, Academic Records)
```

---

## 6. Dataset Specifications
- **Location:** `data/student_data.csv`
- **Total Records:** 1,798 student profiles
- **Acquisition:** Automated acquisition script (`prepare_dataset.py` / `download_dataset.py`) modeled on the UCI Machine Learning Student Performance methodology and accredited university engineering evaluation rubrics.
- **Features (10 Inputs):**
  1. `attendance_percentage`: Classroom attendance (0.0% – 100.0%)
  2. `internal_marks`: Continuous assessment marks (0.0 – 100.0)
  3. `assignment_score`: Coursework assignment average (0.0 – 100.0)
  4. `previous_gpa`: Cumulative Grade Point Average (0.0 – 10.0)
  5. `study_hours_weekly`: Dedicated self-study hours per week (0.0 – 40.0)
  6. `backlogs`: Count of active uncleared course arrears (0 – 10)
  7. `assignment_submission_rate`: Punctual homework completion percentage (0.0% – 100.0%)
  8. `engagement_score`: Classroom & LMS portal interaction rating (1.0 – 10.0)
  9. `previous_failures`: Cumulative historical course failures (0 – 5)
  10. `exam_score`: Midterm or diagnostic examination score (0.0 – 100.0)
- **Target Variable (`risk_level`):**
  - `0`: LOW RISK (36.9% of cohort)
  - `1`: MEDIUM RISK (22.1% of cohort)
  - `2`: HIGH RISK (41.0% of cohort)

---

## 7. Deep Neural Network Architecture & Results

### Model Architecture
- **Input Layer:** 10 normalized features
- **Hidden Layer 1:** 64 units, ReLU activation, $L_2$ Regularization (0.001), BatchNormalization, Dropout (0.3)
- **Hidden Layer 2:** 32 units, ReLU activation, $L_2$ Regularization (0.001), Dropout (0.2)
- **Hidden Layer 3:** 16 units, ReLU activation
- **Output Layer:** 3 units, Softmax activation
- **Compilation:** Adam optimizer ($\text{lr} = 0.001$), Sparse Categorical Cross-Entropy loss, Accuracy metric.

### Measured Test Results (Unseen 20% Stratified Test Split)
| Model | Accuracy | Precision (Weighted) | Recall (Weighted) | F1-Score (Weighted) |
|---|---|---|---|---|
| **Logistic Regression (Baseline)** | 89.17% | 89.56% | 89.17% | 89.21% |
| **Random Forest (Baseline)** | 88.89% | 89.29% | 88.89% | 88.96% |
| **Deep Neural Network (Proposed)** | **91.11%** | **91.68%** | **91.11%** | **91.24%** |

*All generated performance graphs (accuracy curve, loss curve, confusion matrix heatmap, and feature rankings) are stored in `static/plots/`.*

---

## 8. Installation and Execution

### Step 1: Clone or Navigate to Directory
```bash
cd D:\at-risk-student-system
```

### Step 2: Activate Virtual Environment
```bash
# Windows
.\venv\Scripts\activate
```

*(If creating a new virtualenv from scratch: `python -m venv venv`, activate it, then run `pip install -r requirements.txt`)*

### Step 3: Prepare the Dataset
```bash
python prepare_dataset.py
```

### Step 4: Train the Deep Neural Network Model
```bash
python train_model.py
```

### Step 5: Initialize SQLite Database and Seed Realistic Data
```bash
python database/init_db.py
```

### Step 6: Launch the Production Web Application
```bash
python app.py
```

Open your browser and navigate to:
**`http://127.0.0.1:5000`**

---

## 9. Demo Login Credentials

| Role | Username | Password | Notes |
|---|---|---|---|
| **Faculty / Admin** | `admin` | `admin123` | Full access: dashboard, predictions, roster, analytics, reports |
| **Faculty Advisor** | `faculty1` | `faculty123` | AIML department advisor portal |
| **Student** | `student1` | `student123` | Student view for Aarav Sharma (`CSE2024001`) |
| **Student** | `student2` | `student123` | Student view for Priya Patel (`AIML2024015`) |
| **Student** | `student3` | `student123` | Student view for Rohan Gupta (`CSE2024042`) |

*(On the login screen, click **"Fill Faculty (admin)"** or **"Fill Student (student1)"** for one-click credentials filling).*

---

## 10. Automated Testing
Run the comprehensive test suite verifying all 18 routes, forms, APIs, and models:
```bash
python test_app.py
```

---

## 11. Ethical Disclaimer
> **Academic Early Warning Notice:** This software is designed exclusively as an educational decision-support indicator to assist faculty mentors and counselors. Model predictions are probabilistic indicators of academic vulnerability and **must never** be construed as definitive student judgments, character evaluations, or disciplinary sanctions.
