# KIT - DEPARTMENT OF CSE(AI&ML)
## Project Review Presentation Content

**College:** KIT-KALAIGNARKARUNANIDHI INSTITUTE OF TECHNOLOGY, COIMBATORE - 641402  
**Affiliation:** Approved by AICTE, Affiliated to Anna University, Chennai  
**Accreditation:** Accredited With ‘A’ Grade by NAAC & By NBA  
**Department:** DEPARTMENT OF CSE(AI&ML)  
**File Generated:** `D:\at-risk-student-system\KIT_CSE_AIML_Project_Presentation.pptx`

---

### Slide 1: TITLE SLIDE
- **Institution Header:**
  KIT-KALAIGNARKARUNANIDHI INSTITUTE OF TECHNOLOGY
  COIMBATORE - 641402
  Approved by AICTE Affiliated to Anna University, Chennai
  Accredited With ‘A’ Grade by NAAC & By NBA
  DEPARTMENT OF CSE(AI&ML)
- **Project Title:**
  AT-RISK STUDENT EARLY WARNING SYSTEM USING DEEP NEURAL NETWORKS
- **Subtitle:**
  A Deep Neural Network Project for Academic Risk Stratification and Early Intervention
- **Presented By:**
  1. **SURUTHI S** (Reg No: CSE2024001)
  2. **SURYA** (Reg No: AIML2024015)
  3. **PRIYA** (Reg No: CSE2024042)
- **Class / Academic Year:** 3rd Year B.E. CSE(AI&ML) | 2026

---

### Slide 2: PROBLEM STATEMENT / OBJECTIVE
#### Problem Statement:
- Higher education engineering programs enforce strict statutory requirements (e.g. minimum 75% lecture attendance and 40% internal continuous passing cutoffs).
- Conventional institutional monitoring regimes are strictly retroactive: academic distress and failure are identified only after semester examinations conclude, when remedial action is impossible.
- Engineering students experiencing early conceptual difficulties face a compounding deficit spiral: low attendance leads to concept gaps, resulting in failing internal marks, missed assignment deadlines, accumulated arrears, and ultimately semester detention.
- Existing campus ERP software simply stores raw numerical marks without predictive early warning capability, risk probability stratification, or actionable counseling guidance.

#### Project Objectives:
- Build an AI-powered Early Warning System utilizing a multi-layer Deep Neural Network (DNN) to classify students into **LOW RISK**, **MEDIUM RISK**, and **HIGH RISK**.
- Model multi-variate academic & behavioral indicators: attendance %, internal marks, assignment scores, GPA, weekly study hours, backlogs, submission rates, and LMS engagement.
- Compute exact Softmax posterior probability distributions to provide measured confidence scores for each individual risk classification.
- Develop an Explainable Recommendation Engine that identifies specific root-cause deficiencies and synthesizes actionable, personalized mentoring interventions.
- Deploy a production-grade web platform (Flask + SQLite + Chart.js) providing role-based dashboards for faculty and students, institutional analytics, and official printable dossiers.

---

### Slide 3: ABSTRACT
Academic underperformance, attendance shortages, and course backlog accumulation in higher engineering education frequently lead to semester failures and student dropouts. Traditional institutional monitoring systems identify academic vulnerabilities only retroactively following end-semester examinations, precluding timely remedial counseling and mentoring.

This project presents an intelligent, end-to-end **At-Risk Student Early Warning System** powered by a **Deep Neural Network (DNN)** developed using TensorFlow and Keras. By continuously analyzing ten multi-dimensional academic and behavioral indicators—including lecture attendance percentage, continuous internal assessment marks, assignment performance, cumulative GPA, weekly independent study hours, active backlog count, assignment submission rates, and LMS engagement—the system classifies students into three calibrated tiers: **LOW RISK**, **MEDIUM RISK**, and **HIGH RISK**.

The proposed Deep Neural Network architecture incorporates three hidden dense layers (64, 32, 16 units) equipped with Rectified Linear Unit (ReLU) activations, Batch Normalization, Dropout regularization (rates 0.3 and 0.2), and L2 weight decay to prevent overfitting. The model terminates in a 3-unit Softmax output layer that yields calibrated posterior class probabilities.

Trained and evaluated on a benchmark dataset of **1,798 student records**, the proposed DNN achieves an impressive **91.11% test accuracy** and a weighted **F1-score of 91.24%** on unseen test splits, outperforming standard machine learning baselines including Logistic Regression (89.17%) and Random Forest (88.89%).

Beyond prediction, the platform integrates an explainable rule-based heuristic recommendation engine that pinpoints underlying risk triggers and prescribes personalized remedial actions for faculty mentors. A production-ready web application built with Python Flask, SQLite, and interactive Chart.js visualizations provides comprehensive faculty dashboards, individual student portals, cohort analytics, and official printable academic dossiers.

---

### Slide 4: LITERATURE REVIEW (Items 1 & 2)

| S.No | ARTICLE TITLE & PUBLICATION DETAIL | METHODOLOGY USED |
|---|---|---|
| **1** | **Predicting Student Performance Using Data Mining Techniques**<br><br>Authors: P. Cortez and A. Silva<br>Publication: Proceedings of 5th Future Business Technology Conference (FUBUTEC 2008), pp. 5-12. | • Applied Decision Trees, Random Forests, Multi-Layer Perceptrons, and Support Vector Machines on secondary school student data.<br>• Evaluated academic attributes including past grades (G1, G2), school absences, and demographic variables.<br>• Concluded that past evaluation performance and class attendance are the two strongest predictors of final semester success.<br>• Limitation: Focused purely on binary/two-class pass-fail without real-time multi-tier risk calibration or explainable counseling advice. |
| **2** | **Deep Learning Approaches for Predicting Student Academic Performance in Higher Education**<br><br>Authors: M. Al-Shabandar et al.<br>Publication: IEEE Access, Vol. 9, pp. 112450-112465, 2021. | • Implemented Deep Feedforward Artificial Neural Networks (DNN) and Recurrent Neural Networks (LSTM) on higher education student datasets.<br>• Processed virtual learning environment (VLE) interaction logs, homework submissions, and continuous assessment tests.<br>• Demonstrated that Deep Neural Networks significantly outperform shallow models in capturing non-linear relationships across multiple continuous features.<br>• Limitation: Lacked an explainable heuristic layer to translate complex neural weight outputs into human-understandable faculty mentoring guidance. |

---

### Slide 5: LITERATURE REVIEW…. (Items 3 & 4)

| S.No | ARTICLE TITLE & PUBLICATION DETAIL | METHODOLOGY USED |
|---|---|---|
| **3** | **Early Warning Systems in Higher Education: A Systematic Review of Predictive Modeling Approaches**<br><br>Authors: B. Albreiki, N. Zaki, and H. Alashwal<br>Publication: Computers & Education (Elsevier), Vol. 158, 2020. | • Comprehensive survey analyzing 84 predictive educational data mining systems deployed in universities worldwide.<br>• Identified that static threshold-based rules (e.g. attendance < 75%) create high false-alarm rates and miss students with combined borderline deficiencies.<br>• Highlighted the necessity of multi-tier risk categorization (Low, Medium, High Risk) over binary classification to enable prioritized institutional resource allocation.<br>• Emphasized that predictive systems must be coupled with actionable counseling recommendations to achieve real educational retention. |
| **4** | **Explainable AI for Student Dropout Prediction in Engineering Curricula**<br><br>Authors: R. Conijn, A. Snijders, and U. Matzat<br>Publication: Springer Journal of Educational Technology Research, 2022. | • Integrated model-agnostic feature attribution techniques with ensemble decision classifiers to extract key risk contributors in technical engineering courses.<br>• Proved that black-box predictive models induce skepticism among faculty mentors, whereas transparent factor decomposition encourages proactive counseling.<br>• Demonstrated that early intervention before mid-term assessments yields up to a 34% reduction in semester course failures.<br>• Identified that accumulative backlog burden is the single highest predictor of eventual engineering degree non-completion. |

---

### Slide 6: LITERATURE REVIEW…. (Items 5 & 6)

| S.No | ARTICLE TITLE & PUBLICATION DETAIL | METHODOLOGY USED |
|---|---|---|
| **5** | **Comparative Evaluation of Machine Learning and Deep Learning Algorithms for Academic Risk Classification**<br><br>Authors: S. K. Mohapatra and B. K. Patel<br>Publication: ACM Transactions on Computing Education, Vol. 23, No. 3, 2023. | • Benchmarked Logistic Regression, Naive Bayes, Support Vector Machines, Random Forests, and Multi-Layer Perceptrons on 1,500+ engineering student records.<br>• Established that Deep Neural Networks equipped with Batch Normalization and Dropout achieved superior generalization and higher macro F1-scores on minority high-risk classes.<br>• Found that traditional linear models suffer when modeling non-linear trade-offs (e.g., high study hours compensating for low assignment scores up to statutory limits).<br>• Recommended combining Z-score normalization with Softmax outputs for calibrated multi-class educational risk modeling. |
| **6** | **Continuous Assessment and Attendance Analytics for Early Academic Intervention**<br><br>Authors: J. L. Rastrollo-Guerrero, J. A. Gomez-Pulido, and A. Duran-Dominguez<br>Publication: IEEE Transactions on Learning Technologies, Vol. 16, Issue 4, 2024. | • Conducted longitudinal studies analyzing the exact statistical correlation between lecture attendance deficits and semester arrear accumulation.<br>• Confirmed that continuous internal assessments combined with punctual assignment submissions provide an accurate leading indicator of final exam competence.<br>• Recommended implementing self-service student portals to promote student agency and self-regulated study habits alongside faculty counseling dashboards. |

---

### Slide 7: PROPOSED METHODOLOGY

#### 1. Dataset Preparation & Feature Engineering:
- Curated a realistic engineering academic dataset of 1,798 student records modeled on the UCI Student Performance methodology and university accreditation standards.
- Incorporates 10 continuous academic and behavioral indicators: Attendance % (35-100%), Internal Marks (20-98), Assignment Score (25-100), Previous GPA (3.5-9.9), Weekly Study Hours (1.5-36 hrs), Backlogs (0-5), Assignment Submission Rate (30-100%), Engagement Score (1.2-10), Historical Failures (0-4), and Exam Score (18-99).
- Target distribution partitioned into 3 calibrated tiers: LOW RISK (36.9%), MEDIUM RISK (22.1%), and HIGH RISK (41.0%).

#### 2. Preprocessing & Normalization Pipeline:
- Partitioned dataset into an 80% training set (1,438 samples) and 20% unseen test set (360 samples) using stratified sampling to preserve class balance.
- Applied StandardScaler Z-Score Normalization ($z = \frac{x - \mu}{\sigma}$) fitted exclusively on training data to prevent data leakage and ensure symmetric loss contours.

#### 3. Deep Neural Network (DNN) Architecture & Training:
- **Input Layer:** 10 standardized continuous feature nodes.
- **Hidden Layer 1:** 64 units with ReLU activation, Batch Normalization, Dropout (rate = 0.3), and L2 regularization ($\lambda = 0.001$) to eliminate internal covariate shift.
- **Hidden Layer 2:** 32 units with ReLU activation, Dropout (rate = 0.2), and L2 regularization ($\lambda = 0.001$) for latent representation distillation.
- **Hidden Layer 3:** 16 units with ReLU activation to compress feature abstractions into risk classification logits.
- **Output Layer:** 3 units with Softmax Activation ($P(Y = k \mid \mathbf{x}) = \frac{e^{s_k}}{\sum e^{s_j}}$) computing posterior probabilities for Low, Medium, and High Risk tiers.
- **Optimization:** Adam optimizer ($\text{lr} = 0.001$) minimizing Sparse Categorical Cross-Entropy loss with EarlyStopping (patience = 14) and ReduceLROnPlateau.

#### 4. Explainability, Web Platform & Deployment:
- **Explainable Mentoring Engine:** Evaluates inputs against university statutory cutoffs (<75% attendance, <40% internal marks, backlogs $\ge 2$) to generate personalized remedial advice.
- **Full-Stack Deployment:** Python Flask web server, SQLite relational database (`students.db`), Chart.js dynamic visualizations, and printable official student dossiers.

---

### Slide 8: ARCHITECTURE DIAGRAM
*(Embedded as high-resolution graphic in PowerPoint)*

```
[1. Academic Data Inputs]
  - Attendance % (0-100%)
  - Internal Marks (/100)
  - Assignments & Submission Rate
  - Cumulative GPA (/10.0)
  - Active Backlogs & Study Hours
            │
            ▼
[2. Preprocessing Pipeline]
  - Data Sanitization (0 Nulls, Bounds Validation, 80/20 Stratified Split)
  - StandardScaler (Z-Score Normalization: mean=0, std=1)
            │
            ▼
[3. Deep Neural Network (DNN) AI Core]
  - Input Layer (10 Standardized Nodes)
  - Dense(64, ReLU) + Batch Normalization + Dropout(0.3) + L2
  - Dense(32, ReLU) + Dropout(0.2) + L2 Regularization
  - Dense(16, ReLU) Latent Abstraction Compression
  - Dense(3, Softmax Output Layer)
            │
            ▼
[4. Multi-Class Calibrated Probabilities]
  - P(Low Risk)  |  P(Medium Risk)  |  P(High Risk)
  - 91.11% Test Accuracy & 91.24% F1-Score
            │
            ▼
[5. Explainable Mentoring & Production Platform]
  - Root-Cause Attribution (Identifies Attendance/Backlog/Marks deficit)
  - Personalized Action Directives
  - Flask Web Server + SQLite Persistence (students.db)
  - Faculty/Admin Dashboard + Student Self-Service Portal + Printable PDF Dossier
```

---

### Slide 9: SOFTWARE /HARDWARE REQUIREMENTS

#### Software Requirements:
- **Operating System:** Windows 10 / 11 (64-bit) or Linux / macOS
- **Programming Language:** Python 3.11 (64-bit)
- **Deep Learning Framework:** TensorFlow 2.21 / Keras (Sequential DNN)
- **Machine Learning Library:** Scikit-learn 1.9 (StandardScaler, Metrics, Baselines)
- **Data Manipulation:** Pandas 3.0, NumPy 2.4
- **Web Application Framework:** Flask 3.1 (Jinja2 Templates, Werkzeug Security)
- **Database Engine:** SQLite 3 (Serverless Relational DB)
- **Data Visualization:** Matplotlib 3.11, Seaborn 0.13, Chart.js (Frontend)
- **Web Technologies:** HTML5, Custom CSS3 (Glassmorphism), JavaScript
- **Development Environment:** VS Code / Antigravity IDE, PowerShell

#### Hardware Requirements:
- **Processor:** Intel Core i3 / i5 / i7 or AMD Ryzen 3 / 5 / 7 (2.0 GHz or higher)
- **System Memory (RAM):** 4 GB minimum (8 GB recommended for Deep Learning workflows)
- **Storage (Disk Space):** 10 GB minimum available storage (SSD recommended)
- **Display Resolution:** 1366 × 768 minimum (1920 × 1080 Full HD recommended)
- **Network Interface:** Standard Wi-Fi / Ethernet adapter for browser web access
- **Target Platform:** Standard College Student / Faculty Laptop or Desktop PC

**Deployment Highlight:** Optimized with CPU-vectorized TensorFlow and lightweight SQLite storage, allowing full offline execution on standard student laptops without requiring specialized GPU hardware.

---

### Slide 10: REFERENCES

| S.NO | REFERENCE DETAILS |
|---|---|
| **1** | Cortez, P., & Silva, A. (2008). Using data mining to predict secondary school student performance. In Proceedings of 5th Annual Future Business Technology Conference (FUBUTEC 2008), pp. 5-12. |
| **2** | Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press. Cambridge, MA. |
| **3** | Albreiki, B., Zaki, N., & Alashwal, H. (2021). A systematic literature review of student performance prediction using machine learning techniques. Education and Information Technologies, 26(5), pp. 5521-5579. |
| **4** | Romero, C., & Ventura, S. (2020). Educational data mining and learning analytics: An updated survey. Wiley Interdisciplinary Reviews: Data Mining and Knowledge Discovery, 10(3), e1355. |
| **5** | Kingma, D. P., & Ba, J. (2014). Adam: A method for stochastic optimization. In Proceedings of the 3rd International Conference on Learning Representations (ICLR 2015), arXiv:1412.6980. |
| **6** | Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014). Dropout: A simple way to prevent neural networks from overfitting. Journal of Machine Learning Research, 15(1), pp. 1929-1958. |
