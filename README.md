# 🎓 Grad Admission Predictor

This project builds and evaluates several machine learning models to predict a student's chance of being admitted to a graduate program based on standardized test scores, academic performance, and research experience.

## 🔍 Problem Overview

Applying to graduate school is competitive and uncertain. Applicants often wonder how likely they are to get accepted based on their academic profile. This project addresses that question using predictive analytics and machine learning. 

By modeling the probability of admission, the aim is to:
- Help students set realistic expectations and goals
- Enable institutions to explore trends in applicant profiles
- Showcase real-world regression modeling techniques using Python

## 📊 Dataset

- **Source:** [Graduate Admission 2 (Kaggle)](https://www.kaggle.com/datasets/mohansacharya/graduate-admissions)
- **Format:** 500 records × 9 columns (CSV)
- **Target:** `Chance of Admit` (continuous from 0.0 to 1.0)

**Key Features:**
- GRE Score (out of 340)
- TOEFL Score (out of 120)
- University Rating (1–5)
- Statement of Purpose (SOP)
- Letter of Recommendation (LOR)
- CGPA (out of 10)
- Research Experience (0 or 1)

**Cleaning Steps:**
- Renamed target column to remove trailing space
- Verified no null values
- Normalized all features before model training

**Ethical Notes:**
- Dataset is anonymized (CC0 license)
- Does not include race, gender, or income, reducing risk of bias
- Model should not be used for real admissions decisions

## 🧠 Machine Learning Models

We compared multiple regression algorithms:
- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor
- AdaBoost Regressor
- Extra Trees Regressor (Best)
- K-Nearest Neighbors
- Support Vector Regression

### Why Extra Trees?
It handles non-linear relationships well, is robust to overfitting, and provides interpretable feature importances.

## 🧪 Evaluation Approach

- **Data Split:** 80% training / 20% testing
- **Tuning:** RandomizedSearchCV with 5-fold cross-validation
- **Metrics:**
  - RMSE: Root Mean Squared Error
  - MAE: Mean Absolute Error
  - R² Score: Proportion of variance explained

### 📈 Visualized Comparisons:
- Sorted bar charts for RMSE, MAE, and R²
- Feature importance plots from both Extra Trees and Linear Regression

## ✅ Key Results

| Model             | R² Score | RMSE  | MAE   |
|------------------|----------|-------|-------|
| Extra Trees      | **0.86** | 0.05  | 0.04  |
| Gradient Boosting| 0.82     | 0.06  | 0.05  |
| Linear Regression| 0.78     | 0.07  | 0.06  |

- **Top Features:** CGPA, GRE Score, TOEFL Score
- **Insight:** Research experience improves model predictions, especially in tree-based models

## ⚠️ Challenges

- Feature engineering was minimal due to the dataset's simplicity
- Dataset size (500 rows) limited deep model exploration
- Overlap in feature distributions made it difficult for some models (e.g., KNN, SVR) to perform well

## 🔁 Next Steps

- Add qualitative features (e.g., essay scores, diversity metrics)
- Deploy as a web app using Flask or Streamlit
- Extend to multiclass outcomes (e.g., Accepted, Waitlisted, Denied)
- Collaborate with local institutions such as Saint Mary's College, Diablo Valley College, or UC Berkeley to expand the model using recent, real-world admissions data

## 🚀 How to Run

Clone the repository and install dependencies:

```bash
git clone https://github.com/juliocbeckman/grad-admission-predictor.git
cd grad-admission-predictor
pip install -r requirements.txt
