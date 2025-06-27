# 📈 Final Report: Grad Admission Predictor

## 1. Introduction

This project predicts a student's likelihood of being admitted to graduate school using machine learning models trained on academic and profile features. Our goal was to demonstrate modeling techniques, evaluation strategies, and real-world prediction pipelines in Python.

## 2. Dataset

- **Source**: [Kaggle – Graduate Admissions](https://www.kaggle.com/datasets/mohansacharya/graduate-admissions)
- **Size**: 500 rows × 9 columns
- **Target**: `Chance of Admit` (0.0–1.0)

**Key features**: GRE, TOEFL, CGPA, SOP, LOR, University Rating, Research experience.

**Cleaning steps**:
- Removed whitespace in column names
- Dropped "Serial No." field
- Normalized features using `sklearn.preprocessing.normalize`

## 3. Methodology

We trained and compared multiple regression models:
- Linear Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- AdaBoost
- Extra Trees (top performer)
- K-Nearest Neighbors
- Support Vector Regression

### Model Tuning:
- Used `RandomizedSearchCV` with 5-fold cross-validation
- Tuned parameters like number of estimators, depth, learning rate, etc.

### Metrics Used:
- **RMSE** (Root Mean Squared Error)
- **MAE** (Mean Absolute Error)
- **R² Score**

## 4. Results Summary

| Model             | R² Score | RMSE  | MAE   |
|------------------|----------|-------|-------|
| Extra Trees      | **0.86** | 0.05  | 0.04  |
| Gradient Boosting| 0.82     | 0.06  | 0.05  |
| Linear Regression| 0.78     | 0.07  | 0.06  |

### Feature Importance:
- **Top factors**: CGPA, GRE Score, TOEFL Score, Research
- SOP/LOR/Rating showed weaker influence

## 5. Insights

- Tree-based models captured non-linear interactions well.
- CGPA is the strongest single predictor.
- Research experience contributed meaningfully despite being binary.

## 6. Challenges

- Dataset size limited deep learning or more complex ensembling.
- Highly correlated academic features made model selection less discriminative.
- Feature engineering was constrained by limited input variables.

## 7. Future Work

- Add more qualitative features (e.g. essay strength, extracurriculars).
- Deploy as a Flask or Streamlit app.
- Consider classification (Admit vs. Reject) or multi-class labeling.

## 8. Conclusion

This project demonstrates how structured academic data can be used to train regression models that output interpretable predictions. It showcases a full ML pipeline including preprocessing, training, evaluation, and insight extraction using Python and scikit-learn.

📁 Project by Julio Beckman & Jay Ortiz-Pimentel  
