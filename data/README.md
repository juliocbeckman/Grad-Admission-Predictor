# 📂 Data Directory

This folder contains instructions to access the dataset.

The following Kaggle Graduate Admissions dataset was used:
https://www.kaggle.com/datasets/mohansacharya/graduate-admissions

To load the data automatically via Python, use the `kagglehub` library.

```python
import kagglehub
from kagglehub import KaggleDatasetAdapter

df = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "mohansacharya/graduate-admissions",
    "Admission_Predict_Ver1.1.csv"
)
