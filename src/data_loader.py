import pandas as pd
from sklearn.preprocessing import normalize
import kagglehub
from kagglehub import KaggleDatasetAdapter

def load_and_preprocess_data():
    # Load dataset
    file_path = "Admission_Predict_Ver1.1.csv"
    df = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        "mohansacharya/graduate-admissions",
        file_path,
    )

    # Clean column name
    df = df.rename(columns={'Chance of Admit ': 'Chance of Admit'})
    df = df.drop('Serial No.', axis=1)

    # Separate features and target
    X = df.drop('Chance of Admit', axis=1)
    y = df['Chance of Admit']

    # Normalize features
    X_normalized = normalize(X)

    return X_normalized, y, X.columns.tolist()
