import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LinearRegression
from src.data_loader import load_and_preprocess_data

def plot_feature_importances():
    X, y, feature_names = load_and_preprocess_data()

    # Extra Trees
    etr = ExtraTreesRegressor().fit(X, y)
    et_importance = etr.feature_importances_

    plt.figure(figsize=(8, 6))
    plt.barh(feature_names, et_importance, color='teal')
    plt.title("Feature Importances (Extra Trees)")
    plt.tight_layout()
    plt.show()

    # Linear Regression
    lr = LinearRegression().fit(X, y)
    lr_importance = np.abs(lr.coef_)

    plt.figure(figsize=(8, 6))
    plt.barh(feature_names, lr_importance, color='slateblue')
    plt.title("Feature Importances (Linear Regression)")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_feature_importances()
