import matplotlib.pyplot as plt
import pandas as pd

def plot_metrics(results):
    df = pd.DataFrame(results, columns=["Model", "RMSE", "MAE", "R2"])

    # RMSE
    df_rmse = df.sort_values(by="RMSE")
    plt.figure(figsize=(10, 4))
    plt.bar(df_rmse["Model"], df_rmse["RMSE"])
    plt.xticks(rotation=45)
    plt.title("Models Sorted by RMSE")
    plt.tight_layout()
    plt.show()

    # MAE
    df_mae = df.sort_values(by="MAE")
    plt.figure(figsize=(10, 4))
    plt.bar(df_mae["Model"], df_mae["MAE"])
    plt.xticks(rotation=45)
    plt.title("Models Sorted by MAE")
    plt.tight_layout()
    plt.show()

    # R²
    df_r2 = df.sort_values(by="R2", ascending=False)
    plt.figure(figsize=(10, 4))
    plt.bar(df_r2["Model"], df_r2["R2"])
    plt.xticks(rotation=45)
    plt.title("Models Sorted by R²")
    plt.tight_layout()
    plt.show()
