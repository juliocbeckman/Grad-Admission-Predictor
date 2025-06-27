import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def train_and_evaluate_models(models, X_train, y_train, X_test, y_test):
    results = []
    for name, model in models:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        results.append((name, rmse, mae, r2))
    return results

def tune_model(model, param_grid, X_train, y_train):
    search = RandomizedSearchCV(model, param_grid, n_iter=10, scoring='neg_mean_squared_error', cv=5, random_state=42, n_jobs=-1)
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_
