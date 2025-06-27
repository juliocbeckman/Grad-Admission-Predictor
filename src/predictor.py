import numpy as np
from sklearn.preprocessing import normalize
from sklearn.ensemble import ExtraTreesRegressor
from src.data_loader import load_and_preprocess_data

def main():
    X, y, feature_names = load_and_preprocess_data()

    # Train final model
    model = ExtraTreesRegressor(
        n_estimators=200,
        max_depth=20,
        min_samples_split=2,
        min_samples_leaf=2,
        random_state=42
    )
    model.fit(X, y)

    print("Enter your profile information:")
    user_input = []
    for name in feature_names:
        val = float(input(f"{name}: "))
        user_input.append(val)

    input_array = np.array(user_input).reshape(1, -1)
    input_array_norm = normalize(input_array)
    prediction = model.predict(input_array_norm)[0]

    print(f"Estimated Chance of Admission: {prediction:.2f}")

if __name__ == "__main__":
    main()
