"""Train and compare three regression models for house price prediction.

Run:  python train.py
Creates:  models.joblib  and  metrics.json  (used by app.py)
"""
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

DATA_PATH = "data/housing.csv"
TARGET = "median_house_value"
OCEAN_TYPES = ["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw data and create the model features."""
    df = df.copy()
    df["total_bedrooms"] = df["total_bedrooms"].fillna(df["total_bedrooms"].median())
    df["rooms_per_household"] = df["total_rooms"] / df["households"]
    df["bedrooms_per_room"] = df["total_bedrooms"] / df["total_rooms"]
    df["population_per_household"] = df["population"] / df["households"]
    for ocean in OCEAN_TYPES:
        df[f"ocean_{ocean}"] = (df["ocean_proximity"] == ocean).astype(int)
    return df.drop(columns=["ocean_proximity", TARGET], errors="ignore")


def main() -> None:
    raw = pd.read_csv(DATA_PATH)
    X = build_features(raw)
    y = raw[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Linear Regression": make_pipeline(StandardScaler(), LinearRegression()),
        "Decision Tree": DecisionTreeRegressor(max_depth=12, min_samples_leaf=10, random_state=42),
        "Random Forest": RandomForestRegressor(
            n_estimators=100, max_depth=18, min_samples_leaf=3, n_jobs=-1, random_state=42
        ),
    }

    metrics = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        metrics[name] = {
            "r2": round(float(r2_score(y_test, pred)), 4),
            "mae": round(float(mean_absolute_error(y_test, pred)), 0),
            "rmse": round(float(np.sqrt(mean_squared_error(y_test, pred))), 0),
        }
        print(f"{name:18s} R2={metrics[name]['r2']:.3f}  MAE=${metrics[name]['mae']:,.0f}  RMSE=${metrics[name]['rmse']:,.0f}")

    forest = models["Random Forest"]
    importances = dict(sorted(zip(X.columns, forest.feature_importances_.round(4).tolist()), key=lambda kv: -kv[1]))

    joblib.dump({"models": models, "columns": list(X.columns)}, "models.joblib", compress=3)
    with open("metrics.json", "w") as f:
        json.dump(
            {"metrics": metrics, "importances": importances, "train_rows": len(X_train), "test_rows": len(X_test)},
            f,
            indent=2,
        )
    print("Saved models.joblib and metrics.json")


if __name__ == "__main__":
    main()
