# Itahari House Price Prediction & Market Insight System
# Fully corrected version – compatible with Python 3.13/3.14 and all sklearn versions.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

DATA_CSV = "itahari_house_prices_large.csv"

def load_data(path=DATA_CSV):
    return pd.read_csv(path, parse_dates=["date_listed"])

def basic_cleaning(df):
    df = df.copy()
    df["area_sqft"] = df["area_sqft"].fillna(df["area_sqft"].median()).astype(int)
    df["bedrooms"] = df["bedrooms"].fillna(df["bedrooms"].median()).astype(int)
    df["bathrooms"] = df["bathrooms"].fillna(df["bathrooms"].median()).astype(int)
    df["building_age"] = df["building_age"].fillna(df["building_age"].median()).astype(int)
    return df

def feature_engineering(df):
    df = df.copy()
    df["ppsqft"] = df["price_npr"] / df["area_sqft"]
    df["furnished_simple"] = df["furnished"].map({
        "Unfurnished":"Unfurnished",
        "Semi-Furnished":"Semi",
        "Furnished":"Furnished"
    })
    zone_median = df.groupby("zone")["ppsqft"].median().to_dict()
    df["zone_price_rank"] = df["zone"].map(zone_median)
    df["age_bin"] = pd.cut(df["building_age"],
                           bins=[-1,5,15,30,100],
                           labels=["New","Young","Mature","Old"])
    return df

def eda_plots(df, out_dir="figures"):
    os.makedirs(out_dir, exist_ok=True)
    plt.style.use("ggplot")

    # Price distribution
    plt.figure(figsize=(8,5))
    plt.hist(df["price_npr"], bins=60)
    plt.title("Price Distribution (NPR)")
    plt.xlabel("Price (NPR)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "price_distribution.png"))
    plt.close()

    # Price vs Area
    plt.figure(figsize=(8,6))
    plt.scatter(df["area_sqft"], df["price_npr"], alpha=0.4)
    plt.title("Price vs Area (sqft)")
    plt.xlabel("Area (sqft)")
    plt.ylabel("Price (NPR)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "price_vs_area.png"))
    plt.close()

    # Median price by zone
    zone_med = df.groupby("zone")["price_npr"].median().sort_values(ascending=False)
    plt.figure(figsize=(10,6))
    zone_med.plot(kind="bar")
    plt.title("Median Price by Zone")
    plt.xlabel("Zone")
    plt.ylabel("Median Price (NPR)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "median_price_by_zone.png"))
    plt.close()

def prepare_model_data(df):
    df = df.copy()

    features = [
        "zone","house_type","area_sqft","bedrooms","bathrooms",
        "building_age","road_width_m","distance_to_center_km",
        "has_garden","has_parking","furnished_simple","zone_price_rank"
    ]

    df = df.dropna(subset=["price_npr"])
    X = df[features]
    y = df["price_npr"]

    numeric_features = [
        "area_sqft","building_age","road_width_m",
        "distance_to_center_km","zone_price_rank"
    ]

    categorical_features = [
        "zone","house_type","furnished_simple",
        "has_garden","has_parking","bedrooms","bathrooms"
    ]

    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
    categorical_transformer = Pipeline(steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    return X, y, preprocessor

def train_models(X, y, preprocessor, out_dir="models"):
    os.makedirs(out_dir, exist_ok=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    rf = Pipeline(steps=[
        ("pre", preprocessor),
        ("model", RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1))
    ])

    gb = Pipeline(steps=[
        ("pre", preprocessor),
        ("model", GradientBoostingRegressor(n_estimators=200, random_state=42))
    ])

    rf.fit(X_train, y_train)
    gb.fit(X_train, y_train)

    for name, model in [("RandomForest", rf), ("GradientBoosting", gb)]:
        preds = model.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        mse = mean_squared_error(y_test, preds)
        rmse = np.sqrt(mse)  # ✔ compatible and stable
        r2 = r2_score(y_test, preds)

        print(f"\nModel: {name}")
        print(f"MAE : {mae:,.0f} NPR")
        print(f"RMSE: {rmse:,.0f} NPR")
        print(f"R2  : {r2:.4f}")
        print("-" * 40)

    joblib.dump(rf, os.path.join(out_dir, "random_forest_pipeline.joblib"))
    joblib.dump(gb, os.path.join(out_dir, "gradboost_pipeline.joblib"))

    return rf, gb

def simple_forecast(df, out_dir="figures"):
    os.makedirs(out_dir, exist_ok=True)

    yearly = (
        df.groupby("year_listed")["price_npr"]
        .median()
        .reset_index()
        .sort_values("year_listed")
    )

    from sklearn.linear_model import LinearRegression

    X = yearly["year_listed"].values.reshape(-1, 1)
    y = yearly["price_npr"].values

    model = LinearRegression().fit(X, y)

    future_years = np.arange(yearly["year_listed"].max() + 1,
                             yearly["year_listed"].max() + 11).reshape(-1, 1)

    preds = model.predict(np.vstack([X, future_years]))
    all_years = np.vstack([X, future_years]).ravel()

    plt.figure(figsize=(10, 6))
    plt.plot(all_years[:len(X)], y, marker="o", label="Historical median")
    plt.plot(all_years[len(X):], preds[len(X):],
             marker="o", linestyle="--", label="Forecast (next 10 years)")

    plt.title("Median House Price Forecast (Simple Linear Fit)")
    plt.xlabel("Year")
    plt.ylabel("Median Price (NPR)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "median_price_forecast.png"))
    plt.close()

def main():
    print("Loading data...")
    df = load_data(DATA_CSV)
    print("Rows:", len(df))

    df = basic_cleaning(df)
    df = feature_engineering(df)

    print("Running EDA plots...")
    eda_plots(df)

    print("Preparing model data...")
    X, y, preprocessor = prepare_model_data(df)

    print("Training models (this may take a few minutes)...")
    train_models(X, y, preprocessor)

    print("Creating forecast plots...")
    simple_forecast(df)

    print("\nAll done! Check the 'figures/' and 'models/' folders.")
    
if __name__ == "__main__":
    main()
