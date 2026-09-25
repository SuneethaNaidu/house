import os
import warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import (
    ExtraTreesRegressor,
    RandomForestRegressor,
    RandomForestRegressor,
    HistGradientBoostingRegressor
)

from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================

DATA_FILE = "UCI_Real_Estate_Valuation.xlsx"
MODEL_FILE = "house_price_model.pkl"

RANDOM_STATE = 42

# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading dataset...")

df = pd.read_excel(DATA_FILE)

print("Dataset shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

# ============================================================
# CLEAN COLUMN NAMES
# ============================================================

df.columns = (
    df.columns
    .str.strip()
    .str.replace("\n", " ", regex=False)
)

# ============================================================
# AUTOMATIC COLUMN DETECTION
# ============================================================

target_candidates = [
    "Y house price of unit area",
    "house price of unit area",
    "price",
    "Price"
]

target_column = None

for col in target_candidates:
    if col in df.columns:
        target_column = col
        break

if target_column is None:
    raise ValueError(
        "Target column not found. Available columns:\n"
        + "\n".join(df.columns)
    )

print("\nTarget:", target_column)

# ============================================================
# REMOVE ID COLUMN
# ============================================================

id_columns = []

for col in df.columns:
    if "transaction date" in col.lower():
        id_columns.append(col)

    if col.lower() in ["no", "id", "index"]:
        id_columns.append(col)

if id_columns:
    print("Removing:", id_columns)
    df = df.drop(columns=id_columns)

# ============================================================
# NUMERIC CONVERSION
# ============================================================

for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Remove invalid rows
df = df.dropna(subset=[target_column])

# Fill missing numerical values
for col in df.columns:
    if col != target_column:
        df[col] = df[col].fillna(df[col].median())

# ============================================================
# FEATURE ENGINEERING
# ============================================================

print("\nCreating advanced features...")

X = df.drop(columns=[target_column]).copy()
y = df[target_column].copy()

# ----------------------------
# Date features
# ----------------------------

date_columns = [
    col for col in X.columns
    if "date" in col.lower()
]

for col in date_columns:

    try:

        date = pd.to_datetime(X[col])

        X[col + "_year"] = date.dt.year
        X[col + "_month"] = date.dt.month
        X[col + "_day"] = date.dt.day

        X.drop(columns=[col], inplace=True)

    except:
        pass

# ----------------------------
# Location features
# ----------------------------

# UCI dataset has:
# X5 = latitude
# X6 = longitude

latitude_col = None
longitude_col = None

for col in X.columns:

    lower = col.lower()

    if "latitude" in lower:
        latitude_col = col

    if "longitude" in lower:
        longitude_col = col


if latitude_col and longitude_col:

    # Distance from approximate city center
    city_lat = X[latitude_col].median()
    city_lon = X[longitude_col].median()

    X["distance_from_center"] = np.sqrt(
        (X[latitude_col] - city_lat) ** 2 +
        (X[longitude_col] - city_lon) ** 2
    )

# ============================================================
# INTERACTION FEATURES
# ============================================================

numeric_columns = X.select_dtypes(
    include=np.number
).columns.tolist()

for col in numeric_columns:

    if "age" in col.lower():

        X[col + "_squared"] = X[col] ** 2


# ============================================================
# OUTLIER CLIPPING
# ============================================================

for col in X.columns:

    if pd.api.types.is_numeric_dtype(X[col]):

        lower = X[col].quantile(0.01)
        upper = X[col].quantile(0.99)

        X[col] = X[col].clip(lower, upper)

# ============================================================
# TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE
)

print("\nTraining rows:", len(X_train))
print("Testing rows:", len(X_test))

# ============================================================
# MODELS
# ============================================================

models = {

    "XGBoost": XGBRegressor(
        n_estimators=1000,
        learning_rate=0.025,
        max_depth=6,
        min_child_weight=3,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.05,
        reg_lambda=1.0,
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        n_jobs=-1
    ),

    "ExtraTrees": ExtraTreesRegressor(
        n_estimators=700,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=0.9,
        random_state=RANDOM_STATE,
        n_jobs=-1
    ),

    "RandomForest": RandomForestRegressor(
        n_estimators=600,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=0.9,
        random_state=RANDOM_STATE,
        n_jobs=-1
    ),

    "HistGradientBoosting": HistGradientBoostingRegressor(
        max_iter=500,
        learning_rate=0.04,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        random_state=RANDOM_STATE
    )
}

# ============================================================
# TRAIN MODELS
# ============================================================

predictions = {}
trained_models = {}

results = []

print("\n" + "=" * 70)
print("MODEL TRAINING")
print("=" * 70)

for name, model in models.items():

    print(f"\nTraining {name}...")

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    predictions[name] = pred
    trained_models[name] = model

    mae = mean_absolute_error(y_test, pred)

    rmse = np.sqrt(
        mean_squared_error(y_test, pred)
    )

    r2 = r2_score(y_test, pred)

    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })

    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R2   : {r2:.4f}")

# ============================================================
# ENSEMBLE
# ============================================================

print("\n" + "=" * 70)
print("ENSEMBLE MODEL")
print("=" * 70)

# Weight models according to validation performance
r2_values = np.array([
    result["R2"]
    for result in results
])

# Normalize weights
weights = np.maximum(r2_values, 0)

if weights.sum() == 0:
    weights = np.ones(len(weights))

weights = weights / weights.sum()

print("\nEnsemble weights:")

for name, weight in zip(models.keys(), weights):

    print(
        f"{name}: {weight:.3f}"
    )

ensemble_prediction = np.zeros(
    len(X_test)
)

for weight, name in zip(
    weights,
    models.keys()
):

    ensemble_prediction += (
        weight *
        predictions[name]
    )

ensemble_mae = mean_absolute_error(
    y_test,
    ensemble_prediction
)

ensemble_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        ensemble_prediction
    )
)

ensemble_r2 = r2_score(
    y_test,
    ensemble_prediction
)

print("\nENSEMBLE RESULTS")

print(
    "MAE:",
    round(ensemble_mae, 4)
)

print(
    "RMSE:",
    round(ensemble_rmse, 4)
)

print(
    "R2:",
    round(ensemble_r2, 4)
)

results.append({
    "Model": "ENSEMBLE",
    "MAE": ensemble_mae,
    "RMSE": ensemble_rmse,
    "R2": ensemble_r2
})

# ============================================================
# PREDICTION RANGE
# ============================================================

residuals = y_test.values - ensemble_prediction

residual_std = np.std(residuals)

print("\nResidual standard deviation:")
print(residual_std)

# ============================================================
# FEATURE IMPORTANCE
# ============================================================

best_model = trained_models["XGBoost"]

feature_importance = pd.DataFrame({

    "Feature": X.columns,

    "Importance":
        best_model.feature_importances_

}).sort_values(
    "Importance",
    ascending=False
)

print("\nTop Features:")

print(
    feature_importance.head(15)
)

# ============================================================
# SAVE EVERYTHING
# ============================================================

artifact = {

    "models": trained_models,

    "weights": dict(
        zip(
            models.keys(),
            weights
        )
    ),

    "features": X.columns.tolist(),

    "target": target_column,

    "residual_std":
        residual_std,

    "feature_importance":
        feature_importance,

    "results":
        pd.DataFrame(results)

}

joblib.dump(
    artifact,
    MODEL_FILE
)

print(
    f"\nModel saved to: {MODEL_FILE}"
)

print("\nTraining completed successfully.")
