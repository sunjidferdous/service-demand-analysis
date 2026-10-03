import pandas as pd
import numpy as np

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

INPUT_FILE = "data/processed/forecasting_dataset.csv"

FEATURES = [
    "stars",
    "lag_1",
    "lag_2",
    "lag_3",
    "rolling_mean_3",
    "month_sin",
    "month_cos"
]

TARGET = "target"

print("Loading dataset...")

df = pd.read_csv(INPUT_FILE)

train = df[df["year"] <= 2020]
validation = df[df["year"] == 2021]
test = df[df["year"] >= 2022]

# Same sample size and random seed as Random Forest
train_sample = train.sample(
    n=min(1_000_000, len(train)),
    random_state=42
)

X_train = train_sample[FEATURES]
y_train = train_sample[TARGET]

X_val = validation[FEATURES]
y_val = validation[TARGET]

X_test = test[FEATURES]
y_test = test[TARGET]

print("\nDataset sizes")
print("=" * 50)
print(f"Training sample: {len(X_train):,}")
print(f"Validation:      {len(X_val):,}")
print(f"Test:            {len(X_test):,}")

print("\nTraining XGBoost...")

model = XGBRegressor(
    n_estimators=300,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    eval_metric="rmse",
    random_state=42,
    n_jobs=-1,
    tree_method="hist"
)

model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)

print("Training completed.")

# Validation
val_pred = model.predict(X_val)

val_mae = mean_absolute_error(y_val, val_pred)
val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
val_r2 = r2_score(y_val, val_pred)

# Test
test_pred = model.predict(X_test)

test_mae = mean_absolute_error(y_test, test_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
test_r2 = r2_score(y_test, test_pred)

print("\nValidation Results")
print("=" * 50)
print(f"MAE:  {val_mae:.4f}")
print(f"RMSE: {val_rmse:.4f}")
print(f"R²:   {val_r2:.4f}")

print("\nTest Results")
print("=" * 50)
print(f"MAE:  {test_mae:.4f}")
print(f"RMSE: {test_rmse:.4f}")
print(f"R²:   {test_r2:.4f}")

# Feature importance
importance = pd.DataFrame({
    "feature": FEATURES,
    "importance": model.feature_importances_
}).sort_values(
    "importance",
    ascending=False
)

print("\nFeature Importance")
print("=" * 50)
print(importance.to_string(index=False))

# Save results
results = pd.DataFrame({
    "model": ["XGBoost"],
    "validation_mae": [val_mae],
    "validation_rmse": [val_rmse],
    "validation_r2": [val_r2],
    "test_mae": [test_mae],
    "test_rmse": [test_rmse],
    "test_r2": [test_r2]
})

results.to_csv(
    "data/processed/xgboost_results.csv",
    index=False
)

print("\nResults saved to:")
print("data/processed/xgboost_results.csv")