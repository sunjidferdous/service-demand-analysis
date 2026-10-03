import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
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

OUTPUT_FILE = "data/processed/additional_forecasting_results.csv"


print("Loading dataset...")
df = pd.read_csv(INPUT_FILE)

# ============================================================
# TIME-BASED SPLIT
# ============================================================

train = df[df["year"] <= 2020].copy()
validation = df[df["year"] == 2021].copy()
test = df[df["year"] >= 2022].copy()

# Same reproducible training sample as existing models
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
print("=" * 60)
print(f"Training sample: {len(X_train):,}")
print(f"Validation:      {len(X_val):,}")
print(f"Test:            {len(X_test):,}")


# ============================================================
# HELPER FUNCTION
# ============================================================

def evaluate_model(model, model_name):

    print("\n" + "=" * 60)
    print(f"Training {model_name}...")
    print("=" * 60)

    model.fit(X_train, y_train)

    print(f"{model_name} training completed.")

    # Validation prediction
    val_pred = model.predict(X_val)

    val_mae = mean_absolute_error(y_val, val_pred)
    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    val_r2 = r2_score(y_val, val_pred)

    # Test prediction
    test_pred = model.predict(X_test)

    test_mae = mean_absolute_error(y_test, test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    test_r2 = r2_score(y_test, test_pred)

    print("\nValidation Results")
    print("-" * 40)
    print(f"MAE:  {val_mae:.4f}")
    print(f"RMSE: {val_rmse:.4f}")
    print(f"R²:   {val_r2:.4f}")

    print("\nTest Results")
    print("-" * 40)
    print(f"MAE:  {test_mae:.4f}")
    print(f"RMSE: {test_rmse:.4f}")
    print(f"R²:   {test_r2:.4f}")

    return {
        "model": model_name,
        "validation_mae": val_mae,
        "validation_rmse": val_rmse,
        "validation_r2": val_r2,
        "test_mae": test_mae,
        "test_rmse": test_rmse,
        "test_r2": test_r2
    }


# ============================================================
# MODEL 1: LINEAR REGRESSION
# ============================================================

linear_model = LinearRegression()

linear_result = evaluate_model(
    linear_model,
    "Linear Regression"
)


# ============================================================
# MODEL 2: DECISION TREE
# ============================================================

tree_model = DecisionTreeRegressor(
    max_depth=20,
    min_samples_leaf=5,
    random_state=42
)

tree_result = evaluate_model(
    tree_model,
    "Decision Tree"
)


# ============================================================
# MODEL 3: HISTOGRAM GRADIENT BOOSTING
# ============================================================

hist_model = HistGradientBoostingRegressor(
    max_iter=200,
    learning_rate=0.05,
    max_leaf_nodes=31,
    min_samples_leaf=20,
    random_state=42
)

hist_result = evaluate_model(
    hist_model,
    "HistGradientBoosting"
)


# ============================================================
# COMBINE RESULTS
# ============================================================

results = pd.DataFrame([
    linear_result,
    tree_result,
    hist_result
])

print("\n\nFINAL ADDITIONAL MODEL COMPARISON")
print("=" * 80)
print(results.to_string(index=False))

# Save results
results.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nResults saved to:")
print(OUTPUT_FILE)