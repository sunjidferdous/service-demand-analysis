import pandas as pd
import matplotlib.pyplot as plt

rf = pd.read_csv("data/processed/random_forest_results.csv")
xgb = pd.read_csv("data/processed/xgboost_results.csv")

comparison = pd.DataFrame({
    "Model": ["Random Forest", "XGBoost"],
    "Validation MAE": [rf["validation_mae"].iloc[0], xgb["validation_mae"].iloc[0]],
    "Validation RMSE": [rf["validation_rmse"].iloc[0], xgb["validation_rmse"].iloc[0]],
    "Validation R2": [rf["validation_r2"].iloc[0], xgb["validation_r2"].iloc[0]],
    "Test MAE": [rf["test_mae"].iloc[0], xgb["test_mae"].iloc[0]],
    "Test RMSE": [rf["test_rmse"].iloc[0], xgb["test_rmse"].iloc[0]],
    "Test R2": [rf["test_r2"].iloc[0], xgb["test_r2"].iloc[0]],
})

print("\nModel Comparison")
print("=" * 80)
print(comparison.to_string(index=False))

comparison.to_csv(
    "data/processed/forecasting_model_comparison.csv",
    index=False
)

# Validation metrics
comparison.set_index("Model")[["Validation MAE", "Validation RMSE"]].plot(
    kind="bar"
)
plt.title("Forecasting Model Comparison - Validation")
plt.ylabel("Error")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("data/processed/forecasting_validation_comparison.png", dpi=300)
plt.close()

# Test metrics
comparison.set_index("Model")[["Test MAE", "Test RMSE"]].plot(
    kind="bar"
)
plt.title("Forecasting Model Comparison - Test")
plt.ylabel("Error")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("data/processed/forecasting_test_comparison.png", dpi=300)
plt.close()

print("\nFiles saved successfully.")