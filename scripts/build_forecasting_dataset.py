import pandas as pd

input_file = "data/processed/monthly_activity.csv"
output_file = "data/processed/forecasting_dataset.csv"

print("Loading monthly activity data...")

df = pd.read_csv(input_file)

df["year_month"] = pd.to_datetime(
    df["year"].astype(str) + "-" +
    df["month"].astype(str).str.zfill(2)
)

df = df.sort_values(
    ["business_id", "year_month"]
).copy()

# Previous month activity
df["lag_1"] = (
    df.groupby("business_id")["review_count"]
    .shift(1)
)

# Previous 2 months activity
df["lag_2"] = (
    df.groupby("business_id")["review_count"]
    .shift(2)
)

# Previous 3 months activity
df["lag_3"] = (
    df.groupby("business_id")["review_count"]
    .shift(3)
)

# Rolling average of previous 3 months
df["rolling_mean_3"] = (
    df.groupby("business_id")["review_count"]
    .transform(
        lambda x: x.shift(1).rolling(3).mean()
    )
)

# Calendar features
df["month_sin"] = __import__("numpy").sin(
    2 * __import__("numpy").pi * df["month"] / 12
)

df["month_cos"] = __import__("numpy").cos(
    2 * __import__("numpy").pi * df["month"] / 12
)

# Target = current month's review activity
df["target"] = df["review_count"]

# Remove rows where required historical information is unavailable
df = df.dropna(
    subset=[
        "lag_1",
        "lag_2",
        "lag_3",
        "rolling_mean_3"
    ]
).copy()

# Keep useful columns
features = [
    "business_id",
    "year",
    "month",
    "stars",
    "review_count",
    "lag_1",
    "lag_2",
    "lag_3",
    "rolling_mean_3",
    "month_sin",
    "month_cos",
    "target"
]

df = df[features]

print("\nForecasting Dataset")
print("=" * 50)

print(f"Shape: {df.shape}")

print("\nColumns:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum())

print("\nTarget statistics:")
print(df["target"].describe())

print("\nFirst 10 rows:")
print(df.head(10).to_string(index=False))

df.to_csv(
    output_file,
    index=False
)

print("\nSaved to:")
print(output_file)