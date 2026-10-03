import pandas as pd
import numpy as np

input_file = "data/processed/monthly_activity.csv"
output_file = "data/processed/forecasting_dataset.csv"

print("Loading monthly activity data...")

df = pd.read_csv(input_file)

df["date"] = pd.to_datetime(
    df["year"].astype(str) + "-" +
    df["month"].astype(str).str.zfill(2) +
    "-01"
)

# Keep only required columns
df = df[
    ["business_id", "date", "stars", "review_count"]
].copy()

# Business-wise calendar range
print("Creating complete monthly timeline...")

ranges = (
    df.groupby("business_id")["date"]
    .agg(["min", "max"])
    .reset_index()
)

parts = []

for _, row in ranges.iterrows():
    dates = pd.date_range(
        row["min"],
        row["max"],
        freq="MS"
    )

    parts.append(
        pd.DataFrame({
            "business_id": row["business_id"],
            "date": dates
        })
    )

calendar = pd.concat(
    parts,
    ignore_index=True
)

print(f"Calendar rows: {len(calendar):,}")

# Merge actual review activity
df = calendar.merge(
    df,
    on=["business_id", "date"],
    how="left"
)

# Missing months = zero reviews
df["review_count"] = df["review_count"].fillna(0)

# Business rating is constant in this dataset
df["stars"] = (
    df.groupby("business_id")["stars"]
    .transform("first")
)

df = df.sort_values(
    ["business_id", "date"]
).reset_index(drop=True)

# Lag features
df["lag_1"] = (
    df.groupby("business_id")["review_count"]
    .shift(1)
)

df["lag_2"] = (
    df.groupby("business_id")["review_count"]
    .shift(2)
)

df["lag_3"] = (
    df.groupby("business_id")["review_count"]
    .shift(3)
)

# Rolling mean of previous 3 months
df["rolling_mean_3"] = (
    df.groupby("business_id")["review_count"]
    .transform(
        lambda x: x.shift(1).rolling(3).mean()
    )
)

# Calendar features
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month

df["month_sin"] = np.sin(
    2 * np.pi * df["month"] / 12
)

df["month_cos"] = np.cos(
    2 * np.pi * df["month"] / 12
)

# Target = current month's activity
df["target"] = df["review_count"]

# Remove rows without 3 months of history
df = df.dropna(
    subset=[
        "lag_1",
        "lag_2",
        "lag_3",
        "rolling_mean_3"
    ]
)

# IMPORTANT:
# current review_count is removed because it is the target.
features = [
    "business_id",
    "year",
    "month",
    "stars",
    "lag_1",
    "lag_2",
    "lag_3",
    "rolling_mean_3",
    "month_sin",
    "month_cos",
    "target"
]

df = df[features]

print("\nFinal Forecasting Dataset")
print("=" * 60)

print(f"Shape: {df.shape}")

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