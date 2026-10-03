import pandas as pd
import os
import time

review_file = "data/raw/yelp_academic_dataset_review.json"
business_file = "data/raw/yelp_academic_dataset_business.json"
output_file = "data/processed/monthly_activity.csv"

# -----------------------------
# Settings
# -----------------------------
CHUNK_SIZE = 100000

start_time = time.time()

print("=" * 60)
print("STEP 1: Processing Yelp reviews")
print("=" * 60)

all_monthly_data = []

chunk_number = 0
total_reviews = 0

# Read reviews chunk by chunk
for review_df in pd.read_json(
    review_file,
    lines=True,
    chunksize=CHUNK_SIZE
):
    chunk_number += 1
    total_reviews += len(review_df)

    # Convert date
    review_df["date"] = pd.to_datetime(review_df["date"])

    # Extract year and month
    review_df["year"] = review_df["date"].dt.year
    review_df["month"] = review_df["date"].dt.month

    # Aggregate reviews by business and month
    monthly_chunk = (
        review_df
        .groupby(["business_id", "year", "month"])
        .size()
        .reset_index(name="review_count")
    )

    all_monthly_data.append(monthly_chunk)

    print(
        f"Chunk {chunk_number:02d} | "
        f"Reviews processed: {total_reviews:,}"
    )

print("\nAll reviews processed.")

# -----------------------------
# Combine all chunk results
# -----------------------------
print("\nCombining chunk results...")

monthly_activity = pd.concat(
    all_monthly_data,
    ignore_index=True
)

# Same business/month may appear in different chunks,
# so aggregate again.
monthly_activity = (
    monthly_activity
    .groupby(["business_id", "year", "month"], as_index=False)
    ["review_count"]
    .sum()
)

print("\nFinal monthly activity shape:")
print(monthly_activity.shape)

print("\nFirst 10 rows:")
print(monthly_activity.head(10))

# -----------------------------
# Load business information
# -----------------------------
print("\n" + "=" * 60)
print("STEP 2: Loading business data")
print("=" * 60)

business_df = pd.read_json(
    business_file,
    lines=True
)

business_columns = [
    "business_id",
    "name",
    "city",
    "state",
    "latitude",
    "longitude",
    "stars",
    "categories"
]

business_small = business_df[business_columns].copy()

# -----------------------------
# Merge review activity
# with business information
# -----------------------------
print("\nMerging datasets...")

final_df = monthly_activity.merge(
    business_small,
    on="business_id",
    how="left"
)

# Reorder columns
final_df = final_df[
    [
        "business_id",
        "name",
        "city",
        "state",
        "latitude",
        "longitude",
        "categories",
        "stars",
        "year",
        "month",
        "review_count"
    ]
]

# -----------------------------
# Basic validation
# -----------------------------
print("\n" + "=" * 60)
print("STEP 3: Validation")
print("=" * 60)

print("\nFinal shape:")
print(final_df.shape)

print("\nMissing values:")
print(final_df.isnull().sum())

print("\nReview count statistics:")
print(final_df["review_count"].describe())

print("\nYear range:")
print(
    final_df["year"].min(),
    "to",
    final_df["year"].max()
)

print("\nSample rows:")
print(final_df.head(10))

# -----------------------------
# Save
# -----------------------------
os.makedirs(
    "data/processed",
    exist_ok=True
)

final_df.to_csv(
    output_file,
    index=False
)

elapsed = time.time() - start_time

print("\n" + "=" * 60)
print("DONE!")
print("=" * 60)

print(f"Saved to: {output_file}")
print(f"Total reviews processed: {total_reviews:,}")
print(f"Final rows: {len(final_df):,}")
print(f"Time taken: {elapsed / 60:.2f} minutes")