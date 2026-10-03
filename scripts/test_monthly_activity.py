import pandas as pd

review_file = "data/raw/yelp_academic_dataset_review.json"

print("Loading first 100,000 reviews...")

review_df = pd.read_json(
    review_file,
    lines=True,
    nrows=100000
)

print("\nOriginal shape:")
print(review_df.shape)

# Convert date to datetime
review_df["date"] = pd.to_datetime(review_df["date"])

# Extract year and month
review_df["year"] = review_df["date"].dt.year
review_df["month"] = review_df["date"].dt.month

# Count reviews for each business per month
monthly_activity = (
    review_df
    .groupby(["business_id", "year", "month"])
    .size()
    .reset_index(name="review_count")
)

print("\nMonthly activity shape:")
print(monthly_activity.shape)

print("\nFirst 20 rows:")
print(monthly_activity.head(20))

print("\nReview count statistics:")
print(monthly_activity["review_count"].describe())