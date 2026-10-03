import pandas as pd

business_file = "data/raw/yelp_academic_dataset_business.json"
review_file = "data/raw/yelp_academic_dataset_review.json"

# Load business data
business_df = pd.read_json(business_file, lines=True)

# Load only first 10,000 reviews
review_df = pd.read_json(
    review_file,
    lines=True,
    nrows=10000
)

print("Business shape:", business_df.shape)
print("Review sample shape:", review_df.shape)

# Select only useful business columns
business_small = business_df[
    [
        "business_id",
        "name",
        "city",
        "state",
        "latitude",
        "longitude",
        "categories"
    ]
]

# Join review data with business data
merged_df = review_df.merge(
    business_small,
    on="business_id",
    how="inner"
)

print("\nMerged shape:", merged_df.shape)

print("\nMerged columns:")
print(merged_df.columns.tolist())

print("\nFirst 5 merged rows:")
print(
    merged_df[
        [
            "business_id",
            "name",
            "city",
            "state",
            "latitude",
            "longitude",
            "stars",
            "date"
        ]
    ].head()
)