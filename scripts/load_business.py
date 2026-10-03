import pandas as pd

file_path = "data/raw/yelp_academic_dataset_business.json"

business_df = pd.read_json(file_path, lines=True)

print("Shape:", business_df.shape)
print("\nColumns:")
print(business_df.columns.tolist())

print("\nFirst 5 rows:")
print(business_df.head())

print("\nData types:")
print(business_df.dtypes)
print("\nMissing values:")
print(business_df.isnull().sum())
print("\nMissing percentage:")
print((business_df.isnull().sum() / len(business_df) * 100).round(2))
print("\nDuplicate business IDs:")
print(business_df["business_id"].duplicated().sum())
print("\nNumerical summary:")
print(business_df[[
    "latitude",
    "longitude",
    "stars",
    "review_count"
]].describe())
print("\nTop 20 business categories:")
print(business_df["categories"].value_counts().head(20))
all_categories = (
    business_df["categories"]
    .dropna()
    .str.split(", ")
    .explode()
)

print("\nTop 20 individual categories:")
print(all_categories.value_counts().head(20))

print("\nTop 15 states:")
print(business_df["state"].value_counts().head(15))

print("\nTop 20 cities:")
print(business_df["city"].value_counts().head(20))

print("\nBusinesses outside latitude range -90 to 90:")
print(((business_df["latitude"] < -90) | (business_df["latitude"] > 90)).sum())

print("\nBusinesses outside longitude range -180 to 180:")
print(((business_df["longitude"] < -180) | (business_df["longitude"] > 180)).sum())