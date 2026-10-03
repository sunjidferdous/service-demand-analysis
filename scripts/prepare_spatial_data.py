import pandas as pd
import os

file_path = "data/raw/yelp_academic_dataset_business.json"
output_file = "data/processed/spatial_businesses.csv"

print("Loading business dataset...")

business_df = pd.read_json(
    file_path,
    lines=True
)

print("Original shape:", business_df.shape)

# Select columns needed for spatial analysis
spatial_df = business_df[
    [
        "business_id",
        "name",
        "city",
        "state",
        "latitude",
        "longitude",
        "stars",
        "review_count",
        "categories"
    ]
].copy()

print("\nSelected columns:")
print(spatial_df.columns.tolist())

print("\nMissing values:")
print(spatial_df.isnull().sum())

# Validate geographic coordinates
spatial_df = spatial_df[
    spatial_df["latitude"].between(-90, 90)
    & spatial_df["longitude"].between(-180, 180)
].copy()

print("\nShape after coordinate validation:")
print(spatial_df.shape)

# Check duplicate business IDs
print("\nDuplicate business IDs:")
print(spatial_df["business_id"].duplicated().sum())

# Save
os.makedirs("data/processed", exist_ok=True)

spatial_df.to_csv(
    output_file,
    index=False
)

print("\nSaved to:")
print(output_file)

print("\nFirst 10 rows:")
print(spatial_df.head(10))