import pandas as pd

input_file = "data/processed/dbscan_businesses.csv"
output_file = "data/processed/dbscan_cluster_summary.csv"

df = pd.read_csv(input_file)

# Noise বাদ দিচ্ছি
clustered = df[df["cluster_id"] != -1].copy()

cluster_summary = (
    clustered
    .groupby("cluster_id")
    .agg(
        business_count=("business_id", "count"),
        total_review_count=("review_count", "sum"),
        avg_review_count=("review_count", "mean"),
        avg_stars=("stars", "mean"),
        centroid_latitude=("latitude", "mean"),
        centroid_longitude=("longitude", "mean"),
    )
    .reset_index()
)

# Business count অনুযায়ী rank
cluster_summary["business_rank"] = (
    cluster_summary["business_count"]
    .rank(method="dense", ascending=False)
    .astype(int)
)

# Review activity অনুযায়ী rank
cluster_summary["review_activity_rank"] = (
    cluster_summary["total_review_count"]
    .rank(method="dense", ascending=False)
    .astype(int)
)

cluster_summary = cluster_summary.sort_values(
    "business_count",
    ascending=False
)

cluster_summary.to_csv(
    output_file,
    index=False
)

print("Cluster Summary")
print("=" * 60)

print(f"Total clusters: {len(cluster_summary):,}")

print("\nTop 20 clusters by business count:")
print(
    cluster_summary[
        [
            "cluster_id",
            "business_count",
            "total_review_count",
            "avg_review_count",
            "avg_stars",
            "centroid_latitude",
            "centroid_longitude",
        ]
    ].head(20).to_string(index=False)
)

print("\nTop 20 clusters by review activity:")

top_review_activity = cluster_summary.sort_values(
    "total_review_count",
    ascending=False
)

print(
    top_review_activity[
        [
            "cluster_id",
            "business_count",
            "total_review_count",
            "avg_review_count",
            "avg_stars",
            "centroid_latitude",
            "centroid_longitude",
        ]
    ].head(20).to_string(index=False)
)

print("\nSaved to:")
print(output_file)