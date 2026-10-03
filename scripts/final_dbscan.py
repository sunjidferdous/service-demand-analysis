import pandas as pd
import numpy as np

from sklearn.cluster import DBSCAN


file_path = "data/processed/spatial_businesses.csv"
output_file = "data/processed/dbscan_businesses.csv"

df = pd.read_csv(file_path)

coordinates = df[
    ["latitude", "longitude"]
].to_numpy()

coordinates_rad = np.radians(coordinates)

earth_radius_km = 6371.0

eps_km = 0.5
min_samples = 10

eps_radians = eps_km / earth_radius_km

print("Running final DBSCAN...")

dbscan = DBSCAN(
    eps=eps_radians,
    min_samples=min_samples,
    metric="haversine",
    algorithm="ball_tree",
    n_jobs=-1
)

labels = dbscan.fit_predict(
    coordinates_rad
)

df["cluster_id"] = labels

cluster_count = len(
    set(labels) - {-1}
)

noise_count = np.sum(
    labels == -1
)

clustered_count = len(labels) - noise_count

print("\nFinal DBSCAN Results")
print("=" * 50)

print(
    f"eps: {eps_km} km"
)

print(
    f"min_samples: {min_samples}"
)

print(
    f"Number of clusters: "
    f"{cluster_count:,}"
)

print(
    f"Clustered businesses: "
    f"{clustered_count:,}"
)

print(
    f"Noise businesses: "
    f"{noise_count:,}"
)

print(
    f"Noise percentage: "
    f"{noise_count / len(labels) * 100:.2f}%"
)

print("\nCluster sizes:")

cluster_sizes = (
    df[df["cluster_id"] != -1]
    .groupby("cluster_id")
    .size()
    .sort_values(ascending=False)
)

print(
    cluster_sizes.head(20)
)

df.to_csv(
    output_file,
    index=False
)

print("\nSaved to:")
print(output_file)