import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.neighbors import NearestNeighbors


# Step 1: Load spatial dataset

file_path = "data/processed/spatial_businesses.csv"

print("Loading spatial dataset...")

df = pd.read_csv(file_path)

print("Dataset shape:", df.shape)


# Step 2: Select latitude and longitude

coordinates = df[
    ["latitude", "longitude"]
].to_numpy()


# Step 3: Convert degrees to radians

coordinates_rad = np.radians(coordinates)


# Step 4: Calculate nearest neighbors

print("\nCalculating nearest neighbors...")

# We are using the 10th nearest neighbor
k = 10

neighbors = NearestNeighbors(
    n_neighbors=k,
    metric="haversine",
    algorithm="ball_tree"
)

neighbors.fit(coordinates_rad)

distances, indices = neighbors.kneighbors(
    coordinates_rad
)


# Step 5: Get distance to 10th nearest neighbor

k_distances = distances[:, k - 1]


# Step 6: Convert radians to kilometers

earth_radius_km = 6371.0

k_distances_km = (
    k_distances * earth_radius_km
)

# Step 7: Sort distances

k_distances_km = np.sort(k_distances_km)


# Step 8: Display statistics

print("\nK-distance statistics:")

print(
    pd.Series(k_distances_km).describe()
)

# Step 9: Calculate candidate eps values

print("\nCandidate eps values:")

for percentile in [80, 85, 90, 95, 97, 98, 99]:

    value = np.percentile(
        k_distances_km,
        percentile
    )

    print(
        f"{percentile}th percentile: "
        f"{value:.4f} km"
    )

# Step 10: Plot K-distance graph

plt.figure(figsize=(12, 6))

plt.plot(k_distances_km)

plt.title(
    "10-Nearest Neighbor Distance Plot (Zoomed)"
)

plt.xlabel(
    "Points sorted by distance"
)

plt.ylabel(
    "Distance to 10th nearest neighbor (km)"
)

# Zoom into the useful range
plt.ylim(0, 2)

plt.grid(True)

plt.tight_layout()

plt.show()