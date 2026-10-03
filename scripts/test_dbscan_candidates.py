import pandas as pd
import numpy as np

from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score


file_path = "data/processed/spatial_businesses.csv"

df = pd.read_csv(file_path)

coordinates = df[
    ["latitude", "longitude"]
].to_numpy()

coordinates_rad = np.radians(coordinates)

eps_values = [0.5, 1.0, 1.5]

min_samples = 10

earth_radius_km = 6371.0

sample_size = 10000

random_seed = 42


for eps_km in eps_values:

    print("\n" + "=" * 60)

    print(
        f"DBSCAN | eps = {eps_km} km | "
        f"min_samples = {min_samples}"
    )

    print("=" * 60)

    eps_radians = eps_km / earth_radius_km

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

    noise_mask = labels == -1

    clustered_mask = ~noise_mask

    noise_count = np.sum(
        noise_mask
    )

    clustered_count = np.sum(
        clustered_mask
    )

    cluster_count = len(
        set(labels) - {-1}
    )

    noise_percentage = (
        noise_count / len(labels)
    ) * 100

    print(
        f"Clusters: {cluster_count:,}"
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
        f"{noise_percentage:.2f}%"
    )


    # ----------------------------------------------
    # Sample points for Silhouette Score
    # ----------------------------------------------

    clustered_indices = np.where(
        clustered_mask
    )[0]

    rng = np.random.default_rng(
        random_seed
    )

    sample_count = min(
        sample_size,
        len(clustered_indices)
    )

    sample_indices = rng.choice(
        clustered_indices,
        size=sample_count,
        replace=False
    )

    sample_coordinates = coordinates_rad[
        sample_indices
    ]

    sample_labels = labels[
        sample_indices
    ]

    # Make sure sample contains at least 2 clusters
    sample_cluster_count = len(
        np.unique(sample_labels)
    )

    if sample_cluster_count >= 2:

        silhouette = silhouette_score(
            sample_coordinates,
            sample_labels,
            metric="haversine"
        )

        print(
            f"Sample size for Silhouette: "
            f"{sample_count:,}"
        )

        print(
            f"Silhouette Score: "
            f"{silhouette:.4f}"
        )

    else:

        print(
            "Silhouette Score: "
            "Not available"
        )