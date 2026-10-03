import pandas as pd
import matplotlib.pyplot as plt

file_path = "data/processed/spatial_businesses.csv"

print("Loading spatial dataset...")

df = pd.read_csv(file_path)

print("Dataset shape:", df.shape)

print("\nLatitude statistics:")
print(df["latitude"].describe())

print("\nLongitude statistics:")
print(df["longitude"].describe())

# Plot geographic distribution
plt.figure(figsize=(12, 8))

plt.scatter(
    df["longitude"],
    df["latitude"],
    s=2,
    alpha=0.3
)

plt.title("Geographic Distribution of Yelp Businesses")
plt.xlabel("Longitude")
plt.ylabel("Latitude")

plt.grid(True)
plt.tight_layout()

plt.show()