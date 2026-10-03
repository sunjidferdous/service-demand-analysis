import pandas as pd
import matplotlib.pyplot as plt

file_path = "data/processed/monthly_activity.csv"

print("Loading processed dataset...")

df = pd.read_csv(file_path)

print("Dataset shape:", df.shape)

# Aggregate monthly activity by year
yearly_activity = (
    df.groupby("year")["review_count"]
    .sum()
    .reset_index()
)

print("\nYearly activity:")
print(yearly_activity)

# Plot
plt.figure(figsize=(12, 6))

plt.plot(
    yearly_activity["year"],
    yearly_activity["review_count"],
    marker="o"
)

plt.title("Yearly Yelp Review Activity")
plt.xlabel("Year")
plt.ylabel("Number of Reviews")
plt.grid(True)

plt.tight_layout()
plt.show()