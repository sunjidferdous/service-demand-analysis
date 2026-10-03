import pandas as pd
import matplotlib.pyplot as plt

file_path = "data/processed/monthly_activity.csv"

print("Loading processed dataset...")

df = pd.read_csv(file_path)

print("Dataset shape:", df.shape)

# Aggregate total review activity by state
state_activity = (
    df.groupby("state")["review_count"]
    .sum()
    .sort_values(ascending=False)
    .reset_index()
)

print("\nReview activity by state:")
print(state_activity)

# Top 15 states
top_states = state_activity.head(15)

print("\nTop 15 states:")
print(top_states)

# Plot
plt.figure(figsize=(12, 7))

plt.bar(
    top_states["state"],
    top_states["review_count"]
)

plt.title("Top 15 States by Yelp Review Activity")
plt.xlabel("State")
plt.ylabel("Total Review Activity")

plt.xticks(rotation=45)

plt.tight_layout()
plt.show()