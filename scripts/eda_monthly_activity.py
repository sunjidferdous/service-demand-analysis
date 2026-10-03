import pandas as pd
import matplotlib.pyplot as plt

file_path = "data/processed/monthly_activity.csv"

print("Loading processed dataset...")

df = pd.read_csv(file_path)

print("Dataset shape:", df.shape)

# Calculate average review activity for each month
monthly_activity = (
    df.groupby("month")["review_count"]
    .mean()
    .reset_index()
)

# Month names
month_names = {
    1: "January",
    2: "February",
    3: "March",
    4: "April",
    5: "May",
    6: "June",
    7: "July",
    8: "August",
    9: "September",
    10: "October",
    11: "November",
    12: "December"
}

monthly_activity["month_name"] = (
    monthly_activity["month"].map(month_names)
)

print("\nAverage review activity by month:")
print(monthly_activity)

# Plot
plt.figure(figsize=(12, 6))

plt.plot(
    monthly_activity["month_name"],
    monthly_activity["review_count"],
    marker="o"
)

plt.title("Average Monthly Yelp Review Activity")
plt.xlabel("Month")
plt.ylabel("Average Reviews per Business-Month")
plt.xticks(rotation=45)
plt.grid(True)

plt.tight_layout()
plt.show()