import pandas as pd
import matplotlib.pyplot as plt

INPUT_FILE = "data/processed/aspect_sentiment_results.csv"

df = pd.read_csv(INPUT_FILE)

# Aspect × Sentiment counts
table = pd.crosstab(
    df["aspect"],
    df["sentiment"]
)

# Ensure consistent order
table = table.reindex(
    columns=["Positive", "Neutral", "Negative"],
    fill_value=0
)

# Convert to percentage
percentage = table.div(table.sum(axis=1), axis=0) * 100

print("\nAspect Sentiment Percentage")
print("=" * 60)
print(percentage.round(2))

# Stacked percentage chart
percentage.plot(
    kind="bar",
    stacked=True,
    figsize=(9, 6)
)

plt.title("Aspect-Based Sentiment Distribution")
plt.xlabel("Aspect")
plt.ylabel("Percentage (%)")
plt.xticks(rotation=0)
plt.legend(title="Sentiment")
plt.tight_layout()

plt.savefig(
    "data/processed/aspect_sentiment_distribution.png",
    dpi=300
)

plt.close()

# Aspect frequency
frequency = df["aspect"].value_counts()

frequency.plot(
    kind="bar",
    figsize=(9, 6)
)

plt.title("Aspect Frequency in Yelp Reviews")
plt.xlabel("Aspect")
plt.ylabel("Number of Reviews")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(
    "data/processed/aspect_frequency.png",
    dpi=300
)

plt.close()

print("\nCharts saved successfully.")