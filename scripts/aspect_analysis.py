import pandas as pd
import re

INPUT_FILE = "data/raw/yelp_academic_dataset_review.json"
SENTIMENT_FILE = "data/processed/sentiment_results.csv"
OUTPUT_FILE = "data/processed/aspect_sentiment_results.csv"

# Aspect keywords
ASPECTS = {
    "Price": [
        "price", "prices", "priced", "expensive", "cheap",
        "cheaper", "cost", "costly", "value", "worth",
        "overpriced", "affordable", "deal"
    ],

    "Service Quality": [
        "service", "staff", "employee", "employees",
        "waiter", "waitress", "server", "friendly",
        "rude", "helpful", "unhelpful", "customer service",
        "service was", "staff was"
    ],

    "Punctuality": [
        "wait", "waited", "waiting", "late", "delay",
        "delayed", "slow", "quick", "fast", "on time",
        "timely", "minutes", "hour", "hours",
        "delivery time", "delivery was"
    ]
}

# Load sentiment sample
sentiment_df = pd.read_csv(SENTIMENT_FILE)

# Read original review text only for sampled review IDs
sample_ids = set(sentiment_df["review_id"])

print("Reading review text...")

reviews = []

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        import json
        review = json.loads(line)

        if review["review_id"] in sample_ids:
            reviews.append({
                "review_id": review["review_id"],
                "text": review["text"]
            })

        if (i + 1) % 1_000_000 == 0:
            print(f"Processed {i + 1:,} reviews...")

        if len(reviews) == len(sample_ids):
            break

text_df = pd.DataFrame(reviews)

df = sentiment_df.merge(
    text_df,
    on="review_id",
    how="left"
)

print(f"\nReviews matched: {len(df):,}")


def find_aspects(text):
    text = str(text).lower()

    found = []

    for aspect, keywords in ASPECTS.items():
        for keyword in keywords:
            pattern = r"\b" + re.escape(keyword) + r"\b"

            if re.search(pattern, text):
                found.append(aspect)
                break

    return found


df["aspects"] = df["text"].apply(find_aspects)

# One row per detected aspect
rows = []

for _, row in df.iterrows():
    for aspect in row["aspects"]:
        rows.append({
            "review_id": row["review_id"],
            "stars": row["stars"],
            "sentiment": row["sentiment"],
            "aspect": aspect
        })

aspect_df = pd.DataFrame(rows)

aspect_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nAspect Frequency")
print("=" * 50)

print(
    aspect_df["aspect"]
    .value_counts()
)

print("\nAspect × Sentiment")
print("=" * 50)

print(
    pd.crosstab(
        aspect_df["aspect"],
        aspect_df["sentiment"]
    )
)

print("\nResults saved to:")
print(OUTPUT_FILE)