import json
import random
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer

INPUT_FILE = "data/raw/yelp_academic_dataset_review.json"
OUTPUT_FILE = "data/processed/sentiment_results.csv"

SAMPLE_SIZE = 100_000
RANDOM_STATE = 42

random.seed(RANDOM_STATE)

print("Reading Yelp reviews...")

reservoir = []

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        review = json.loads(line)

        if len(reservoir) < SAMPLE_SIZE:
            reservoir.append(review)
        else:
            j = random.randint(0, i)
            if j < SAMPLE_SIZE:
                reservoir[j] = review

        if (i + 1) % 1_000_000 == 0:
            print(f"Processed {i + 1:,} reviews...")

print(f"\nSample collected: {len(reservoir):,}")

df = pd.DataFrame(reservoir)

print("Running VADER sentiment analysis...")

analyzer = SentimentIntensityAnalyzer()

scores = df["text"].apply(analyzer.polarity_scores)

df["compound"] = scores.apply(lambda x: x["compound"])

def classify_sentiment(score):
    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    else:
        return "Neutral"

df["sentiment"] = df["compound"].apply(classify_sentiment)

# Star-based weak labels for comparison
def star_label(stars):
    if stars >= 4:
        return "Positive"
    elif stars <= 2:
        return "Negative"
    else:
        return "Neutral"

df["star_sentiment"] = df["stars"].apply(star_label)

result = df[
    [
        "review_id",
        "business_id",
        "stars",
        "date",
        "compound",
        "sentiment",
        "star_sentiment"
    ]
]

result.to_csv(OUTPUT_FILE, index=False)

print("\nSentiment Distribution")
print("=" * 40)
print(result["sentiment"].value_counts())

print("\nStar-based Distribution")
print("=" * 40)
print(result["star_sentiment"].value_counts())

print("\nResults saved to:")
print(OUTPUT_FILE)