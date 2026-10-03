import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

INPUT_FILE = "data/processed/sentiment_results.csv"

df = pd.read_csv(INPUT_FILE)

y_true = df["star_sentiment"]
y_pred = df["sentiment"]

print("Sentiment Evaluation")
print("=" * 60)

print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
print(f"Precision: {precision_score(y_true, y_pred, average='weighted'):.4f}")
print(f"Recall:    {recall_score(y_true, y_pred, average='weighted'):.4f}")
print(f"F1-score:  {f1_score(y_true, y_pred, average='weighted'):.4f}")

print("\nClassification Report")
print("=" * 60)
print(classification_report(y_true, y_pred))

print("\nConfusion Matrix")
print("=" * 60)

labels = ["Negative", "Neutral", "Positive"]

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=labels
)

print(pd.DataFrame(
    cm,
    index=[f"Actual {x}" for x in labels],
    columns=[f"Predicted {x}" for x in labels]
))