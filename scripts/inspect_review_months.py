import json
from collections import Counter

file_path = "data/raw/yelp_academic_dataset_review.json"

month_counts = Counter()

with open(file_path, "r", encoding="utf-8") as file:
    for line in file:
        review = json.loads(line)

        month = review["date"][:7]

        month_counts[month] += 1

print("Reviews by month:")

for month, count in sorted(month_counts.items()):
    print(month, ":", count)