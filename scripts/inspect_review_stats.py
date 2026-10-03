import json
from collections import Counter

file_path = "data/raw/yelp_academic_dataset_review.json"

year_counts = Counter()

with open(file_path, "r", encoding="utf-8") as file:
    for line in file:
        review = json.loads(line)

        year = review["date"][:4]

        year_counts[year] += 1

print("Reviews by year:")

for year, count in sorted(year_counts.items()):
    print(year, ":", count)