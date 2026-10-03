import json
from collections import Counter
from datetime import datetime

file_path = "data/raw/yelp_academic_dataset_review.json"

weekday_counts = Counter()

with open(file_path, "r", encoding="utf-8") as file:
    for line in file:
        review = json.loads(line)

        date = datetime.strptime(
            review["date"],
            "%Y-%m-%d %H:%M:%S"
        )

        weekday_counts[date.strftime("%A")] += 1

print("Reviews by weekday:")

for weekday, count in weekday_counts.items():
    print(weekday, ":", count)