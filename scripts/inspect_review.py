import json

file_path = "data/raw/yelp_academic_dataset_review.json"

with open(file_path, "r", encoding="utf-8") as file:
    for i in range(3):
        line = file.readline()
        review = json.loads(line)

        print(f"\n--- Review {i + 1} ---")
        print(review)