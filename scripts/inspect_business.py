import json

file_path = "data/raw/yelp_academic_dataset_business.json"

with open(file_path, "r", encoding="utf-8") as file:
    for i in range(3):
        line = file.readline()
        business = json.loads(line)

        print(f"\n--- Business {i + 1} ---")
        print(business)