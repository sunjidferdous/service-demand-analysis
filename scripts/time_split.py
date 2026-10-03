import pandas as pd

input_file = "data/processed/forecasting_dataset.csv"

df = pd.read_csv(input_file)

print("Dataset loaded.")
print(f"Shape: {df.shape}")

print("\nYear distribution:")
print(df["year"].value_counts().sort_index())

train = df[df["year"] <= 2020].copy()
validation = df[df["year"] == 2021].copy()
test = df[df["year"] >= 2022].copy()

print("\nTime-based Split")
print("=" * 50)

print(f"Train:      {train.shape}")
print(f"Validation: {validation.shape}")
print(f"Test:       {test.shape}")

print("\nTrain years:")
print(train["year"].min(), "to", train["year"].max())

print("\nValidation years:")
print(validation["year"].min(), "to", validation["year"].max())

print("\nTest years:")
print(test["year"].min(), "to", test["year"].max())