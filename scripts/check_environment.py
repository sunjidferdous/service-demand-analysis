import importlib.util

packages = {
    "numpy": "NumPy",
    "pandas": "Pandas",
    "matplotlib": "Matplotlib",
    "seaborn": "Seaborn",
    "sklearn": "Scikit-learn",
    "xgboost": "XGBoost",
    "nltk": "NLTK",
    "textblob": "TextBlob",
    "spacy": "spaCy",
    "folium": "Folium",
    "plotly": "Plotly",
    "streamlit": "Streamlit",
    "torch": "PyTorch",
    "transformers": "Transformers",
    "datasets": "Hugging Face Datasets",
    "accelerate": "Accelerate",
}

print("=" * 55)
print("PROJECT ENVIRONMENT CHECK")
print("=" * 55)

missing = []

for module, name in packages.items():
    if importlib.util.find_spec(module) is not None:
        print(f"[OK]      {name}")
    else:
        print(f"[MISSING] {name}")
        missing.append(module)

print("=" * 55)

if missing:
    print("Missing packages:")
    print(", ".join(missing))
else:
    print("Everything is installed!")

print("=" * 55)