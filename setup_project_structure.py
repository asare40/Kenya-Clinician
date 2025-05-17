import os

folders = [
    "data",
    "outputs"
]
files = [
    "01_eda.py",
    "02_feature_engineering_and_preprocessing.py",
    "03_baseline_model.py",
    "04_advanced_modeling.py",
    "05_analysis_and_report.py",
    "README.md"
]

for folder in folders:
    os.makedirs(folder, exist_ok=True)

for file in files:
    with open(file, "w") as f:
        f.write("# " + file + "\n")

print("Project structure created. Please copy your datasets into the 'data' folder.")