import os

# Define the directory structure
folders = [
    "data",
    "notebooks",
    "src",
    "outputs",
    "docs"
]
files = [
    "README.md",
    "requirements.txt",
]

# Create folders
for folder in folders:
    os.makedirs(folder, exist_ok=True)

# Create files if they do not exist
for file in files:
    if not os.path.exists(file):
        with open(file, "w", encoding="utf-8") as f:
            if file == "README.md":
                f.write("# Kenya Clinician Response Prediction\n")
            else:
                f.write("")

print("Project structure created successfully.")