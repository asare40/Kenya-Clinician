"""
Auto-generates a requirements.txt file based on the imports found in your notebooks and scripts.

Usage:
    python src/generate_requirements.py

It will scan the notebooks/ and src/ directories, extract all import statements,
map them to PyPI package names, and write a requirements.txt file in the project root.
"""

import os
import re
from glob import glob

# Packages that need mapping (module name -> pip package name)
PACKAGE_FIXES = {
    "sklearn": "scikit-learn",
    "cv2": "opencv-python",
    "PIL": "pillow",
    "yaml": "pyyaml",
    "bs4": "beautifulsoup4",
    "Bio": "biopython",
}

# Skip these standard libraries
STANDARD_LIBS = set([
    "os", "sys", "re", "math", "json", "csv", "glob", "logging", "collections",
    "datetime", "itertools", "random", "string", "subprocess", "pathlib",
    "functools", "shutil", "typing", "unittest", "threading", "time", "argparse"
])

def extract_imports_from_py(filename):
    imports = set()
    with open(filename, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            m = re.match(r"(from|import) (\S+)", line)
            if m:
                pkg = m.group(2).split(".")[0]
                imports.add(pkg)
    return imports

def extract_imports_from_ipynb(filename):
    import json
    imports = set()
    with open(filename, encoding="utf-8") as f:
        nb = json.load(f)
        for cell in nb.get("cells", []):
            if cell.get("cell_type") == "code":
                for line in cell.get("source", []):
                    line = line.strip()
                    m = re.match(r"(from|import) (\S+)", line)
                    if m:
                        pkg = m.group(2).split(".")[0]
                        imports.add(pkg)
    return imports

def main():
    all_imports = set()

    # Scan .py files in src/
    for py_file in glob("src/**/*.py", recursive=True):
        all_imports.update(extract_imports_from_py(py_file))

    # Scan .ipynb files in notebooks/
    for nb_file in glob("notebooks/**/*.ipynb", recursive=True):
        all_imports.update(extract_imports_from_ipynb(nb_file))

    # Remove standard libraries
    reqs = [PACKAGE_FIXES.get(pkg, pkg) for pkg in all_imports if pkg not in STANDARD_LIBS]
    reqs = sorted(set(reqs))

    # Add must-have data science stack if not present (for most ML projects)
    default_pkgs = ["numpy", "pandas", "scikit-learn", "matplotlib", "seaborn", "jupyter", "notebook"]
    for pkg in default_pkgs:
        if pkg not in reqs:
            reqs.append(pkg)

    reqs = sorted(set(reqs))

    with open("requirements.txt", "w") as f:
        for pkg in reqs:
            f.write(pkg + "\n")

    print("requirements.txt generated with packages:")
    for pkg in reqs:
        print(" -", pkg)

if __name__ == "__main__":
    main()