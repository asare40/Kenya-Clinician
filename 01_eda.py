import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    # Load data
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')
    print(f"Train shape: {train.shape}")
    print(f"Test shape: {test.shape}")

    print("\nTrain columns:", train.columns.tolist())
    print("\nTest columns:", test.columns.tolist())

    print("\nFirst 5 rows of train:")
    print(train.head())

    print("\nMissing values in train:")
    print(train.isnull().sum())

    print("\nLabel distribution in train:")
    if 'Clinician' in train.columns:
        print(train['Clinician'].value_counts().head(10))

    # Plot prompt length if present
    if 'Prompt' in train.columns:
        train['prompt_len'] = train['Prompt'].astype(str).apply(len)
        sns.histplot(train['prompt_len'], bins=30)
        plt.title("Prompt length distribution")
        plt.savefig("outputs/prompt_length_distribution.png")
        print("Saved outputs/prompt_length_distribution.png")

if __name__ == "__main__":
    main()