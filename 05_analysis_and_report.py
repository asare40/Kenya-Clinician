import pandas as pd
from difflib import SequenceMatcher
import matplotlib.pyplot as plt
import seaborn as sns

def string_similarity(a, b):
    if pd.isna(a) or pd.isna(b):
        return 0
    return SequenceMatcher(None, str(a), str(b)).ratio()

def main():
    baseline = pd.read_csv('outputs/baseline_submission.csv')
    advanced = pd.read_csv('outputs/advanced_submission.csv')
    test = pd.read_csv('data/test_processed.csv')
    train = pd.read_csv('data/train_processed.csv')

    # Merge for side-by-side comparison
    comparison = test[['Master_Index', 'Prompt_clean']].copy()
    comparison = comparison.merge(baseline, on='Master_Index', how='left', suffixes=('', '_baseline'))
    comparison = comparison.merge(advanced, on='Master_Index', how='left', suffixes=('', '_advanced'))
    comparison.rename(columns={'Clinician': 'Baseline', 'Clinician_advanced': 'Advanced'}, inplace=True)

    # Compute similarity
    comparison['sim_score'] = comparison.apply(lambda row: string_similarity(row['Baseline'], row['Advanced']), axis=1)
    print("Average similarity between Baseline and Advanced:", comparison['sim_score'].mean())

    # Plot histogram (saves as PNG)
    plt.figure(figsize=(8, 5))
    sns.histplot(comparison['sim_score'], bins=20)
    plt.title('Similarity between Baseline and Advanced predictions')
    plt.xlabel('SequenceMatcher Similarity')
    plt.savefig('outputs/similarity_histogram.png')
    print("Saved outputs/similarity_histogram.png.")

    # Show 10 most different cases
    diff_cases = comparison.sort_values('sim_score').head(10)
    for idx, row in diff_cases.iterrows():
        print("="*50)
        print(f"Prompt: {row['Prompt_clean']}\n")
        print(f"Baseline: {row['Baseline']}\n")
        print(f"Advanced: {row['Advanced']}\n")
        print(f"Similarity: {row['sim_score']:.2f}")

if __name__ == "__main__":
    main()