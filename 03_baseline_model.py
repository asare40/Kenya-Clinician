import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

def main():
    # Load processed train, test, and embeddings
    train = pd.read_csv('data/train_processed.csv')
    test = pd.read_csv('data/test_processed.csv')
    train_embeddings = np.load('data/train_prompt_embeddings.npy')
    test_embeddings = np.load('data/test_prompt_embeddings.npy')

    # For each test prompt, find the most similar train prompt
    preds = []
    for i in tqdm(range(test_embeddings.shape[0])):
        sims = cosine_similarity([test_embeddings[i]], train_embeddings)[0]
        best_idx = np.argmax(sims)
        pred = train.iloc[best_idx]['Clinician'] if 'Clinician' in train.columns else ''
        preds.append(pred)
    test['Clinician'] = preds

    # Save in submission format
    submission = test[['Master_Index', 'Clinician']]
    submission.to_csv('outputs/baseline_submission.csv', index=False)
    print("Saved outputs/baseline_submission.csv")

if __name__ == "__main__":
    main()