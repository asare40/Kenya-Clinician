import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
import torch
from tqdm import tqdm

def rag_context_prompt(test_emb, train, train_embeddings, topk=3):
    sims = np.dot(train_embeddings, test_emb) / (np.linalg.norm(train_embeddings, axis=1) * np.linalg.norm(test_emb) + 1e-8)
    topk_idx = np.argsort(sims)[-topk:][::-1]
    context = '\n'.join(train.iloc[idx]['Clinician'] for idx in topk_idx)
    return context

def main():
    train = pd.read_csv('data/train_processed.csv')
    test = pd.read_csv('data/test_processed.csv')
    train_embeddings = np.load('data/train_prompt_embeddings.npy')
    test_embeddings = np.load('data/test_prompt_embeddings.npy')

    model_name = 'google/flan-t5-base'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    device = 0 if torch.cuda.is_available() else -1
    gen_pipe = pipeline('text2text-generation', model=model, tokenizer=tokenizer, device=device)

    # RAG predictions
    rag_preds = []
    for i in tqdm(range(len(test))):
        context = rag_context_prompt(test_embeddings[i], train, train_embeddings, topk=3)
        inp = f"Instruction: {test.iloc[i]['Prompt_clean']}\nPrevious similar cases: {context}\nResponse:"
        out = gen_pipe(inp, max_new_tokens=128, do_sample=False)[0]['generated_text']
        answer = out.replace(inp, '').strip()
        rag_preds.append(answer)
    test['Clinician_RAG'] = rag_preds

    submission = test[['Master_Index', 'Clinician_RAG']].rename(columns={'Clinician_RAG': 'Clinician'})
    submission.to_csv('outputs/advanced_submission.csv', index=False)
    print("Saved outputs/advanced_submission.csv")

if __name__ == "__main__":
    main()