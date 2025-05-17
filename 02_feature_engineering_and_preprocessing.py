import pandas as pd
import numpy as np
import re
import string
from sklearn.preprocessing import LabelEncoder
import spacy
from sentence_transformers import SentenceTransformer

def clean_text(text):
    if pd.isnull(text):
        return ""
    text = text.lower()
    text = text.replace('\n', ' ').replace('\r', ' ')
    text = re.sub(r'[^a-z0-9\s.,?-]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_age(prompt):
    match = re.search(r'(\d{1,3})[- ]?year[s]?(?:[- ]old)?', prompt.lower())
    if match:
        return int(match.group(1))
    return np.nan

def extract_sex(prompt):
    prompt = prompt.lower()
    if 'female' in prompt:
        return 'female'
    if 'male' in prompt:
        return 'male'
    return np.nan

def extract_vitals(prompt):
    prompt = prompt.lower()
    vitals = {}
    match = re.search(r'bp[:=\s]+(\d{2,3})[\s/.,-]+(\d{2,3})', prompt)
    if match:
        vitals['bp_sys'] = int(match.group(1))
        vitals['bp_dia'] = int(match.group(2))
    match = re.search(r'(pr|pulse)[:=\s]+(\d{2,3})', prompt)
    if match:
        vitals['pr'] = int(match.group(2))
    match = re.search(r'(rr|respirations?|resp)[:=\s]+(\d{2,3})', prompt)
    if match:
        vitals['rr'] = int(match.group(2))
    match = re.search(r'(t|temp(?:erature)?)[:=\s]+(\d{2}(?:\.\d)?)', prompt)
    if match:
        vitals['temp'] = float(match.group(2))
    match = re.search(r'(spo2)[:=\s]+(\d{2,3})', prompt)
    if match:
        vitals['spo2'] = int(match.group(2))
    return vitals

def extract_symptoms(prompt):
    symptoms = ['pain', 'fever', 'cough', 'vomit', 'bleed', 'diarrhea', 'weak', 'swelling', 'headache', 'dizziness', 'chills', 'convulsion', 'twitch']
    found = [s for s in symptoms if s in prompt.lower()]
    return ','.join(found) if found else np.nan

def extract_entities(text, nlp):
    doc = nlp(text)
    return [ent.text for ent in doc.ents]

def extract_question_type(prompt):
    prompt = prompt.lower()
    if 'diagnos' in prompt:
        return 'diagnosis'
    if 'manage' in prompt:
        return 'management'
    if 'investigat' in prompt:
        return 'investigation'
    if 'educat' in prompt:
        return 'education'
    if 'counsel' in prompt:
        return 'counseling'
    return 'other'

def main():
    # Load data
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')

    # Clean text
    train['Prompt_clean'] = train['Prompt'].apply(clean_text)
    test['Prompt_clean'] = test['Prompt'].apply(clean_text)
    if 'Clinician' in train.columns:
        train['Clinician_clean'] = train['Clinician'].apply(clean_text)

    # NLP model
    nlp = spacy.load("en_core_web_sm")

    # Feature extraction
    for df in [train, test]:
        df['patient_age'] = df['Prompt_clean'].apply(extract_age)
        df['patient_sex'] = df['Prompt_clean'].apply(extract_sex)
        df['symptoms'] = df['Prompt_clean'].apply(extract_symptoms)
        vitals = df['Prompt_clean'].apply(extract_vitals)
        df['bp_sys'] = vitals.apply(lambda x: x.get('bp_sys') if isinstance(x, dict) else np.nan)
        df['bp_dia'] = vitals.apply(lambda x: x.get('bp_dia') if isinstance(x, dict) else np.nan)
        df['pr'] = vitals.apply(lambda x: x.get('pr') if isinstance(x, dict) else np.nan)
        df['rr'] = vitals.apply(lambda x: x.get('rr') if isinstance(x, dict) else np.nan)
        df['temp'] = vitals.apply(lambda x: x.get('temp') if isinstance(x, dict) else np.nan)
        df['spo2'] = vitals.apply(lambda x: x.get('spo2') if isinstance(x, dict) else np.nan)
        df['entities'] = df['Prompt_clean'].apply(lambda x: extract_entities(x, nlp))
        df['question_type'] = df['Prompt_clean'].apply(extract_question_type)

    # Encode categorical
    cat_cols = ['County', 'Health level', 'Nursing Competency', 'Clinical Panel', 'patient_sex', 'question_type']
    label_encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        all_vals = pd.concat([train[col], test[col]], axis=0).astype(str).fillna('unknown')
        le.fit(all_vals)
        train[col+'_enc'] = le.transform(train[col].astype(str).fillna('unknown'))
        test[col+'_enc'] = le.transform(test[col].astype(str).fillna('unknown'))
        label_encoders[col] = le

    # Sentence Transformer Embeddings
    embedder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    train_prompts = train['Prompt_clean'].tolist()
    test_prompts = test['Prompt_clean'].tolist()
    train_embeddings = embedder.encode(train_prompts, show_progress_bar=True)
    test_embeddings = embedder.encode(test_prompts, show_progress_bar=True)
    np.save('data/train_prompt_embeddings.npy', train_embeddings)
    np.save('data/test_prompt_embeddings.npy', test_embeddings)

    # Save processed data
    train.to_csv('data/train_processed.csv', index=False)
    test.to_csv('data/test_processed.csv', index=False)
    print("Saved processed train and test data and embeddings.")

if __name__ == "__main__":
    main()