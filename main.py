import pandas as pd
import re
from sentence_transformers import SentenceTransformer, util
from sklearn.utils import shuffle
import torch
import numpy
from collections import Counter

# GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# pentru fiecare label avem un dictionar cu ce cuvinte contine
def extract_keywords(labels):
    keywords_dict = {}

    for label in labels['label']:
        good_label = label.lower()
        good_label = re.sub(r'[^a-z0-9]+', ' ', good_label).strip()
        words = good_label.split()
        keywords_dict[label] = words
    
    return keywords_dict

# pregatim campurile relevante pentru a cauta cuvinte cheie in el
def combine_and_normalize_fields(row):
    fields = [
        str(row.get('business_tags', "")),
        str(row.get('sector', "")),
        str(row.get('category', "")),
        str(row.get('niche', ""))
    ]
    full_text = " ".join(fields).lower()
    full_text = re.sub(r'[^a-z0-9]+', ' ', full_text).strip()
    return full_text

# verific daca toate cuvintele cheie apar in text
def keyword_match_score(label_tokens, tokens_set):
    return all(tok in tokens_set for tok in label_tokens)

labels = pd.read_csv('labels.csv')
businesses = pd.read_csv('businesses.csv')

keywords_dict = extract_keywords(labels)

# similaritati intre text si label uri
model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

label_list = labels['label'].tolist()
label_embeddings = model.encode(label_list, convert_to_tensor=True)

business_texts = businesses.apply(combine_and_normalize_fields, axis=1).tolist()
business_tokens = [text.split() for text in business_texts]
business_token_sets = [set(toks) for toks in business_tokens]
business_embeddings = model.encode(business_texts, convert_to_tensor=True)

cosine_matrix = util.cos_sim(business_embeddings, label_embeddings)

# adauga label urile care au un scor de similaritate ok si gaseste toate cuvintele cheie in text
# sau are un scor mai mare dar nu gaseste tot in text
# trial and error la praguri
results = []
for i, scores in enumerate(cosine_matrix):
    row_results = []
    tokens_set = business_token_sets[i]
    for j, label in enumerate(label_list):
        score = float(scores[j])
        label_tokens = re.sub(r'[^a-z0-9]+', ' ', label.lower()).split()
        has_kw_match = keyword_match_score(label_tokens, tokens_set)
        if (score > 0.6 and has_kw_match) or score > 0.65:
            row_results.append(label)
    results.append(row_results)

businesses['label'] = results

businesses.to_csv('businesses_with_matched_labels.csv', index=False)

# am creat articificial date pentru label urile care nu au fost deloc atribuite
made_labels = pd.read_csv("made_labels.csv")
made_labels['label'] = made_labels['label'].apply(lambda x: [x])

# le multiplicam si adaugam la date
made_labels = pd.concat([made_labels]*10, ignore_index=True)
combined_df = pd.concat([businesses, made_labels], ignore_index=True)
combined_df = shuffle(combined_df)

# separam datele cu label uri atribuite de celelalte si construim textul
df_with_labels = combined_df[combined_df['label'].apply(lambda x: len(x) > 0)].copy()
df_without_labels = combined_df[combined_df['label'].apply(lambda x: len(x) == 0)].copy()
df_with_labels['text'] = (df_with_labels['description'].fillna("") + ' - ' + df_with_labels['business_tags'].fillna("").astype(str) + 
                        ' - ' + df_with_labels['sector'].fillna("") + ' - ' + df_with_labels['category'].fillna("") + ' - ' + df_with_labels['niche'].fillna(""))
df_without_labels['text'] = (df_without_labels['description'].fillna("") + ' - ' + df_without_labels['business_tags'].fillna("").astype(str) + 
                        ' - ' + df_without_labels['sector'].fillna("") + ' - ' + df_without_labels['category'].fillna("") + ' - ' + df_without_labels['niche'].fillna(""))

df_with_labels.to_csv("df_with_labels.csv", index=False)
df_without_labels.to_csv("df_without_labels.csv", index=False)
combined_df.to_csv("businesses_with_made_labels.csv", index=False)

# calculam label ul cel mai des aparut pentru a echilibra datele
flat_labels = [label for sublist in df_with_labels['label'] for label in sublist]
label_counts = Counter(flat_labels)
max_count = max(label_counts.values())
label_to_rows = {label: df_with_labels[df_with_labels['label'].apply(lambda labels: label in labels)] for label in label_counts}

balanced_rows = []
# multiplicam datele cu label uri pentru a ajunge la un numar egal intre ele
for label, rows in label_to_rows.items():
    count = len(rows)
    needed = max_count - count
    if needed > 0:
        multiplier = needed // count
        remainder = needed % count

        duplicated_rows = pd.concat([rows]*multiplier + [rows.sample(remainder, replace=True)], ignore_index=True)
        balanced_rows.append(pd.concat([rows, duplicated_rows], ignore_index=True))
    else:
        balanced_rows.append(rows)

balanced_df = pd.concat(balanced_rows, ignore_index=True)
balanced_df = shuffle(balanced_df).reset_index(drop=True)
balanced_df.to_csv("balanced_df_with_labels.csv", index=False)

# dupa ce le am echilibrat au fost prea multe, asa ca le am redus
reduced_df = balanced_df.sample(frac=0.25, random_state=42).reset_index(drop=True)
reduced_df.to_csv("reduced_balanced_df.csv", index=False)
