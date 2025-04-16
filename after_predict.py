import ast
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch

df_pred_labels = pd.read_csv("df_without_labels_with_predictions_qwen.csv")
df_pred_labels = df_pred_labels.rename(columns={"label": "label_predicted"})

labels = pd.read_csv("labels.csv")
labels_list = labels['label'].tolist()

# extragem label/text din predictie
def extract_after_assistant(label_text):
    if not isinstance(label_text, str):
        return ""
    words = label_text.split()
    if "assistant" in words:
        index = words.index("assistant")
        return " ".join(words[index + 1:])  
    return label_text  

df_pred_labels["label_predicted"] = df_pred_labels["label_predicted"].apply(extract_after_assistant)

# perfect match intre label prezis si lista de labeluri 
def match_labels(pred, label_list):
    pred = pred.lower()
    return [label for label in label_list if label.lower() in pred]

matched_labels = [", ".join(match_labels(p, labels_list)) for p in df_pred_labels["label_predicted"]]
df_pred_labels["insurance_label"] = matched_labels

# similaritate intre label prezis si lista de label uri
# am ales un prag care sa treaca doar daca sunt aproape perfect asemanatoare
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
label_list_embeddings = model.encode(labels_list, convert_to_tensor=True)

def find_best_label(pred_label, threshold=0.75):
    if not isinstance(pred_label, str) or not pred_label.strip():
        return ""

    pred_embedding = model.encode(pred_label, convert_to_tensor=True)
    similarities = util.cos_sim(pred_embedding, label_list_embeddings)[0]

    scores = similarities.cpu().tolist()
    best_index = scores.index(max(scores))
    best_score = scores[best_index]
    
    if best_score >= threshold:
        return labels_list[best_index]
    else:
        return ""

# adaugam cel mai bun label din lista daca nu exista deja 
updated_matched_labels = []
for pred_label, current_match in zip(df_pred_labels["label_predicted"], df_pred_labels["insurance_label"]):
    if current_match.strip(): 
        updated_matched_labels.append(current_match)
    else:
        best_label = find_best_label(pred_label)
        updated_matched_labels.append(best_label)

df_pred_labels["insurance_label"] = updated_matched_labels

# verificare similaritate intre label uri prezise si text
descriptions = df_pred_labels["text"].tolist()  
predicted_labels = df_pred_labels["insurance_label"].tolist()

desc_embeddings = model.encode(descriptions, convert_to_tensor=True)
label_embeddings = model.encode(predicted_labels, convert_to_tensor=True)

similarities = util.cos_sim(desc_embeddings, label_embeddings).diagonal().cpu().tolist()

df_pred_labels["similarity_score"] = similarities
df_pred_labels.to_csv("df_predicted_labels.csv", index=False)

# pregatire pentru concatenare 
df_matched_labels = pd.read_csv("businesses_with_matched_labels.csv")
df_matched_labels = df_matched_labels.drop(df_matched_labels[df_matched_labels['label'] == '[]'].index)

def clean_label_cell(cell):
    try:
        items = ast.literal_eval(cell)
        if isinstance(items, list):
            return ", ".join(items)
        return str(items)
    except:
        return cell

df_matched_labels['label'] = df_matched_labels['label'].apply(clean_label_cell)
df_matched_labels = df_matched_labels.rename(columns={"label": "insurance_label"})
df_matched_labels.to_csv('df_matched_labels.csv', index=False)

final = pd.concat([df_pred_labels, df_matched_labels], ignore_index=True)
final = final.drop(columns=["label_predicted", "similarity_score", "text"])
final.to_csv("labeled_businesses.csv", index=False)