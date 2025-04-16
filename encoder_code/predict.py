from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import torch
import ast

batch_size = 8
max_length = 4096
threshold = 0.1

#GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

df_without_labels = pd.read_csv("df_without_labels.csv")
balanced_df_with_labels = pd.read_csv("reduced_balanced_df.csv")
balanced_df_with_labels['label'] = balanced_df_with_labels['label'].apply(
    lambda x: ast.literal_eval(x) if isinstance(x, str) else x
)

mlb = MultiLabelBinarizer()
matrix = mlb.fit_transform(balanced_df_with_labels['label']).astype("float32")
num_labels = len(mlb.classes_)

model = AutoModelForSequenceClassification.from_pretrained(
    "merged_model",
    num_labels=num_labels,
    problem_type="multi_label_classification"
).to(device)

tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")

texts = df_without_labels['text'].tolist()

class TextDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]

text_dataset = TextDataset(texts)
text_loader = DataLoader(text_dataset, batch_size=batch_size)

all_preds = []

with torch.no_grad():
    for batch_texts in text_loader:
        tokenized = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )

        tokenized = {k: v.to(device) for k, v in tokenized.items()}

        outputs = model(**tokenized)
        logits = outputs.logits  

        probs = torch.sigmoid(logits)

        k = 3
        topk_indices = torch.topk(probs, k, dim=1).indices
        topk_preds = torch.zeros_like(probs)
        for i in range(probs.shape[0]):
            topk_preds[i, topk_indices[i]] = 1
        preds = topk_preds.int().cpu().numpy()
        # preds = (probs > threshold).int().cpu().numpy()
        # print("Min prob:", probs.min().item())
        # print("Max prob:", probs.max().item())
        # print("Mean prob:", probs.mean().item())

        all_preds.extend(preds)

        del tokenized, outputs, logits, probs, preds
        torch.cuda.empty_cache()

all_preds = np.array(all_preds, dtype=float)
predicted_labels = mlb.inverse_transform(all_preds)

df_without_labels['label'] = predicted_labels
df_without_labels.to_csv("df_without_labels_with_predictions.csv", index=False)
