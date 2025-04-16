from transformers import LongformerTokenizer, LongformerForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.preprocessing import MultiLabelBinarizer
from peft import get_peft_model, LoraConfig, TaskType
import torch
import pandas as pd
import numpy as np
import ast

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

balanced_df_with_labels = pd.read_csv("reduced_balanced_df.csv")
balanced_df_with_labels['label'] = balanced_df_with_labels['label'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
text_train = balanced_df_with_labels['text'].tolist()

model_name = "allenai/longformer-base-4096"
tokenizer = LongformerTokenizer.from_pretrained(model_name)

mlb = MultiLabelBinarizer()
labels = mlb.fit_transform(balanced_df_with_labels['label']).astype("float32")
num_labels = len(mlb.classes_)

balanced_df_with_labels["labels"] = list(labels)

dataset = Dataset.from_pandas(balanced_df_with_labels)

tokenized = tokenizer(
    dataset["text"],
    padding="max_length",
    truncation=True,
    max_length=4096,
    return_tensors="pt"
)

global_attention_mask = torch.zeros_like(tokenized["input_ids"])
global_attention_mask[:, 0] = 1 

for col in ["text", "label"]:
    if col in dataset.column_names:
        dataset = dataset.remove_columns(col)

dataset = dataset.add_column("input_ids", tokenized["input_ids"].tolist())
dataset = dataset.add_column("attention_mask", tokenized["attention_mask"].tolist())
dataset = dataset.add_column("global_attention_mask", global_attention_mask.tolist())

model = LongformerForSequenceClassification.from_pretrained(
    model_name,
    num_labels=num_labels,
    problem_type="multi_label_classification"
).to(device)


# print("\n--- Available submodules (relevant for LoRA): ---")
# for name, _ in model.named_modules():
#     if any(keyword in name.lower() for keyword in ["query", "q_proj", "key", "k_proj", "value", "v_proj", "attn"]):
#         print(name)
# print("--- End of submodules list ---\n")

lora_config = LoraConfig(
    r=16,
    lora_alpha=8,
    target_modules=[ "query", "key", "value",
        "query_global", "key_global", "value_global"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.SEQ_CLS
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

training_args = TrainingArguments(
    output_dir="output",
    eval_strategy="no",
    learning_rate=2e-5,
    loss_function="binary_cross_entropy_with_logits",
    per_device_train_batch_size=8,
    num_train_epochs=2,
    weight_decay=0.01,
    fp16=torch.cuda.is_available(),
)

def data_collator(features):
    return {
        "input_ids": torch.tensor([f["input_ids"] for f in features], dtype=torch.long),
        "attention_mask": torch.tensor([f["attention_mask"] for f in features], dtype=torch.long),
        "global_attention_mask": torch.tensor([f["global_attention_mask"] for f in features], dtype=torch.long),
        "labels": torch.tensor([f["labels"] for f in features], dtype=torch.float32),
    }

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

trainer.train()

trainer.save_model("output")
trainer.save_state()
tokenizer.save_pretrained("output")
