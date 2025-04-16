from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import Dataset
from peft import get_peft_model, LoraConfig, TaskType
import torch
import pandas as pd
import ast
import gc

# GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
gc.collect()
torch.cuda.empty_cache()

# pregatim datele
balanced_df_with_labels = pd.read_csv("reduced_balanced_df.csv")
balanced_df_with_labels['label'] = balanced_df_with_labels['label'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
balanced_df_with_labels["labels_str"] = balanced_df_with_labels["label"].apply(lambda labels: ", ".join(labels))
balanced_df_with_labels = balanced_df_with_labels.rename(columns={"text": "input", "labels_str": "output"})

dataset = Dataset.from_pandas(balanced_df_with_labels[["input", "output"]])

# incarcam model si tokenizer
model_name = "Qwen/Qwen2.5-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto")

# aplicam lora cu alpha jumatate din rank pentru a evita overfitting
lora_config = LoraConfig(
    r=16,
    lora_alpha=8,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
model.gradient_checkpointing_enable()
model.enable_input_require_grads()

# ii dam modelului lista din care sa aleaga pentru a fi mai precis
labels = pd.read_csv("labels.csv")
labels = labels["label"].to_list()
label_str = ", ".join(labels)

def tokenize_fn(batch):
    # prompt pentru model
    prompt_header = "<|im_start|>user\n Predict the label(s) for this input from the following list: " + label_str + " Text: "
    assistant_prefix = " <|im_end|>\n<|im_start|>assistant\n"

    full_prompts = [prompt_header + text + assistant_prefix for text in batch["input"]]
    full_texts = [prompt + output for prompt, output in zip(full_prompts, batch["output"])]

    tokenized = tokenizer(
        full_texts,
        padding="max_length",
        truncation=True,
        max_length=2048,
        return_tensors="pt"
    )
    prompt_tokenized = tokenizer(
        full_prompts,
        padding="max_length",
        truncation=True,
        max_length=2048,
        return_tensors="pt"
    )

    labels = tokenized["input_ids"].clone()
    prompt_lens = prompt_tokenized["attention_mask"].sum(dim=1)

    # mascam inputul pentru memorie si acuratete ( ne intereseaza generarea)
    for i, l in enumerate(prompt_lens):
        labels[i, :l] = -100  

    tokenized["labels"] = labels
    return tokenized

tokenized_dataset = dataset.map(tokenize_fn, batched=True, remove_columns=dataset.column_names)

# argumente necesare
training_args = TrainingArguments(
    output_dir="qwen2.5-lora-multilabel-instruct",
    eval_strategy="no",
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=6,
    weight_decay=0.01,
    bf16=True,
    remove_unused_columns=False,
    report_to="none",
    logging_steps=20,
    save_strategy="epoch",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
)

trainer.train()

trainer.save_model("qwen2.5-lora-multilabel")
tokenizer.save_pretrained("qwen2.5-lora-multilabel")
