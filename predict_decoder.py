from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import torch
from tqdm import tqdm

# configuratii
max_length = 2048
max_new_tokens = 128
batch_size = 1

# GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# datele pentru predict
df_without_labels = pd.read_csv("df_without_labels.csv")
texts = df_without_labels['text'].tolist()
labels = pd.read_csv("labels.csv")
labels = labels["label"].to_list()
label_str = ", ".join(labels)

# modelul antrenat
model_name = "qwen2.5-lora-multilabel-instruct/merged_model"  
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto")
model.eval()

# predict
predictions = []
for i in tqdm(range(0, len(texts), batch_size)):
    batch_texts = texts[i:i+batch_size]
    # folosim promptul pentru a prezice corect
    prompt_header = "<|im_start|>user\n Predict the label(s) for this input from the following list: " + label_str + " Text: "
    assistant_prefix = " <|im_end|>\n<|im_start|>assistant\n"

    full_prompts = [prompt_header + text + assistant_prefix for text in batch_texts]

    inputs = tokenizer(
        full_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    for j in range(len(batch_texts)):
        generated_text = tokenizer.decode(outputs[j], skip_special_tokens=True)
        predicted_label = generated_text.replace(full_prompts[j], "").strip()
        predictions.append(predicted_label)

df_without_labels['label'] = predictions
df_without_labels.to_csv("df_without_labels_with_predictions_qwen.csv", index=False)
