import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
model = PeftModel.from_pretrained(base_model, "qwen2.5-lora-multilabel-instruct/checkpoint-2352")

# imbinam lora cu modelul
model = model.merge_and_unload()

model.save_pretrained("qwen2.5-lora-multilabel-instruct/merged_model")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
tokenizer.save_pretrained("qwen2.5-lora-multilabel-instruct/merged_model")
