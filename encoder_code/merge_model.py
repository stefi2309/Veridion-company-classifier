import ast
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel

df = pd.read_csv("reduced_balanced_df.csv")
df['label'] = df['label'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
mlb = MultiLabelBinarizer()
_ = mlb.fit_transform(df['label'])
num_labels = len(mlb.classes_)

base_model = AutoModelForSequenceClassification.from_pretrained(
    "allenai/longformer-base-4096",
    num_labels=num_labels,  
    problem_type="multi_label_classification"
)

model = PeftModel.from_pretrained(base_model, "output/checkpoint-12540")

model = model.merge_and_unload()

model.save_pretrained("merged_model")
tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")
tokenizer.save_pretrained("merged_model")
