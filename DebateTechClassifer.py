import pandas as pd
from datasets import Dataset
from sklearn.preprocessing import MultiLabelBinarizer
import json
print("Loading libraries")
from tensorflow import keras
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch


examples = []
with open("C:/Users/samue/Downloads/Blackrock Assessment/DebateAnalysis/CleanData.jsonl", "r") as f:
    for line in f:
        row = json.loads(line)
        examples.append({
            "text": row["text"],
            "labels": row["labels"]
        })
print("data Loaded")
rhetorical_mapping = {
    "Loaded Language": "pathos",
    "Name Calling,Labeling": "ethos",
    "Repetition": "pathos",
    "Exaggeration,Minimization": "pathos",
    "Doubt": "logos",
    "Appeal to fear-prejudice": "pathos",
    "Flag-Waving": "ethos",
    "Causal Oversimplification": "logos",
    "Slogans": "pathos",
    "Appeal to Authority": "ethos",
    "Black-and-White Fallacy": "logos",
    "Thought-terminating Cliches": "pathos",
    "Whataboutism": "ethos",
    "Reductio ad Hitlerum": "ethos",
    "Red Herring": "logos",
    "Bandwagon": "pathos",
    "Obfuscation,Intentional Vagueness,Confusion": "ethos",
    "Straw Men": "logos"
}


def map_labels(label_list):
    return list(set(rhetorical_mapping[label] for label in label_list if label in rhetorical_mapping))
df = pd.DataFrame(examples)
df['mapped_labels'] = df['labels'].apply(map_labels)


mlb = MultiLabelBinarizer(classes=["pathos", "ethos", "logos", "statistics", "anecdote"])
label_matrix = mlb.fit_transform(df['mapped_labels'])

df['label_vector'] = label_matrix.tolist()

dataset = Dataset.from_pandas(df[['text', 'label_vector']])


tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize_fn(example):
    return tokenizer(example["text"], padding="max_length", truncation=True, max_length=256)
tokenized_ds = dataset.map(tokenize_fn)

tokenized_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label_vector"])

class MultiLabelDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset):
        self.ds = hf_dataset
    def __len__(self):
        return len(self.ds)
    def __getitem__(self, idx):
        item = self.ds[idx]
        return {
            "input_ids": item["input_ids"],
            "attention_mask": item["attention_mask"],
            "labels": torch.tensor(item["label_vector"], dtype=torch.float)
        }

train_dataset = MultiLabelDataset(tokenized_ds)

from transformers import BertModel
class MultiLabelBERT(torch.nn.Module):
    def __init__(self, base_model_name, num_labels):
        super().__init__()
        self.bert = BertModel.from_pretrained(base_model_name)
        self.classifier = torch.nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.classifier(outputs.pooler_output)
        loss = None
        if labels is not None:
            loss_fct = torch.nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels)
        return {"loss": loss, "logits": logits}
model = MultiLabelBERT("bert-base-uncased", num_labels=5)
training_args = TrainingArguments(
    output_dir="C:/Users/samue/Downloads/Blackrock Assessment/DebateAnalysis/rhetoric_model",
    per_device_train_batch_size=16,
    num_train_epochs=4,
    save_strategy="epoch",
    logging_dir="C:/Users/samue/Downloads/Blackrock Assessment/DebateAnalysis/logs",
    logging_steps=100
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)
trainer.train()
torch.save(model.state_dict(), "C:/Users/samue/Downloads/Blackrock Assessment/DebateAnalysis/rhetoric_model.pt")