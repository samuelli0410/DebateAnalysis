from transformers import AutoTokenizer
import torch
from transformers import BertModel
import nltk
nltk.download("punkt")
from nltk.tokenize import PunktSentenceTokenizer, sent_tokenize

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
labels = [
    "Loaded Language",
    "Name Calling,Labeling",
    "Repetition",
    "Exaggeration,Minimization",
    "Doubt",
    "Appeal to fear-prejudice",
    "Flag-Waving",
    "Causal Oversimplification",
    "Slogans",
    "Appeal to Authority",
    "Black-and-White Fallacy",
    "Thought-terminating Cliches",
    "Whataboutism",
    "Reductio ad Hitlerum",
    "Red Herring",
    "Bandwagon",
    "Obfuscation,Intentional Vagueness,Confusion",
    "Straw Men"
]




model = MultiLabelBERT("bert-base-uncased", num_labels=len(labels))
model.load_state_dict(torch.load("C:/Users/samue/Downloads/Blackrock Assessment/DebateAnalysis/rhetoric_model.pt"))
model.eval()

with open("C:/Users/samue/Downloads/Blackrock Assessment/DebateAnalysis/transcript.txt", "r", encoding="utf-8") as f:
    full_text = f.read()

tokenizer = PunktSentenceTokenizer()  # default English
sentences = tokenizer.tokenize(full_text)
hf_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
results = []

from tqdm import tqdm
for sentence in tqdm(sentences, desc="Classifying sentences"):
    inputs = hf_tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=256)
    with torch.no_grad():
        outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        probs = torch.sigmoid(outputs["logits"])
        predictions = (probs > 0.5).int()

    predicted_labels = [label for i, label in enumerate(labels) if predictions[0][i] == 1]
    results.append(f"{sentence} --> {predicted_labels}")

with open("C:/Users/samue/Downloads/Blackrock Assessment/DebateAnalysis/classified_transcript.txt", "w", encoding="utf-8") as f:
    for line in results:
        f.write(line + "\n")

print("Done! Saved to 'classified_transcript.txt'")