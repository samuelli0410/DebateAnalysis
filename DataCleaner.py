import os
import re
import nltk
import pandas as pd
from collections import defaultdict
from nltk.tokenize import PunktSentenceTokenizer, sent_tokenize
print("1")

ARTICLES_DIR = "C:/Users/samue/Downloads/Blackrock Assessment/DebateAnalysis/PropagandaDetection-master/datasets/train-articles"
LABELS_DIR = "C:/Users/samue/Downloads/Blackrock Assessment/DebateAnalysis/PropagandaDetection-master/datasets/train-labels-FLC"

data = []
print("1.5")

def load_labels(label_path):
    spans = []
    with open(label_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) != 4:
                continue
            _, technique, start, end = parts
            spans.append((technique, int(start), int(end)))
    return spans
print("2")
import nltk
nltk.download('punkt')
for fname in os.listdir(ARTICLES_DIR):
    if not fname.endswith('.txt'):
        continue

    article_id = fname.replace('article', '').replace('.txt', '')
    text_path = os.path.join(ARTICLES_DIR, fname)
    label_path = os.path.join(LABELS_DIR, f'article{article_id}.task-FLC.labels')

    with open(text_path, 'r', encoding='utf-8') as f:
        raw_text = f.read()

    # Article title is first line, skip empty second line
    lines = raw_text.strip().split("\n")
    title = lines[0]
    article_body = "\n".join(lines[2:])
    full_text = f"{title}. {article_body}"

    # Sentence tokenize
    tokenizer = PunktSentenceTokenizer()  # default English
    sentences = tokenizer.tokenize(full_text)

    # Track sentence spans in char offsets
    offset = 0
    sentence_spans = []
    for sentence in sentences:
        start = full_text.find(sentence, offset)
        end = start + len(sentence)
        sentence_spans.append((start, end, sentence))
        offset = end

    # Load and assign labels
    span_labels = load_labels(label_path) if os.path.exists(label_path) else []
    sentence_labels = defaultdict(set)

    for label, span_start, span_end in span_labels:
        for i, (sent_start, sent_end, _) in enumerate(sentence_spans):
            if sent_start <= span_start < sent_end or sent_start < span_end <= sent_end or (span_start <= sent_start and span_end >= sent_end):
                sentence_labels[i].add(label)

    for i, (sent_start, sent_end, sent_text) in enumerate(sentence_spans):
        labels = list(sentence_labels.get(i, []))
        # if labels != []:
        #     data.append({
        #         "article_id": article_id,
        #         "sentence_id": i,
        #         "text": sent_text.strip(),
        #         "labels": labels
        #     })
        data.append({
            "article_id": article_id,
            "sentence_id": i,
            "text": sent_text.strip(),
            "labels": labels
        })


df = pd.DataFrame(data)
print(df.head(20))
print(df.size)

# Optional: Save to CSV or JSONL
# df.to_json("C:/Users/samue/Downloads/Blackrock Assessment/DebateAnalysis/CleanData.jsonl", orient="records", lines=True)
print("done")