
import torch
import numpy as np
from datasets import load_dataset
from transformers import BertTokenizer, BertForSequenceClassification
from peft import PeftModel
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)

# 1. Configuration
MODEL_NAME = "yiyanghkust/finbert-tone"
ADAPTER_PATH = "../models/fin_lora_adapter"
DATA_PATH = "../data/datasets.csv"

LABEL_MAP = {
    "negative": 0,
    "neutral": 1,
    "positive": 2
}

ID2LABEL = {
    0: "Negative",
    1: "Neutral",
    2: "Positive"
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Load dataset and reproduce original split
dataset = load_dataset("csv", data_files=DATA_PATH)["train"]

dataset = dataset.train_test_split(
    test_size=0.1,
    seed=42
)

test_data = dataset["test"]

def encode_labels(example):
    example["label"] = LABEL_MAP[
        example["Sentiment"].lower()
    ]
    return example

test_data = test_data.map(encode_labels)

# 3. Load tokenizer
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

def preprocess(batch):
    return tokenizer(
        batch["Sentence"],
        padding="max_length",
        truncation=True,
        max_length=128
    )

test_data = test_data.map(preprocess, batched=True)

test_data.set_format(
    "torch",
    columns=["input_ids", "attention_mask", "label"]
)

test_loader = DataLoader(test_data, batch_size=16)

# 4. Load trained FinBERT + LoRA model
base_model = BertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=3
)

model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_PATH
)

model.to(DEVICE)
model.eval()

# 5. Evaluate
all_predictions = []
all_labels = []

with torch.no_grad():
    for batch in test_loader:

        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["label"]

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        predictions = torch.argmax(
            outputs.logits,
            dim=1
        )

        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.numpy())

# 6. Calculate metrics
accuracy = accuracy_score(all_labels, all_predictions)
macro_f1 = f1_score(
    all_labels,
    all_predictions,
    average="macro"
)

print("\n========== MODEL EVALUATION ==========")
print(f"Device: {DEVICE}")
print(f"Evaluation samples: {len(all_labels)}")
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Macro F1 Score: {macro_f1:.4f}")

print("\n========== CLASSIFICATION REPORT ==========")

print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=["Negative", "Neutral", "Positive"],
        digits=4
    )
)

print("\n========== CONFUSION MATRIX ==========")
print(
    confusion_matrix(
        all_labels,
        all_predictions
    )
)
