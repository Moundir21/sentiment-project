import torch
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments
from transformers import EarlyStoppingCallback
from transformers import DataCollatorWithPadding

from preprocess import load_data
from transformer_model import load_transformer_model

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

import numpy as np

import os
print(os.getcwd())

def compute_metrics(pred):

    labels = pred.label_ids

    preds = np.argmax(pred.predictions, axis=1)

    acc = accuracy_score(labels, preds)

    f1 = f1_score(labels, preds, average="weighted")

    return {
        "accuracy": acc,
        "f1": f1
    }


class SentimentDataset(Dataset):

    def __init__(self, texts, labels, tokenizer):

        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):

        return len(self.texts)

    def __getitem__(self, idx):

        text = self.texts[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=256
        )

        encoding["labels"] = self.labels[idx]

        return encoding


# تحميل البيانات
df = load_data("data/dataset.csv")

texts = df["clean_text"].tolist()

labels = df["sentiment"].replace({-1:0,0:1,1:2}).tolist()


X_train, X_test, y_train, y_test = train_test_split(

    texts,
    labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)


tokenizer, model = load_transformer_model()


train_dataset = SentimentDataset(
    X_train,
    y_train,
    tokenizer
)

test_dataset = SentimentDataset(
    X_test,
    y_test,
    tokenizer
)


data_collator = DataCollatorWithPadding(tokenizer)


training_args = TrainingArguments(

    output_dir="./results",

    num_train_epochs=6,

    learning_rate=2e-5,

    per_device_train_batch_size=16,

    per_device_eval_batch_size=16,

    gradient_accumulation_steps=2,

    warmup_ratio=0.1,

    weight_decay=0.01,

    logging_dir="./logs",

    evaluation_strategy="epoch",

    save_strategy="epoch",

    load_best_model_at_end=True,

    metric_for_best_model="f1",

    greater_is_better=True,

    fp16=True,

    logging_steps=100,

    report_to=[],  # هنا نوقف WandB

)


trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=train_dataset,

    eval_dataset=test_dataset,

    tokenizer=tokenizer,

    data_collator=data_collator,

    compute_metrics=compute_metrics,

    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)


trainer.train()


trainer.save_model("arabert_sentiment_model")

tokenizer.save_pretrained("arabert_sentiment_model")


print("Training Finished")
