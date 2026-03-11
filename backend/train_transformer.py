import torch
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments

from preprocess import load_data
from transformer_model import load_transformer_model

from sklearn.model_selection import train_test_split


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
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )

        return {

            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(self.labels[idx])
        }


# تحميل البيانات
df = load_data("data/dataset.csv")

texts = df["clean_text"].tolist()

labels = df["sentiment"].replace({-1:0,0:1,1:2}).tolist()


X_train, X_test, y_train, y_test = train_test_split(

    texts,
    labels,
    test_size=0.2,
    random_state=42
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


training_args = TrainingArguments(

    output_dir="./results",

    num_train_epochs=4,

    per_device_train_batch_size=16,

    per_device_eval_batch_size=16,

    learning_rate=2e-5,

    warmup_steps=500,

    weight_decay=0.01,

    logging_dir="./logs",

    evaluation_strategy="epoch",

    save_strategy="epoch",
 
    report_to=[],  # هنا نوقف WandB

)


trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=train_dataset,

    eval_dataset=test_dataset
)


trainer.train()


trainer.save_model("arabert_sentiment_model")

tokenizer.save_pretrained("arabert_sentiment_model")

print("Training Finished")
