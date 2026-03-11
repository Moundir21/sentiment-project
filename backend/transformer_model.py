from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification


def load_transformer_model():

    model_name = "aubmindlab/bert-base-arabertv2"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=3
    )

    return tokenizer, model