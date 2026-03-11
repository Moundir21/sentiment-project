
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# ------------------------------
# تحميل نموذج AraBERT المدرب مسبقاً
# ------------------------------
model_dir = "arabert_sentiment_model"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSequenceClassification.from_pretrained(model_dir)

# ------------------------------
# Load API keys
# ------------------------------
with open("api_keys.json", "r") as f:
    api_keys = json.load(f)

# ------------------------------
# FastAPI app
# ------------------------------
app = FastAPI(title="Algerian Dialect Sentiment API")

# ------------------------------
# Request model
# ------------------------------
class TextRequest(BaseModel):
    text: str

# ------------------------------
# Helper function: predict sentiment
# ------------------------------
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)
    pred_class = torch.argmax(probs).item()
    confidence = probs[0][pred_class].item()
    
    label_map = {0: "negative", 1: "neutral", 2: "positive"}
    
    return {
        "sentiment": label_map[pred_class],
        "confidence": round(confidence, 2)
    }

# ------------------------------
# API route with API Key check
# ------------------------------
@app.post("/predict")
def predict(request: TextRequest, x_api_key: str = Header(...)):
    # تحقق من مفتاح الـ API
    if x_api_key not in api_keys.values():
        raise HTTPException(status_code=401, detail="Invalid API Key")
    
    result = predict_sentiment(request.text)
    return result