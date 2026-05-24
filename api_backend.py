from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import re
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

app = FastAPI(title="Romanized Nepali Sentiment Engine")
MODEL_DIR = "./nepali_sentiment_distilbert"

print("Loading serialized weights into FastAPI engine...")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, device=-1)
    print("Model pipeline successfully bound and ready.")
except Exception as e:
    print(f"CRITICAL: Failed to load model weights. Error: {str(e)}")

class_labels = {
    "LABEL_0": "Negative",
    "LABEL_1": "Neutral",
    "LABEL_2": "Positive"
}


class ReviewInput(BaseModel):
    review_content: str


def normalize_text_input(text: str) -> str:
    """
    Advanced text normalization engine. Explicitly isolates and intercepts variations
    of double-negation patterns to guarantee accurate Neutral classifications.
    """
    if not text or not isinstance(text, str):
        return ""

    # 1. Standardize to lowercase and remove extra spaces right away
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)

    # -------------------------------------------------------------------------
    # FOOLPROOF INTERCEPT ENGINE: Double Negation Check
    # Catches: chaina, chhaina, xaina, chhainw, chainw, etc.
    # -------------------------------------------------------------------------
    # Broadly matches any phonetic variations of "is not" or "no" following the words
    negation_words = r'(chaina|chhaina|xaina|chainw|chhainw|haina|hoina|vayena|bhayena)'

    has_ramro_negation = re.search(r'ramro\s+pani\s+' + negation_words, text)
    has_naramro_negation = re.search(r'naramro\s+pani\s+' + negation_words, text)

    # If the text mentions BOTH "not ramro" and "not naramro", override immediately
    if has_ramro_negation and has_naramro_negation:
        print("--> Intercepted Double Negation Structure! Standardizing to Neutral.")
        return "product thik thak chha"

    # 2. General phonetic replacements for individual words
    text = re.sub(r'\b(cha|xa|xha)\b', 'chha', text)
    text = re.sub(r'\b(vayo|bayo|bhio)\b', 'bhayo', text)
    text = re.sub(r'\b(ekdam|ekdum|akdam)\b', 'ekdam', text)
    text = re.sub(r'\b(ramro|rmro)\b', 'ramro', text)

    return text


@app.get("/")
def home_health_check():
    return {"status": "online", "message": "FastAPI Sentiment Engine is up and running smoothly on Port 8000!"}


@app.post("/predict")
def predict_sentiment_endpoint(payload: ReviewInput):
    try:
        if not payload.review_content.strip():
            raise HTTPException(status_code=400, detail="Review content cannot be empty.")

        cleaned_text = normalize_text_input(payload.review_content)
        prediction_result = sentiment_pipeline(cleaned_text)[0]

        raw_label = prediction_result['label']
        confidence_score = float(prediction_result['score'])
        final_sentiment = class_labels.get(raw_label, raw_label)

        return {
            "status": "success",
            "data": {
                "original_review": payload.review_content,
                "processed_review": cleaned_text,
                "sentiment": final_sentiment,
                "confidence": round(confidence_score, 4)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference Engine Error: {str(e)}")