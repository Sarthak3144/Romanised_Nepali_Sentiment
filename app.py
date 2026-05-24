import streamlit as st
import re
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# 1. Page Configuration Setup
st.set_page_config(page_title="Nepali Sentiment Analyzer", page_icon="🇳", layout="centered")

st.title("🇳Romanized Nepali Sentiment Analyzer")
st.write("An end-to-end DistilBERT system designed to analyze and normalize code-mixed Romanized Nepali comments.")


# 2. Optimized Caching for Neural Network Weights
@st.cache_resource
def load_production_pipeline():
    model_directory = "./nepali_sentiment_distilbert"
    tokenizer = AutoTokenizer.from_pretrained(model_directory)
    model = AutoModelForSequenceClassification.from_pretrained(model_directory)
    return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, device=-1)


# Initialize the pipeline
with st.spinner("Loading Transformer model weights into cloud memory..."):
    sentiment_engine = load_production_pipeline()


# 3. Enhanced Normalization Engine
def clean_phonetic_text(text: str) -> str:
    if not text or not isinstance(text, str):
        return ""

    # Standardize basic spaces and cases first
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)

    # Standardize common spelling variants to create a predictable text stream
    text = re.sub(r'\b(cha|xa|xha)\b', 'chha', text)
    text = re.sub(r'\b(vayo|bayo|bhio)\b', 'bhayo', text)
    text = re.sub(r'\b(ekdam|ekdum|akdam)\b', 'ekdam', text)
    text = re.sub(r'\b(ramro|rmro)\b', 'ramro', text)
    text = re.sub(r'\b(chaina|xaina|chainw|haina|hoina)\b', 'chhaina', text)

    # -------------------------------------------------------------------------
    # NEW LOGICAL OVERRIDE: Order-Independent Keyword Interception
    # Checks if the text negates BOTH "ramro" and "naramro" regardless of their order.
    # -------------------------------------------------------------------------
    has_ramro_negation = bool(re.search(r'ramro\s+pani\s+chhaina', text) or re.search(r'ramro\s+chhaina', text))
    has_naramro_negation = bool(re.search(r'naramro\s+pani\s+chhaina', text) or re.search(r'naramro\s+chhaina', text))

    if has_ramro_negation and has_naramro_negation:
        return "product thik thak chha"

    return text


class_labels = {"LABEL_0": "Negative", "LABEL_1": "Neutral", "LABEL_2": "Positive"}

# 4. Interactive User Form
user_input = st.text_area("Customer Review Text:", placeholder="Type romanized comment here...")

if st.button("Run Model Inference", type="primary"):
    if user_input.strip() == "":
        st.warning("Please type a valid review message first.")
    else:
        # Preprocess input phrase
        cleaned_review = clean_phonetic_text(user_input)

        # Calculate text inferences
        raw_prediction = sentiment_engine(cleaned_review)[0]
        predicted_sentiment = class_labels.get(raw_prediction['label'], raw_prediction['label'])
        confidence_metric = raw_prediction['score']

        # If our rule caught an override, force an appropriate high-confidence display
        if cleaned_review == "product thik thak chha":
            predicted_sentiment = "Neutral"
            confidence_metric = 1.0

        st.success("Analysis Complete!")


        st.markdown("###  Text Preprocessing & Cleaning Pipeline")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Before (Raw User Input):**")
            st.caption(user_input)
        with col2:
            st.markdown("**After (Standardized Engine Input):**")
            st.code(cleaned_review, language="text")
        st.write("---")

        # Display Prediction Cards
        st.markdown("###  Model Inference Prediction")
        if predicted_sentiment == "Positive":
            st.metric(label="Inferred Label State", value="🟢 Positive Feedback")
            st.info(
                f"The structural pipeline evaluates this statement as positive with {confidence_metric * 100:.2f}% confidence.")
        elif predicted_sentiment == "Negative":
            st.metric(label="Inferred Label State", value="🔴 Negative Complaint")
            st.error(
                f"The text profile contains severe user dissatisfaction metrics. Confidence: {confidence_metric * 100:.2f}%.")
        else:
            st.metric(label="Inferred Label State", value="🟡 Neutral Response")
            st.warning(
                f"Ambiguous or objective structural patterns identified. Confidence: {confidence_metric * 100:.2f}%.")

        st.write(f"**Confidence Score:** {confidence_metric * 100:.2f}%")

        # Metadata Expander Package Payload
        with st.expander("Show Production JSON Response Package Payload"):
            st.json({
                "status": "success",
                "metrics": {
                    "raw_input": user_input,
                    "normalized_token_stream": cleaned_review,
                    "target_class": predicted_sentiment,
                    "confidence_score": round(float(confidence_metric), 4)
                }
            })