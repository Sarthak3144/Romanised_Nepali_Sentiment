import streamlit as st
import requests

# Set page visibility attributes layout controls definitions - KEEPING EXACT ORIGINAL
st.set_page_config(page_title="Nepali Sentiment Dashboard", page_icon="🇳🇵", layout="centered")

st.title("🇳🇵 Romanized Nepali Sentiment Analyzer")
st.write(
    "Enter e-commerce feedback logs to analyze operational targets using fine-tuned DistilBERT via FastAPI routes.")

# User prompt interface window initialization
user_input = st.text_area("Customer Review Text:",
                          placeholder="Type romanized comment... (e.g., product ekdam jhyau lagne raixa)")

# Trigger operations evaluations processing boundaries rule checks
if st.button("Run Analytics Pass", type="primary"):
    if user_input.strip() == "":
        st.warning("Please specify a valid sequence of characters first.")
    else:
        # Define internal connection routing paths matching FastAPI standard boundaries configuration
        backend_url = "http://127.0.0.1:8000/predict"
        request_payload = {"review_content": str(user_input)}

        with st.spinner("Streaming calculations over background network hooks..."):
            try:
                # Dispatched HTTP transmission call
                api_response = requests.post(backend_url, json=request_payload, timeout=10)

                if api_response.status_code == 200:
                    response_json = api_response.json()
                    result_data = response_json["data"]

                    # Extract variables from the response
                    orig_text = result_data["original_review"]
                    proc_text = result_data["processed_review"]
                    sentiment_output = result_data["sentiment"]
                    confidence_pct = result_data["confidence"] * 100

                    st.success("API Transaction Successfully Completed!")

                    # -------------------------------------------------------------
                    # ADDITION: Recruiter Comparison Panel (Fully Added)
                    # -------------------------------------------------------------
                    st.markdown("### 🔍 Text Preprocessing & Cleaning Pipeline")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Before (Raw User Input):**")
                        st.caption(orig_text)
                    with col2:
                        st.markdown("**After (Standardized Engine Input):**")
                        st.code(proc_text, language="text")
                    st.write("---")

                    # -------------------------------------------------------------
                    # RESTORED: All Original Sentiment Output Cards & Messages
                    # -------------------------------------------------------------
                    st.markdown("### 📊 Model Inference Prediction")

                    if sentiment_output == "Positive":
                        st.metric(label="Inferred Label State", value="🟢 Positive Feedback")
                        st.info(
                            f"The structural pipeline evaluates this statement as positive with {confidence_pct:.2f}% accuracy configuration bounds.")
                    elif sentiment_output == "Negative":
                        st.metric(label="Inferred Label State", value="🔴 Negative Complaint")
                        st.error(
                            f"The text profile contains severe user dissatisfaction metrics. Confidence: {confidence_pct:.2f}%.")
                    else:
                        st.metric(label="Inferred Label State", value="🟡 Neutral Response")
                        st.warning(
                            f"Ambiguous or objective structural patterns identified. Confidence: {confidence_pct:.2f}%.")

                    # KEEPING ORIGINAL METRIC SCORE SUMMARY BELOW CARDS
                    st.write(f"**Confidence Score:** {confidence_pct:.2f}%")

                    # RESTORED: Full original production JSON expander block
                    with st.expander("Show Production JSON Response Package Payload"):
                        st.json(response_json)
                else:
                    st.error(f"Backend Server returned an error code: {api_response.status_code}")
                    st.info(api_response.text)

            except requests.exceptions.Timeout:
                st.error("Infrastructure Gateway Timeout: The FastAPI server took too long to process.")
            except requests.exceptions.ConnectionError:
                st.error(
                    "Infrastructure Gateway Error: Cannot connect to FastAPI on http://127.0.0.1:8000. Please launch your backend using 'uvicorn api_backend:app --reload' in your first terminal window.")