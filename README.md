# PrivacyGuard

Find and hide personal information (PII) in text and images. NLP lab project demo: a fine-tuned DistilBERT plus a rules layer,
with OCR (RapidOCR) for screenshots. English only, not a replacement for a real compliance tool.

The model is downloaded at start-up from the Hugging Face Hub: `viralk/privacyguard-distilbert`
(set another repo with the `HUB_MODEL_ID` secret/environment variable).

Run locally: `pip install -r requirements.txt && streamlit run streamlit_app.py`
