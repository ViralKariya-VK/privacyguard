"""
Logic shared by the two front-ends (app.py = Gradio, streamlit_app.py = Streamlit).
No UI library is imported here; problems are raised as ValueError and each front-end shows them its own way.
"""
import os
from pathlib import Path

from src.config import ENTITY_TYPES
from src.data import merge_adjacent
from src.postprocess import mark_birth_dates
from src.redact import mask_text

# Hub repo with the fine-tuned model (used when models/pii-distilbert is not next to the app).
HUB_MODEL_ID = os.environ.get("HUB_MODEL_ID", "viralk/privacyguard-distilbert")
MAX_CHARS = 20000          # free CPU hosting: keep requests small
MAX_IMAGE_SIDE = 3000      # bigger photos are shrunk before OCR
LOCAL_MODEL = Path("models/pii-distilbert")
CLASSIC_ML_PKL = Path("models/classic_ml.pkl")

HYBRID = "Hybrid: fine-tuned DistilBERT + rules"
BERT_ONLY = "Transformer only (fine-tuned DistilBERT)"
CLASSIC_ML = "Classic ML (SVM on token features)"
REGEX = "Classic: regex + spaCy"
MODEL_CHOICES = [HYBRID, BERT_ONLY, REGEX]
if CLASSIC_ML_PKL.exists():
    MODEL_CHOICES.insert(2, CLASSIC_ML)

# What to mask by default. DATE and LOCATION are switched off: on a resume or an invoice the dates and
# cities are not private and hiding them makes the document useless (found while testing a resume).
# DOB = a date that comes after "born / birth date / DOB", that one IS private so it stays on.
# Everything is always DETECTED; this list only decides what gets masked.
MASK_CHOICES = ENTITY_TYPES + ["DOB"]
DEFAULT_MASK = [t for t in MASK_CHOICES if t not in ("DATE", "LOCATION")]

COLORS = {"NAME": "#6366f1", "EMAIL": "#0ea5e9", "PHONE": "#10b981", "ID_NUMBER": "#ef4444",
          "USERNAME": "#8b5cf6", "PASSWORD": "#dc2626", "IP_ADDRESS": "#f59e0b",
          "LOCATION": "#14b8a6", "DATE": "#94a3b8", "DOB": "#ec4899"}

TEXT_EXAMPLES = [
    "Hi, this is Rahul Verma. Please call me on +91 98765 43210 or write to rahul.verma@gmail.com. My Aadhaar is 4521 8890 1234.",
    "Server log: user admin_kiran failed login from 10.24.3.77 on 2024-05-11, password tried: Kiran@2024!",
    "Ship to Fatima Sheikh, Flat 402, Sunrise Apartments, Andheri West, Mumbai 400058, India.",
]

_cache = {}   # models are heavy so load them only once per process


def get_model(choice):
    if choice in (HYBRID, BERT_ONLY):
        if "bert" not in _cache:      # hybrid and plain share the same weights
            from src.transformer_infer import TransformerPredictor
            path = os.environ.get("PII_MODEL") or (str(LOCAL_MODEL) if LOCAL_MODEL.exists() else HUB_MODEL_ID)
            if path.startswith("YOUR_HF_USERNAME"):
                raise ValueError("Model not configured: set HUB_MODEL_ID to your Hub model repo.")
            _cache["bert"] = TransformerPredictor(path)
        _cache["bert"].refine = (choice == HYBRID)
        return _cache["bert"]
    if choice not in _cache:
        if choice == CLASSIC_ML:
            from src.classic_ml import ClassicMLPredictor
            _cache[choice] = ClassicMLPredictor.load(CLASSIC_ML_PKL)
        else:
            from src.classic_rules import RegexSpacyPredictor
            _cache[choice] = RegexSpacyPredictor()
    return _cache[choice]


def to_highlight(text, spans):
    """spans -> list of (text, label or None) pieces"""
    pieces, pos = [], 0
    for s, e, label in spans:
        if s > pos:
            pieces.append((text[pos:s], None))
        pieces.append((text[s:e], label))
        pos = e
    if pos < len(text):
        pieces.append((text[pos:], None))
    return pieces


def summary(spans, n_masked=None):
    if not spans:
        return "✅ No PII found"
    n_masked = len(spans) if n_masked is None else n_masked
    counts = {}
    for _, _, label in spans:
        counts[label] = counts.get(label, 0) + 1
    parts = ", ".join(f"{n} {lab}" for lab, n in sorted(counts.items(), key=lambda kv: -kv[1]))
    return f"🔒 {len(spans)} PII span(s) found ({n_masked} masked): {parts}"


def analyse_text(text, choice, mask):
    """returns (all spans, masked text, summary line)"""
    if len(text) > MAX_CHARS:
        raise ValueError(f"Text too long ({len(text):,} characters). The limit is {MAX_CHARS:,}.")
    spans = mark_birth_dates(text, merge_adjacent(get_model(choice).predict(text), text))
    to_mask = [sp for sp in spans if sp[2] in mask]
    return spans, mask_text(text, to_mask), summary(spans, len(to_mask))


def analyse_image(image, choice, tag_style, mask):
    """returns the process_image result dict plus 'summary'; image is shrunk when it is very large"""
    from src.image_pipeline import process_image
    if max(image.size) > MAX_IMAGE_SIDE:
        image = image.copy()
        image.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
    res = process_image(image, get_model(choice), mode="tag" if tag_style else "mask", entities=set(mask))
    res["summary"] = summary(res["spans"], len(res["masked_spans"])) if res["text"].strip() else "No text found in the image."
    res["image"] = image
    return res
