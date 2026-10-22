"""
Classic baseline 1: regex patterns + spaCy's pretrained NER. No training at all.

Idea:
  - things with a fixed format (email, ip, phone, ssn...) -> regex
  - names / places / dates -> spaCy NER (PERSON, GPE, DATE ...)
Regex wins when both find something at the same place.
"""
import re

import spacy

# (label, pattern). order doesn't matter much, overlaps are cleaned later
PATTERNS = [
    ("EMAIL", re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")),
    ("IP_ADDRESS", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
    ("IP_ADDRESS", re.compile(r"\b(?:[0-9a-fA-F]{1,4}:){2,7}[0-9a-fA-F]{1,4}\b")),
    ("ID_NUMBER", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),            # US ssn
    ("ID_NUMBER", re.compile(r"\b\d{4}\s\d{4}\s\d{4}\b")),          # aadhaar style
    ("ID_NUMBER", re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b")),           # PAN style
    ("DATE", re.compile(r"\b\d{1,4}[/-]\d{1,2}[/-]\d{1,4}\b")),
]
PHONE_RE = re.compile(r"(?<!\w)\+?\d[\d\s().-]{7,}\d(?!\w)")

SPACY_MAP = {"PERSON": "NAME", "GPE": "LOCATION", "LOC": "LOCATION",
             "FAC": "LOCATION", "DATE": "DATE"}


class RegexSpacyPredictor:
    def __init__(self, model_name="en_core_web_sm"):
        try:
            self.nlp = spacy.load(model_name, disable=["lemmatizer"])
        except OSError:
            raise SystemExit(f"spaCy model missing, run: python -m spacy download {model_name}")

    def predict(self, text):
        spans = []

        for label, pat in PATTERNS:
            for m in pat.finditer(text):
                spans.append((m.start(), m.end(), label))

        # phone numbers: loose regex then check that there are enough digits
        # (otherwise dates like 12-05-1990 would also match)
        for m in PHONE_RE.finditer(text):
            digits = sum(c.isdigit() for c in m.group())
            if digits >= 9:
                spans.append((m.start(), m.end(), "PHONE"))

        regex_spans = list(spans)

        for ent in self.nlp(text).ents:
            label = SPACY_MAP.get(ent.label_)
            if label is None:
                continue
            # skip if regex already covers this region
            if any(ent.start_char < e and ent.end_char > s for s, e, _ in regex_spans):
                continue
            spans.append((ent.start_char, ent.end_char, label))

        return sorted(spans)
