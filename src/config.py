"""
Central place for paths + the label scheme, so every other file imports from here.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "raw"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

TRAIN_FILE = DATA_DIR / "train_en.jsonl"
TEST_FILE = DATA_DIR / "val_en.jsonl"      # official "validation" split, we use it as our held-out TEST set
OOD_FILE = ROOT / "data" / "ood_test.json"  # hand-written out-of-domain examples

DATA_URLS = {
    TRAIN_FILE: "https://huggingface.co/datasets/ai4privacy/pii-masking-300k/resolve/main/data/train/1english_openpii_30k.jsonl",
    TEST_FILE: "https://huggingface.co/datasets/ai4privacy/pii-masking-300k/resolve/main/data/validation/1english_openpii_8k.jsonl",
}

# The dataset has ~27 fine-grained labels (GIVENNAME1, LASTNAME2, STREET, ...).
# Too fine for our purpose, so we group them into 9 types.
# Labels that are not really "private" (TIME, SEX, TITLE, ...) are dropped -> treated as 'O'.
LABEL_MAP = {
    "GIVENNAME1": "NAME", "GIVENNAME2": "NAME",
    "LASTNAME1": "NAME", "LASTNAME2": "NAME", "LASTNAME3": "NAME",
    "EMAIL": "EMAIL",
    "TEL": "PHONE",
    "IDCARD": "ID_NUMBER", "SOCIALNUMBER": "ID_NUMBER",
    "PASSPORT": "ID_NUMBER", "DRIVERLICENSE": "ID_NUMBER",
    "USERNAME": "USERNAME",
    "PASS": "PASSWORD",
    "IP": "IP_ADDRESS",
    "STREET": "LOCATION", "BUILDING": "LOCATION", "POSTCODE": "LOCATION",
    "SECADDRESS": "LOCATION", "CITY": "LOCATION", "STATE": "LOCATION",
    "COUNTRY": "LOCATION",
    "DATE": "DATE", "BOD": "DATE",
}

ENTITY_TYPES = ["NAME", "EMAIL", "PHONE", "ID_NUMBER", "USERNAME",
                "PASSWORD", "IP_ADDRESS", "LOCATION", "DATE"]

# BIO tag list used by the token classifiers (classic ML + transformer)
TAGS = ["O"] + [f"{p}-{t}" for t in ENTITY_TYPES for p in ("B", "I")]
TAG2ID = {t: i for i, t in enumerate(TAGS)}
ID2TAG = {i: t for t, i in TAG2ID.items()}
