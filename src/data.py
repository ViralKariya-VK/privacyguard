"""
Data loading + the span <-> BIO helpers that all models share.

A "span" everywhere in this project is a tuple (start_char, end_char, label).
Using character offsets (and not token ids) means we can compare very different
models (regex, spaCy, BERT, LLM) with the exact same evaluation code.
"""
import json

from src.config import LABEL_MAP


# gaps that are allowed between two spans of the same type so that we merge them
# e.g. "John" + "Smith" -> "John Smith"
def _allowed_gaps(label):
    if label == "LOCATION":
        return ("", " ", ", ")
    return ("", " ")


def merge_adjacent(spans, text):
    """Sort spans, drop overlaps of different labels and glue neighbouring same-label spans."""
    merged = []
    for s, e, lab in sorted(spans):
        if merged:
            ps, pe, plab = merged[-1]
            if s < pe and lab != plab:
                continue                      # overlapping + different label -> skip the later one
            if lab == plab and (s <= pe or text[pe:s] in _allowed_gaps(lab)):
                merged[-1] = (ps, max(pe, e), lab)
                continue
        merged.append((s, e, lab))
    return merged


def load_docs(path, n=None):
    """Reads the jsonl file, maps labels to our 9 types and returns list of {'text', 'spans'}."""
    docs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            text = row["source_text"]
            spans = []
            for m in row["privacy_mask"]:
                label = LABEL_MAP.get(m["label"])
                if label is None:
                    continue                  # label we don't care about
                spans.append((m["start"], m["end"], label))
            docs.append({"text": text, "spans": merge_adjacent(spans, text)})
            if n and len(docs) >= n:
                break
    return docs


def load_ood(path):
    """
    OOD file is written by hand with entity *strings*, not offsets (easier to write),
    so here we find the offsets ourselves.
    """
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    docs = []
    for item in raw:
        text = item["text"]
        spans = []
        for ent in item["entities"]:
            start = text.find(ent["text"])
            assert start != -1, f"entity not in text: {ent['text']}"
            # label every occurrence (a username can appear many times in a chat log)
            while start != -1:
                spans.append((start, start + len(ent["text"]), ent["label"]))
                start = text.find(ent["text"], start + 1)
        docs.append({"text": text, "spans": merge_adjacent(spans, text)})
    return docs


# ---------------------------------------------------------------- BIO helpers

def spans_to_bio(offsets, spans):
    """For every token (start, end) give a BIO tag based on which gold span it overlaps."""
    tags = []
    current = None                             # index of the span the previous token was in
    for s, e in offsets:
        hit = None
        for i, (ss, se, _) in enumerate(spans):
            if s < se and e > ss:
                hit = i
                break
        if hit is None:
            tags.append("O")
        elif hit == current:
            tags.append("I-" + spans[hit][2])
        else:
            tags.append("B-" + spans[hit][2])
        current = hit
    return tags


def bio_to_spans(offsets, tags):
    """Opposite of spans_to_bio: group tokens back into character spans."""
    spans = []
    cur = None
    for (s, e), tag in zip(offsets, tags):
        if s == e:                             # special tokens like [CLS] have (0, 0)
            continue
        if tag == "O":
            if cur:
                spans.append(tuple(cur))
                cur = None
            continue
        prefix, label = tag.split("-", 1)
        if prefix == "B" or cur is None or cur[2] != label:
            if cur:
                spans.append(tuple(cur))
            cur = [s, e, label]
        else:
            cur[1] = e
    if cur:
        spans.append(tuple(cur))
    return spans


def chunk_text(text, max_chars=400):
    """
    Splits long text into pieces (at whitespace) so the transformer never sees more than
    max_chars at once. Returns list of (offset, chunk). Quick and dirty, entities that
    sit exactly on a boundary might get cut - didn't have time to do overlapping windows.
    """
    chunks = []
    pos = 0
    while pos < len(text):
        end = min(pos + max_chars, len(text))
        if end < len(text):
            cut = text.rfind(" ", pos, end)
            cut2 = text.rfind("\n", pos, end)
            cut = max(cut, cut2)
            if cut > pos:
                end = cut
        chunks.append((pos, text[pos:end]))
        pos = end
    return chunks
