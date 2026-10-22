"""Loads the fine-tuned token classifier and predicts spans for raw text."""
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from src.data import bio_to_spans, chunk_text
from src.device import get_device
from src.postprocess import lowercase_lines, refine_spans, uppercase_lines


class TransformerPredictor:
    def __init__(self, model_path, refine=False):
        self.refine = refine                 # True = hybrid model (transformer + rules)
        # model_path can be a local folder or a HF hub id like "yourname/privacyguard-distilbert"
        self.device = get_device()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, text):
        spans = self._predict_raw(text)
        if self.refine:
            # headings in ALL CAPS: run again on the title-cased line and keep only NAME spans
            for offset, line in uppercase_lines(text):
                found = [sp for sp in self._predict_raw(line) if sp[2] == "NAME"]
                if found:
                    # the model often tags only the first word of "Vanshika Ahuja", short heading
                    # that contains a name -> take the whole line as the name
                    end = offset + len(line.rstrip())
                    spans = [sp for sp in spans if not (sp[0] < end and offset < sp[1])]
                    spans.append((offset, end, "NAME"))
            # lower-case lines: same idea, but only keep NAME spans that cover 2+ words (fewer false alarms)
            for offset, line in lowercase_lines(text):
                for s, e, lab in self._predict_raw(line):
                    if lab == "NAME" and len(line[s:e].split()) >= 2 and not any(
                            sp[0] < offset + e and offset + s < sp[1] for sp in spans):
                        spans.append((offset + s, offset + e, "NAME"))
            spans = refine_spans(text, spans)
        return spans

    def _predict_raw(self, text):
        spans = []
        for offset, chunk in chunk_text(text):
            enc = self.tokenizer(chunk, truncation=True, max_length=256,
                                 return_offsets_mapping=True, return_tensors="pt")
            offsets = enc.pop("offset_mapping")[0].tolist()
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.no_grad():
                logits = self.model(**enc).logits[0]
            tag_ids = logits.argmax(-1).tolist()
            tags = [self.model.config.id2label[i] for i in tag_ids]
            for s, e, label in bio_to_spans(offsets, tags):
                spans.append((s + offset, e + offset, label))
        return spans
