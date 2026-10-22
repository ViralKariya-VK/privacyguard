"""image -> OCR -> PII detection -> text outputs + redacted image. Used by the app and by eval_ocr.py"""
from src.data import merge_adjacent
from src.image_redact import draw_detections, redact_image
from src.ocr import read_image
from src.postprocess import mark_birth_dates
from src.redact import mask_text


def process_image(pil_image, predictor, mode="mask", entities=None):
    """entities: set of labels to mask, None = mask everything. Detection always reports everything."""
    ocr = read_image(pil_image)
    text = ocr["text"]
    spans = merge_adjacent(predictor.predict(text), text) if text.strip() else []
    spans = mark_birth_dates(text, spans)
    to_mask = spans if entities is None else [sp for sp in spans if sp[2] in entities]
    return {
        "text": text,
        "spans": spans,
        "masked_spans": to_mask,
        "masked_text": mask_text(text, to_mask),
        "detected_image": draw_detections(pil_image, ocr["chars"], spans),
        "redacted_image": redact_image(pil_image, ocr["chars"], to_mask, mode=mode),
    }
