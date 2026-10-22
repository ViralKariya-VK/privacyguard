"""
Hide PII inside the image.

Input : the OCR characters (with boxes) + the PII spans found in the OCR text.
Output: a copy of the image where the PII regions are covered
        - mode "mask": solid black box
        - mode "tag" : coloured box with the label written on it, e.g. [NAME]

The box of a character is estimated by cutting the OCR word box into equal slices
(exact for monospace fonts, a bit off for normal fonts), so we add some padding.
"""
from PIL import Image, ImageDraw, ImageFont

TAG_COLORS = {
    "NAME": "#2563eb", "EMAIL": "#16a34a", "PHONE": "#ea580c", "ID_NUMBER": "#dc2626",
    "USERNAME": "#7c3aed", "PASSWORD": "#be123c", "IP_ADDRESS": "#0891b2",
    "LOCATION": "#ca8a04", "DATE": "#475569", "DOB": "#db2777",
}


def span_boxes(chars, spans):
    """For every span return list of (label, (x0,y0,x1,y1)) with one merged box per text line."""
    result = []
    for s, e, label in spans:
        per_line = {}
        for c in chars[s:e]:
            if c is None:
                continue
            x0, y0, x1, y1, line = c
            if line in per_line:
                ox0, oy0, ox1, oy1 = per_line[line]
                per_line[line] = (min(ox0, x0), min(oy0, y0), max(ox1, x1), max(oy1, y1))
            else:
                per_line[line] = (x0, y0, x1, y1)
        for box in per_line.values():
            result.append((label, box))
    return result


def _fit_font(draw, text, max_w, max_h):
    size = max(8, int(max_h * 0.7))
    while size >= 8:
        font = ImageFont.load_default(size=size)
        l, t, r, b = draw.textbbox((0, 0), text, font=font)
        if r - l <= max_w and b - t <= max_h:
            return font
        size -= 1
    return None          # does not fit -> draw the box without text


def redact_image(pil_image, chars, spans, mode="mask", pad=3):
    # box positions inside a word are estimates, so pad more sideways (proportional fonts)
    img = pil_image.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    for label, (x0, y0, x1, y1) in span_boxes(chars, spans):
        pad_x = max(pad, 0.3 * (y1 - y0))
        box = (x0 - pad_x, y0 - pad, x1 + pad_x, y1 + pad)
        if mode == "tag":
            draw.rectangle(box, fill=TAG_COLORS.get(label, "#000000"))
            tag = f"[{label}]"
            font = _fit_font(draw, tag, box[2] - box[0], box[3] - box[1])
            if font:
                draw.text(((box[0] + box[2]) / 2, (box[1] + box[3]) / 2), tag,
                          fill="white", font=font, anchor="mm")
        else:
            draw.rectangle(box, fill="black")
    return img


def draw_detections(pil_image, chars, spans):
    """just outlines + label above the box, to SEE what was detected (original pixels stay visible)"""
    img = pil_image.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=14)
    for label, (x0, y0, x1, y1) in span_boxes(chars, spans):
        color = TAG_COLORS.get(label, "#000000")
        draw.rectangle((x0 - 2, y0 - 2, x1 + 2, y1 + 2), outline=color, width=3)
        draw.text((x0, max(0, y0 - 16)), label, fill=color, font=font)
    return img
