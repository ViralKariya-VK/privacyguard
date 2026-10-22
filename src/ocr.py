"""
OCR step for images: picture -> text + the position of every character.

Uses RapidOCR (free, pip install, runs on CPU with onnxruntime, models are bundled).
Why we need positions: the PII model works on the text, but to hide something INSIDE the
image we must know where on the picture each character of the text is.

Result: {"text": str, "chars": [None or (x0, y0, x1, y1, line_no) for every character of text]}
(spaces and newlines have None)
"""
import numpy as np
from PIL import Image

_engine = None   # load once, creating the engine is slow


def get_engine():
    global _engine
    if _engine is None:
        from rapidocr import RapidOCR
        _engine = RapidOCR()
    return _engine


def _word_box(poly, scale):
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return (min(xs) / scale, min(ys) / scale, max(xs) / scale, max(ys) / scale)


def read_image(pil_image, upscale=None):
    pil_image = pil_image.convert("RGB")

    # Found with a real resume screenshot: at normal size the OCR read the (small, blue) header
    # line as "Va-  ika  ik  ika" so the name/phone/email were never seen by the PII model.
    # The same page upscaled 1.5-2x was read perfectly. So we enlarge not-so-big images first
    # and divide the box coordinates by the factor afterwards.
    if upscale is None:
        upscale = 2.0 if max(pil_image.size) < 2200 else 1.0
    if upscale != 1.0:
        big = pil_image.resize((int(pil_image.width * upscale), int(pil_image.height * upscale)),
                               Image.LANCZOS)
    else:
        big = pil_image

    res = get_engine()(np.array(big)[:, :, ::-1], return_word_box=True)   # RGB -> BGR (opencv style)
    if res.txts is None or len(res.txts) == 0:
        return {"text": "", "chars": []}

    text = ""
    chars = []
    for line_no, line_text in enumerate(res.txts):
        line_words = res.word_results[line_no] if res.word_results else None
        if not line_words:
            line_words = [(line_text, 1.0, res.boxes[line_no])]

        # Box of every non-space character of the line: the OCR word boxes are cut into equal
        # slices per character (fine for most fonts, small errors for narrow letters like 'i').
        slices = []
        for w_text, _score, poly in line_words:
            x0, y0, x1, y1 = _word_box(poly, upscale)
            n = max(1, len(w_text))
            for k in range(len(w_text)):
                slices.append((x0 + k / n * (x1 - x0), y0, x0 + (k + 1) / n * (x1 - x0), y1))

        # Use the OCR *line* text for the text itself. The word list is sometimes split in the
        # middle of words when the letters are far apart (VANSHIKA -> "VA NSHIKA").
        # If the characters don't line up with the words we fall back to joining the words.
        stripped = "".join(line_text.split())
        if len(stripped) != len(slices):
            line_text = " ".join(w[0] for w in line_words)
            slices = []
            for w_text, _score, poly in line_words:
                x0, y0, x1, y1 = _word_box(poly, upscale)
                n = max(1, len(w_text))
                for k in range(len(w_text)):
                    slices.append((x0 + k / n * (x1 - x0), y0, x0 + (k + 1) / n * (x1 - x0), y1))

        if line_no > 0:
            text += "\n"
            chars.append(None)
        k = 0
        for ch in line_text:
            text += ch
            if ch.isspace():
                chars.append(None)
            else:
                x0, y0, x1, y1 = slices[k]
                chars.append((x0, y0, x1, y1, line_no))
                k += 1
    return {"text": text, "chars": chars}
