"""Turns spans into masked text. Used by the app and for demos."""


def mask_text(text, spans):
    out = []
    pos = 0
    for s, e, label in sorted(spans):
        if s < pos:          # overlapping span, ignore
            continue
        out.append(text[pos:s])
        out.append(f"[{label}]")
        pos = e
    out.append(text[pos:])
    return "".join(out)
