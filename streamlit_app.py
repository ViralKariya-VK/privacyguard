"""
PrivacyGuard demo app (Streamlit). Same detection code as app.py (Gradio), see src/ui_core.py.

Local:   streamlit run streamlit_app.py
Tabs:    1) paste text   2) upload an image (screenshot, photo of a document ...)
"""
import html
from io import BytesIO
from pathlib import Path

import streamlit as st
from PIL import Image

from src.ui_core import (COLORS, DEFAULT_MASK, HYBRID, MASK_CHOICES, MODEL_CHOICES, TEXT_EXAMPLES, analyse_image,
                         analyse_text, get_model, to_highlight)

st.set_page_config(page_title="PrivacyGuard", page_icon="🛡️", layout="wide")

CSS = """
<style>
.block-container {max-width: 1180px; padding-top: 1.6rem;}
.hero {background: linear-gradient(135deg,#4f46e5 0%,#2563eb 60%,#0ea5e9 100%); color: white;
       padding: 26px 30px; border-radius: 16px; margin-bottom: 14px;}
.hero h1 {color: white; margin: 0 0 6px 0; font-size: 2rem; padding: 0;}
.hero p {color: #e0e7ff; margin: 0; font-size: 1.02rem;}
.pill {display: inline-block; background: rgba(255,255,255,.18); border-radius: 999px; color: white;
       padding: 3px 12px; margin: 12px 6px 0 0; font-size: .82rem;}
.hl {line-height: 2.1; padding: 12px 14px; border: 1px solid rgba(128,128,128,.3); border-radius: 10px;
     white-space: pre-wrap; word-break: break-word;}
.tag {border-radius: 6px; padding: 2px 6px; color: white; font-weight: 500;}
.tag small {font-size: .62rem; opacity: .85; margin-left: 4px; font-weight: 700;}
.legend span {display: inline-block; border-radius: 6px; padding: 1px 8px; margin: 2px 4px 2px 0; color: white; font-size: .75rem;}
.info {font-weight: 600; color: #6366f1; margin: 4px 0 8px 0;}
.foot {text-align: center; color: #94a3b8; font-size: .85rem; margin-top: 28px;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)
st.markdown(
    "<div class='hero'><h1>🛡️ PrivacyGuard</h1>"
    "<p>Find and hide personal information in <b>text</b> or <b>images</b> before you share it or paste it into an LLM.</p>"
    "<span class='pill'>Fine-tuned DistilBERT + rules</span><span class='pill'>OCR for screenshots</span>"
    "<span class='pill'>10 PII types</span><span class='pill'>Runs on CPU</span></div>", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading the model (first time only, about a minute) ...")
def load_model(choice):
    return get_model(choice)


def highlight_html(text, spans):
    parts = []
    for piece, label in to_highlight(text, spans):
        if label is None:
            parts.append(html.escape(piece))
        else:
            parts.append(f"<span class='tag' style='background:{COLORS[label]}'>{html.escape(piece)}"
                         f"<small>{label}</small></span>")
    return "<div class='hl'>" + "".join(parts) + "</div>"


def legend_html():
    return "<div class='legend'>" + "".join(
        f"<span style='background:{c}'>{lab}</span>" for lab, c in COLORS.items()) + "</div>"


# ------------------------------------------------------------------ sidebar: shared settings
with st.sidebar:
    st.header("Settings")
    choice = st.radio("Model", MODEL_CHOICES, index=0)
    st.markdown("**What to mask**")
    st.caption("Everything is always detected and highlighted. These boxes only decide what gets masked.")
    mask = [t for t in MASK_CHOICES if st.checkbox(t, value=t in DEFAULT_MASK, key=f"mask_{t}")]
    st.caption("DATE and LOCATION are off by default: on a resume or invoice they are usually not private.")

tab_text, tab_image, tab_about = st.tabs(["📝 Text", "🖼️ Image", "ℹ️ About"])

# ------------------------------------------------------------------ text tab
with tab_text:
    st.caption("Try an example:")
    cols = st.columns(len(TEXT_EXAMPLES))
    for i, (col, ex) in enumerate(zip(cols, TEXT_EXAMPLES)):
        if col.button(ex[:42] + " ...", key=f"ex{i}", width="stretch"):
            st.session_state["text_in"] = ex
    text = st.text_area("Input text", height=200, key="text_in",
                        placeholder="Paste an email, chat log, ticket or document here ...")
    if st.button("🔍 Find PII", type="primary", key="go_text"):
        if not text.strip():
            st.warning("Nothing to analyse.")
        else:
            try:
                load_model(choice)
                with st.spinner("Analysing ..."):
                    spans, masked, info = analyse_text(text, choice, mask)
            except ValueError as err:
                st.error(str(err))
            else:
                st.markdown(f"<div class='info'>{info}</div>", unsafe_allow_html=True)
                left, right = st.columns(2)
                with left:
                    st.markdown("**Detected PII**")
                    st.markdown(legend_html(), unsafe_allow_html=True)
                    st.markdown(highlight_html(text, spans), unsafe_allow_html=True)
                with right:
                    st.markdown("**Redacted text**")
                    st.text_area("Redacted text", masked, height=200, label_visibility="collapsed")
                    st.download_button("⬇️ Download redacted text", masked, file_name="redacted.txt")

# ------------------------------------------------------------------ image tab
with tab_image:
    st.caption("Upload a screenshot or photo of a document. Text is read with OCR, PII is detected in that text and then "
               "covered **in the image**. OCR mistakes can cause missed PII, always check the result.")
    style = st.radio("How to hide PII in the image", ["Black box (mask)", "Tag with label, e.g. [NAME]"], horizontal=True)
    up = st.file_uploader("Input image", type=["png", "jpg", "jpeg", "webp"])
    sample = Path("data/samples/sample_screenshot.png")
    use_sample = sample.exists() and st.button("Use the sample screenshot")
    if use_sample:
        st.session_state["use_sample"] = True
    image = None
    if up is not None:
        st.session_state["use_sample"] = False
        image = Image.open(up)
    elif st.session_state.get("use_sample") and sample.exists():
        image = Image.open(sample)
    if image is not None:
        image = image.convert("RGB")
        st.image(image, caption="Input image", width="stretch")
    if st.button("🔍 Find and hide PII", type="primary", key="go_img", disabled=image is None):
        try:
            load_model(choice)
            with st.spinner("Reading the image (OCR) and analysing ..."):
                res = analyse_image(image, choice, style.startswith("Tag"), mask)
        except ValueError as err:
            st.error(str(err))
        else:
            st.markdown(f"<div class='info'>{res['summary']}</div>", unsafe_allow_html=True)
            if res["text"].strip():
                a, b = st.columns(2)
                a.image(res["redacted_image"], caption="Redacted image", width="stretch")
                b.image(res["detected_image"], caption="What was detected", width="stretch")
                buf = BytesIO()
                res["redacted_image"].save(buf, format="PNG")
                st.download_button("⬇️ Download redacted image", buf.getvalue(), file_name="redacted.png", mime="image/png")
                with st.expander("Text read from the image"):
                    c1, c2 = st.columns(2)
                    c1.text_area("OCR text", res["text"], height=200)
                    c2.text_area("Redacted text", res["masked_text"], height=200)

# ------------------------------------------------------------------ about tab
with tab_about:
    st.markdown("""
**How it works.** A DistilBERT token classifier (fine-tuned on the ai4privacy dataset) tags PII in the text.
A rules layer fixes known weak spots (emails, phones, ID numbers, passwords, usernames, addresses, names after
"this is / regards", ALL-CAPS and lower-case names). For images, RapidOCR reads the text and its position, so the PII
can be covered in the picture.

**Benchmark (1000 held-out docs).** Hybrid model: strict F1 0.93, leak recall 0.98. Regex + spaCy and Presidio: F1 about 0.35.
On a small hand-written out-of-domain set the F1 is about 0.80, so real-world texts are harder.

**Limits.** Synthetic training data, English only, Indian-style addresses and names in the middle of lists are sometimes missed,
no overlap between long-text chunks. Not a replacement for a real compliance tool.""")

st.markdown("<div class='foot'>NLP lab project demo · English only · AI can miss PII, always review the output before sharing."
            "<br>Your input is processed on the server for this request only and is not stored by the app.</div>",
            unsafe_allow_html=True)
