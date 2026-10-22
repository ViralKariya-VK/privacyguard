"""
PrivacyGuard demo app (Streamlit). Same detection code as app.py (Gradio), see src/ui_core.py.

Local:   streamlit run streamlit_app.py
Tabs:    Text, Image, Architecture
"""
import html
from io import BytesIO
from pathlib import Path

import streamlit as st
from PIL import Image

from src.ui_core import (BERT_ONLY, DEFAULT_MASK, HYBRID, MASK_CHOICES, REGEX, TEXT_EXAMPLES, analyse_image,
                         analyse_text, get_model, to_highlight)

st.set_page_config(page_title="PrivacyGuard", layout="wide")

# ------------------------------------------------------------------ look
# Paper background, ink text, one muted tint per PII type (soft fill + stronger underline, dark text).
TINTS = {
    "NAME": ("#f3ddd0", "#c2562b"), "EMAIL": ("#d9e6dc", "#2f6b4f"), "PHONE": ("#dce3ee", "#3d5a8a"),
    "ID_NUMBER": ("#f0d4d4", "#a32d2d"), "USERNAME": ("#e4daee", "#6b4c9a"), "PASSWORD": ("#f2d0dc", "#9c2f5b"),
    "IP_ADDRESS": ("#f1e3bd", "#a67b10"), "LOCATION": ("#d3e5e3", "#2b7a78"), "DATE": ("#e3e0d8", "#7a7566"),
    "DOB": ("#f2d9c7", "#b5651d"),
}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');
html, body, [class*="css"], .stApp {font-family: 'Inter', system-ui, sans-serif;}
.block-container {max-width: 1120px; padding-top: 3.2rem; padding-bottom: 3rem;}
header[data-testid="stHeader"] {background: transparent;}
h1, h2, h3 {font-family: 'Fraunces', Georgia, serif; font-weight: 600; letter-spacing: -0.01em;}
h3 {font-size: 1.25rem; margin-top: 1.6rem;}

.brand {display: flex; align-items: baseline; gap: 14px; margin-bottom: 2px;}
.brand .bar {display: inline-block; width: 38px; height: 14px; background: #1d1b18; border-radius: 2px; transform: translateY(1px);}
.brand .name {font-family: 'Fraunces', Georgia, serif; font-size: 2.3rem; font-weight: 600; letter-spacing: -0.02em; line-height: 1.1;}
.tagline {color: #5e5a52; font-size: 1.02rem; margin: 2px 0 18px 0; max-width: 46rem;}
.rule {border: 0; border-top: 1px solid #cfc8b8; margin: 0 0 18px 0;}

.label {font-family: 'JetBrains Mono', monospace; font-size: .72rem; letter-spacing: .08em; text-transform: uppercase; color: #6d6859; margin: 2px 0 6px 0;}
.stats {display: flex; gap: 28px; margin: 6px 0 2px 0;}
.stat .v {font-family: 'Fraunces', Georgia, serif; font-size: 1.45rem; font-weight: 600; line-height: 1.1;}
.stat .k {font-size: .74rem; color: #6d6859;}
.desc {color: #3b3832; font-size: .93rem; margin: 6px 0 4px 0; max-width: 34rem;}

.summary {display: flex; flex-wrap: wrap; gap: 8px 14px; align-items: center; margin: 4px 0 14px 0; font-size: .92rem;}
.summary b {font-weight: 600;}
.chip {border-left: 3px solid; padding: 1px 8px; border-radius: 2px; font-size: .8rem;}

.hl {line-height: 2.15; padding: 16px 18px; background: #fbf9f4; border: 1px solid #d8d1c1; border-radius: 6px;
     white-space: pre-wrap; word-break: break-word; font-size: .97rem;}
.tag {border-bottom: 2px solid; padding: 1px 3px; border-radius: 2px;}
.tag small {font-family: 'JetBrains Mono', monospace; font-size: .6rem; margin-left: 5px; opacity: .75; letter-spacing: .04em;}

.steps {display: flex; flex-wrap: wrap; gap: 0; margin: 8px 0 6px 0;}
.step {flex: 1 1 150px; border: 1px solid #cfc8b8; border-right: 0; padding: 12px 14px; background: #fbf9f4; min-width: 150px;}
.step:last-child {border-right: 1px solid #cfc8b8;}
.step .n {font-family: 'JetBrains Mono', monospace; font-size: .68rem; color: #8a8473;}
.step .t {font-weight: 600; font-size: .92rem; margin: 2px 0 3px 0;}
.step .d {font-size: .8rem; color: #5e5a52; line-height: 1.4;}
.step.dark {background: #1d1b18; border-color: #1d1b18;} .step.dark .n, .step.dark .d {color: #b9b3a3;} .step.dark .t {color: #f4f1ea;}

table.tbl {border-collapse: collapse; width: 100%; font-size: .86rem; margin: 6px 0 4px 0;}
table.tbl th {text-align: left; font-family: 'JetBrains Mono', monospace; font-size: .68rem; letter-spacing: .06em;
              text-transform: uppercase; color: #6d6859; border-bottom: 1.5px solid #1d1b18; padding: 6px 10px 6px 0;}
table.tbl td {border-bottom: 1px solid #d8d1c1; padding: 8px 10px 8px 0; vertical-align: top; color: #2b2925;}
table.tbl tr.hot td {background: #ece6d6;}
.pillr {display: inline-block; border: 1px solid #1d1b18; border-radius: 99px; padding: 0 8px; font-size: .7rem; margin-left: 6px;}

.card {border: 1px solid #d8d1c1; background: #fbf9f4; border-radius: 6px; padding: 14px 16px; height: 100%;}
.card .t {font-family: 'Fraunces', Georgia, serif; font-weight: 600; font-size: 1.05rem; margin-bottom: 6px;}
.card ul {margin: 0; padding-left: 1.1rem; font-size: .86rem; color: #2b2925; line-height: 1.55;}
.foot {color: #8a8473; font-size: .8rem; margin-top: 34px; border-top: 1px solid #cfc8b8; padding-top: 12px;}

.stButton > button, .stDownloadButton > button {border-radius: 5px; font-weight: 500;}
[data-testid="stButtonGroup"] > div {flex-wrap: wrap !important; overflow-x: visible !important; row-gap: 6px;}
button[data-baseweb="tab"] {font-family: 'Inter', sans-serif; font-weight: 500;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)
st.markdown(
    "<div class='brand'><span class='bar'></span><span class='name'>PrivacyGuard</span></div>"
    "<div class='tagline'>Find personal information in text and images, then cover it before the content is shared "
    "or pasted into an LLM.</div><hr class='rule'>", unsafe_allow_html=True)

# ------------------------------------------------------------------ methods (numbers: results/summary_*.csv)
METHODS = {
    "Hybrid": dict(key=HYBRID, f1="0.93", leak="98%", ms="19 ms",
                   desc="Fine-tuned DistilBERT plus a rules layer for fixed formats (emails, phones, IDs, passwords). "
                        "Best overall and the most robust on text it was not trained on. Recommended."),
    "Transformer": dict(key=BERT_ONLY, f1="0.94", leak="97%", ms="19 ms",
                        desc="The fine-tuned DistilBERT on its own, without the rules layer. Strong on text like its "
                             "training data, weaker on real-world formats."),
    "Regex + spaCy": dict(key=REGEX, f1="0.36", leak="60%", ms="10 ms",
                          desc="Hand-written patterns plus spaCy's pretrained entity recogniser. No training. "
                               "Kept as a classic baseline to compare against."),
}


@st.cache_resource(show_spinner="Loading the model (first run only, about a minute) ...")
def load_model(choice):
    return get_model(choice)


def tag_html(piece, label):
    bg, border = TINTS[label]
    return (f"<span class='tag' style='background:{bg};border-color:{border}'>{html.escape(piece)}"
            f"<small>{label}</small></span>")


def highlight_html(text, spans):
    body = "".join(html.escape(p) if lab is None else tag_html(p, lab) for p, lab in to_highlight(text, spans))
    return f"<div class='hl'>{body}</div>"


def summary_html(spans, n_masked):
    if not spans:
        return "<div class='summary'><b>No PII found.</b></div>"
    counts = {}
    for _, _, lab in spans:
        counts[lab] = counts.get(lab, 0) + 1
    chips = "".join(f"<span class='chip' style='background:{TINTS[l][0]};border-color:{TINTS[l][1]}'>{n} {l}</span>"
                    for l, n in sorted(counts.items(), key=lambda kv: -kv[1]))
    return f"<div class='summary'><span><b>{len(spans)}</b> found, <b>{n_masked}</b> masked</span>{chips}</div>"


# ------------------------------------------------------------------ configuration (shared by Text and Image)
with st.container(border=True):
    c_method, c_mask = st.columns([1, 1.15], gap="large")
    with c_method:
        st.markdown("<div class='label'>Detection method</div>", unsafe_allow_html=True)
        method_name = st.segmented_control("Detection method", list(METHODS), default="Hybrid",
                                           label_visibility="collapsed") or "Hybrid"
        m = METHODS[method_name]
        st.markdown(f"<div class='desc'>{m['desc']}</div>"
                    f"<div class='stats'><div class='stat'><div class='v'>{m['f1']}</div><div class='k'>F1, 1000 test docs</div></div>"
                    f"<div class='stat'><div class='v'>{m['leak']}</div><div class='k'>PII characters covered</div></div>"
                    f"<div class='stat'><div class='v'>{m['ms']}</div><div class='k'>per document</div></div></div>",
                    unsafe_allow_html=True)
    with c_mask:
        st.markdown("<div class='label'>Mask these types</div>", unsafe_allow_html=True)
        mask = st.pills("Mask these types", MASK_CHOICES, selection_mode="multi", default=DEFAULT_MASK,
                        label_visibility="collapsed") or []
        st.caption("Every type is always detected and highlighted. This choice only decides what gets covered. "
                   "Dates and locations start unselected because on a resume or invoice they are usually not private.")
choice = m["key"]

tab_text, tab_image, tab_arch = st.tabs(["Text", "Image", "Architecture"])

# ------------------------------------------------------------------ text tab
with tab_text:
    st.markdown("<div class='label'>Examples</div>", unsafe_allow_html=True)
    names = ["Contact message", "Server log", "Shipping address"]
    cols = st.columns(len(TEXT_EXAMPLES) + 2)
    for i, ex in enumerate(TEXT_EXAMPLES):
        if cols[i].button(names[i], key=f"ex{i}", use_container_width=True):
            st.session_state["text_in"] = ex
    text = st.text_area("Input text", height=180, key="text_in", label_visibility="collapsed",
                        placeholder="Paste an email, chat log, ticket or document here.")
    if st.button("Analyse text", type="primary", key="go_text"):
        if not text.strip():
            st.warning("Paste some text first.")
        else:
            try:
                load_model(choice)
                with st.spinner("Analysing"):
                    spans, masked, _ = analyse_text(text, choice, mask)
            except ValueError as err:
                st.error(str(err))
            else:
                st.markdown(summary_html(spans, sum(sp[2] in mask for sp in spans)), unsafe_allow_html=True)
                left, right = st.columns(2, gap="large")
                with left:
                    st.markdown("<div class='label'>Detected</div>", unsafe_allow_html=True)
                    st.markdown(highlight_html(text, spans), unsafe_allow_html=True)
                with right:
                    st.markdown("<div class='label'>Redacted output</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='hl'>{html.escape(masked)}</div>", unsafe_allow_html=True)
                    st.download_button("Download redacted text", masked, file_name="redacted.txt")

# ------------------------------------------------------------------ image tab
with tab_image:
    st.caption("Upload a screenshot or a photo of a document. The text is read with OCR, PII is detected in it and then "
               "covered in the image. OCR mistakes can cause missed PII, so always check the result.")
    style = st.segmented_control("How to cover PII", ["Black box", "Label tag"], default="Black box",
                                 help="Label tag writes the type, for example [NAME], over the covered area.") or "Black box"
    up = st.file_uploader("Image", type=["png", "jpg", "jpeg", "webp"], label_visibility="collapsed")
    sample = Path("data/samples/sample_screenshot.png")
    if sample.exists() and st.button("Use the sample screenshot"):
        st.session_state["use_sample"] = True
    image = None
    if up is not None:
        st.session_state["use_sample"] = False
        image = Image.open(up)
    elif st.session_state.get("use_sample") and sample.exists():
        image = Image.open(sample)
    if image is not None:
        image = image.convert("RGB")
        st.image(image, caption="Input", width="stretch")
    if st.button("Analyse image", type="primary", key="go_img", disabled=image is None):
        try:
            load_model(choice)
            with st.spinner("Reading the image and analysing"):
                res = analyse_image(image, choice, style == "Label tag", mask)
        except ValueError as err:
            st.error(str(err))
        else:
            if not res["text"].strip():
                st.info("No text was found in the image.")
            else:
                st.markdown(summary_html(res["spans"], len(res["masked_spans"])), unsafe_allow_html=True)
                a, b = st.columns(2, gap="large")
                a.image(res["redacted_image"], caption="Redacted", width="stretch")
                b.image(res["detected_image"], caption="Everything that was detected", width="stretch")
                buf = BytesIO()
                res["redacted_image"].save(buf, format="PNG")
                st.download_button("Download redacted image", buf.getvalue(), file_name="redacted.png", mime="image/png")
                with st.expander("Text read from the image"):
                    c1, c2 = st.columns(2)
                    c1.text_area("OCR text", res["text"], height=200)
                    c2.text_area("Redacted text", res["masked_text"], height=200)

# ------------------------------------------------------------------ architecture tab
with tab_arch:
    st.markdown("### How it works")
    st.markdown(
        "<div class='steps'>"
        "<div class='step'><div class='n'>01</div><div class='t'>Input</div><div class='d'>Pasted text, or an image of a document.</div></div>"
        "<div class='step'><div class='n'>02</div><div class='t'>OCR</div><div class='d'>Images only. RapidOCR returns the text and the position of every character.</div></div>"
        "<div class='step dark'><div class='n'>03</div><div class='t'>Detector</div><div class='d'>The chosen method returns spans: start, end and PII type.</div></div>"
        "<div class='step dark'><div class='n'>04</div><div class='t'>Rules layer</div><div class='d'>Fixes known weak spots: formats, context words, word boundaries.</div></div>"
        "<div class='step'><div class='n'>05</div><div class='t'>Cover</div><div class='d'>Text gets [TYPE] tags. Images get a box or label at the stored positions.</div></div>"
        "</div>", unsafe_allow_html=True)
    st.caption("Every method returns the same span list, so all of them are scored with the same evaluation code. "
               "Detection always reports every type; the mask selection is applied only in the last step.")

    st.markdown("### Models")
    st.markdown("""
<table class='tbl'>
<tr><th>Method</th><th>Type</th><th>How it works</th><th>Status</th></tr>
<tr class='hot'><td><b>Hybrid</b></td><td>Transformer + rules</td><td>DistilBERT token classifier fine-tuned on PII, then a rules layer for emails, phones, IDs, passwords, usernames, addresses and names.</td><td>Deployed<span class='pillr'>default</span></td></tr>
<tr><td><b>Transformer</b></td><td>Transformer</td><td>The same fine-tuned DistilBERT without the rules layer.</td><td>Deployed</td></tr>
<tr><td><b>Regex + spaCy</b></td><td>Classic NLP</td><td>Regular expressions for fixed formats and spaCy's pretrained named-entity model for names, places and dates. No training.</td><td>Deployed</td></tr>
<tr><td>Token features + SVM</td><td>Classic ML</td><td>Hand-made features per token (word shape, suffixes, neighbours, 199k sparse features) and a LinearSVC that predicts the tag.</td><td>Experiment</td></tr>
<tr><td>Presidio</td><td>Industry baseline</td><td>Microsoft's open-source PII analyser with its default recognisers.</td><td>Experiment</td></tr>
<tr><td>Qwen2.5-0.5B + LoRA</td><td>LLM</td><td>A small instruction-tuned LLM fine-tuned to answer with a JSON list of PII. Too slow for a free CPU host.</td><td>Experiment</td></tr>
</table>""", unsafe_allow_html=True)

    st.markdown("### Fine-tuning, in short")
    c1, c2, c3 = st.columns(3, gap="medium")
    c1.markdown("""<div class='card'><div class='t'>Data</div><ul>
<li>ai4privacy pii-masking-300k, English</li>
<li>27 fine labels grouped into 9 types, birth dates added by a rule</li>
<li>Labels converted to BIO tags (begin, inside, outside)</li>
<li>10,000 training documents, 1,000 held-out test documents</li>
<li>Plus 11 hand-written out-of-domain texts and a 66-case test suite written for this project</li></ul></div>""", unsafe_allow_html=True)
    c2.markdown("""<div class='card'><div class='t'>DistilBERT</div><ul>
<li>distilbert-base-cased with a token-classification head</li>
<li>3 epochs, learning rate 5e-5, batch size 16, weight decay 0.01</li>
<li>256 tokens per chunk, long texts are split</li>
<li>About 22 minutes on an Apple-silicon GPU</li>
<li>Validation F1 0.934</li></ul></div>""", unsafe_allow_html=True)
    c3.markdown("""<div class='card'><div class='t'>Rules layer</div><ul>
<li>Added after testing by hand and with the suite</li>
<li>Patterns for emails, phones, Aadhaar, PAN and SSN</li>
<li>Context words: password, user, login, "this is", "regards"</li>
<li>Snaps spans to whole words, filters tool names and order numbers</li>
<li>Each rule was checked against the 1000-document benchmark so it did not hurt accuracy</li></ul></div>""", unsafe_allow_html=True)
    st.caption("LoRA experiment: Qwen2.5-0.5B-Instruct, rank 16, alpha 32, attention projections only, 2,000 examples, one epoch, "
               "learning rate 2e-4. It reached F1 0.71 at about 3 seconds per document, so it was not deployed.")

    st.markdown("### Results")
    st.markdown("""
<table class='tbl'>
<tr><th>Method</th><th>F1 (1000 test docs)</th><th>PII characters covered</th><th>F1 (out-of-domain)</th><th>Speed per doc</th></tr>
<tr class='hot'><td><b>Hybrid</b></td><td>0.93</td><td>98%</td><td>0.80</td><td>19 ms</td></tr>
<tr><td>Transformer</td><td>0.94</td><td>97%</td><td>0.56</td><td>19 ms</td></tr>
<tr><td>Token features + SVM</td><td>0.71</td><td>95%</td><td>0.25</td><td>1 ms</td></tr>
<tr><td>Qwen2.5-0.5B + LoRA</td><td>0.71</td><td>80%</td><td>0.70</td><td>3,078 ms</td></tr>
<tr><td>Regex + spaCy</td><td>0.36</td><td>60%</td><td>0.68</td><td>10 ms</td></tr>
<tr><td>Presidio</td><td>0.35</td><td>53%</td><td>0.57</td><td>12 ms</td></tr>
</table>""", unsafe_allow_html=True)
    st.caption("F1 is the strict span score: the boundaries and the type must both match. The test set is synthetic, which is why "
               "the transformer looks better there than it is on the small out-of-domain set. The rules layer is what makes "
               "the hybrid hold up on text it was not trained on.")

    st.markdown("### Limits")
    st.markdown("- Trained on synthetic English data; real documents are harder.\n"
                "- Indian-style addresses and names inside long lists are sometimes missed or only partly covered.\n"
                "- OCR errors on small or low-contrast text can hide PII from the detector.\n"
                "- Not a replacement for a real compliance tool. Always review the output before sharing.")

st.markdown("<div class='foot'>NLP lab project. Your input is processed on the server for the current request only and is not stored by the app.</div>",
            unsafe_allow_html=True)
