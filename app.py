import streamlit as st
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import time

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="NLP as Digital Veda",
    page_icon="🕊️",
    layout="wide"
)

# ── Constants ─────────────────────────────────────────────────
M1_REPO   = "kb1084/ahimsa-scorer"
M2_REPO   = "kb1084/multilingual-aligner"
THRESHOLD = 0.7
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"

# ── Load models ───────────────────────────────────────────────
@st.cache_resource
def load_ahimsa_model():
    tokenizer = AutoTokenizer.from_pretrained(M1_REPO)
    model     = AutoModelForSequenceClassification.from_pretrained(M1_REPO)
    model.eval()
    return tokenizer, model

@st.cache_resource
def load_multilingual_model():
    return SentenceTransformer(M2_REPO)

# ── Helper functions ──────────────────────────────────────────
def get_ahimsa_score(text, tokenizer, model):
    inputs = tokenizer(
        text, return_tensors="pt",
        truncation=True, max_length=128
    )
    with torch.no_grad():
        logits = model(**inputs).logits
    probs   = torch.softmax(logits, dim=1)[0]
    p_toxic = probs[1].item()
    return round(1 - p_toxic, 4), round(p_toxic, 4)

def get_similarity(text1, text2, sbert_model):
    emb1 = sbert_model.encode([text1])
    emb2 = sbert_model.encode([text2])
    return float(cosine_similarity(emb1, emb2)[0][0])

def score_color(score):
    if score >= 0.8:
        return "🟢"
    elif score >= 0.7:
        return "🟡"
    else:
        return "🔴"

# ── Header ────────────────────────────────────────────────────
st.title("🕊️ NLP as Digital Veda")
st.markdown(
    "**Ethical AI Pipeline** — Ahimsa Scoring + Multilingual Semantic Alignment"
)
st.markdown(
    "> *Inspired by the Digital Veda framework | "
    "Implemented using Borkan et al. (2023) + Reimers & Gurevych (2020)*"
)
st.divider()

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    threshold = st.slider(
        "Ahimsa Filter Threshold",
        min_value=0.0, max_value=1.0,
        value=THRESHOLD, step=0.05
    )
    st.markdown("---")
    st.markdown("### 📚 Papers")
    st.markdown("- [Borkan et al. 2023](https://arxiv.org/abs/2301.11125)")
    st.markdown("- [Reimers & Gurevych 2020](https://arxiv.org/abs/2004.09813)")
    st.markdown("---")
    st.markdown("### 🤗 Models")
    st.markdown(f"- [{M1_REPO}](https://huggingface.co/{M1_REPO})")
    st.markdown(f"- [{M2_REPO}](https://huggingface.co/{M2_REPO})")

# ── Tabs ──────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🕊️ Ahimsa Score",
    "⚖️ Compare Models",
    "✍️ Rewrite Text",
    "🌐 Multilingual Checker"
])

# ═══════════════════════════════════════════════════════════════
# TAB 1 — Ahimsa Score
# ═══════════════════════════════════════════════════════════════
with tab1:
    st.header("🕊️ Ahimsa Ethical Score")
    st.markdown(
        "Enter any text to compute its **Ahimsa score**. "
        "Formula: `Score = 1 - P(toxic)`"
    )

    user_input = st.text_area(
        "Enter text here:",
        placeholder="Type something...",
        height=120
    )

    if st.button("Compute Ahimsa Score", type="primary"):
        if user_input.strip():
            with st.spinner("Loading model..."):
                tokenizer, model = load_ahimsa_model()
            with st.spinner("Scoring..."):
                score, p_toxic = get_ahimsa_score(
                    user_input, tokenizer, model
                )

            st.divider()
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    label="🕊️ Ahimsa Score",
                    value=f"{score:.4f}",
                    delta="Safe" if score >= threshold else "Harmful"
                )
            with col2:
                st.metric(
                    label="☠️ Toxicity Probability",
                    value=f"{p_toxic:.4f}"
                )
            with col3:
                verdict = "✅ ALLOWED" if score >= threshold else "❌ BLOCKED"
                st.metric(label="Filter Result", value=verdict)

            st.progress(score, text=f"Ahimsa Score: {score:.4f}")

            if score < threshold:
                st.error(
                    f"⚠️ Blocked — score {score:.4f} "
                    f"below threshold {threshold}"
                )
            else:
                st.success(
                    f"✅ Allowed — score {score:.4f} "
                    f"above threshold {threshold}"
                )
        else:
            st.warning("Please enter some text first.")

# ═══════════════════════════════════════════════════════════════
# TAB 2 — Compare Models
# ═══════════════════════════════════════════════════════════════
with tab2:
    st.header("⚖️ Compare Models Side by Side")
    st.markdown(
        "See how **DistilBERT, BERT-base, and RoBERTa** "
        "score the same text differently."
    )

    compare_input = st.text_area(
        "Enter text to compare:",
        placeholder="Type something...",
        height=120,
        key="compare_input"
    )

    if st.button("Compare All Models", type="primary"):
        if compare_input.strip():
            compare_models = [
                ("DistilBERT", "distilbert-base-uncased"),
                ("BERT-base",  "bert-base-uncased"),
                ("RoBERTa",    "roberta-base"),
            ]
            cols = st.columns(3)

            for i, (display, model_name) in enumerate(compare_models):
                with cols[i]:
                    st.subheader(f"🤖 {display}")
                    with st.spinner(f"Loading {display}..."):
                        tok = AutoTokenizer.from_pretrained(model_name)
                        mdl = AutoModelForSequenceClassification.from_pretrained(
                            model_name, num_labels=2
                        )
                        mdl.eval()

                    start          = time.time()
                    score, p_toxic = get_ahimsa_score(compare_input, tok, mdl)
                    elapsed        = (time.time() - start) * 1000

                    st.metric("🕊️ Ahimsa Score", f"{score:.4f}")
                    st.metric("☠️ P(toxic)",      f"{p_toxic:.4f}")
                    st.metric("⚡ Speed",          f"{elapsed:.1f} ms")
                    st.markdown(f"**Verdict:** {score_color(score)}")
                    st.progress(score)

                    del mdl

            st.info(
                "📌 Borkan et al. (2023) compared these exact models "
                "on Civil Comments. DistilBERT offers best "
                "speed-accuracy tradeoff."
            )
        else:
            st.warning("Please enter some text first.")

# ═══════════════════════════════════════════════════════════════
# TAB 3 — Rewrite Text
# ═══════════════════════════════════════════════════════════════
with tab3:
    st.header("✍️ Ethical Text Rewriting")
    st.markdown(
        "Harmful text is **rewritten** into constructive language. "
        "*(Digital Veda — Ahimsa principle)*"
    )

    rewrite_input = st.text_area(
        "Enter harmful text to rewrite:",
        placeholder="e.g. You are useless.",
        height=120,
        key="rewrite_input"
    )

    if st.button("Rewrite Text", type="primary"):
        if rewrite_input.strip():
            with st.spinner("Loading scorer..."):
                tokenizer, model = load_ahimsa_model()

            orig_score, _ = get_ahimsa_score(rewrite_input, tokenizer, model)

            st.divider()
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("📝 Original")
                st.text_area(
                    "", value=rewrite_input,
                    height=100, disabled=True,
                    key="orig_display"
                )
                st.metric("Ahimsa Score", f"{orig_score:.4f}")
                st.progress(orig_score)

            with col2:
                st.subheader("✨ Rewritten")
                with st.spinner("Rewriting with T5..."):
                    from transformers import (
                        T5ForConditionalGeneration, T5Tokenizer
                    )
                    t5_tok = T5Tokenizer.from_pretrained("t5-small")
                    t5_mdl = T5ForConditionalGeneration.from_pretrained(
                        "t5-small"
                    )
                    t5_mdl.eval()

                    prompt = (
                        f"rewrite this text to be non-toxic "
                        f"and constructive: {rewrite_input}"
                    )
                    inputs = t5_tok(
                        prompt, return_tensors="pt",
                        max_length=128, truncation=True
                    )
                    with torch.no_grad():
                        output = t5_mdl.generate(
                            **inputs, max_new_tokens=64,
                            num_beams=4, early_stopping=True
                        )
                    rewritten = t5_tok.decode(
                        output[0], skip_special_tokens=True
                    )

                new_score, _ = get_ahimsa_score(rewritten, tokenizer, model)

                st.text_area(
                    "", value=rewritten,
                    height=100, disabled=True,
                    key="rewritten_display"
                )
                st.metric(
                    "Ahimsa Score", f"{new_score:.4f}",
                    delta=f"+{new_score - orig_score:.4f}"
                )
                st.progress(new_score)

            if new_score > orig_score:
                st.success(
                    f"✅ Score improved by "
                    f"{new_score - orig_score:.4f}"
                )
            else:
                st.warning(
                    "⚠️ T5-small has limited rewriting capability."
                )

            del t5_mdl
        else:
            st.warning("Please enter some text first.")

# ═══════════════════════════════════════════════════════════════
# TAB 4 — Multilingual Checker
# ═══════════════════════════════════════════════════════════════
with tab4:
    st.header("🌐 Multilingual Semantic Similarity")
    st.markdown(
        "Check how well meaning is **preserved across languages**. "
        "Based on Reimers & Gurevych (2020)."
    )

    col1, col2 = st.columns(2)
    with col1:
        text_lang1  = st.text_area(
            "Text in Language 1:",
            placeholder="Sanskrit / Hindi / any language...",
            height=150, key="lang1"
        )
        lang1_label = st.text_input("Label:", value="Sanskrit")

    with col2:
        text_lang2  = st.text_area(
            "Text in Language 2:",
            placeholder="Translation or meaning...",
            height=150, key="lang2"
        )
        lang2_label = st.text_input("Label:", value="English")

    if st.button("Check Semantic Alignment", type="primary"):
        if text_lang1.strip() and text_lang2.strip():
            with st.spinner("Loading multilingual model..."):
                sbert = load_multilingual_model()
            with st.spinner("Computing alignment..."):
                similarity = get_similarity(text_lang1, text_lang2, sbert)

            st.divider()
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    label=f"🌐 {lang1_label}–{lang2_label} Similarity",
                    value=f"{similarity:.4f}"
                )
            with col2:
                st.metric(
                    label="📊 Random Baseline",
                    value="~0.10–0.30"
                )
            with col3:
                if similarity > 0.5:
                    verdict = "✅ Strong alignment"
                elif similarity > 0.3:
                    verdict = "🟡 Moderate alignment"
                else:
                    verdict = "🔴 Weak alignment"
                st.metric(label="Verdict", value=verdict)

            st.progress(
                min(similarity, 1.0),
                text=f"Similarity: {similarity:.4f}"
            )
            st.info(
                "📌 Reimers & Gurevych (2020): correct pairs score "
                f"above random baseline. Your score: {similarity:.4f} — "
                + ("meaningful alignment ✅" if similarity > 0.3
                   else "weak alignment ⚠️")
            )
        else:
            st.warning("Please enter text in both fields.")

# ── Footer ────────────────────────────────────────────────────
st.divider()
st.markdown(
    "**NLP as Digital Veda** | "
    "Kaustubh Bhalerao · Sai Subham Jenna · Yogesh Saini"
)
