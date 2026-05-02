"""
T7.5 — Indian Wildlife Identifier — Streamlit App
====================================================
Upload a photo → get top-3 species predictions →
Wikipedia summary + IUCN conservation status badge.
"""

import os, json, io
import streamlit as st
import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from pathlib import Path

# ── Page config ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Indian Wildlife Identifier",
    page_icon="https://em-content.zobj.net/source/twitter/408/paw-prints_1f43e.png",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Constants ─────────────────────────────────────────────────────────
MODEL_NAME = "openai/clip-vit-base-patch32"

CLASSES = ["cheetah", "fox", "hyena", "lion", "tiger", "wolf"]

TEXT_PROMPTS = [
    "a photo of a cheetah, a large spotted African cat",
    "a photo of a fox, a small reddish canine with a bushy tail",
    "a photo of a hyena, an African carnivore with rounded ears",
    "a photo of a lion, a large maned African cat",
    "a photo of a tiger, a large striped Asian cat",
    "a photo of a gray wolf, a large wild canine",
]

IUCN_COLORS = {
    "LC": ("#2E7D32", "#A5D6A7"),
    "NT": ("#558B2F", "#C5E1A5"),
    "VU": ("#F57F17", "#FFF176"),
    "EN": ("#E65100", "#FFAB91"),
    "CR": ("#B71C1C", "#EF9A9A"),
    "EW": ("#6A1B9A", "#CE93D8"),
    "EX": ("#212121", "#BDBDBD"),
}

IUCN_FULL = {
    "LC": "Least Concern",
    "NT": "Near Threatened",
    "VU": "Vulnerable",
    "EN": "Endangered",
    "CR": "Critically Endangered",
    "EW": "Extinct in the Wild",
    "EX": "Extinct",
}

SPECIES_EMOJI = {
    "cheetah": "&#x1F406;",
    "fox": "&#x1F98A;",
    "hyena": "&#x1F43E;",
    "lion": "&#x1F981;",
    "tiger": "&#x1F405;",
    "wolf": "&#x1F43A;",
}

CLASS_COLORS = {
    "cheetah": "#FF6B6B",
    "fox": "#FFA94D",
    "hyena": "#FFD43B",
    "lion": "#69DB7C",
    "tiger": "#4DABF7",
    "wolf": "#9775FA",
}


# ── Load model (cached) ──────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    model = CLIPModel.from_pretrained(MODEL_NAME)
    model.eval()
    return model, processor


@st.cache_data(show_spinner=False)
def load_species_cache():
    cache_path = Path(__file__).parent / "species_cache.json"
    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


# ── Inference ─────────────────────────────────────────────────────────
def predict(model, processor, image: Image.Image):
    """Return (class_names, probabilities) sorted descending."""
    text_inputs = processor(text=TEXT_PROMPTS, return_tensors="pt", padding=True)
    image_inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(
            **{k: v for k, v in image_inputs.items()},
            input_ids=text_inputs["input_ids"],
            attention_mask=text_inputs["attention_mask"],
        )
        probs = outputs.logits_per_image.softmax(dim=-1).cpu().numpy()[0]

    order = probs.argsort()[::-1]
    return [(CLASSES[i], float(probs[i])) for i in order]


# ── Custom CSS ────────────────────────────────────────────────────────
def inject_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

        /* ── Global ──────────────────────────────────────── */
        html, body, .stApp {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        .stApp {
            background: linear-gradient(160deg, #0a0e1a 0%, #111827 40%, #0f172a 100%);
        }

        section[data-testid="stSidebar"] { display: none; }

        /* hide default streamlit header/footer */
        header[data-testid="stHeader"] { background: transparent; }
        footer { display: none; }
        #MainMenu { display: none; }

        /* ── Hero Section ────────────────────────────────── */
        .hero {
            text-align: center;
            padding: 2.5rem 1rem 1.5rem;
            margin-bottom: 1rem;
        }
        .hero-icon {
            font-size: 3.5rem;
            margin-bottom: 0.5rem;
            filter: drop-shadow(0 0 20px rgba(255, 165, 0, 0.4));
        }
        .hero h1 {
            font-size: 2.8rem;
            font-weight: 900;
            background: linear-gradient(135deg, #f59e0b, #ef4444, #ec4899, #8b5cf6, #3b82f6);
            background-size: 200% 200%;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            animation: gradient-shift 4s ease infinite;
            margin: 0 0 0.5rem 0;
            letter-spacing: -0.5px;
        }
        @keyframes gradient-shift {
            0%, 100% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
        }
        .hero-subtitle {
            color: #94a3b8;
            font-size: 1.05rem;
            font-weight: 400;
            max-width: 550px;
            margin: 0 auto;
            line-height: 1.6;
        }

        /* ── Glass Card ──────────────────────────────────── */
        .glass-card {
            background: rgba(255, 255, 255, 0.04);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 16px;
            padding: 1.5rem;
            margin-bottom: 1rem;
            transition: all 0.3s ease;
        }
        .glass-card:hover {
            border-color: rgba(255, 255, 255, 0.15);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        }
        .glass-card h3 {
            color: #f1f5f9;
            font-weight: 700;
            font-size: 1.15rem;
            margin: 0 0 0.8rem 0;
        }
        .glass-card p {
            color: #cbd5e1;
            line-height: 1.7;
            font-size: 0.92rem;
            margin: 0;
        }

        /* ── Section Headers ─────────────────────────────── */
        .section-label {
            color: #64748b;
            font-size: 0.75rem;
            font-weight: 700;
            letter-spacing: 2px;
            text-transform: uppercase;
            margin-bottom: 0.8rem;
        }

        /* ── Confidence Bars ─────────────────────────────── */
        .conf-row {
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 10px;
        }
        .conf-rank {
            width: 28px;
            height: 28px;
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 800;
            font-size: 0.75rem;
            flex-shrink: 0;
        }
        .conf-rank.gold   { background: rgba(245,158,11,0.2); color: #f59e0b; }
        .conf-rank.silver { background: rgba(148,163,184,0.15); color: #94a3b8; }
        .conf-rank.bronze { background: rgba(120,113,108,0.15); color: #78716c; }
        .conf-info {
            flex: 1;
            min-width: 0;
        }
        .conf-label-row {
            display: flex;
            justify-content: space-between;
            align-items: baseline;
            margin-bottom: 4px;
        }
        .conf-name {
            font-weight: 600;
            font-size: 0.95rem;
            color: #f1f5f9;
        }
        .conf-pct {
            font-weight: 700;
            font-size: 0.9rem;
            font-variant-numeric: tabular-nums;
        }
        .conf-track {
            background: rgba(255,255,255,0.06);
            border-radius: 6px;
            height: 8px;
            overflow: hidden;
        }
        .conf-fill {
            height: 100%;
            border-radius: 6px;
            transition: width 0.8s cubic-bezier(0.25, 0.46, 0.45, 0.94);
        }

        /* ── IUCN Badge ──────────────────────────────────── */
        .iucn-badge {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 8px 18px;
            border-radius: 10px;
            font-weight: 700;
            font-size: 0.88rem;
            letter-spacing: 0.5px;
        }
        .iucn-dot {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            display: inline-block;
        }

        /* ── Fun Fact Card ────────────────────────────────── */
        .fun-fact-card {
            background: linear-gradient(135deg, rgba(245,158,11,0.08), rgba(249,115,22,0.06));
            border: 1px solid rgba(245,158,11,0.2);
            border-radius: 12px;
            padding: 1rem 1.2rem;
            margin-top: 0.8rem;
        }
        .fun-fact-card .ff-label {
            color: #f59e0b;
            font-weight: 700;
            font-size: 0.8rem;
            letter-spacing: 1px;
            text-transform: uppercase;
            margin-bottom: 6px;
        }
        .fun-fact-card .ff-text {
            color: #e2e8f0;
            line-height: 1.65;
            font-size: 0.9rem;
        }

        /* ── Habitat / Diet Pills ─────────────────────────── */
        .info-pills {
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
            margin-top: 0.6rem;
        }
        .pill {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 20px;
            padding: 5px 14px;
            font-size: 0.78rem;
            color: #94a3b8;
        }
        .pill strong { color: #cbd5e1; }

        /* ── Waiting State ────────────────────────────────── */
        .empty-state {
            text-align: center;
            padding: 3rem 2rem;
            color: #475569;
        }
        .empty-state .es-icon {
            font-size: 3rem;
            margin-bottom: 1rem;
            opacity: 0.5;
        }
        .empty-state h3 {
            color: #64748b;
            font-weight: 600;
            margin-bottom: 0.5rem;
        }
        .empty-state p {
            color: #475569;
            font-size: 0.9rem;
        }

        /* ── Divider ──────────────────────────────────────── */
        .divider {
            height: 1px;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.08), transparent);
            margin: 1.5rem 0;
        }

        /* ── Batch Section ────────────────────────────────── */
        .batch-header {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 0.5rem;
        }
        .batch-header h2 {
            color: #f1f5f9;
            font-weight: 700;
            font-size: 1.3rem;
            margin: 0;
        }
        .batch-tag {
            background: rgba(139, 92, 246, 0.15);
            color: #a78bfa;
            font-size: 0.7rem;
            font-weight: 700;
            padding: 3px 10px;
            border-radius: 6px;
            letter-spacing: 1px;
        }

        /* ── Footer ───────────────────────────────────────── */
        .app-footer {
            text-align: center;
            color: #334155;
            font-size: 0.78rem;
            padding: 2rem 0 1rem;
            letter-spacing: 0.5px;
        }

        /* ── Streamlit file uploader deep overrides ──────── */
        /* outer wrapper */
        [data-testid="stFileUploader"] {
            background: transparent !important;
        }
        /* dropzone container */
        [data-testid="stFileUploaderDropzone"] {
            background: rgba(255,255,255,0.03) !important;
            border: 2px dashed rgba(255,255,255,0.15) !important;
            border-radius: 14px !important;
            padding: 2rem 1.5rem !important;
            transition: all 0.3s ease !important;
        }
        [data-testid="stFileUploaderDropzone"]:hover {
            border-color: rgba(245,158,11,0.5) !important;
            background: rgba(245,158,11,0.04) !important;
        }
        /* "Drag and drop" + "Limit" text */
        [data-testid="stFileUploaderDropzone"] span,
        [data-testid="stFileUploaderDropzone"] small,
        [data-testid="stFileUploaderDropzone"] div {
            color: #64748b !important;
        }
        /* the Browse/Upload button inside dropzone */
        [data-testid="stFileUploaderDropzone"] button,
        [data-testid="stBaseButton-secondary"] {
            background: linear-gradient(135deg, #f59e0b, #d97706) !important;
            color: #000 !important;
            border: none !important;
            border-radius: 8px !important;
            padding: 0.45rem 1.3rem !important;
            font-weight: 700 !important;
            font-size: 0.85rem !important;
            letter-spacing: 0.3px !important;
            cursor: pointer !important;
            min-height: 38px !important;
            line-height: 1.4 !important;
        }
        [data-testid="stFileUploaderDropzone"] button:hover,
        [data-testid="stBaseButton-secondary"]:hover {
            background: linear-gradient(135deg, #d97706, #b45309) !important;
        }
        /* fix any overlapping text inside the button */
        [data-testid="stFileUploaderDropzone"] button * {
            position: relative !important;
            z-index: 1 !important;
        }
        /* Streamlit renders TWO labels inside button:
             button > div > span > span (duplicate text)
             button > div > span > div > p (the actual "Upload" text)
           Hide the first nested span that has the duplicate text */
        [data-testid="stFileUploaderDropzone"] button > div > span > span {
            display: none !important;
        }
        /* uploaded file name chip */
        [data-testid="stFileUploaderFile"] {
            background: rgba(255,255,255,0.05) !important;
            border: 1px solid rgba(255,255,255,0.1) !important;
            border-radius: 8px !important;
            color: #e2e8f0 !important;
        }
        [data-testid="stFileUploaderFile"] span {
            color: #e2e8f0 !important;
        }
        [data-testid="stFileUploaderFile"] button {
            background: transparent !important;
            color: #ef4444 !important;
        }

        /* ── Download button ─────────────────────────────── */
        .stDownloadButton > button {
            background: linear-gradient(135deg, #8b5cf6, #6366f1) !important;
            color: white !important;
            border: none !important;
            border-radius: 10px !important;
            padding: 0.6rem 1.5rem !important;
            font-weight: 600 !important;
        }
        .stDownloadButton > button:hover {
            background: linear-gradient(135deg, #7c3aed, #4f46e5) !important;
        }

        .stSpinner > div { color: #f59e0b !important; }

        /* ── Expander ────────────────────────────────────── */
        div[data-testid="stExpander"] {
            background: rgba(255,255,255,0.02);
            border: 1px solid rgba(255,255,255,0.06);
            border-radius: 12px;
        }
        div[data-testid="stExpander"] summary {
            color: #94a3b8 !important;
            font-size: 0.85rem;
        }

        /* ── Dataframe ───────────────────────────────────── */
        .stDataFrame { border-radius: 12px; overflow: hidden; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ── UI Builders ───────────────────────────────────────────────────────
def render_confidence_bars(preds):
    """Render top-3 animated confidence bars."""
    rank_classes = ["gold", "silver", "bronze"]
    rank_labels = ["1", "2", "3"]

    html = '<div class="section-label">TOP 3 PREDICTIONS</div>'

    for i, (cls_name, prob) in enumerate(preds[:3]):
        pct = prob * 100
        color = CLASS_COLORS.get(cls_name, "#64748b")
        emoji = SPECIES_EMOJI.get(cls_name, "")
        html += f"""
        <div class="conf-row">
            <div class="conf-rank {rank_classes[i]}">{rank_labels[i]}</div>
            <div class="conf-info">
                <div class="conf-label-row">
                    <span class="conf-name">{emoji} {cls_name.title()}</span>
                    <span class="conf-pct" style="color:{color};">{pct:.1f}%</span>
                </div>
                <div class="conf-track">
                    <div class="conf-fill" style="width:{pct}%; background: linear-gradient(90deg, {color}, {color}88);"></div>
                </div>
            </div>
        </div>"""

    st.markdown(html, unsafe_allow_html=True)


def render_species_card(cls_name, info):
    """Render the species info glass card."""
    emoji = SPECIES_EMOJI.get(cls_name, "")
    common = info.get("common_name", cls_name.title())
    sci = info.get("scientific_name", "")
    summary = info.get("wikipedia_summary", "")

    st.markdown(f"""
    <div class="glass-card">
        <h3>{emoji} {common} <span style="color:#64748b; font-weight:400; font-size:0.85rem;">({sci})</span></h3>
        <p>{summary}</p>
    </div>
    """, unsafe_allow_html=True)


def render_iucn_badge(code):
    """Render IUCN conservation status badge."""
    bg_color, dot_color = IUCN_COLORS.get(code, ("#334155", "#64748b"))
    full = IUCN_FULL.get(code, code)

    st.markdown(f"""
    <div class="section-label" style="margin-top:1rem;">CONSERVATION STATUS</div>
    <div class="iucn-badge" style="background: {bg_color}22; border: 1px solid {bg_color}44; color: {dot_color};">
        <span class="iucn-dot" style="background: {bg_color};"></span>
        {code} &mdash; {full}
    </div>
    """, unsafe_allow_html=True)


def render_fun_fact(fact):
    """Render the fun fact card."""
    st.markdown(f"""
    <div class="fun-fact-card">
        <div class="ff-label">&#x1F4A1; Fun Fact</div>
        <div class="ff-text">{fact}</div>
    </div>
    """, unsafe_allow_html=True)


def render_habitat_diet(info):
    """Render habitat and diet pills."""
    habitat = info.get("habitat", "")
    diet = info.get("diet", "")
    if habitat or diet:
        pills_html = '<div class="info-pills">'
        if habitat:
            pills_html += f'<span class="pill">&#x1F30D; <strong>Habitat:</strong> {habitat}</span>'
        if diet:
            pills_html += f'<span class="pill">&#x1F356; <strong>Diet:</strong> {diet}</span>'
        pills_html += '</div>'
        st.markdown(pills_html, unsafe_allow_html=True)


def render_empty_state():
    """Render the waiting/empty state."""
    st.markdown("""
    <div class="empty-state">
        <div class="es-icon">&#x1F4F7;</div>
        <h3>Upload a Wildlife Photo</h3>
        <p>Drop an image of a cheetah, fox, hyena, lion, tiger, or wolf<br>
        to identify the species and learn about it.</p>
    </div>
    """, unsafe_allow_html=True)


# ── Main App ──────────────────────────────────────────────────────────
def main():
    inject_css()

    # ── Hero ──────────────────────────────────────────────────────────
    st.markdown("""
    <div class="hero">
        <div class="hero-icon">&#x1F43E;</div>
        <h1>Indian Wildlife Identifier</h1>
        <p class="hero-subtitle">
            Snap a photo on a forest trek &mdash; identify the species,
            learn fun facts &amp; check IUCN conservation status
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Load resources ────────────────────────────────────────────────
    with st.spinner("Loading CLIP model..."):
        model, processor = load_model()
    species_cache = load_species_cache()

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── Main Content ──────────────────────────────────────────────────
    col_upload, col_gap, col_results = st.columns([4, 0.3, 6])

    with col_upload:
        st.markdown('<div class="section-label">UPLOAD IMAGE</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Choose a wildlife photo",
            type=["png", "jpg", "jpeg", "webp"],
            label_visibility="collapsed",
        )

        if uploaded is not None:
            image = Image.open(uploaded).convert("RGB")
            st.image(image, use_container_width=True)

    with col_results:
        if uploaded is not None:
            with st.spinner("Classifying..."):
                preds = predict(model, processor, image)

            # Confidence bars
            render_confidence_bars(preds)

            # Remaining classes
            with st.expander("All class probabilities"):
                for cls_name, prob in preds:
                    st.markdown(
                        f"<span style='color:#94a3b8;'>{cls_name.title()}</span>: "
                        f"<span style='color:#f1f5f9; font-weight:600;'>{prob*100:.2f}%</span>",
                        unsafe_allow_html=True,
                    )

            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

            # Species info
            top_cls = preds[0][0]
            info = species_cache.get(top_cls, {})

            if info:
                render_species_card(top_cls, info)

                # IUCN badge
                iucn_code = info.get("iucn_code", "")
                if iucn_code:
                    render_iucn_badge(iucn_code)

                # Habitat & diet
                render_habitat_diet(info)

                # Fun fact
                fun_fact = info.get("fun_fact", "")
                if fun_fact:
                    render_fun_fact(fun_fact)
        else:
            render_empty_state()

    # ── Batch Mode ────────────────────────────────────────────────────
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="batch-header">
        <h2>&#x1F4C2; Batch Mode</h2>
        <span class="batch-tag">CAMERA-TRAP</span>
    </div>
    <p style="color:#64748b; font-size:0.88rem; margin-bottom:1rem;">
        Upload multiple images at once &mdash; classify all &mdash; download results as CSV
    </p>
    """, unsafe_allow_html=True)

    batch_files = st.file_uploader(
        "Upload multiple images",
        type=["png", "jpg", "jpeg", "webp"],
        accept_multiple_files=True,
        key="batch_uploader",
        label_visibility="collapsed",
    )

    if batch_files:
        import pandas as pd

        rows = []
        progress = st.progress(0, text="Classifying...")
        for idx, f in enumerate(batch_files):
            img = Image.open(f).convert("RGB")
            preds = predict(model, processor, img)
            top3 = preds[:3]
            rows.append({
                "Filename": f.name,
                "Prediction 1": top3[0][0].title(),
                "Confidence 1": f"{top3[0][1]*100:.1f}%",
                "Prediction 2": top3[1][0].title(),
                "Confidence 2": f"{top3[1][1]*100:.1f}%",
                "Prediction 3": top3[2][0].title(),
                "Confidence 3": f"{top3[2][1]*100:.1f}%",
            })
            progress.progress(
                (idx + 1) / len(batch_files),
                text=f"Classified {idx+1}/{len(batch_files)}"
            )

        progress.empty()
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

        csv = df.to_csv(index=False)
        st.download_button(
            "Download CSV",
            data=csv,
            file_name="wildlife_predictions.csv",
            mime="text/csv",
        )

    # ── Footer ────────────────────────────────────────────────────────
    st.markdown("""
    <div class="app-footer">
        CLIP ViT-B/32 Zero-Shot &middot; T7.5 Indian Wildlife Identifier &middot; SMAI Assignment 3
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
