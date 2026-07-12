"""
Brand Sentiment Analysis — Streamlit Demo App
===============================================
Interactive web interface for the sentiment analysis pipeline.

Run with:
    streamlit run app.py

Features:
    - Paste any tweet/article text
    - See detected brands, aspects, and sentiments
    - Visual pipeline flow diagram
    - Sample texts for quick demo
"""

import streamlit as st
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from brands import get_brands, replace_hin_to_eng
from detect_script import detect as detect_script
from aspect_sentiment import analyze_aspects, detect_aspects, ASPECT_KEYWORDS
from fuzzy_brands import fuzzy_match_brand, get_brands_fuzzy


# ─── Page Configuration ───
st.set_page_config(
    page_title="Brand Sentiment Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ───
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0;
    }
    .sub-header {
        text-align: center;
        color: #888;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 1.2rem;
        border-radius: 12px;
        border: 1px solid #333;
        text-align: center;
    }
    .positive { color: #00c853; font-weight: bold; }
    .negative { color: #ff1744; font-weight: bold; }
    .neutral { color: #ffd600; font-weight: bold; }
    .brand-tag {
        display: inline-block;
        background: #667eea;
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        margin: 2px;
        font-size: 0.9rem;
    }
    .aspect-card {
        background: #1a1a2e;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid;
    }
    .aspect-positive { border-left-color: #00c853; }
    .aspect-negative { border-left-color: #ff1744; }
    .aspect-neutral { border-left-color: #ffd600; }
</style>
""", unsafe_allow_html=True)


# ─── Header ───
st.markdown('<h1 class="main-header">📊 Brand Sentiment Analysis</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Multilingual NLP Pipeline • BERT + XGBoost + T5 • English / Hindi / Hinglish</p>', unsafe_allow_html=True)


# ─── Sidebar ───
with st.sidebar:
    st.header("⚙️ Pipeline Settings")
    
    fuzzy_threshold = st.slider(
        "Fuzzy Match Threshold",
        min_value=60, max_value=100, value=80,
        help="Lower = more lenient matching (catches more typos but may have false positives)"
    )
    
    st.divider()
    
    st.header("📋 Sample Texts")
    sample_texts = {
        "English (positive + negative)": "Samsung camera is absolutely amazing but the battery drains so fast. Worst battery life ever!",
        "Hinglish": "Bhai OnePlus ka phone ekdum mast hai, camera bhi accha hai lekin price thoda zyada hai",
        "Brand typos": "I just bought a new Samsng phone and the Xiomi one my friend has. Both are great!",
        "Multi-brand comparison": "iPhone display is gorgeous but too expensive. OnePlus offers better value with smooth performance.",
        "Hindi (Devanagari)": "शाओमी का नया फोन बहुत अच्छा है लेकिन बैटरी जल्दी खत्म हो जाती है",
        "Aspect-rich review": "Pixel 8 camera is the best in class, display is crisp and bright, but the design looks cheap and plastic. Software is clean though.",
    }
    
    selected_sample = st.selectbox("Try a sample:", ["(Type your own)"] + list(sample_texts.keys()))


# ─── Main Input ───
if selected_sample != "(Type your own)":
    input_text = st.text_area(
        "📝 Enter tweet or article text:",
        value=sample_texts[selected_sample],
        height=120
    )
else:
    input_text = st.text_area(
        "📝 Enter tweet or article text:",
        placeholder="Paste any tweet, review, or article about mobile phones...",
        height=120
    )


# ─── Analysis ───
if st.button("🔍 Analyze", type="primary", use_container_width=True):
    if not input_text.strip():
        st.warning("Please enter some text to analyze!")
    else:
        st.divider()
        
        # ── Stage 1: Script Detection ──
        with st.spinner("🔤 Detecting language..."):
            script = detect_script(input_text)
            
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("🔤 Script Detected", script)
        
        # ── Stage 2: Hindi Brand Replacement ──
        processed_text = replace_hin_to_eng([input_text])[0]
        if processed_text != input_text:
            with col2:
                st.metric("🔄 Hindi Brands", "Translated")
        
        # ── Stage 3: Brand Detection ──
        with st.spinner("🏷️ Detecting brands..."):
            # Exact matching
            exact_brands = get_brands([processed_text], verbose=False)[0]
            # Fuzzy matching
            fuzzy_brands = get_brands_fuzzy([processed_text], threshold=fuzzy_threshold, verbose=False)[0]
            # New brands found only by fuzzy matching
            new_fuzzy = [b for b in fuzzy_brands if b not in exact_brands]
        
        with col3:
            st.metric("🏷️ Brands Found", len(fuzzy_brands))
        
        # ── Stage 4: Aspect-Based Sentiment ──
        with st.spinner("😊 Analyzing sentiment..."):
            absa_result = analyze_aspects(processed_text)
        
        with col4:
            sentiment = absa_result['overall']['sentiment']
            emoji = '😊' if sentiment == 'Positive' else '😡' if sentiment == 'Negative' else '😐'
            st.metric(f"{emoji} Overall Sentiment", sentiment)
        
        st.divider()
        
        # ── Results Display ──
        left_col, right_col = st.columns([1, 1])
        
        with left_col:
            # Brands Section
            st.subheader("🏷️ Brands Detected")
            if fuzzy_brands:
                brand_html = ""
                for brand in fuzzy_brands:
                    label = f"{brand.title()}"
                    if brand in new_fuzzy:
                        label += " 🔍"  # Fuzzy match indicator
                    brand_html += f'<span class="brand-tag">{label}</span> '
                st.markdown(brand_html, unsafe_allow_html=True)
                
                if new_fuzzy:
                    st.caption("🔍 = Found via fuzzy matching (typo correction)")
            else:
                st.info("No mobile brands detected in this text.")
        
        with right_col:
            # Overall Sentiment
            st.subheader(f"{emoji} Overall Sentiment")
            sentiment_class = sentiment.lower()
            confidence = absa_result['overall']['confidence']
            st.progress(confidence)
            st.markdown(
                f'<span class="{sentiment_class}">{sentiment}</span> '
                f'— Confidence: **{confidence:.0%}**',
                unsafe_allow_html=True
            )
        
        # ── Aspect-Based Sentiment ──
        st.divider()
        st.subheader("🎯 Aspect-Based Sentiment Breakdown")
        
        if absa_result['aspects']:
            aspect_cols = st.columns(min(len(absa_result['aspects']), 3))
            
            for i, (aspect, info) in enumerate(absa_result['aspects'].items()):
                col_idx = i % min(len(absa_result['aspects']), 3)
                with aspect_cols[col_idx]:
                    s = info['sentiment']
                    emoji_a = '✅' if s == 'Positive' else '❌' if s == 'Negative' else '😐'
                    border_class = f"aspect-{s.lower()}"
                    
                    st.markdown(
                        f'<div class="aspect-card {border_class}">'
                        f'<strong>{emoji_a} {aspect.upper()}</strong><br/>'
                        f'{s} ({info["confidence"]:.0%})<br/>'
                        f'<small>"{info["text"][:80]}..."</small>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
        else:
            st.info("No specific product aspects detected. Try mentioning camera, battery, display, price, etc.")
        
        # ── Pipeline Flow ──
        st.divider()
        st.subheader("🔄 Pipeline Flow")
        
        flow_data = {
            "Stage": ["1. Script Detection", "2. Brand Translation", 
                      "3. Brand Extraction", "4. Sentiment Analysis"],
            "Input": [input_text[:50] + "...", 
                      "Hindi brand names",
                      "Processed text",
                      "Brand-segmented text"],
            "Output": [f"{script}",
                       f"{'Translated' if processed_text != input_text else 'No translation needed'}",
                       f"{len(fuzzy_brands)} brands: {', '.join(b.title() for b in fuzzy_brands[:5])}",
                       f"{sentiment} ({confidence:.0%})"],
            "Model/Method": ["Unicode Ranges", "Dictionary Mapping", 
                            "Regex + Fuzzy (rapidfuzz)", "Rule-based ABSA"]
        }
        
        st.table(flow_data)


# ─── Footer ───
st.divider()
st.markdown(
    "<center><small>Built with PyTorch, HuggingFace Transformers, XGBoost, and Streamlit<br/>"
    "Brand Sentiment Analysis Pipeline • NLP Project</small></center>",
    unsafe_allow_html=True
)
