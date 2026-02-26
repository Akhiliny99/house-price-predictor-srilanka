import streamlit as st
import pickle
import numpy as np
import shap
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import io
import datetime
import gdown
import os
import warnings
warnings.filterwarnings('ignore')

# ── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="HouseAI · Sri Lanka",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Master CSS ────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;600;700;800&family=Courier+Prime:wght@400;700&family=Nunito:wght@300;400;500;600;700&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, .stApp {
    font-family: 'Nunito', sans-serif;
    background: #020c06 !important;
    color: #d4f5d4;
    min-height: 100vh;
    font-weight: 600 !important;
}

/* ── Animated Mesh Background ── */
.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background:
        radial-gradient(ellipse 70% 60% at 10% 10%, rgba(0,200,80,0.18) 0%, transparent 55%),
        radial-gradient(ellipse 50% 50% at 90% 90%, rgba(0,255,100,0.12) 0%, transparent 55%),
        radial-gradient(ellipse 40% 40% at 50% 50%, rgba(50,200,50,0.06) 0%, transparent 60%),
        radial-gradient(ellipse 30% 30% at 80% 20%, rgba(100,255,100,0.08) 0%, transparent 50%);
    pointer-events: none;
    z-index: 0;
}

/* ── Grid Pattern Overlay ── */
.stApp::after {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
        linear-gradient(rgba(0,200,80,0.04) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,200,80,0.04) 1px, transparent 1px);
    background-size: 40px 40px;
    pointer-events: none;
    z-index: 0;
}

.main .block-container {
    position: relative;
    z-index: 1;
    max-width: 1100px !important;
    padding: 2rem 2rem 4rem !important;
}

/* ── Hero ── */
.hero { text-align: center; padding: 3rem 0 2rem; }

.hero-eyebrow {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(0,200,80,0.08);
    border: 1px solid rgba(0,200,80,0.3);
    border-radius: 100px;
    padding: 6px 18px;
    font-family: 'Courier Prime', monospace;
    font-size: 0.7rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #39ff14;
    margin-bottom: 1.5rem;
}

.hero-dot {
    width: 7px; height: 7px;
    border-radius: 50%;
    background: #39ff14;
    display: inline-block;
    animation: blink 1.5s ease-in-out infinite;
    box-shadow: 0 0 8px #39ff14;
}

@keyframes blink {
    0%, 100% { opacity: 1; box-shadow: 0 0 8px #39ff14; }
    50%       { opacity: 0.2; box-shadow: none; }
}

.hero-title {
    font-family: 'Orbitron', sans-serif;
    font-size: clamp(2rem, 5vw, 3.8rem);
    font-weight: 800;
    line-height: 1.1;
    letter-spacing: 2px;
    margin-bottom: 1rem;
}

.hero-title .line1 { color: #e8ffe8; display: block; }
.hero-title .line2 {
    display: block;
    background: linear-gradient(135deg, #39ff14 0%, #00c850 40%, #a8ff78 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    background-size: 200% auto;
    animation: shimmer 4s linear infinite;
}

@keyframes shimmer {
    0%   { background-position: 0% center; }
    100% { background-position: 200% center; }
}

.hero-sub { font-size: 0.9rem; color: #7acc7a; font-weight: 700; letter-spacing: 0.5px; }

/* ── Stats Strip ── */
.stats-strip {
    display: flex;
    justify-content: center;
    gap: 1.5rem;
    margin: 2rem 0;
    flex-wrap: wrap;
}

.stat-pill {
    background: rgba(0,200,80,0.06);
    border: 1px solid rgba(0,200,80,0.2);
    border-radius: 12px;
    padding: 12px 22px;
    text-align: center;
    transition: all 0.3s ease;
}

.stat-num {
    font-family: 'Orbitron', sans-serif;
    font-size: 1.3rem;
    font-weight: 700;
    color: #39ff14;
    text-shadow: 0 0 10px rgba(57,255,20,0.4);
}

.stat-lbl {
    font-size: 0.68rem;
    color: #7acc7a;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-top: 3px;
}

/* ── Card Label ── */
.card-label {
    font-family: 'Courier Prime', monospace;
    font-size: 0.68rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #39ff14;
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    gap: 10px;
}

.card-label::before {
    content: '';
    display: inline-block;
    width: 24px; height: 2px;
    background: linear-gradient(90deg, #39ff14, #00c850);
    box-shadow: 0 0 6px #39ff14;
}

/* ── Widget Overrides ── */
/* FIX 1: input labels were #4d8c4d + font-weight:400 → now bright + bold */
div[data-testid="stSelectbox"] > label,
div[data-testid="stNumberInput"] > label {
    font-family: 'Courier Prime', monospace !important;
    font-size: 0.65rem !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    color: #7acc7a !important;
    font-weight: 700 !important;
}

div[data-testid="stSelectbox"] > div > div,
div[data-testid="stNumberInput"] input {
    background: rgba(0,10,3,0.8) !important;
    border: 1px solid rgba(0,200,80,0.2) !important;
    border-radius: 10px !important;
    color: #d4f5d4 !important;
    font-family: 'Nunito', sans-serif !important;
    font-size: 0.95rem !important;
}

div[data-testid="stSelectbox"] > div > div:focus-within,
div[data-testid="stNumberInput"] input:focus {
    border-color: #39ff14 !important;
    box-shadow: 0 0 0 3px rgba(57,255,20,0.1) !important;
}

/* ── Button ── */
.stButton > button {
    width: 100% !important;
    background: linear-gradient(135deg, #006620, #00c850) !important;
    color: #e8ffe8 !important;
    border: 1px solid rgba(57,255,20,0.3) !important;
    border-radius: 12px !important;
    padding: 16px 24px !important;
    font-family: 'Orbitron', sans-serif !important;
    font-size: 0.9rem !important;
    font-weight: 700 !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    transition: all 0.3s ease !important;
    margin-top: 8px !important;
    box-shadow: 0 4px 20px rgba(0,200,80,0.2) !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(57,255,20,0.35) !important;
    background: linear-gradient(135deg, #00c850, #39ff14) !important;
    color: #020c06 !important;
}

/* ── Price Card ── */
.price-result {
    background: linear-gradient(135deg,
        rgba(0,100,30,0.25) 0%,
        rgba(0,200,80,0.12) 100%);
    border: 1px solid rgba(57,255,20,0.3);
    border-radius: 20px;
    padding: 40px;
    text-align: center;
    position: relative;
    overflow: hidden;
    margin: 10px 0 24px;
    box-shadow: 0 0 40px rgba(0,200,80,0.08), inset 0 0 40px rgba(0,200,80,0.03);
}

.price-result::before {
    content: '';
    position: absolute;
    inset: 0;
    background: radial-gradient(ellipse at 50% 0%,
        rgba(57,255,20,0.08), transparent 60%);
}

.price-eyebrow {
    font-family: 'Courier Prime', monospace;
    font-size: 0.65rem;
    letter-spacing: 4px;
    text-transform: uppercase;
    color: #39ff14;
    margin-bottom: 14px;
    text-shadow: 0 0 10px rgba(57,255,20,0.5);
}

.price-main {
    font-family: 'Orbitron', sans-serif;
    font-size: clamp(2.5rem, 6vw, 4.5rem);
    font-weight: 800;
    letter-spacing: 2px;
    background: linear-gradient(135deg, #a8ff78, #39ff14, #00c850);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1;
    margin-bottom: 14px;
    filter: drop-shadow(0 0 20px rgba(57,255,20,0.3));
}

/* FIX 2: price-range was #4d8c4d → now #7acc7a */
.price-range { font-size: 0.85rem; color: #7acc7a; font-weight: 700; }
.price-range span { color: #a8ff78; font-weight: 600; }

/* ── Metrics Row ── */
.metrics-row {
    display: flex;
    gap: 10px;
    margin: 16px 0;
    flex-wrap: wrap;
}

.metric-chip {
    flex: 1;
    min-width: 110px;
    background: rgba(0,200,80,0.05);
    border: 1px solid rgba(0,200,80,0.15);
    border-radius: 12px;
    padding: 12px 14px;
    text-align: center;
}

.metric-val {
    font-family: 'Orbitron', sans-serif;
    font-size: 0.9rem;
    font-weight: 700;
    color: #39ff14;
}

.metric-key {
    font-size: 0.65rem;
    color: #7acc7a;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 4px;
}

/* ── SHAP Header ── */
.shap-header {
    font-family: 'Courier Prime', monospace;
    font-size: 0.65rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #39ff14;
    margin: 24px 0 16px;
    display: flex;
    align-items: center;
    gap: 10px;
}

.shap-header::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, rgba(57,255,20,0.4), transparent);
}

/* ── Feature Cards ── */
.feature-card {
    background: rgba(0,200,80,0.03);
    border: 1px solid rgba(0,200,80,0.12);
    border-radius: 16px;
    padding: 20px;
    margin: 10px 0;
    height: 100%;
    position: relative;
    overflow: hidden;
}

.feature-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent,
        rgba(57,255,20,0.5), transparent);
}

.feature-title {
    font-family: 'Courier Prime', monospace;
    font-size: 0.65rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #39ff14;
    margin-bottom: 16px;
    text-shadow: 0 0 8px rgba(57,255,20,0.3);
}

.insight-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 0;
    border-bottom: 1px solid rgba(0,200,80,0.07);
}

.insight-label { font-size: 0.8rem; color: #7acc7a; font-weight: 700; }

.insight-value {
    font-family: 'Orbitron', sans-serif;
    font-size: 0.82rem;
    font-weight: 600;
    color: #d4f5d4;
}

/* ── Score ── */
.score-ring { text-align: center; padding: 12px 0; }

.score-num {
    font-family: 'Orbitron', sans-serif;
    font-size: 3.8rem;
    font-weight: 800;
    line-height: 1;
}

/* FIX 3: score-label was #4d8c4d → now #7acc7a */
.score-label {
    font-size: 0.72rem;
    color: #7acc7a;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-top: 4px;
}

/* ── Download Button ── */
.stDownloadButton > button {
    background: rgba(0,200,80,0.08) !important;
    color: #39ff14 !important;
    border: 1px solid rgba(57,255,20,0.3) !important;
    border-radius: 10px !important;
    font-family: 'Orbitron', sans-serif !important;
    font-size: 0.75rem !important;
    letter-spacing: 2px !important;
    padding: 12px 24px !important;
    transition: all 0.3s ease !important;
}

.stDownloadButton > button:hover {
    background: rgba(57,255,20,0.15) !important;
    box-shadow: 0 0 20px rgba(57,255,20,0.2) !important;
}

/* ── Divider ── */
hr {
    border: none !important;
    height: 1px !important;
    background: linear-gradient(90deg, transparent,
        rgba(0,200,80,0.3) 30%,
        rgba(57,255,20,0.3) 70%,
        transparent) !important;
    margin: 24px 0 !important;
}

/* ── Footer ── */
.app-footer {
    text-align: center;
    padding: 30px 0 10px;
    font-family: 'Courier Prime', monospace;
    font-size: 0.65rem;
    letter-spacing: 2px;
    color: #7acc7a;
    font-weight: 700;
    text-transform: uppercase;
}

/* ── Dataframe ── */
div[data-testid="stDataFrame"] {
    border-radius: 12px !important;
    overflow: hidden !important;
    border: 1px solid rgba(0,200,80,0.15) !important;
}

#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Google Drive Model Loading ────────────────────────────────
MODEL_ID   = "1-NRS70L2VE1BrAhXjfaIw1FSXHA8Nfkq"
ENCODER_ID = "173BPUAWNr3_SEfJeesQQzCIv6wxMWsm0"

@st.cache_resource
def load_assets():
    if not os.path.exists("best_model.pkl"):
        with st.spinner("⏳ Loading AI model... please wait"):
            gdown.download(
                f"https://drive.google.com/uc?id={MODEL_ID}",
                "best_model.pkl", quiet=False
            )
    if not os.path.exists("encoders.pkl"):
        with st.spinner("⏳ Loading encoders... please wait"):
            gdown.download(
                f"https://drive.google.com/uc?id={ENCODER_ID}",
                "encoders.pkl", quiet=False
            )
    with open("best_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("encoders.pkl", "rb") as f:
        encoders = pickle.load(f)
    explainer = shap.TreeExplainer(model)
    return model, encoders, explainer

model, encoders, explainer = load_assets()

# ── Province → District Mapping ───────────────────────────────
province_districts = {
    'Western':       ['Colombo', 'Gampaha', 'Kalutara'],
    'Central':       ['Kandy', 'Matale', 'Nuwara Eliya'],
    'Southern':      ['Galle', 'Matara', 'Hambantota'],
    'Eastern':       ['Batticaloa', 'Ampara', 'Trincomalee'],
    'Northern':      ['Jaffna', 'Kilinochchi', 'Mannar', 'Vavuniya', 'Mullaitivu'],
    'North Western': ['Kurunegala', 'Puttalam'],
    'North Central': ['Anuradhapura', 'Polonnaruwa'],
    'Sabaragamuwa':  ['Ratnapura', 'Kegalle'],
    'Uva':           ['Badulla', 'Moneragala']
}

province_keys = {
    'Western': 'western', 'Central': 'central', 'Southern': 'southern',
    'Eastern': 'eastern', 'Northern': 'northern',
    'North Western': 'north_western', 'North Central': 'north_central',
    'Sabaragamuwa': 'sabaragamuwa', 'Uva': 'uva'
}

# ── Market Data ───────────────────────────────────────────────
market_data = {
    'Western':       {'avg': 68_000_000, 'min': 5_000_000,  'max': 500_000_000, 'growth': '+12.4%', 'demand': 'Very High'},
    'Central':       {'avg': 28_000_000, 'min': 3_000_000,  'max': 200_000_000, 'growth': '+8.1%',  'demand': 'High'},
    'Southern':      {'avg': 32_000_000, 'min': 4_000_000,  'max': 250_000_000, 'growth': '+9.5%',  'demand': 'High'},
    'Eastern':       {'avg': 18_000_000, 'min': 2_000_000,  'max': 120_000_000, 'growth': '+6.2%',  'demand': 'Moderate'},
    'Northern':      {'avg': 15_000_000, 'min': 1_500_000,  'max': 100_000_000, 'growth': '+5.8%',  'demand': 'Moderate'},
    'North Western': {'avg': 22_000_000, 'min': 2_500_000,  'max': 150_000_000, 'growth': '+7.3%',  'demand': 'Moderate'},
    'North Central': {'avg': 16_000_000, 'min': 1_500_000,  'max': 100_000_000, 'growth': '+5.1%',  'demand': 'Low'},
    'Sabaragamuwa':  {'avg': 20_000_000, 'min': 2_000_000,  'max': 130_000_000, 'growth': '+6.9%',  'demand': 'Moderate'},
    'Uva':           {'avg': 14_000_000, 'min': 1_000_000,  'max': 90_000_000,  'growth': '+4.5%',  'demand': 'Low'},
}

# ── Helper Functions ──────────────────────────────────────────
def format_rs(val):
    val = round(val)
    if val >= 1e6: return f"Rs {val/1e6:.1f}M"
    if val >= 1e3: return f"Rs {val/1e3:.0f}K"
    return f"Rs {val:,}"

def get_investment_score(prediction, province, bedrooms,
                          bathrooms, house_size, land_size, verified):
    score = 5.0
    mkt   = market_data[province]
    demand_scores = {'Very High': 2.0, 'High': 1.5, 'Moderate': 0.5, 'Low': 0.0}
    score += demand_scores.get(mkt['demand'], 0)
    if prediction < mkt['avg'] * 0.85:  score += 1.5
    elif prediction < mkt['avg']:        score += 0.8
    elif prediction > mkt['avg'] * 1.2: score -= 1.0
    if bathrooms >= bedrooms:   score += 0.5
    if land_size > house_size:  score += 0.5
    if verified:                score += 0.3
    if bedrooms >= 3:           score += 0.2
    growth_val = float(mkt['growth'].replace('%','').replace('+',''))
    score += growth_val / 10
    return min(10.0, max(1.0, round(score, 1)))

def get_score_color(score):
    if score >= 8: return '#39ff14'
    if score >= 6: return '#00c850'
    if score >= 4: return '#a8ff78'
    return '#7acc7a'   # FIX 4: was #4d8c4d (too dark)

def get_score_label(score):
    if score >= 8: return 'Excellent Investment'
    if score >= 6: return 'Good Investment'
    if score >= 4: return 'Fair Investment'
    return 'Below Average'

# ── PDF Report ────────────────────────────────────────────────
def generate_pdf_report(prediction, range_low, range_high,
                         province, district, bedrooms, bathrooms,
                         house_size, land_size, property_type,
                         verified, inv_score, shap_data, mkt):
    buffer = io.BytesIO()
    doc    = SimpleDocTemplate(buffer, pagesize=A4,
                               rightMargin=2*cm, leftMargin=2*cm,
                               topMargin=2*cm, bottomMargin=2*cm)
    GREEN  = colors.HexColor('#006620')
    LGREEN = colors.HexColor('#00c850')
    DARK   = colors.HexColor('#020c06')
    LIGHT  = colors.HexColor('#e8ffe8')
    GRAY   = colors.HexColor('#4d8c4d')
    WHITE  = colors.white
    styles = getSampleStyleSheet()
    story  = []

    ht = Table([[Paragraph(
        "<b>SABARAGAMUWA UNIVERSITY OF SRI LANKA</b><br/>"
        "Sri Lanka House Price Prediction System — AI Generated Report",
        ParagraphStyle('H', fontSize=12, fontName='Helvetica-Bold',
                       textColor=WHITE, alignment=TA_CENTER)
    )]], colWidths=[17*cm])
    ht.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), DARK),
        ('TOPPADDING', (0,0), (-1,-1), 14),
        ('BOTTOMPADDING', (0,0), (-1,-1), 14),
    ]))
    story.append(ht)
    story.append(Spacer(1, 0.4*cm))
    story.append(Paragraph(
        f"Report Generated: {datetime.datetime.now().strftime('%d %B %Y, %H:%M')} "
        f"| Student: Akhiliny Vijeyagumar | Index: 20APSE4875",
        ParagraphStyle('meta', fontSize=8, fontName='Helvetica',
                       textColor=GRAY, alignment=TA_CENTER)
    ))
    story.append(Spacer(1, 0.5*cm))

    pt = Table([[
        Paragraph("PREDICTED PROPERTY VALUE",
                  ParagraphStyle('pl', fontSize=8, fontName='Helvetica',
                                 textColor=LGREEN, alignment=TA_CENTER))
    ],[
        Paragraph(format_rs(prediction),
                  ParagraphStyle('pv', fontSize=28, fontName='Helvetica-Bold',
                                 textColor=GREEN, alignment=TA_CENTER))
    ],[
        Paragraph(f"Confidence Range: {format_rs(range_low)} — {format_rs(range_high)}",
                  ParagraphStyle('pr', fontSize=9, fontName='Helvetica',
                                 textColor=GRAY, alignment=TA_CENTER))
    ]], colWidths=[17*cm])
    pt.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), LIGHT),
        ('BOX', (0,0), (-1,-1), 1.5, GREEN),
        ('TOPPADDING', (0,0), (-1,-1), 10),
        ('BOTTOMPADDING', (0,0), (-1,-1), 10),
    ]))
    story.append(pt)
    story.append(Spacer(1, 0.4*cm))

    prop_details = [
        ['Property Details', ''],
        ['Property Type',  property_type],
        ['Province',       province],
        ['District',       district],
        ['Bedrooms',       str(bedrooms)],
        ['Bathrooms',      str(bathrooms)],
        ['House Size',     f'{house_size} perches'],
        ['Land Size',      f'{land_size} perches'],
        ['Price / Perch',  format_rs(prediction / land_size if land_size > 0 else 0)],
        ['Verified Seller','Yes' if verified else 'No'],
    ]
    prop_t = Table(prop_details, colWidths=[4*cm, 4.5*cm])
    prop_t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), DARK),
        ('TEXTCOLOR', (0,0), (-1,0), WHITE),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('SPAN', (0,0), (-1,0)),
        ('ALIGN', (0,0), (-1,0), 'CENTER'),
        ('FONTSIZE', (0,0), (-1,-1), 9),
        ('FONTNAME', (0,1), (0,-1), 'Helvetica-Bold'),
        ('TEXTCOLOR', (0,1), (0,-1), GREEN),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [LIGHT, WHITE]),
        ('BOX', (0,0), (-1,-1), 0.5, GREEN),
        ('INNERGRID', (0,0), (-1,-1), 0.3, colors.HexColor('#d4f5d4')),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
    ]))

    sc = get_score_color(inv_score)
    sl = get_score_label(inv_score)
    score_color_map = {
        '#39ff14': colors.HexColor('#006620'),
        '#00c850': colors.HexColor('#00c850'),
        '#a8ff78': colors.HexColor('#4d8c4d'),
        '#7acc7a': colors.HexColor('#4d8c4d'),  # FIX 5: updated key
    }
    invest_data = [
        ['Investment Score', ''],
        ['', ''],
        [Paragraph(f"<b>{inv_score}/10</b>",
                   ParagraphStyle('sc', fontSize=28, fontName='Helvetica-Bold',
                                  textColor=score_color_map.get(sc, LGREEN),
                                  alignment=TA_CENTER)), ''],
        [Paragraph(sl, ParagraphStyle('sl', fontSize=10, fontName='Helvetica',
                                      textColor=GRAY, alignment=TA_CENTER)), ''],
        ['', ''],
        ['Market Growth',  mkt['growth']],
        ['Market Demand',  mkt['demand']],
        ['Market Avg',     format_rs(mkt['avg'])],
    ]
    inv_t = Table(invest_data, colWidths=[4.5*cm, 4*cm])
    inv_t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), GREEN),
        ('TEXTCOLOR', (0,0), (-1,0), WHITE),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('SPAN', (0,0), (-1,0)),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('FONTSIZE', (0,0), (-1,-1), 9),
        ('SPAN', (0,2), (-1,2)),
        ('SPAN', (0,3), (-1,3)),
        ('FONTNAME', (0,5), (0,-1), 'Helvetica-Bold'),
        ('TEXTCOLOR', (0,5), (0,-1), GREEN),
        ('ROWBACKGROUNDS', (0,5), (-1,-1), [LIGHT, WHITE]),
        ('BOX', (0,0), (-1,-1), 0.5, GREEN),
        ('INNERGRID', (0,5), (-1,-1), 0.3, colors.HexColor('#d4f5d4')),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
    ]))

    combined = Table([[prop_t, inv_t]], colWidths=[8.7*cm, 8.7*cm])
    combined.setStyle(TableStyle([('VALIGN', (0,0), (-1,-1), 'TOP')]))
    story.append(combined)
    story.append(Spacer(1, 0.4*cm))

    story.append(Paragraph(
        "SHAP Explainability — Feature Contributions",
        ParagraphStyle('sh', fontSize=10, fontName='Helvetica-Bold',
                       textColor=GREEN, spaceBefore=8)
    ))
    story.append(Spacer(1, 0.2*cm))
    shap_rows = [['Feature', 'Impact Direction', 'Contribution Amount']]
    for name, val in shap_data:
        shap_rows.append([
            name,
            '⬆ Increases Price' if val >= 0 else '⬇ Decreases Price',
            f"{'+'if val>=0 else '-'}{format_rs(abs(val))}"
        ])
    shap_t = Table(shap_rows, colWidths=[5*cm, 6*cm, 6*cm])
    shap_t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), DARK),
        ('TEXTCOLOR', (0,0), (-1,0), WHITE),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 9),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [LIGHT, WHITE]),
        ('BOX', (0,0), (-1,-1), 0.5, GREEN),
        ('INNERGRID', (0,0), (-1,-1), 0.3, colors.HexColor('#d4f5d4')),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('ALIGN', (2,0), (2,-1), 'RIGHT'),
    ]))
    story.append(shap_t)
    story.append(Spacer(1, 0.4*cm))
    story.append(HRFlowable(width="100%", thickness=1, color=GREEN))
    story.append(Spacer(1, 0.2*cm))
    story.append(Paragraph(
        "This report was generated by an AI-powered explainable system. "
        "Predictions are estimates based on 56,862 Sri Lankan property listings. "
        "Not a substitute for professional valuation.",
        ParagraphStyle('disc', fontSize=7.5, fontName='Helvetica-Oblique',
                       textColor=GRAY, alignment=TA_CENTER)
    ))
    doc.build(story)
    buffer.seek(0)
    return buffer

# ══════════════════════════════════════════════════════════════
# ── HERO ──────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">
        <span class="hero-dot"></span>
        Explainable AI · Random Forest · SHAP
    </div>
    <h1 class="hero-title">
        <span class="line1">Sri Lanka</span>
        <span class="line2">House Price Intelligence</span>
    </h1>
    <p class="hero-sub">
        Sabaragamuwa University of Sri Lanka &nbsp;·&nbsp;
        Faculty of Computing &nbsp;·&nbsp; 20APSE4875
    </p>
</div>
<div class="stats-strip">
    <div class="stat-pill">
        <div class="stat-num">56,862</div>
        <div class="stat-lbl">Properties Trained</div>
    </div>
    <div class="stat-pill">
        <div class="stat-num">82.2%</div>
        <div class="stat-lbl">R² Accuracy</div>
    </div>
    <div class="stat-pill">
        <div class="stat-num">9</div>
        <div class="stat-lbl">Provinces Covered</div>
    </div>
    <div class="stat-pill">
        <div class="stat-num">SHAP</div>
        <div class="stat-lbl">XAI Powered</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ── INPUT FORM ────────────────────────────────────────────────
st.markdown('<div class="card-label">Property Details</div>',
            unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    bedrooms   = st.number_input("Bedrooms",             min_value=1, max_value=20,  value=3)
    house_size = st.number_input("House Size (perches)",  min_value=1, max_value=500, value=3)
with col2:
    bathrooms  = st.number_input("Bathrooms",            min_value=1, max_value=20,  value=2)
    land_size  = st.number_input("Land Size (perches)",   min_value=1, max_value=500, value=10)
with col3:
    property_type = st.selectbox("Property Type",   ["House", "Apartment"])
    verified      = st.selectbox("Verified Seller", ["Yes", "No"])

col4, col5 = st.columns(2)
with col4:
    province_display = st.selectbox("Province", list(province_districts.keys()))
with col5:
    district = st.selectbox("District", province_districts[province_display])

st.markdown("<br>", unsafe_allow_html=True)
predict_btn = st.button("⬡  ANALYSE & PREDICT PRICE")

# ══════════════════════════════════════════════════════════════
# ── RESULTS ───────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════
if predict_btn:
    province_key = province_keys[province_display]
    district_key = district.lower()
    proptype_key = property_type.lower()
    verified_val = 1 if verified == "Yes" else 0
    mkt          = market_data[province_display]

    def safe_encode(encoder, val):
        classes = list(encoder.classes_)
        return encoder.transform([val])[0] if val in classes else 0

    features = np.array([[
        float(bedrooms), float(bathrooms),
        float(house_size), float(land_size),
        safe_encode(encoders['location'], district_key),
        safe_encode(encoders['area'],     district_key),
        safe_encode(encoders['province'], province_key),
        safe_encode(encoders['property_type'], proptype_key),
        float(verified_val)
    ]])

    prediction      = float(model.predict(features)[0])
    range_low       = prediction * 0.85
    range_high      = prediction * 1.15
    inv_score       = get_investment_score(prediction, province_display,
                                           bedrooms, bathrooms,
                                           house_size, land_size, verified_val)
    price_per_perch = prediction / land_size if land_size > 0 else 0

    feature_names = ['Bedrooms','Bathrooms','House Size','Land Size',
                     'Location','Area','Province','Property Type','Verified Seller']
    shap_vals = explainer.shap_values(features)[0]
    shap_data = sorted(zip(feature_names, shap_vals),
                       key=lambda x: abs(x[1]), reverse=True)
    names  = [d[0] for d in shap_data]
    values = [d[1] for d in shap_data]

    st.markdown("---")

    # ── Price Card ────────────────────────────────────────────
    st.markdown(f"""
    <div class="price-result">
        <div class="price-eyebrow">⬡ Estimated Market Value</div>
        <div class="price-main">{format_rs(prediction)}</div>
        <div class="price-range">
            Confidence Range &nbsp;
            <span>{format_rs(range_low)}</span>
            &nbsp;—&nbsp;
            <span>{format_rs(range_high)}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Summary Chips ─────────────────────────────────────────
    st.markdown(f"""
    <div class="metrics-row">
        <div class="metric-chip">
            <div class="metric-val">{int(bedrooms)} / {int(bathrooms)}</div>
            <div class="metric-key">Bed / Bath</div>
        </div>
        <div class="metric-chip">
            <div class="metric-val">{house_size}p</div>
            <div class="metric-key">House Size</div>
        </div>
        <div class="metric-chip">
            <div class="metric-val">{land_size}p</div>
            <div class="metric-key">Land Size</div>
        </div>
        <div class="metric-chip">
            <div class="metric-val">{format_rs(price_per_perch)}</div>
            <div class="metric-key">Price / Perch</div>
        </div>
        <div class="metric-chip">
            <div class="metric-val">{district}</div>
            <div class="metric-key">{province_display}</div>
        </div>
        <div class="metric-chip">
            <div class="metric-val">{property_type}</div>
            <div class="metric-key">Type</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ── 3 Feature Cards ───────────────────────────────────────
    fc1, fc2, fc3 = st.columns(3)

    with fc1:
        sc = get_score_color(inv_score)
        sl = get_score_label(inv_score)
        st.markdown(f"""
        <div class="feature-card">
            <div class="feature-title">— Investment Score</div>
            <div class="score-ring">
                <div class="score-num" style="color:{sc};
                     text-shadow:0 0 20px {sc}80;">{inv_score}</div>
                <div style="font-size:0.7rem;color:{sc};font-weight:600;
                            letter-spacing:1px;margin:4px 0;">/ 10</div>
                <div class="score-label">{sl}</div>
            </div>
            <div style="margin-top:14px;">
                <div class="insight-row">
                    <span class="insight-label">Market Growth</span>
                    <span class="insight-value" style="color:#39ff14;">
                        {mkt['growth']}
                    </span>
                </div>
                <div class="insight-row">
                    <span class="insight-label">Demand Level</span>
                    <span class="insight-value">{mkt['demand']}</span>
                </div>
                <div class="insight-row">
                    <span class="insight-label">vs Market Avg</span>
                    <span class="insight-value"
                        style="color:{'#39ff14' if prediction < mkt['avg'] else '#a8ff78'};">
                        {'Below Avg ✓' if prediction < mkt['avg'] else 'Above Avg'}
                    </span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with fc2:
        st.markdown(f"""
        <div class="feature-card">
            <div class="feature-title">— Market Insights</div>
            <div class="insight-row">
                <span class="insight-label">Province Avg</span>
                <span class="insight-value">{format_rs(mkt['avg'])}</span>
            </div>
            <div class="insight-row">
                <span class="insight-label">Price Range</span>
                <span class="insight-value" style="font-size:0.75rem;">
                    {format_rs(mkt['min'])} – {format_rs(mkt['max'])}
                </span>
            </div>
            <div class="insight-row">
                <span class="insight-label">Your Price</span>
                <span class="insight-value" style="color:#39ff14;">
                    {format_rs(prediction)}
                </span>
            </div>
            <div class="insight-row">
                <span class="insight-label">Price / Perch</span>
                <span class="insight-value">{format_rs(price_per_perch)}</span>
            </div>
            <div class="insight-row">
                <span class="insight-label">Annual Growth</span>
                <span class="insight-value" style="color:#39ff14;">
                    {mkt['growth']}
                </span>
            </div>
            <div class="insight-row">
                <span class="insight-label">Market Demand</span>
                <span class="insight-value">{mkt['demand']}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with fc3:
        down_20    = prediction * 0.20
        loan_80    = prediction * 0.80
        monthly_8  = (loan_80 * 0.08) / 12
        monthly_10 = (loan_80 * 0.10) / 12
        monthly_12 = (loan_80 * 0.12) / 12
        st.markdown(f"""
        <div class="feature-card">
            <div class="feature-title">— Affordability</div>
            <div class="insight-row">
                <span class="insight-label">Property Price</span>
                <span class="insight-value">{format_rs(prediction)}</span>
            </div>
            <div class="insight-row">
                <span class="insight-label">Down Payment 20%</span>
                <span class="insight-value">{format_rs(down_20)}</span>
            </div>
            <div class="insight-row">
                <span class="insight-label">Loan Amount 80%</span>
                <span class="insight-value">{format_rs(loan_80)}</span>
            </div>
            <div style="margin-top:10px;padding-top:10px;
                        border-top:1px solid rgba(0,200,80,0.1);">
                <!-- FIX 6: was #4d8c4d → now #7acc7a -->
                <div style="font-size:0.62rem;color:#7acc7a;font-weight:700;
                            letter-spacing:1.5px;text-transform:uppercase;
                            margin-bottom:10px;font-family:'Courier Prime',monospace;">
                    Monthly Mortgage Est.
                </div>
                <div class="insight-row">
                    <span class="insight-label">@ 8% interest</span>
                    <span class="insight-value" style="color:#39ff14;">
                        {format_rs(monthly_8)}/mo
                    </span>
                </div>
                <div class="insight-row">
                    <span class="insight-label">@ 10% interest</span>
                    <span class="insight-value" style="color:#00c850;">
                        {format_rs(monthly_10)}/mo
                    </span>
                </div>
                <div class="insight-row">
                    <span class="insight-label">@ 12% interest</span>
                    <span class="insight-value" style="color:#a8ff78;">
                        {format_rs(monthly_12)}/mo
                    </span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ── SHAP Chart + Table ────────────────────────────────────
    res_col1, res_col2 = st.columns([3, 2])

    with res_col1:
        st.markdown('<div class="shap-header">Feature Contributions · SHAP Analysis</div>',
                    unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(8, 5))
        fig.patch.set_facecolor('#020c06')
        ax.set_facecolor('#030f07')
        bar_colors = ['#39ff14' if v >= 0 else '#00c850' for v in values]
        bars = ax.barh(names, values, color=bar_colors, height=0.55, edgecolor='none')

        # FIX 7: white labels with dark background box — always visible over bars
        for bar, val in zip(bars, values):
            w      = bar.get_width()
            label  = f"+{format_rs(abs(val))}" if val >= 0 else f"-{format_rs(abs(val))}"
            offset = max(abs(v) for v in values) * 0.04
            x_pos  = w + offset if w >= 0 else w - offset
            ha     = 'left' if w >= 0 else 'right'
            ax.text(x_pos, bar.get_y() + bar.get_height()/2,
                    label, va='center', ha=ha,
                    color='#ffffff', fontsize=9,
                    fontweight='bold', fontfamily='monospace',
                    bbox=dict(boxstyle='round,pad=0.2',
                              facecolor='#020c06',
                              edgecolor='none',
                              alpha=0.85))

        ax.axvline(x=0, color='#39ff14', linewidth=1, linestyle='-', alpha=0.2)

        # FIX 8: axis label was #1a4d1a (nearly invisible) → white
        ax.set_xlabel('SHAP Value — Price Impact (Rs)',
                      color='#ffffff', fontsize=9,
                      fontweight='bold', labelpad=10)

        # FIX 9: x-axis tick numbers were #4d8c4d → white
        ax.tick_params(colors='#ffffff', labelsize=9)

        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        ax.spines['bottom'].set_color('#0a2010')
        ax.spines['left'].set_color('#0a2010')
        ax.invert_yaxis()

        # FIX 10: y-axis feature names — keep neon green, make bold
        ax.yaxis.set_tick_params(labelcolor='#39ff14', labelsize=9)
        for lbl in ax.get_yticklabels():
            lbl.set_fontweight('bold')

        legend_els = [
            mpatches.Patch(color='#39ff14', label='↑ Increases Price'),
            mpatches.Patch(color='#00c850', label='↓ Decreases Price')
        ]
        # FIX 11: legend text was #4d8c4d → white
        ax.legend(handles=legend_els, facecolor='#030f07',
                  labelcolor='#ffffff', fontsize=8,
                  framealpha=0.9, edgecolor='#0a2010',
                  prop={'weight': 'bold'})
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with res_col2:
        st.markdown('<div class="shap-header">Contribution Breakdown</div>',
                    unsafe_allow_html=True)
        shap_df = pd.DataFrame({
            'Feature': names,
            'Impact':  ['⬆ Adds' if v >= 0 else '⬇ Reduces' for v in values],
            'Amount':  [f"{'+'if v>=0 else '-'}{format_rs(abs(v))}" for v in values]
        })
        st.dataframe(shap_df, use_container_width=True, hide_index=True, height=320)

        top_name = names[0]
        top_val  = values[0]
        top_dir  = "increases" if top_val >= 0 else "decreases"
        top_col  = "#39ff14" if top_val >= 0 else "#00c850"
        # FIX 12: "Top Price Driver" label was #1a4d1a, body was #4d8c4d → both #7acc7a
        st.markdown(f"""
        <div style="background:rgba(57,255,20,0.04);
                    border:1px solid rgba(57,255,20,0.15);
                    border-radius:12px;padding:16px;margin-top:12px;">
            <div style="font-family:'Courier Prime',monospace;font-size:0.6rem;
                        letter-spacing:2px;color:#7acc7a;font-weight:700;
                        text-transform:uppercase;margin-bottom:8px;">Top Price Driver</div>
            <div style="font-family:'Orbitron',sans-serif;font-size:1rem;
                        font-weight:700;color:{top_col};
                        text-shadow:0 0 10px {top_col}60;">{top_name}</div>
            <div style="font-size:0.8rem;color:#7acc7a;font-weight:700;margin-top:6px;">
                {top_dir.capitalize()} price by
                <strong style="color:{top_col};">
                    {format_rs(abs(top_val))}
                </strong>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ── PDF Export ────────────────────────────────────────────
    st.markdown('<div class="card-label">Export Report</div>',
                unsafe_allow_html=True)
    pdf_buffer = generate_pdf_report(
        prediction, range_low, range_high,
        province_display, district,
        bedrooms, bathrooms, house_size, land_size,
        property_type, verified_val, inv_score,
        list(shap_data), mkt
    )
    st.download_button(
        label="⬇  DOWNLOAD PDF REPORT",
        data=pdf_buffer,
        file_name=f"HousePrice_{district}_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
        mime="application/pdf"
    )

# ── Footer ────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div class="app-footer">
    Akhiliny Vijeyagumar &nbsp;·&nbsp; 20APSE4875 &nbsp;·&nbsp;
    BSc (Hons) Software Engineering &nbsp;·&nbsp;
    Sabaragamuwa University of Sri Lanka &nbsp;·&nbsp; 2026
</div>
""", unsafe_allow_html=True)