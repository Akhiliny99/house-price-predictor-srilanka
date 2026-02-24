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
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;500;600;700&family=Share+Tech+Mono&family=Exo+2:wght@300;400;500;600;700&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, .stApp {
    font-family: 'Exo 2', sans-serif;
    background: #050510 !important;
    color: #e8e0f5;
    min-height: 100vh;
}

.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background:
        radial-gradient(ellipse 70% 60% at 15% 10%, rgba(108,43,217,0.35) 0%, transparent 55%),
        radial-gradient(ellipse 50% 50% at 85% 85%, rgba(6,182,212,0.25) 0%, transparent 55%),
        radial-gradient(ellipse 40% 40% at 50% 50%, rgba(139,92,246,0.1) 0%, transparent 60%),
        radial-gradient(ellipse 30% 30% at 80% 20%, rgba(34,211,238,0.15) 0%, transparent 50%);
    pointer-events: none;
    z-index: 0;
}

.stApp::after {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
        linear-gradient(rgba(139,92,246,0.04) 1px, transparent 1px),
        linear-gradient(90deg, rgba(139,92,246,0.04) 1px, transparent 1px);
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

.hero { text-align: center; padding: 3rem 0 2rem; }

.hero-eyebrow {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(139,92,246,0.12);
    border: 1px solid rgba(139,92,246,0.3);
    border-radius: 100px;
    padding: 6px 18px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #a78bfa;
    margin-bottom: 1.5rem;
}

.hero-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: #06b6d4;
    animation: blink 2s ease-in-out infinite;
    display: inline-block;
}

@keyframes blink {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.2; }
}

.hero-title {
    font-family: 'Rajdhani', sans-serif;
    font-size: clamp(2.5rem, 5vw, 4rem);
    font-weight: 700;
    line-height: 1.05;
    letter-spacing: -1px;
    margin-bottom: 1rem;
}

.hero-title .line1 { color: #f0e6ff; display: block; }
.hero-title .line2 {
    display: block;
    background: linear-gradient(135deg, #a78bfa 0%, #06b6d4 50%, #34d399 100%);
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

.hero-sub { font-size: 0.95rem; color: #7c6b9e; font-weight: 300; }

.stats-strip {
    display: flex;
    justify-content: center;
    gap: 2rem;
    margin: 2rem 0;
    flex-wrap: wrap;
}

.stat-pill {
    background: rgba(139,92,246,0.08);
    border: 1px solid rgba(139,92,246,0.2);
    border-radius: 12px;
    padding: 10px 20px;
    text-align: center;
}

.stat-num {
    font-family: 'Rajdhani', sans-serif;
    font-size: 1.3rem;
    font-weight: 700;
    color: #06b6d4;
}

.stat-lbl { font-size: 0.7rem; color: #7c6b9e; text-transform: uppercase; letter-spacing: 1px; }

.glass-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(139,92,246,0.15);
    border-radius: 20px;
    padding: 28px;
    margin-bottom: 20px;
    backdrop-filter: blur(20px);
    position: relative;
    overflow: hidden;
}

.glass-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent,
        rgba(139,92,246,0.6) 30%, rgba(6,182,212,0.6) 70%, transparent);
}

.card-label {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #06b6d4;
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    gap: 8px;
}

.card-label::before {
    content: '';
    display: inline-block;
    width: 20px; height: 1px;
    background: #06b6d4;
}

div[data-testid="stSelectbox"] > label,
div[data-testid="stNumberInput"] > label {
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.65rem !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    color: #7c6b9e !important;
    font-weight: 400 !important;
}

div[data-testid="stSelectbox"] > div > div,
div[data-testid="stNumberInput"] input {
    background: rgba(5,5,16,0.7) !important;
    border: 1px solid rgba(139,92,246,0.2) !important;
    border-radius: 10px !important;
    color: #e8e0f5 !important;
    font-family: 'Exo 2', sans-serif !important;
}

.stButton > button {
    width: 100% !important;
    background: linear-gradient(135deg, #6c2bd9, #0891b2) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 16px 24px !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 1.1rem !important;
    font-weight: 700 !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    transition: all 0.3s ease !important;
    margin-top: 8px !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 12px 40px rgba(108,43,217,0.5) !important;
}

.price-result {
    background: linear-gradient(135deg,
        rgba(108,43,217,0.2) 0%, rgba(6,182,212,0.15) 100%);
    border: 1px solid rgba(6,182,212,0.3);
    border-radius: 20px;
    padding: 36px;
    text-align: center;
    position: relative;
    overflow: hidden;
    margin: 10px 0 24px;
}

.price-eyebrow {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 4px;
    text-transform: uppercase;
    color: #06b6d4;
    margin-bottom: 12px;
}

.price-main {
    font-family: 'Rajdhani', sans-serif;
    font-size: clamp(2.5rem, 6vw, 4.5rem);
    font-weight: 700;
    letter-spacing: -1px;
    background: linear-gradient(135deg, #c4b5fd, #06b6d4, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1;
    margin-bottom: 12px;
}

.price-range { font-size: 0.82rem; color: #7c6b9e; }
.price-range span { color: #a78bfa; font-weight: 500; }

.metrics-row {
    display: flex;
    gap: 12px;
    margin: 16px 0;
    flex-wrap: wrap;
}

.metric-chip {
    flex: 1;
    min-width: 110px;
    background: rgba(139,92,246,0.08);
    border: 1px solid rgba(139,92,246,0.2);
    border-radius: 12px;
    padding: 12px 16px;
    text-align: center;
}

.metric-val {
    font-family: 'Rajdhani', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: #a78bfa;
}

.metric-key { font-size: 0.68rem; color: #7c6b9e; text-transform: uppercase; letter-spacing: 1px; margin-top: 2px; }

.shap-header {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #a78bfa;
    margin: 24px 0 16px;
    display: flex;
    align-items: center;
    gap: 10px;
}

.shap-header::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, rgba(139,92,246,0.3), transparent);
}

.feature-card {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(139,92,246,0.12);
    border-radius: 16px;
    padding: 20px;
    margin: 10px 0;
}

.feature-title {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #06b6d4;
    margin-bottom: 14px;
}

.insight-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 0;
    border-bottom: 1px solid rgba(139,92,246,0.08);
}

.insight-label { font-size: 0.82rem; color: #7c6b9e; }
.insight-value { font-family: 'Rajdhani', sans-serif; font-size: 1rem; font-weight: 600; color: #e8e0f5; }

.score-ring {
    text-align: center;
    padding: 10px;
}

.score-num {
    font-family: 'Rajdhani', sans-serif;
    font-size: 3.5rem;
    font-weight: 700;
    line-height: 1;
}

.score-label { font-size: 0.75rem; color: #7c6b9e; text-transform: uppercase; letter-spacing: 1px; }

.app-footer {
    text-align: center;
    padding: 30px 0 10px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 2px;
    color: #a78bfa;
    text-transform: uppercase;
    opacity: 1;
}

hr {
    border: none !important;
    height: 1px !important;
    background: linear-gradient(90deg, transparent,
        rgba(139,92,246,0.3) 30%, rgba(6,182,212,0.3) 70%, transparent) !important;
    margin: 24px 0 !important;
}

#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Load Model ────────────────────────────────────────────────
BASE = r'C:\Users\Akhiliny Vijeyagumar\OneDrive\Desktop\20APSE4875 Akhiliny Research\System Building'

@st.cache_resource
def load_assets():
    with open(f'{BASE}\\best_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open(f'{BASE}\\encoders.pkl', 'rb') as f:
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

# ── Market Data (based on your real dataset findings) ─────────
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

def format_rs(val):
    val = round(val)
    if val >= 1e6: return f"Rs {val/1e6:.1f}M"
    if val >= 1e3: return f"Rs {val/1e3:.0f}K"
    return f"Rs {val:,}"

def get_investment_score(prediction, province, bedrooms, bathrooms, house_size, land_size, verified):
    score = 5.0
    mkt   = market_data[province]

    # Location demand
    demand_scores = {'Very High': 2.0, 'High': 1.5, 'Moderate': 0.5, 'Low': 0.0}
    score += demand_scores.get(mkt['demand'], 0)

    # Price vs market avg
    if prediction < mkt['avg'] * 0.85:  score += 1.5  # Below market = good buy
    elif prediction < mkt['avg']:        score += 0.8
    elif prediction > mkt['avg'] * 1.2: score -= 1.0  # Overpriced

    # Property features
    if bathrooms >= bedrooms:   score += 0.5
    if land_size > house_size:  score += 0.5
    if verified:                score += 0.3
    if bedrooms >= 3:           score += 0.2

    # Growth bonus
    growth_val = float(mkt['growth'].replace('%','').replace('+',''))
    score += growth_val / 10

    return min(10.0, max(1.0, round(score, 1)))

def get_score_color(score):
    if score >= 8: return '#34d399'
    if score >= 6: return '#06b6d4'
    if score >= 4: return '#f59e0b'
    return '#ef4444'

def get_score_label(score):
    if score >= 8: return 'Excellent Investment'
    if score >= 6: return 'Good Investment'
    if score >= 4: return 'Fair Investment'
    return 'Below Average'

# ── Generate PDF Report ───────────────────────────────────────
def generate_pdf_report(prediction, range_low, range_high,
                         province, district, bedrooms, bathrooms,
                         house_size, land_size, property_type,
                         verified, inv_score, shap_data, mkt):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4,
                            rightMargin=2*cm, leftMargin=2*cm,
                            topMargin=2*cm, bottomMargin=2*cm)

    PURPLE = colors.HexColor('#7c3aed')
    TEAL   = colors.HexColor('#0f766e')
    DARK   = colors.HexColor('#1e1b4b')
    LIGHT  = colors.HexColor('#f3e8ff')
    GRAY   = colors.HexColor('#6b7280')
    WHITE  = colors.white

    styles = getSampleStyleSheet()
    story  = []

    # Header
    header_data = [[Paragraph(
        "<b>SABARAGAMUWA UNIVERSITY OF SRI LANKA</b><br/>"
        "Sri Lanka House Price Prediction System — AI Generated Report",
        ParagraphStyle('H', fontSize=12, fontName='Helvetica-Bold',
                       textColor=WHITE, alignment=TA_CENTER)
    )]]
    ht = Table(header_data, colWidths=[17*cm])
    ht.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), DARK),
        ('TOPPADDING', (0,0), (-1,-1), 14),
        ('BOTTOMPADDING', (0,0), (-1,-1), 14),
    ]))
    story.append(ht)
    story.append(Spacer(1, 0.4*cm))

    # Date & ID
    story.append(Paragraph(
        f"Report Generated: {datetime.datetime.now().strftime('%d %B %Y, %H:%M')} &nbsp;|&nbsp; Student: Akhiliny Vijeyagumar &nbsp;|&nbsp; Index: 20APSE4875",
        ParagraphStyle('meta', fontSize=8, fontName='Helvetica',
                       textColor=GRAY, alignment=TA_CENTER)
    ))
    story.append(Spacer(1, 0.5*cm))

    # Predicted Price Box
    price_data = [[
        Paragraph("PREDICTED PROPERTY VALUE",
                  ParagraphStyle('pl', fontSize=8, fontName='Helvetica',
                                 textColor=TEAL, alignment=TA_CENTER)),
    ],[
        Paragraph(format_rs(prediction),
                  ParagraphStyle('pv', fontSize=28, fontName='Helvetica-Bold',
                                 textColor=PURPLE, alignment=TA_CENTER)),
    ],[
        Paragraph(f"Confidence Range: {format_rs(range_low)} — {format_rs(range_high)}",
                  ParagraphStyle('pr', fontSize=9, fontName='Helvetica',
                                 textColor=GRAY, alignment=TA_CENTER)),
    ]]
    pt = Table(price_data, colWidths=[17*cm])
    pt.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), LIGHT),
        ('BOX', (0,0), (-1,-1), 1.5, PURPLE),
        ('TOPPADDING', (0,0), (-1,-1), 10),
        ('BOTTOMPADDING', (0,0), (-1,-1), 10),
        ('ROUNDEDCORNERS', [8,8,8,8]),
    ]))
    story.append(pt)
    story.append(Spacer(1, 0.4*cm))

    # Property Details + Investment Score side by side
    prop_details = [
        ['Property Type', property_type],
        ['Province',      province],
        ['District',      district],
        ['Bedrooms',      str(bedrooms)],
        ['Bathrooms',     str(bathrooms)],
        ['House Size',    f'{house_size} perches'],
        ['Land Size',     f'{land_size} perches'],
        ['Price/perch',   format_rs(prediction / land_size if land_size > 0 else 0)],
        ['Verified Seller', 'Yes' if verified else 'No'],
    ]

    prop_table_data = [['Property Details', '']] + prop_details
    prop_t = Table(prop_table_data, colWidths=[4*cm, 4.5*cm])
    prop_t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), DARK),
        ('TEXTCOLOR', (0,0), (-1,0), WHITE),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('SPAN', (0,0), (-1,0)),
        ('ALIGN', (0,0), (-1,0), 'CENTER'),
        ('FONTSIZE', (0,0), (-1,-1), 9),
        ('FONTNAME', (0,1), (0,-1), 'Helvetica-Bold'),
        ('TEXTCOLOR', (0,1), (0,-1), PURPLE),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [LIGHT, WHITE]),
        ('BOX', (0,0), (-1,-1), 0.5, PURPLE),
        ('INNERGRID', (0,0), (-1,-1), 0.3, colors.HexColor('#e5e7eb')),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
    ]))

    score_color_map = {
        '#34d399': colors.HexColor('#34d399'),
        '#06b6d4': colors.HexColor('#06b6d4'),
        '#f59e0b': colors.HexColor('#f59e0b'),
        '#ef4444': colors.HexColor('#ef4444'),
    }
    sc = get_score_color(inv_score)
    sl = get_score_label(inv_score)

    invest_data = [
        ['Investment Score', ''],
        ['', ''],
        [Paragraph(f"<b>{inv_score}/10</b>",
                   ParagraphStyle('sc', fontSize=28, fontName='Helvetica-Bold',
                                  textColor=score_color_map.get(sc, TEAL),
                                  alignment=TA_CENTER)), ''],
        [Paragraph(sl, ParagraphStyle('sl', fontSize=10, fontName='Helvetica',
                                      textColor=GRAY, alignment=TA_CENTER)), ''],
        ['', ''],
        ['Market Growth', mkt['growth']],
        ['Market Demand', mkt['demand']],
        ['Market Avg',    format_rs(mkt['avg'])],
    ]
    inv_t = Table(invest_data, colWidths=[4.5*cm, 4*cm])
    inv_t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), TEAL),
        ('TEXTCOLOR', (0,0), (-1,0), WHITE),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('SPAN', (0,0), (-1,0)),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('FONTSIZE', (0,0), (-1,-1), 9),
        ('SPAN', (0,2), (-1,2)),
        ('SPAN', (0,3), (-1,3)),
        ('FONTNAME', (0,5), (0,-1), 'Helvetica-Bold'),
        ('TEXTCOLOR', (0,5), (0,-1), TEAL),
        ('ROWBACKGROUNDS', (0,5), (-1,-1), [LIGHT, WHITE]),
        ('BOX', (0,0), (-1,-1), 0.5, TEAL),
        ('INNERGRID', (0,5), (-1,-1), 0.3, colors.HexColor('#e5e7eb')),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
    ]))

    combined = Table([[prop_t, inv_t]], colWidths=[8.7*cm, 8.7*cm])
    combined.setStyle(TableStyle([('VALIGN', (0,0), (-1,-1), 'TOP')]))
    story.append(combined)
    story.append(Spacer(1, 0.4*cm))

    # SHAP Table
    story.append(Paragraph("SHAP Explainability — Feature Contributions",
                            ParagraphStyle('sh', fontSize=10,
                                           fontName='Helvetica-Bold',
                                           textColor=PURPLE, spaceBefore=8)))
    story.append(Spacer(1, 0.2*cm))

    shap_rows = [['Feature', 'Impact Direction', 'Contribution Amount']]
    for name, val in shap_data:
        direction = '⬆ Increases Price' if val >= 0 else '⬇ Decreases Price'
        amount    = f"{'+'if val>=0 else '-'}{format_rs(abs(val))}"
        shap_rows.append([name, direction, amount])

    shap_t = Table(shap_rows, colWidths=[5*cm, 6*cm, 6*cm])
    shap_t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), DARK),
        ('TEXTCOLOR', (0,0), (-1,0), WHITE),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 9),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [LIGHT, WHITE]),
        ('BOX', (0,0), (-1,-1), 0.5, PURPLE),
        ('INNERGRID', (0,0), (-1,-1), 0.3, colors.HexColor('#e5e7eb')),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('ALIGN', (2,0), (2,-1), 'RIGHT'),
    ]))
    story.append(shap_t)
    story.append(Spacer(1, 0.4*cm))

    # Footer
    story.append(HRFlowable(width="100%", thickness=1, color=PURPLE))
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
# ── HERO SECTION ──────────────────────────────────────────────
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
    <div class="stat-pill"><div class="stat-num">56,862</div><div class="stat-lbl">Properties Trained</div></div>
    <div class="stat-pill"><div class="stat-num">82.2%</div><div class="stat-lbl">R² Accuracy</div></div>
    <div class="stat-pill"><div class="stat-num">9</div><div class="stat-lbl">Provinces Covered</div></div>
    <div class="stat-pill"><div class="stat-num">SHAP</div><div class="stat-lbl">XAI Powered</div></div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ── INPUT FORM ────────────────────────────────────────────────
st.markdown('<div class="card-label">— Property Details</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    bedrooms   = st.number_input("Bedrooms",             min_value=1, max_value=20,  value=3)
    house_size = st.number_input("House Size (perches)",  min_value=1, max_value=500, value=3)
with col2:
    bathrooms  = st.number_input("Bathrooms",            min_value=1, max_value=20,  value=2)
    land_size  = st.number_input("Land Size (perches)",   min_value=1, max_value=500, value=10)
with col3:
    property_type    = st.selectbox("Property Type",    ["House", "Apartment"])
    verified         = st.selectbox("Verified Seller",  ["Yes", "No"])

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

    prediction = float(model.predict(features)[0])
    range_low  = prediction * 0.85
    range_high = prediction * 1.15
    inv_score  = get_investment_score(prediction, province_display,
                                      bedrooms, bathrooms,
                                      house_size, land_size,
                                      verified_val)
    price_per_perch = prediction / land_size if land_size > 0 else 0

    feature_names = ['Bedrooms','Bathrooms','House Size','Land Size',
                     'Location','Area','Province','Property Type','Verified Seller']
    shap_vals = explainer.shap_values(features)[0]
    shap_data = sorted(zip(feature_names, shap_vals),
                       key=lambda x: abs(x[1]), reverse=True)
    names  = [d[0] for d in shap_data]
    values = [d[1] for d in shap_data]

    st.markdown("---")

    # ── 1. PRICE RESULT ───────────────────────────────────────
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

    # ── 2. PROPERTY SUMMARY CHIPS ─────────────────────────────
    st.markdown(f"""
    <div class="metrics-row">
        <div class="metric-chip">
            <div class="metric-val">{int(bedrooms)} bd / {int(bathrooms)} ba</div>
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

    # ── 3. THREE COLUMN FEATURES ──────────────────────────────
    fc1, fc2, fc3 = st.columns(3)

    # ── 3a. Investment Score ──────────────────────────────────
    with fc1:
        sc = get_score_color(inv_score)
        sl = get_score_label(inv_score)
        st.markdown(f"""
        <div class="feature-card">
            <div class="feature-title">— Investment Score</div>
            <div class="score-ring">
                <div class="score-num" style="color:{sc};">{inv_score}</div>
                <div style="font-size:0.7rem;color:{sc};font-weight:600;
                            letter-spacing:1px;margin:4px 0;">/ 10</div>
                <div class="score-label">{sl}</div>
            </div>
            <div style="margin-top:14px;">
                <div class="insight-row">
                    <span class="insight-label">Market Growth</span>
                    <span class="insight-value" style="color:#34d399;">
                        {mkt['growth']}
                    </span>
                </div>
                <div class="insight-row">
                    <span class="insight-label">Demand Level</span>
                    <span class="insight-value">{mkt['demand']}</span>
                </div>
                <div class="insight-row">
                    <span class="insight-label">vs Market Avg</span>
                    <span class="insight-value" style="color:{'#34d399' if prediction < mkt['avg'] else '#f59e0b'};">
                        {'Below Avg ✓' if prediction < mkt['avg'] else 'Above Avg'}
                    </span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── 3b. Market Insights ───────────────────────────────────
    with fc2:
        mortgage_monthly = (prediction * 0.8 * 0.08) / 12
        st.markdown(f"""
        <div class="feature-card">
            <div class="feature-title">— Market Insights</div>
            <div class="insight-row">
                <span class="insight-label">Province Avg Price</span>
                <span class="insight-value">{format_rs(mkt['avg'])}</span>
            </div>
            <div class="insight-row">
                <span class="insight-label">Price Range</span>
                <span class="insight-value" style="font-size:0.85rem;">
                    {format_rs(mkt['min'])} – {format_rs(mkt['max'])}
                </span>
            </div>
            <div class="insight-row">
                <span class="insight-label">Your Price</span>
                <span class="insight-value" style="color:#a78bfa;">
                    {format_rs(prediction)}
                </span>
            </div>
            <div class="insight-row">
                <span class="insight-label">Price / Perch</span>
                <span class="insight-value">{format_rs(price_per_perch)}</span>
            </div>
            <div class="insight-row">
                <span class="insight-label">Annual Growth</span>
                <span class="insight-value" style="color:#34d399;">{mkt['growth']}</span>
            </div>
            <div class="insight-row">
                <span class="insight-label">Market Demand</span>
                <span class="insight-value">{mkt['demand']}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── 3c. Affordability Calculator ─────────────────────────
    with fc3:
        down_20   = prediction * 0.20
        loan_80   = prediction * 0.80
        monthly_8 = (loan_80 * 0.08) / 12
        monthly_10= (loan_80 * 0.10) / 12
        monthly_12= (loan_80 * 0.12) / 12
        st.markdown(f"""
        <div class="feature-card">
            <div class="feature-title">— Affordability Calculator</div>
            <div class="insight-row">
                <span class="insight-label">Property Price</span>
                <span class="insight-value">{format_rs(prediction)}</span>
            </div>
            <div class="insight-row">
                <span class="insight-label">Down Payment (20%)</span>
                <span class="insight-value">{format_rs(down_20)}</span>
            </div>
            <div class="insight-row">
                <span class="insight-label">Loan Amount (80%)</span>
                <span class="insight-value">{format_rs(loan_80)}</span>
            </div>
            <div style="margin-top:10px;padding-top:10px;
                        border-top:1px solid rgba(139,92,246,0.15);">
                <div style="font-size:0.65rem;color:#7c6b9e;
                            letter-spacing:1px;text-transform:uppercase;
                            margin-bottom:8px;">Monthly Mortgage Est.</div>
                <div class="insight-row">
                    <span class="insight-label">@ 8% interest</span>
                    <span class="insight-value" style="color:#34d399;">
                        {format_rs(monthly_8)}/mo
                    </span>
                </div>
                <div class="insight-row">
                    <span class="insight-label">@ 10% interest</span>
                    <span class="insight-value" style="color:#06b6d4;">
                        {format_rs(monthly_10)}/mo
                    </span>
                </div>
                <div class="insight-row">
                    <span class="insight-label">@ 12% interest</span>
                    <span class="insight-value" style="color:#f59e0b;">
                        {format_rs(monthly_12)}/mo
                    </span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ── 4. SHAP CHART + TABLE ─────────────────────────────────
    res_col1, res_col2 = st.columns([3, 2])

    with res_col1:
        st.markdown('<div class="shap-header">Feature Contributions · SHAP Analysis</div>',
                    unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(8, 5))
        fig.patch.set_facecolor('#050510')
        ax.set_facecolor('#0a0a1a')

        bar_colors = ['#06b6d4' if v >= 0 else '#a78bfa' for v in values]
        bars = ax.barh(names, values, color=bar_colors, height=0.6, edgecolor='none')

        for bar, val in zip(bars, values):
            w     = bar.get_width()
            label = f"+{format_rs(abs(val))}" if val >= 0 else f"-{format_rs(abs(val))}"
            offset = max(abs(v) for v in values) * 0.02
            x_pos = w + offset if w >= 0 else w - offset
            ha    = 'left' if w >= 0 else 'right'
            ax.text(x_pos, bar.get_y() + bar.get_height()/2,
                    label, va='center', ha=ha,
                    color='#7c6b9e', fontsize=8, fontfamily='monospace')

        ax.axvline(x=0, color='#8b5cf6', linewidth=1, linestyle='-', alpha=0.3)
        ax.set_xlabel('SHAP Value — Price Impact (Rs)',
                      color='#3d3556', fontsize=9, labelpad=10)
        ax.tick_params(colors='#7c6b9e', labelsize=9)
        for spine in ['top','right']:
            ax.spines[spine].set_visible(False)
        ax.spines['bottom'].set_color('#1a1030')
        ax.spines['left'].set_color('#1a1030')
        ax.invert_yaxis()
        ax.yaxis.set_tick_params(labelcolor='#a78bfa', labelsize=9)

        legend_els = [
            mpatches.Patch(color='#06b6d4', label='↑ Increases Price'),
            mpatches.Patch(color='#a78bfa', label='↓ Decreases Price')
        ]
        ax.legend(handles=legend_els, facecolor='#0a0a1a',
                  labelcolor='#7c6b9e', fontsize=8,
                  framealpha=0.8, edgecolor='#1a1030')
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
        top_col  = "#06b6d4" if top_val >= 0 else "#a78bfa"
        st.markdown(f"""
        <div style="background:rgba(6,182,212,0.06);border:1px solid
                    rgba(6,182,212,0.2);border-radius:12px;padding:16px;margin-top:12px;">
            <div style="font-family:'Share Tech Mono',monospace;font-size:0.6rem;
                        letter-spacing:2px;color:#3d3556;text-transform:uppercase;
                        margin-bottom:8px;">Top Price Driver</div>
            <div style="font-family:'Rajdhani',sans-serif;font-size:1rem;
                        font-weight:700;color:{top_col};">{top_name}</div>
            <div style="font-size:0.8rem;color:#7c6b9e;margin-top:4px;">
                {top_dir.capitalize()} price by
                <strong style="color:{top_col};">{format_rs(abs(top_val))}</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ── 5. PDF EXPORT ─────────────────────────────────────────
    st.markdown('<div class="card-label">— Export Report</div>', unsafe_allow_html=True)

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

# ── FOOTER ────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div class="app-footer">
    Akhiliny Vijeyagumar &nbsp;·&nbsp; 20APSE4875 &nbsp;·&nbsp;
    BSc (Hons) Software Engineering &nbsp;·&nbsp;
    Sabaragamuwa University of Sri Lanka &nbsp;·&nbsp; 2026
</div>
""", unsafe_allow_html=True)