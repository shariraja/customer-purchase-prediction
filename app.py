import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from datetime import datetime
from groq import Groq

# ════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="PurchaseIQ — AI Marketing Engine",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ════════════════════════════════════════════════════════════════
# GROQ CLIENT
# ════════════════════════════════════════════════════════════════
def get_groq_client():
    try:
        return Groq(api_key=st.secrets["GROQ_API_KEY"])
    except Exception:
        return None

groq_client = get_groq_client()

def groq_chat(prompt, max_tokens=300, temperature=0.85):
    try:
        client = get_groq_client()
        if not client:
            return "Error: API key not found."
        resp = client.chat.completions.create(
            model       = "llama-3.1-8b-instant",
            messages    = [{"role": "user", "content": prompt}],
            max_tokens  = max_tokens,
            temperature = temperature,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {str(e)}"

# ════════════════════════════════════════════════════════════════
# LOAD MODEL
# ════════════════════════════════════════════════════════════════
@st.cache_resource
def load_model():
    try:
        return joblib.load("rf_model.pkl")
    except Exception:
        return None

model = load_model()

# ════════════════════════════════════════════════════════════════
# SESSION STATE INIT — Variables save rahein between clicks
# ════════════════════════════════════════════════════════════════
if 'batch_done'    not in st.session_state: st.session_state.batch_done    = False
if 'probs'         not in st.session_state: st.session_state.probs         = None
if 'preds'         not in st.session_state: st.session_state.preds         = None
if 'res_df'        not in st.session_state: st.session_state.res_df        = None
if 'df_batch'      not in st.session_state: st.session_state.df_batch      = None
if 'high'          not in st.session_state: st.session_state.high          = 0
if 'medium'        not in st.session_state: st.session_state.medium        = 0
if 'low'           not in st.session_state: st.session_state.low           = 0
if 't_rev'         not in st.session_state: st.session_state.t_rev         = 0
if 'disc_cost'     not in st.session_state: st.session_state.disc_cost     = 0
if 'msgs_out'      not in st.session_state: st.session_state.msgs_out      = []
if 'report_text'   not in st.session_state: st.session_state.report_text   = ""
if 'msgs_done'     not in st.session_state: st.session_state.msgs_done     = False
if 'report_done'   not in st.session_state: st.session_state.report_done   = False
if 'ind_done'      not in st.session_state: st.session_state['ind_done']   = False
if 'ind_pred'      not in st.session_state: st.session_state['ind_pred']   = 0
if 'ind_age'       not in st.session_state: st.session_state['ind_age']    = 30
if 'ind_visitor'   not in st.session_state: st.session_state['ind_visitor']= "Returning Visitor"
if 'ind_product'   not in st.session_state: st.session_state['ind_product']= 20
if 'ind_page_val'  not in st.session_state: st.session_state['ind_page_val']= 10.0
if 'ind_exit'      not in st.session_state: st.session_state['ind_exit']   = 0.05
if 'ind_bounce'    not in st.session_state: st.session_state['ind_bounce'] = 0.02
if 'ind_order'     not in st.session_state: st.session_state['ind_order']  = 3000
if 'ind_last_purchase' not in st.session_state: st.session_state['ind_last_purchase'] = 30
if 'ind_msg'       not in st.session_state: st.session_state['ind_msg']    = ""
if 'ind_rep'       not in st.session_state: st.session_state['ind_rep']    = ""
if 'ind_prob'      not in st.session_state: st.session_state['ind_prob']   = None
if 'ind_act'       not in st.session_state: st.session_state['ind_act']    = ""

# ════════════════════════════════════════════════════════════════
# GLOBAL CSS
# ════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
    --bg:           #f0f2f8;
    --surface:      #ffffff;
    --surface-2:    #f8f9fc;
    --border:       #e2e5f0;
    --border-2:     #c7cde3;
    --accent:       #2351d4;
    --accent-light: #eff3ff;
    --accent-glow:  rgba(35,81,212,0.15);
    --success:      #0d9e6e;
    --success-bg:   #ecfdf5;
    --warning:      #c77700;
    --warning-bg:   #fffbeb;
    --danger:       #d13030;
    --danger-bg:    #fff1f1;
    --text-1:       #0d1117;
    --text-2:       #424b63;
    --text-3:       #8892a4;
    --radius:       16px;
    --radius-sm:    10px;
    --shadow-sm:    0 1px 2px rgba(0,0,0,.04), 0 2px 8px rgba(0,0,0,.04);
    --shadow:       0 2px 4px rgba(0,0,0,.04), 0 8px 24px rgba(0,0,0,.06);
    --shadow-md:    0 4px 8px rgba(0,0,0,.05), 0 16px 40px rgba(0,0,0,.08);
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    background-color: var(--bg) !important;
    color: var(--text-1) !important;
}
* { box-sizing: border-box; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.75rem 2.5rem 4rem !important; max-width: 1380px !important; }

[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text-1) !important; }
[data-testid="stSidebar"] .stMarkdown h3 {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: .68rem !important; font-weight: 600 !important;
    letter-spacing: .12em !important; text-transform: uppercase !important;
    color: var(--text-3) !important; margin: 1.5rem 0 .6rem !important;
    padding-bottom: .4rem !important; border-bottom: 1px solid var(--border) !important;
}

[data-testid="stSlider"] > div > div > div { background: var(--accent) !important; }
[data-testid="stNumberInput"] input {
    background: var(--surface-2) !important; border: 1.5px solid var(--border) !important;
    border-radius: var(--radius-sm) !important; color: var(--text-1) !important;
    font-family: 'JetBrains Mono', monospace !important; font-size: .85rem !important;
}
[data-testid="stNumberInput"] input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(35,81,212,.12) !important;
}
div[data-baseweb="select"] > div {
    background: var(--surface-2) !important; border: 1.5px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
}

[data-testid="stMetric"] {
    background: var(--surface) !important; border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important; padding: 1.25rem 1.4rem !important;
    box-shadow: var(--shadow-sm) !important;
    transition: box-shadow .25s ease, transform .25s ease !important;
    position: relative !important; overflow: hidden !important;
}
[data-testid="stMetric"]::before {
    content: '' !important; position: absolute !important;
    top: 0 !important; left: 0 !important;
    width: 3px !important; height: 100% !important;
    background: var(--accent) !important; border-radius: 3px 0 0 3px !important;
}
[data-testid="stMetric"]:hover {
    box-shadow: var(--shadow-md) !important; transform: translateY(-3px) !important;
}
[data-testid="stMetricLabel"] {
    font-size: .68rem !important; font-weight: 600 !important;
    letter-spacing: .1em !important; text-transform: uppercase !important;
    color: var(--text-3) !important; font-family: 'JetBrains Mono', monospace !important;
}
[data-testid="stMetricValue"] {
    font-family: 'Syne', sans-serif !important; font-size: 1.75rem !important;
    font-weight: 800 !important; color: var(--text-1) !important;
    letter-spacing: -.02em !important;
}

.stTabs [data-baseweb="tab-list"] {
    background: var(--surface); border: 1.5px solid var(--border);
    border-radius: var(--radius-sm); padding: 4px; gap: 2px;
    box-shadow: var(--shadow-sm);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 7px; font-family: 'Inter', sans-serif;
    font-weight: 500; font-size: .88rem; color: var(--text-2);
    padding: .5rem 1.25rem; transition: all .2s !important;
}
.stTabs [aria-selected="true"] {
    background: var(--accent) !important; color: #ffffff !important;
    font-weight: 600 !important; box-shadow: 0 2px 8px rgba(35,81,212,.3) !important;
}

.stButton > button {
    background: var(--accent) !important; color: #ffffff !important;
    border: none !important; border-radius: var(--radius-sm) !important;
    font-family: 'Inter', sans-serif !important; font-weight: 600 !important;
    font-size: .9rem !important; padding: .65rem 1.5rem !important;
    box-shadow: 0 2px 8px rgba(35,81,212,.25) !important;
    transition: all .2s ease !important;
}
.stButton > button:hover {
    background: #1a3fb5 !important;
    box-shadow: 0 6px 20px rgba(35,81,212,.4) !important;
    transform: translateY(-2px) !important;
}

[data-testid="stDownloadButton"] > button {
    background: var(--surface) !important; color: var(--accent) !important;
    border: 1.5px solid var(--accent) !important;
    border-radius: var(--radius-sm) !important; font-weight: 600 !important;
}
[data-testid="stDownloadButton"] > button:hover {
    background: var(--accent) !important; color: #fff !important;
}

[data-testid="stFileUploader"] {
    background: var(--surface) !important;
    border: 2px dashed var(--border-2) !important;
    border-radius: var(--radius) !important;
}

.stSuccess > div {
    background: var(--success-bg) !important; border: 1px solid #a7f3d0 !important;
    border-radius: var(--radius-sm) !important; color: var(--success) !important;
}
.stError > div {
    background: var(--danger-bg) !important; border: 1px solid #fca5a5 !important;
    border-radius: var(--radius-sm) !important; color: var(--danger) !important;
}
.stInfo > div {
    background: var(--accent-light) !important; border: 1px solid #bfdbfe !important;
    border-radius: var(--radius-sm) !important; color: var(--accent) !important;
}
.stWarning > div {
    background: var(--warning-bg) !important; border: 1px solid #fde68a !important;
    border-radius: var(--radius-sm) !important; color: var(--warning) !important;
}

hr { border-color: var(--border) !important; margin: 2rem 0 !important; }

.card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: var(--radius); padding: 1.5rem;
    box-shadow: var(--shadow-sm); transition: box-shadow .25s, transform .25s;
    height: 100%;
}
.card:hover { box-shadow: var(--shadow-md); transform: translateY(-3px); }
.card-label {
    font-family: 'JetBrains Mono', monospace; font-size: .68rem; font-weight: 600;
    letter-spacing: .12em; text-transform: uppercase; color: var(--text-3);
    margin-bottom: .75rem;
}
.section-label {
    font-family: 'JetBrains Mono', monospace; font-size: .68rem; font-weight: 600;
    letter-spacing: .12em; text-transform: uppercase; color: var(--text-3);
    margin-bottom: .75rem; display: flex; align-items: center; gap: 6px;
}
.section-label::after { content:''; flex:1; height:1px; background:var(--border); }
.section-heading {
    font-family: 'Syne', sans-serif; font-size: 1.1rem; font-weight: 700;
    color: var(--text-1); margin-bottom: .25rem; letter-spacing: -.01em;
}
.section-sub {
    font-size: .85rem; color: var(--text-2); margin-bottom: 1.25rem; line-height: 1.55;
}
.accent-line {
    width: 40px; height: 3px;
    background: linear-gradient(90deg, var(--accent), #60a5fa);
    border-radius: 99px; margin: .4rem 0 1.25rem;
}
.badge {
    display: inline-block; padding: 3px 10px; border-radius: 99px;
    font-size: .7rem; font-weight: 600;
    font-family: 'JetBrains Mono', monospace; letter-spacing: .02em;
}
.badge-blue   { background:var(--accent-light); color:var(--accent);   border:1px solid #bfdbfe; }
.badge-green  { background:var(--success-bg);   color:var(--success);  border:1px solid #a7f3d0; }
.badge-yellow { background:var(--warning-bg);   color:var(--warning);  border:1px solid #fde68a; }
.badge-red    { background:var(--danger-bg);    color:var(--danger);   border:1px solid #fca5a5; }
.badge-gray   { background:#f1f5f9;             color:#64748b;         border:1px solid #e2e8f0; }
.stat-row {
    display:flex; justify-content:space-between; align-items:center;
    padding:.6rem 0; border-bottom:1px solid var(--border); font-size:.875rem;
}
.stat-row:last-child { border-bottom:none; }
.stat-key { color:var(--text-2); }
.stat-val { font-family:'JetBrains Mono',monospace; font-weight:500; color:var(--text-1); font-size:.83rem; }
.msg-card {
    background:var(--surface); border:1px solid var(--border);
    border-radius:var(--radius); padding:1.25rem 1.5rem;
    margin-bottom:1rem; box-shadow:var(--shadow-sm);
    transition:box-shadow .2s, transform .2s;
}
.msg-card:hover { box-shadow:var(--shadow); transform:translateY(-2px); }
.msg-body {
    background:var(--surface-2); border:1px solid var(--border);
    border-left:3px solid var(--accent);
    border-radius:0 var(--radius-sm) var(--radius-sm) 0;
    padding:1rem 1.25rem; font-size:.875rem; color:var(--text-2);
    line-height:1.7; white-space:pre-wrap; font-family:'Inter',sans-serif;
    margin-top:.75rem;
}

/* ══════════════════════════════════════════════════════
   SIDEBAR TOGGLE BUTTONS — NATIVE STREAMLIT CONTROLS
   Sidebar collapse arrow (when closed) + button (when open)
   Both made bright blue and clearly visible.
══════════════════════════════════════════════════════ */

/* ── Arrow tab shown on LEFT EDGE when sidebar is COLLAPSED ── */
[data-testid="collapsedControl"] {
    opacity: 1 !important;
    visibility: visible !important;
    background: #2351d4 !important;
    border-radius: 0 10px 10px 0 !important;
    width: 26px !important;
    height: 60px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    box-shadow: 3px 0 16px rgba(35,81,212,.5) !important;
    border: none !important;
    cursor: pointer !important;
    transition: width .18s ease, box-shadow .18s ease !important;
}
[data-testid="collapsedControl"]:hover {
    width: 32px !important;
    box-shadow: 4px 0 24px rgba(35,81,212,.7) !important;
}
[data-testid="collapsedControl"] svg {
    fill: #ffffff !important;
    stroke: #ffffff !important;
    color: #ffffff !important;
    opacity: 1 !important;
    width: 13px !important;
    height: 13px !important;
}

/* ── Collapse button INSIDE sidebar (top of sidebar panel) ── */
[data-testid="stSidebarCollapseButton"] button,
[data-testid="stSidebarCollapseButton"] > button,
section[data-testid="stSidebar"] button[kind="header"],
section[data-testid="stSidebar"] [data-testid="stBaseButton-header"] {
    opacity: 1 !important;
    visibility: visible !important;
    background: #2351d4 !important;
    border: none !important;
    border-radius: 8px !important;
    width: 30px !important;
    height: 30px !important;
    min-width: 30px !important;
    min-height: 30px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    box-shadow: 0 2px 10px rgba(35,81,212,.4) !important;
    cursor: pointer !important;
    padding: 0 !important;
    transition: background .18s, box-shadow .18s !important;
}
[data-testid="stSidebarCollapseButton"] button:hover,
[data-testid="stSidebarCollapseButton"] > button:hover,
section[data-testid="stSidebar"] button[kind="header"]:hover,
section[data-testid="stSidebar"] [data-testid="stBaseButton-header"]:hover {
    background: #1a3fb5 !important;
    box-shadow: 0 4px 18px rgba(35,81,212,.55) !important;
}
[data-testid="stSidebarCollapseButton"] button svg,
[data-testid="stSidebarCollapseButton"] > button svg,
section[data-testid="stSidebar"] button[kind="header"] svg,
section[data-testid="stSidebar"] [data-testid="stBaseButton-header"] svg {
    fill: #ffffff !important;
    stroke: #ffffff !important;
    color: #ffffff !important;
    opacity: 1 !important;
    width: 13px !important;
    height: 13px !important;
}
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════
def get_action(p, o):
    if p >= .75 and o >= 2000: return "🎯 10% Discount"
    elif p >= .50:             return "📱 Reminder SMS"
    else:                      return "⏭️ No Action"

def get_priority(p):
    if p >= .75:   return "🔴 HIGH"
    elif p >= .50: return "🟡 MEDIUM"
    else:          return "⚪ LOW"

REQUIRED_COLS = [
    'Administrative','Administrative_Duration',
    'Informational','Informational_Duration',
    'ProductRelated','ProductRelated_Duration',
    'BounceRates','ExitRates','PageValues',
    'SpecialDay','Month','OperatingSystems',
    'Browser','Region','TrafficType',
    'VisitorType','Weekend',
    'Last_Purchase_Days_Ago','Age','Avg_Order_Value'
]

SAMPLE_DATA = {
    'Customer_Name'          : ['Ahmed Khan','Sara Ali','Usman Raza','Fatima Malik','Bilal Ahmed'],
    'Administrative'         : [2,5,0,3,8],
    'Administrative_Duration': [90,225,0,135,360],
    'Informational'          : [0,1,0,2,1],
    'Informational_Duration' : [0,45,0,90,30],
    'ProductRelated'         : [20,45,5,30,60],
    'ProductRelated_Duration': [600,1350,150,900,1800],
    'BounceRates'            : [.05,.02,.15,.03,.01],
    'ExitRates'              : [.05,.03,.18,.04,.02],
    'PageValues'             : [10.,85.,0.,45.,120.],
    'SpecialDay'             : [0,0,0,0,0],
    'Month'                  : [5,11,3,8,11],
    'OperatingSystems'       : [2,2,1,3,2],
    'Browser'                : [2,2,1,4,2],
    'Region'                 : [3,1,3,2,1],
    'TrafficType'            : [4,2,4,1,2],
    'VisitorType'            : [2,2,1,0,2],
    'Weekend'                : [0,1,0,0,1],
    'Last_Purchase_Days_Ago' : [30,10,200,45,5],
    'Age'                    : [28,35,52,24,31],
    'Avg_Order_Value'        : [3000,8500,750,5200,12000],
}


# ════════════════════════════════════════════════════════════════
# HEADER WITH LOGO
# ════════════════════════════════════════════════════════════════
model_ok = model is not None
groq_ok  = groq_client is not None

st.markdown(f"""
<div style="
    background:linear-gradient(135deg,#0d1117 0%,#1a2035 100%);
    border-radius:18px;
    padding:1.5rem 2rem;
    margin-bottom:1.5rem;
    border:1px solid #1e2d4a;
    display:flex;
    align-items:center;
    justify-content:space-between;
    flex-wrap:wrap;
    gap:1rem;
">
    <div style="display:flex;align-items:center;gap:1.25rem;">
        <div style="
            width:60px;height:60px;
            background:#1e2d4a;
            border-radius:14px;
            border:1.5px solid #2351d4;
            display:flex;align-items:center;justify-content:center;
            font-size:2rem;font-weight:900;
            font-family:'Helvetica Neue',Arial,sans-serif;
            color:#60a5fa;
            letter-spacing:-2px;
            flex-shrink:0;
        ">S</div>
        <div>
            <div style="
                font-family:'Helvetica Neue',Arial,sans-serif;
                font-size:2rem;font-weight:900;
                letter-spacing:-1px;line-height:1;
                color:#ffffff;
            ">S.S<span style="
                background:linear-gradient(90deg,#2351d4,#60a5fa);
                -webkit-background-clip:text;
                -webkit-text-fill-color:transparent;
                background-clip:text;
            ">_AI</span></div>
            <div style="
                font-family:'JetBrains Mono',monospace;
                font-size:.65rem;color:#8892a4;
                letter-spacing:.18em;margin-top:3px;
            ">ARTIFICIAL INTELLIGENCE SOLUTIONS</div>
            <div style="
                width:160px;height:2px;
                background:linear-gradient(90deg,#2351d4,#60a5fa);
                border-radius:99px;margin-top:6px;
            "></div>
        </div>
    </div>
    <div style="text-align:center;">
        <div style="
            font-family:'JetBrains Mono',monospace;
            font-size:.65rem;color:#424b63;
            letter-spacing:.15em;margin-bottom:4px;
        ">PRODUCT</div>
        <div style="
            font-family:'Syne',sans-serif;
            font-size:1.2rem;font-weight:800;
            color:#ffffff;letter-spacing:-.01em;
        ">Purchase<span style="color:#60a5fa;">IQ</span></div>
        <div style="
            font-family:'JetBrains Mono',monospace;
            font-size:.62rem;color:#424b63;
            letter-spacing:.1em;margin-top:2px;
        ">ML + GEN AI · v3.0</div>
    </div>
    <div style="display:flex;flex-direction:column;gap:6px;align-items:flex-end;">
        <span style="
            background:{'rgba(13,158,110,.15)' if model_ok else 'rgba(209,48,48,.15)'};
            color:{'#4ade80' if model_ok else '#f87171'};
            border:1px solid {'#166534' if model_ok else '#7f1d1d'};
            border-radius:99px;padding:4px 12px;
            font-family:'JetBrains Mono',monospace;
            font-size:.68rem;font-weight:600;
        ">{'✓ Model · RF 0.91 AUC' if model_ok else '✗ Model Missing'}</span>
        <span style="
            background:{'rgba(13,158,110,.15)' if groq_ok else 'rgba(199,119,0,.15)'};
            color:{'#4ade80' if groq_ok else '#fbbf24'};
            border:1px solid {'#166534' if groq_ok else '#78350f'};
            border-radius:99px;padding:4px 12px;
            font-family:'JetBrains Mono',monospace;
            font-size:.68rem;font-weight:600;
        ">{'✓ LLaMA 3.1 · AI Ready' if groq_ok else '⚠ Add GROQ_API_KEY'}</span>
        <span style="
            font-family:'JetBrains Mono',monospace;
            font-size:.62rem;color:#424b63;
            letter-spacing:.08em;
        ">{datetime.now().strftime('%B %d, %Y')}</span>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="display:flex;align-items:center;gap:10px;padding:.25rem 0 1rem;
                border-bottom:1px solid #e2e5f0;margin-bottom:.5rem;">
        <div style="width:34px;height:34px;
                    background:linear-gradient(135deg,#2351d4,#60a5fa);
                    border-radius:9px;display:flex;align-items:center;
                    justify-content:center;font-size:1rem;">🎯</div>
        <div>
            <div style="font-family:'Syne',sans-serif;font-weight:700;
                        font-size:.95rem;color:#0d1117;">Customer Profile</div>
            <div style="font-family:'JetBrains Mono',monospace;
                        font-size:.65rem;color:#8892a4;">Individual Prediction</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 👤 Demographics")
    age          = st.slider("Age", 18, 65, 30)
    visitor_type = st.selectbox("Visitor Type",
                                ["Returning Visitor","New Visitor","Other"])

    st.markdown("### 🛍️ Browsing Behavior")
    product_related = st.number_input("Products Browsed",    0, 700,   20)
    administrative  = st.number_input("Admin Pages Visited", 0, 27,    2)
    page_values     = st.number_input("Page Value Score",    0.0, 400.0, 10.0)

    st.markdown("### 📊 Session Quality")
    exit_rates   = st.slider("Exit Rate",   0.0, 0.20, 0.05, 0.01)
    bounce_rates = st.slider("Bounce Rate", 0.0, 0.20, 0.02, 0.01)

    st.markdown("### 💰 Purchase History")
    avg_order_value = st.number_input("Avg Order Value (Rs.)", 500, 40000, 3000)
    last_purchase   = st.slider("Days Since Last Purchase",   1,   365,   30)

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("🎯  Run Prediction", use_container_width=True)

    st.markdown(f"""
    <div style="text-align:center;margin-top:1.5rem;padding-top:1rem;
                border-top:1px solid #e2e5f0;font-family:'JetBrains Mono',monospace;
                font-size:.65rem;color:#8892a4;line-height:1.8;">
        PurchaseIQ · AI Marketing Engine<br>
        Random Forest + LLaMA 3 · {datetime.now().year}
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# TABS
# ════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "  🎯  Individual Prediction  ",
    "  📋  Batch Prediction + AI  ",
    "  📊  Analytics Dashboard  ",
    "  ℹ️  Model Intelligence  "
])


# ════════════════════════════════════════════════════════════════
# TAB 1 — INDIVIDUAL PREDICTION
# ════════════════════════════════════════════════════════════════
with tab1:

    # When predict button clicked — compute and save everything to session_state
    if predict_btn:
        if model is None:
            st.error("Model file not found. Ensure rf_model.pkl is in the project folder.")
        else:
            vm = {"Returning Visitor":2,"New Visitor":1,"Other":0}
            X  = pd.DataFrame({
                'Administrative'          : [administrative],
                'Administrative_Duration' : [administrative*45],
                'Informational'           : [0],
                'Informational_Duration'  : [0],
                'ProductRelated'          : [product_related],
                'ProductRelated_Duration' : [product_related*30],
                'BounceRates'             : [bounce_rates],
                'ExitRates'               : [exit_rates],
                'PageValues'              : [page_values],
                'SpecialDay'              : [0],
                'Month'                   : [5],
                'OperatingSystems'        : [2],
                'Browser'                 : [2],
                'Region'                  : [3],
                'TrafficType'             : [4],
                'VisitorType'             : [vm[visitor_type]],
                'Weekend'                 : [0],
                'Last_Purchase_Days_Ago'  : [last_purchase],
                'Age'                     : [age],
                'Avg_Order_Value'         : [avg_order_value],
            })
            prob = model.predict_proba(X)[0][1]
            pred = model.predict(X)[0]
            act  = get_action(prob, avg_order_value)

            st.session_state['ind_prob']          = prob
            st.session_state['ind_pred']          = int(pred)
            st.session_state['ind_act']           = act
            st.session_state['ind_age']           = age
            st.session_state['ind_visitor']       = visitor_type
            st.session_state['ind_product']       = product_related
            st.session_state['ind_admin']         = administrative
            st.session_state['ind_page_val']      = page_values
            st.session_state['ind_exit']          = exit_rates
            st.session_state['ind_bounce']        = bounce_rates
            st.session_state['ind_order']         = avg_order_value
            st.session_state['ind_last_purchase'] = last_purchase
            st.session_state['ind_done']          = True
            st.session_state['ind_msg'] = ""
            st.session_state['ind_rep'] = ""

    # Show welcome cards if no prediction yet
    if not st.session_state.get('ind_done', False):
        c1,c2,c3 = st.columns(3)
        for col, icon, title, desc, t1, t2 in zip(
            [c1,c2,c3],
            ["🎯","💡","✉️"],
            ["Purchase Intelligence","Smart Campaign Engine","Generative AI Layer"],
            [
                "Predict 30-day purchase probability using Random Forest with 91% ROC-AUC.",
                "3-tier discount strategy reduces marketing spend by 90% while maximizing ROI.",
                "LLaMA 3 generates unique personalized emails and SMS — no templates, pure AI.",
            ],
            ["Random Forest","Cost Optimized","LLaMA 3"],
            ["91% AUC","ROI Driven","Personalized"],
        ):
            with col:
                st.markdown(f"""
                <div class="card">
                    <div style="width:44px;height:44px;
                                background:linear-gradient(135deg,#eff3ff,#dbeafe);
                                border-radius:12px;display:flex;align-items:center;
                                justify-content:center;font-size:1.35rem;
                                margin-bottom:1rem;">{icon}</div>
                    <div style="font-family:\'Syne\',sans-serif;font-weight:700;
                                font-size:1rem;color:#0d1117;margin-bottom:.5rem;">{title}</div>
                    <div class="accent-line"></div>
                    <div style="font-size:.84rem;color:#424b63;line-height:1.6;
                                margin-bottom:1rem;">{desc}</div>
                    <span class="badge badge-blue">{t1}</span>
                    <span class="badge badge-blue" style="margin-left:5px;">{t2}</span>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        s1,s2,s3,s4 = st.columns(4)
        with s1: st.metric("Dataset Size",   "12,330",  "Sessions")
        with s2: st.metric("Model Accuracy", "88.3%",   "+3.2% vs baseline")
        with s3: st.metric("ROC-AUC Score",  "0.910",   "Top Performer")
        with s4: st.metric("Cost Reduction", "90%",     "vs No-ML approach")

        st.markdown("""
        <div style="text-align:center;padding:2.5rem;color:#8892a4;
                    font-family:\'JetBrains Mono\',monospace;font-size:.78rem;
                    border:2px dashed #c7cde3;border-radius:16px;
                    background:#f8f9fc;margin-top:1.5rem;line-height:2;">
            ← Configure customer profile in the sidebar<br>
            then click <strong style="color:#2351d4;">Run Prediction</strong>
        </div>
        """, unsafe_allow_html=True)

    # Show results from session_state
    else:
        prob      = st.session_state['ind_prob']
        pred      = st.session_state['ind_pred']
        act       = st.session_state['ind_act']
        age_s     = st.session_state['ind_age']
        vt_s      = st.session_state['ind_visitor']
        pr_s      = st.session_state['ind_product']
        pv_s      = st.session_state['ind_page_val']
        ex_s      = st.session_state['ind_exit']
        bo_s      = st.session_state['ind_bounce']
        ov_s      = st.session_state['ind_order']
        lp_s      = st.session_state['ind_last_purchase']

        k1,k2,k3,k4 = st.columns(4)
        with k1: st.metric("Purchase Probability", f"{prob:.1%}")
        with k2: st.metric("Priority Level",       get_priority(prob).split()[-1])
        with k3: st.metric("Expected Revenue",     f"Rs. {prob*ov_s:,.0f}")
        with k4: st.metric("Customer Recency",     "Recent" if lp_s<60 else "Lapsed")

        st.markdown("<br>", unsafe_allow_html=True)
        cl, cr = st.columns(2)

        with cl:
            bc = "#0d9e6e" if prob>=.75 else "#c77700" if prob>=.5 else "#2351d4"
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number", value=prob*100,
                number={"suffix":"%","font":{"size":42,"color":"#0d1117","family":"Syne"}},
                title={"text":"Purchase Probability",
                       "font":{"size":13,"color":"#8892a4","family":"Inter"}},
                gauge={
                    "axis":{"range":[0,100],
                            "tickfont":{"color":"#8892a4","size":9},
                            "tickcolor":"#e2e5f0"},
                    "bar":{"color":bc,"thickness":.2},
                    "bgcolor":"#f0f2f8","borderwidth":0,
                    "steps":[
                        {"range":[0,50],  "color":"#eff3ff"},
                        {"range":[50,75], "color":"#fefce8"},
                        {"range":[75,100],"color":"#ecfdf5"},
                    ],
                    "threshold":{"line":{"color":"#0d9e6e","width":3},
                                 "thickness":.85,"value":75}
                }
            ))
            fig_g.update_layout(height=280,
                                margin=dict(l=20,r=20,t=45,b=10),
                                paper_bgcolor="rgba(0,0,0,0)",
                                plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_g, use_container_width=True)

        with cr:
            if pred==1:
                rb,rbd,rc = "#ecfdf5","#a7f3d0","#0d9e6e"
                ri,rt,rd  = "✅","WILL PURCHASE",\
                    "High conversion likelihood detected. Customer shows strong buying intent."
            else:
                rb,rbd,rc = "#fff1f1","#fca5a5","#d13030"
                ri,rt,rd  = "❌","WILL NOT PURCHASE",\
                    "Low conversion probability. Save budget for higher-intent customers."

            st.markdown(f"""
            <div style="background:{rb};border:1.5px solid {rbd};
                        border-radius:16px;padding:1.5rem;
                        text-align:center;margin-bottom:.85rem;">
                <div style="font-size:2.75rem;margin-bottom:.5rem;">{ri}</div>
                <div style="font-family:\'Syne\',sans-serif;font-size:1.05rem;
                            font-weight:800;color:{rc};letter-spacing:.05em;">{rt}</div>
                <div style="font-size:.82rem;color:#424b63;margin-top:.5rem;
                            line-height:1.55;">{rd}</div>
            </div>
            """, unsafe_allow_html=True)

            if prob>=.75 and ov_s>=2000:
                ab,abd,ac = "#eff3ff","#bfdbfe","#2351d4"
                ai,at,ad  = "🎯","10% DISCOUNT — HIGH VALUE",\
                            f"Discount value: Rs. {ov_s*.1:,.0f}"
            elif prob>=.5:
                ab,abd,ac = "#fffbeb","#fde68a","#c77700"
                ai,at,ad  = "📱","SEND REMINDER SMS","Low-cost touchpoint"
            else:
                ab,abd,ac = "#f8fafc","#e2e8f0","#64748b"
                ai,at,ad  = "⏭️","NO ACTION REQUIRED","Conserve budget"

            st.markdown(f"""
            <div style="background:{ab};border:1.5px solid {abd};
                        border-radius:12px;padding:1.1rem 1.25rem;text-align:center;">
                <span style="font-size:1.4rem;">{ai}</span>
                <div style="font-family:\'Syne\',sans-serif;font-size:.88rem;
                            font-weight:700;color:{ac};margin:.4rem 0 .2rem;
                            letter-spacing:.04em;">{at}</div>
                <div style="font-size:.78rem;color:#64748b;">{ad}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-label">Input Signal Analysis</div>',
                    unsafe_allow_html=True)
        sigs = {
            "Page Value Score" : min(pv_s/400, 1.0),
            "Avg Order Value"  : min(ov_s/40000,1.0),
            "Products Browsed" : min(pr_s/700,  1.0),
            "Recency Score"    : 1-min(lp_s/365, 1.0),
            "Session Quality"  : max(0, 1-ex_s/0.20),
        }
        sc = ["#2351d4" if v>=.5 else "#c7cde3" for v in sigs.values()]
        fig_b = go.Figure(go.Bar(
            x=list(sigs.values()), y=list(sigs.keys()), orientation="h",
            marker=dict(color=sc,line=dict(width=0)),
            text=[f"{v:.0%}" for v in sigs.values()], textposition="outside",
            textfont=dict(color="#424b63",size=11,family="JetBrains Mono")
        ))
        fig_b.update_layout(
            height=220, margin=dict(l=5,r=60,t=5,b=5),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(range=[0,1.25],showgrid=False,showticklabels=False,zeroline=False),
            yaxis=dict(tickfont=dict(color="#424b63",size=11,family="Inter"),
                       gridcolor="#f0f2f8"),
            bargap=.38
        )
        st.plotly_chart(fig_b, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<hr>", unsafe_allow_html=True)

        ai_col1, ai_col2 = st.columns(2)

        with ai_col1:
            st.markdown("""
            <div class="section-heading">✉️ AI Personalized Message</div>
            <div class="accent-line"></div>
            <div class="section-sub">
                LLaMA 3 generates a unique personalized message
                based on this customer behavior and prediction.
            </div>
            """, unsafe_allow_html=True)

            if not groq_client:
                st.warning("⚠️ Add GROQ_API_KEY in .streamlit/secrets.toml")
            else:
                ind_msg_type = st.radio(
                    "Channel", ["📧 Email","📱 SMS"],
                    horizontal=True, key="ind_msg_type"
                )
                ind_name = st.text_input(
                    "Customer Name", value="Customer", key="ind_name"
                )
                if st.button("✨  Generate Message", use_container_width=True,
                             key="ind_msg_btn"):
                    is_email = "Email" in ind_msg_type
                    vt_label = "returning" if vt_s=="Returning Visitor" else "new"
                    prompt   = f"""You are an expert marketing copywriter for ShopSmart, a premium e-commerce platform.

Customer Profile:
- Name: {ind_name}
- Purchase Probability: {prob:.0%} ({"HIGH" if prob>=.75 else "MEDIUM" if prob>=.5 else "LOW"} intent)
- Average Order Value: Rs.{ov_s:,.0f}
- Products Browsed: {int(pr_s)} items
- Page Value Score: {pv_s:.1f}
- Days Since Last Purchase: {int(lp_s)} days
- Age: {age_s} years old
- Visitor Type: {vt_label}
- Recommendation: {act}

Write a {"personalized marketing email with subject line" if is_email else "personalized SMS (max 160 chars)"}.
- Use their name naturally
- Reference their browsing or purchase behavior
- {"Offer exclusive 10% discount" if "10%" in act else "Create gentle urgency" if "SMS" in act else "Re-engage warmly"}
- Friendly, professional tone. Company: ShopSmart
- {"Subject line + 3-4 sentence body" if is_email else "One SMS under 160 chars"}
- No placeholder brackets. Write only the message."""

                    with st.spinner("Writing personalized message..."):
                        result = groq_chat(prompt, max_tokens=300, temperature=0.85)
                        st.session_state["ind_msg"] = result

                if st.session_state.get("ind_msg",""):
                    st.markdown(f"""
                    <div class="msg-body" style="margin-top:.75rem;">
                        {st.session_state["ind_msg"]}
                    </div>
                    """, unsafe_allow_html=True)
                    st.download_button(
                        "📥 Download Message",
                        data=st.session_state["ind_msg"],
                        file_name="customer_message.txt",
                        mime="text/plain",
                        use_container_width=True,
                        key="ind_msg_dl"
                    )

        with ai_col2:
            st.markdown("""
            <div class="section-heading">📄 AI Customer Report</div>
            <div class="accent-line"></div>
            <div class="section-sub">
                LLaMA 3 generates a detailed customer intelligence
                report with strategy and ROI recommendation.
            </div>
            """, unsafe_allow_html=True)

            if not groq_client:
                st.warning("⚠️ Add GROQ_API_KEY in .streamlit/secrets.toml")
            else:
                if st.button("📊  Generate Report", use_container_width=True,
                             key="ind_rep_btn"):
                    report_prompt = f"""You are a Senior Marketing Analyst.

Customer Intelligence Data:
- Purchase Probability: {prob:.1%}
- Priority: {get_priority(prob)}
- Age: {age_s} years
- Visitor Type: {vt_s}
- Products Browsed: {int(pr_s)}
- Page Value Score: {pv_s:.1f}
- Exit Rate: {ex_s:.1%}
- Avg Order Value: Rs.{ov_s:,.0f}
- Last Purchase: {int(lp_s)} days ago
- Expected Revenue: Rs.{prob*ov_s:,.0f}
- Recommended Action: {act}

Write a concise customer intelligence report:
1. CUSTOMER SUMMARY (2 sentences)
2. KEY SIGNALS (2-3 bullet points)
3. RECOMMENDED ACTION (specific with reasoning)
4. EXPECTED ROI (use the numbers above)

Professional tone. Be specific with numbers."""

                    with st.spinner("Generating customer report..."):
                        result2 = groq_chat(report_prompt, max_tokens=400, temperature=0.6)
                        st.session_state["ind_rep"] = result2

                if st.session_state.get("ind_rep",""):
                    st.markdown(f"""
                    <div style="background:#f8f9fc;border:1px solid #e2e5f0;
                                border-left:3px solid #2351d4;
                                border-radius:0 10px 10px 0;
                                padding:1rem 1.25rem;font-size:.82rem;
                                color:#424b63;line-height:1.75;
                                white-space:pre-wrap;font-family:\'Inter\',sans-serif;
                                margin-top:.75rem;">
                        {st.session_state["ind_rep"]}
                    </div>
                    """, unsafe_allow_html=True)
                    st.download_button(
                        "📥 Download Report",
                        data=st.session_state["ind_rep"],
                        file_name="customer_report.txt",
                        mime="text/plain",
                        use_container_width=True,
                        key="ind_rep_dl"
                    )

# ════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-label">Batch Prediction + AI Campaign Engine</div>',
                unsafe_allow_html=True)

    c1,c2,c3 = st.columns(3)
    for col, num, ico, title, desc in zip(
        [c1,c2,c3], ["01","02","03"], ["📥","⚡","✉️"],
        ["Download Template","Predict All Customers","AI Message + Report"],
        ["Sample CSV with all required columns.",
         "Upload any size list — ML predicts instantly.",
         "LLaMA 3 writes personalized emails/SMS + executive report."]
    ):
        with col:
            st.markdown(f"""
            <div class="card" style="text-align:center;">
                <div style="font-family:'JetBrains Mono',monospace;font-size:1.6rem;
                            font-weight:700;color:#e2e5f0;line-height:1;
                            margin-bottom:.5rem;">{num}</div>
                <div style="font-size:1.4rem;margin-bottom:.4rem;">{ico}</div>
                <div style="font-family:'Syne',sans-serif;font-weight:700;
                            font-size:.92rem;color:#0d1117;margin-bottom:.4rem;">{title}</div>
                <div style="font-size:.82rem;color:#424b63;line-height:1.55;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('<div class="section-label">Step 1 — Download Sample Template</div>',
                unsafe_allow_html=True)
    sample_df = pd.DataFrame(SAMPLE_DATA)
    cp, cd = st.columns([3,1])
    with cp:
        with st.expander("👁️ Preview Template Format"):
            st.dataframe(sample_df, use_container_width=True, height=210)
    with cd:
        st.markdown("<br>", unsafe_allow_html=True)
        st.download_button(
            "⬇️  Download Template",
            data=sample_df.to_csv(index=False),
            file_name="purchaseiq_template.csv",
            mime="text/csv",
            use_container_width=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('<div class="section-label">Step 2 — Upload Customer List</div>',
                unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Drag & drop CSV here — or click to browse",
        type=['csv'],
        help="Must match the sample template columns."
    )

    if uploaded:
        try:
            df_b = pd.read_csv(uploaded)
            has_name = 'Customer_Name' in df_b.columns

            m1,m2,m3 = st.columns(3)
            with m1: st.metric("Customers Loaded",  f"{len(df_b):,}")
            with m2: st.metric("Columns Detected",  f"{len(df_b.columns)}")
            with m3: st.metric("Has Name Column",
                               "✅ Yes" if has_name else "⚠️ No")

            with st.expander("👁️ Preview Uploaded Data"):
                st.dataframe(df_b.head(), use_container_width=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="section-label">Step 3 — Run ML Prediction</div>',
                        unsafe_allow_html=True)

            if st.button("⚡  Predict All Customers", use_container_width=True):
                if model is None:
                    st.error("Model not loaded. Check rf_model.pkl.")
                else:
                    miss = [c for c in REQUIRED_COLS if c not in df_b.columns]
                    if miss:
                        st.error(f"Missing columns: {miss}")
                    else:
                        with st.spinner("Running ML predictions on all customers..."):
                            probs = model.predict_proba(df_b[REQUIRED_COLS])[:,1]
                            preds = model.predict(df_b[REQUIRED_COLS])

                        names = (df_b['Customer_Name'].tolist() if has_name
                                 else [f"Customer {i+1}" for i in range(len(df_b))])

                        res = pd.DataFrame({
                            'Customer'        : names,
                            'Probability'     : [f"{p:.1%}" for p in probs],
                            'Priority'        : [get_priority(p) for p in probs],
                            'Action'          : [get_action(p,o) for p,o in
                                                 zip(probs,df_b['Avg_Order_Value'])],
                            'Expected Revenue': [f"Rs. {p*o:,.0f}" for p,o in
                                                 zip(probs,df_b['Avg_Order_Value'])],
                            'Prediction'      : ["✅ Will Buy" if p==1 else "❌ Won't Buy"
                                                 for p in preds]
                        })

                        st.session_state.probs     = probs
                        st.session_state.preds     = preds
                        st.session_state.res_df    = res
                        st.session_state.df_batch  = df_b
                        st.session_state.high      = sum(1 for p in probs if p>=.75)
                        st.session_state.medium    = sum(1 for p in probs if .5<=p<.75)
                        st.session_state.low       = sum(1 for p in probs if p<.5)
                        st.session_state.t_rev     = sum(p*o for p,o in zip(probs,df_b['Avg_Order_Value']))
                        st.session_state.disc_cost = sum(
                            o*.1 for p,o in zip(probs,df_b['Avg_Order_Value'])
                            if p>=.75 and o>=2000
                        )
                        st.session_state.batch_done  = True
                        st.session_state.msgs_done   = False
                        st.session_state.report_done = False
                        st.session_state.msgs_out    = []
                        st.session_state.report_text = ""

            if st.session_state.batch_done and st.session_state.res_df is not None:
                probs     = st.session_state.probs
                preds     = st.session_state.preds
                res       = st.session_state.res_df
                df_b_s    = st.session_state.df_batch
                high      = st.session_state.high
                medium    = st.session_state.medium
                low       = st.session_state.low
                t_rev     = st.session_state.t_rev
                disc_cost = st.session_state.disc_cost
                has_name_s= 'Customer_Name' in df_b_s.columns
                names_s   = (df_b_s['Customer_Name'].tolist() if has_name_s
                             else [f"Customer {i+1}" for i in range(len(df_b_s))])

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown('<div class="section-label">Prediction Results</div>',
                            unsafe_allow_html=True)

                s1,s2,s3,s4,s5 = st.columns(5)
                with s1: st.metric("Total Processed",    f"{len(res):,}")
                with s2: st.metric("🔴 High Priority",   f"{high:,}")
                with s3: st.metric("🟡 Medium Priority", f"{medium:,}")
                with s4: st.metric("⚪ Low Priority",    f"{low:,}")
                with s5: st.metric("Exp. Revenue",       f"Rs. {t_rev:,.0f}")

                st.markdown("<br>", unsafe_allow_html=True)
                col_pie, col_tbl = st.columns([1,2])

                with col_pie:
                    st.markdown('<div class="section-label">Priority Distribution</div>',
                                unsafe_allow_html=True)
                    fig_pie = go.Figure(go.Pie(
                        labels=['High','Medium','Low'],
                        values=[high,medium,low], hole=.58,
                        marker=dict(colors=['#d13030','#c77700','#94a3b8'],
                                    line=dict(color='#ffffff',width=4)),
                        textfont=dict(size=12,family='Inter')
                    ))
                    fig_pie.update_layout(
                        height=270, paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        legend=dict(font=dict(color='#424b63',size=11),
                                    orientation='h',y=-.12),
                        annotations=[dict(
                            text=f"<b>{len(res)}</b>",x=.5,y=.5,
                            font=dict(size=28,color='#0d1117',family='Syne'),
                            showarrow=False
                        )],
                        margin=dict(l=0,r=0,t=5,b=30)
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)

                with col_tbl:
                    st.markdown('<div class="section-label">Full Results Table</div>',
                                unsafe_allow_html=True)
                    st.dataframe(res, use_container_width=True, height=280)

                st.markdown("<br>", unsafe_allow_html=True)
                st.download_button(
                    "📥  Download Predictions CSV",
                    data=res.to_csv(index=False),
                    file_name="purchaseiq_predictions.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                st.success(
                    f"✅ {len(res):,} customers processed — "
                    f"{high} high-priority targets · "
                    f"Campaign budget: Rs. {disc_cost:,.0f}"
                )

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("<hr>", unsafe_allow_html=True)
                st.markdown("""
                <div class="section-heading">✉️ AI Personalized Message Generator</div>
                <div class="accent-line"></div>
                <div class="section-sub">
                    LLaMA 3 generates unique personalized emails or SMS for each customer —
                    based on their behavior, purchase probability, and recommended action.
                    No templates. Pure Generative AI.
                </div>
                """, unsafe_allow_html=True)

                if not groq_client:
                    st.warning("⚠️ Add GROQ_API_KEY in .streamlit/secrets.toml to enable AI.")
                else:
                    col_o1, col_o2, col_o3 = st.columns(3)
                    with col_o1:
                        msg_type = st.radio("Channel",
                                            ["📧 Email","📱 SMS"], horizontal=False)
                    with col_o2:
                        sort_by = st.radio("Target Segment",
                                           ["Top by Probability",
                                            "High Priority Only",
                                            "All Customers"],
                                           horizontal=False)
                    with col_o3:
                        max_msgs = st.slider(
                            "Number of Messages", 1,
                            min(30,len(df_b_s)), min(5,len(df_b_s)),
                            help="Groq free: 30 req/min"
                        )

                    if st.button("✨  Generate AI Messages", use_container_width=True):
                        if "High Priority Only" in sort_by:
                            idxs = [i for i,p in enumerate(probs) if p>=.75][:max_msgs]
                        elif "Top by Probability" in sort_by:
                            idxs = sorted(range(len(probs)),
                                          key=lambda i: probs[i],
                                          reverse=True)[:max_msgs]
                        else:
                            idxs = list(range(min(max_msgs,len(df_b_s))))

                        if not idxs:
                            st.warning("No customers match selected filter.")
                        else:
                            msgs_out  = []
                            prog_bar  = st.progress(0)
                            prog_text = st.empty()
                            is_email  = "Email" in msg_type

                            for i, idx in enumerate(idxs):
                                p    = float(probs[idx])
                                o    = float(df_b_s['Avg_Order_Value'].iloc[idx])
                                act  = get_action(p, o)
                                name = names_s[idx]
                                pv   = float(df_b_s['PageValues'].iloc[idx])
                                pr   = int(df_b_s['ProductRelated'].iloc[idx])
                                lp   = int(df_b_s['Last_Purchase_Days_Ago'].iloc[idx])
                                vt   = int(df_b_s['VisitorType'].iloc[idx])
                                v_lb = "returning" if vt==2 else "new"

                                prog_text.markdown(
                                    f'<div class="section-label">'
                                    f'Writing message {i+1}/{len(idxs)} for {name}...</div>',
                                    unsafe_allow_html=True
                                )

                                prompt = f"""You are an expert marketing copywriter for ShopSmart, a premium e-commerce platform.

Customer Profile:
- Name: {name}
- Purchase Probability: {p:.0%} ({'HIGH' if p>=.75 else 'MEDIUM' if p>=.5 else 'LOW'} intent)
- Average Order Value: Rs.{o:,.0f}
- Products Browsed: {pr} items
- Page Value Score: {pv:.1f}
- Days Since Last Purchase: {lp} days
- Visitor Type: {v_lb}
- Recommendation: {act}

Write a {'personalized marketing email with subject line' if is_email else 'personalized SMS (max 160 chars)'}.
- Use their name naturally
- Reference browsing or purchase behavior
- {'Offer exclusive 10% discount' if '10%' in act else 'Create urgency reminder' if 'SMS' in act else 'Re-engage warmly'}
- Friendly, professional tone. Company: ShopSmart
- {'Subject + 3-4 sentence body' if is_email else 'One SMS under 160 chars'}
- No placeholder brackets. Write only the message."""

                                msg = groq_chat(prompt, max_tokens=300, temperature=0.85)

                                msgs_out.append({
                                    'Customer'   : name,
                                    'Probability': f"{p:.1%}",
                                    'Priority'   : get_priority(p),
                                    'Action'     : act,
                                    'Channel'    : "Email" if is_email else "SMS",
                                    'Message'    : msg
                                })
                                prog_bar.progress((i+1)/len(idxs))

                            prog_text.empty()
                            prog_bar.empty()

                            st.session_state.msgs_out  = msgs_out
                            st.session_state.msgs_done = True

                    if st.session_state.msgs_done and st.session_state.msgs_out:
                        st.markdown("<br>", unsafe_allow_html=True)
                        st.markdown(
                            f'<div class="section-label">'
                            f'{len(st.session_state.msgs_out)} Personalized Messages Ready</div>',
                            unsafe_allow_html=True
                        )
                        for m in st.session_state.msgs_out:
                            pv2 = float(m['Probability'].strip('%'))/100
                            if pv2>=.75:   pc,pb,pbd = "#d13030","#fff1f1","#fca5a5"
                            elif pv2>=.5:  pc,pb,pbd = "#c77700","#fffbeb","#fde68a"
                            else:          pc,pb,pbd = "#64748b","#f8fafc","#e2e8f0"

                            st.markdown(f"""
                            <div class="msg-card">
                                <div style="display:flex;justify-content:space-between;
                                            align-items:center;flex-wrap:wrap;gap:8px;">
                                    <div style="font-family:'Syne',sans-serif;font-weight:700;
                                                font-size:.98rem;color:#0d1117;">
                                        👤 {m['Customer']}</div>
                                    <div style="display:flex;gap:6px;flex-wrap:wrap;">
                                        <span style="background:{pb};color:{pc};
                                                     border:1px solid {pbd};
                                                     border-radius:99px;padding:3px 10px;
                                                     font-size:.68rem;font-weight:600;
                                                     font-family:'JetBrains Mono',monospace;">
                                            {m['Priority']}</span>
                                        <span class="badge badge-blue">{m['Probability']}</span>
                                        <span class="badge badge-green">{m['Action']}</span>
                                        <span class="badge badge-gray">{m['Channel']}</span>
                                    </div>
                                </div>
                                <div class="msg-body">{m['Message']}</div>
                            </div>
                            """, unsafe_allow_html=True)

                        msg_df = pd.DataFrame(st.session_state.msgs_out)
                        st.download_button(
                            "📥  Download All Messages CSV",
                            data=msg_df.to_csv(index=False),
                            file_name="purchaseiq_messages.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                        st.success(
                            f"✅ {len(st.session_state.msgs_out)} AI messages generated!"
                        )

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("<hr>", unsafe_allow_html=True)
                st.markdown("""
                <div class="section-heading">📄 AI Marketing Intelligence Report</div>
                <div class="accent-line"></div>
                <div class="section-sub">
                    LLaMA 3 analyzes your entire customer batch and generates a professional
                    executive report — with strategic recommendations, budget allocation, and ROI.
                </div>
                """, unsafe_allow_html=True)

                if not groq_client:
                    st.warning("⚠️ Add GROQ_API_KEY in .streamlit/secrets.toml")
                else:
                    if st.button("📊  Generate AI Intelligence Report",
                                 use_container_width=True):
                        with st.spinner("Analyzing data and writing report..."):
                            avg_prob  = float(np.mean(probs))
                            sms_count = int(medium)
                            no_act    = int(low)
                            top5      = sorted(
                                zip(names_s, probs,
                                    df_b_s['Avg_Order_Value'].tolist()),
                                key=lambda x: x[1], reverse=True
                            )[:5]
                            top5_str = "\n".join([
                                f"  - {n}: {p:.0%} probability, Rs.{o:,.0f} avg order"
                                for n,p,o in top5
                            ])

                            report_prompt = f"""You are a Senior Marketing Intelligence Analyst at a top AI consulting firm.

BATCH ANALYSIS RESULTS:
- Total Customers: {len(res)}
- High Priority (>=75%): {high}
- Medium Priority (50-75%): {medium}
- Low Priority (<50%): {low}
- Average Purchase Probability: {avg_prob:.1%}
- Total Expected Revenue: Rs. {t_rev:,.0f}
- Estimated Campaign Cost: Rs. {disc_cost:,.0f}
- Discount Targets: {high}
- SMS Reminder Targets: {sms_count}
- No Action: {no_act}

TOP 5 CUSTOMERS:
{top5_str}

Write a professional Marketing Intelligence Report with:
1. EXECUTIVE SUMMARY (2-3 sentences)
2. KEY FINDINGS (3-4 bullet points with numbers)
3. STRATEGIC RECOMMENDATIONS (specific actions)
4. BUDGET & ROI PROJECTION (use actual numbers)
5. RISK CONSIDERATIONS (1-2 points)

Be specific with numbers. Professional business language."""

                            report_text = groq_chat(report_prompt, max_tokens=700, temperature=0.6)
                            st.session_state.report_text = report_text
                            st.session_state.report_done = True

                    if st.session_state.report_done and st.session_state.report_text:
                        st.markdown(f"""
                        <div class="card" style="margin-top:1rem;">
                            <div style="display:flex;justify-content:space-between;
                                        align-items:center;margin-bottom:1rem;
                                        padding-bottom:.75rem;
                                        border-bottom:1px solid #e2e5f0;">
                                <div>
                                    <div style="font-family:'Syne',sans-serif;
                                                font-weight:800;font-size:1.1rem;
                                                color:#0d1117;">
                                        📄 Marketing Intelligence Report</div>
                                    <div style="font-family:'JetBrains Mono',monospace;
                                                font-size:.68rem;color:#8892a4;margin-top:2px;">
                                        LLaMA 3 · {datetime.now().strftime("%B %d, %Y %H:%M")}
                                    </div>
                                </div>
                                <span class="badge badge-blue">AI Generated</span>
                            </div>
                            <div style="font-size:.875rem;color:#424b63;line-height:1.8;
                                        white-space:pre-wrap;font-family:'Inter',sans-serif;">
                                {st.session_state.report_text}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        st.download_button(
                            "📥  Download Report TXT",
                            data=st.session_state.report_text,
                            file_name="purchaseiq_report.txt",
                            mime="text/plain",
                            use_container_width=True
                        )
                        st.success("✅ AI Intelligence Report ready!")

        except Exception as e:
            st.error(f"Error: {e}")
            st.info("Ensure your CSV matches the template format.")
    else:
        st.session_state.batch_done  = False
        st.session_state.msgs_done   = False
        st.session_state.report_done = False

        st.markdown("""
        <div style="text-align:center;padding:3rem 2rem;color:#8892a4;
                    font-family:'JetBrains Mono',monospace;font-size:.78rem;
                    line-height:2;border:2px dashed #c7cde3;border-radius:16px;
                    background:#f8f9fc;margin-top:.5rem;">
            Upload template · Fill with your data · Upload here<br>
            Then get <strong style="color:#2351d4;">
            ML predictions + AI messages + AI report</strong>
        </div>
        """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# TAB 3 — ANALYTICS
# ════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-label">Business Performance Analytics</div>',
                unsafe_allow_html=True)

    k1,k2,k3,k4,k5 = st.columns(5)
    with k1: st.metric("Model Accuracy",  "88.3%",     "+3.2%")
    with k2: st.metric("ROC-AUC Score",   "0.910",     "+0.043")
    with k3: st.metric("Recall Score",    "71.5%",     "+8.1%")
    with k4: st.metric("Cost Reduction",  "90%",       "vs No-ML")
    with k5: st.metric("Net Profit",      "Rs.81,240", "per batch")

    st.markdown("<br>", unsafe_allow_html=True)
    ca, cb = st.columns(2)

    with ca:
        st.markdown('<div class="section-label">Model Comparison — All Metrics</div>',
                    unsafe_allow_html=True)
        fig_c = go.Figure()
        for m, v, c in [
            ('Accuracy',[.872,.883,.882],'#2351d4'),
            ('Recall',  [.699,.715,.686],'#0d9e6e'),
            ('F1-Score',[.629,.655,.644],'#c77700'),
            ('ROC-AUC', [.867,.910,.900],'#d13030'),
        ]:
            fig_c.add_trace(go.Bar(
                name=m, x=['Logistic Reg','Random Forest','XGBoost'],
                y=v, marker_color=c, marker_line=dict(width=0), opacity=.88
            ))
        fig_c.update_layout(
            barmode='group', height=310,
            margin=dict(l=5,r=5,t=10,b=5),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            legend=dict(orientation='h',y=1.1,x=.5,xanchor='center',
                        font=dict(color='#424b63',size=10,family='Inter')),
            xaxis=dict(tickfont=dict(color='#424b63',size=11),gridcolor='#f0f2f8'),
            yaxis=dict(range=[.5,1.0],tickfont=dict(color='#424b63',size=10),
                       gridcolor='#f0f2f8',zeroline=False),
            bargap=.2,bargroupgap=.05
        )
        st.plotly_chart(fig_c, use_container_width=True)

    with cb:
        st.markdown('<div class="section-label">Marketing ROI — ML vs No-ML</div>',
                    unsafe_allow_html=True)
        fig_roi = go.Figure(go.Bar(
            x=['Without ML','With ML'], y=[493200,46760],
            marker=dict(color=['rgba(209,48,48,.15)','rgba(13,158,110,.15)'],
                        line=dict(color=['#d13030','#0d9e6e'],width=2.5)),
            text=['Rs. 4,93,200','Rs. 46,760'], textposition='outside',
            textfont=dict(color=['#d13030','#0d9e6e'],size=12,family='JetBrains Mono'),
            width=.45
        ))
        fig_roi.add_annotation(x=.5,y=280000,text="<b>90% Cost Reduction</b>",
                               showarrow=False,
                               font=dict(color='#2351d4',size=14,family='Syne'))
        fig_roi.update_layout(
            height=310, margin=dict(l=5,r=5,t=10,b=5),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(tickfont=dict(color='#424b63',size=12),gridcolor='#f0f2f8'),
            yaxis=dict(tickfont=dict(color='#424b63',size=10),
                       gridcolor='#f0f2f8',zeroline=False),
            showlegend=False
        )
        st.plotly_chart(fig_roi, use_container_width=True)

    st.markdown('<div class="section-label">Feature Importance — Random Forest</div>',
                unsafe_allow_html=True)
    feats = ['SpecialDay','Weekend','Browser','Informational','VisitorType','Age',
             'BounceRates','Administrative','ProductRelated','ExitRates',
             'Admin_Duration','Avg_Order_Value','PageValues']
    imps  = [.005,.008,.011,.012,.014,.015,.022,.031,.042,.065,.070,.125,.373]
    fig_fi = go.Figure(go.Bar(
        x=imps, y=feats, orientation='h',
        marker=dict(color=['#2351d4' if v>=.05 else '#c7cde3' for v in imps],
                    line=dict(width=0)),
        text=[f"{v:.3f}" for v in imps], textposition='outside',
        textfont=dict(color='#424b63',size=10,family='JetBrains Mono')
    ))
    fig_fi.update_layout(
        height=380, margin=dict(l=5,r=70,t=5,b=5),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False,showticklabels=False,zeroline=False),
        yaxis=dict(tickfont=dict(color='#424b63',size=11,family='Inter'),
                   gridcolor='#f0f2f8'),
        bargap=.32
    )
    st.plotly_chart(fig_fi, use_container_width=True)

    st.markdown('<div class="section-label">3-Tier Smart Discount Strategy</div>',
                unsafe_allow_html=True)
    dl1,dl2,dl3 = st.columns(3)
    for col, icon, title, cond, desc, color, bg, bdr in zip(
        [dl1,dl2,dl3],
        ["🎯","📱","⏭️"],
        ["10% Discount","Reminder SMS","No Action"],
        ["Prob >= 75% AND Order >= Rs.2,000","Prob >= 50%","Prob < 50%"],
        ["Immediate discount — high ROI target",
         "Low-cost SMS touchpoint",
         "Save budget entirely"],
        ["#2351d4","#c77700","#64748b"],
        ["#eff3ff","#fffbeb","#f8fafc"],
        ["#bfdbfe","#fde68a","#e2e8f0"]
    ):
        with col:
            st.markdown(f"""
            <div class="card" style="border-top:3px solid {color};text-align:center;">
                <div style="font-size:1.5rem;margin-bottom:.5rem;">{icon}</div>
                <div style="font-family:'Syne',sans-serif;font-weight:700;
                            font-size:.95rem;color:#0d1117;margin-bottom:.4rem;">{title}</div>
                <span style="background:{bg};color:{color};border:1px solid {bdr};
                             border-radius:99px;padding:3px 10px;font-size:.7rem;
                             font-weight:600;font-family:'JetBrains Mono',monospace;
                             display:inline-block;margin-bottom:.6rem;">{cond}</span>
                <div style="font-size:.82rem;color:#424b63;line-height:1.5;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# TAB 4 — MODEL INTELLIGENCE
# ════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-label">Model Architecture · Training Pipeline · Results</div>',
                unsafe_allow_html=True)

    c1,c2 = st.columns(2)
    with c1:
        config = [
            ("Algorithm",     "Random Forest Classifier"),
            ("Estimators",    "100 Decision Trees"),
            ("Imbalance Fix", "SMOTE Oversampling"),
            ("Train / Test",  "80% / 20% Split"),
            ("Random State",  "42"),
            ("Features",      "20 Columns"),
        ]
        st.markdown(
            '<div class="card"><div class="card-label">Model Configuration</div>' +
            "".join([f'<div class="stat-row"><span class="stat-key">{k}</span>'
                     f'<span class="stat-val">{v}</span></div>' for k,v in config]) +
            "</div>", unsafe_allow_html=True
        )

    with c2:
        evals = [
            ("Accuracy",        "88.3%",      "#0d9e6e"),
            ("Precision",       "60.4%",      "#0d9e6e"),
            ("Recall",          "71.5%",      "#0d9e6e"),
            ("F1-Score",        "65.5%",      "#0d9e6e"),
            ("ROC-AUC Best",    "0.910",       "#2351d4"),
            ("Confusion TN/TP", "1905 / 273", "#0d1117"),
        ]
        st.markdown(
            '<div class="card"><div class="card-label">Evaluation Metrics</div>' +
            "".join([f'<div class="stat-row"><span class="stat-key">{k}</span>'
                     f'<span class="stat-val" style="color:{c};">{v}</span></div>'
                     for k,v,c in evals]) +
            "</div>", unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)
    ds_info = [
        ("Source",        "Kaggle — Online Shoppers Intention"),
        ("Total Samples", "12,330 Sessions"),
        ("Class Split",   "84.5% No Purchase · 15.5% Purchase"),
        ("Synthetic",     "Age · Last_Purchase_Days_Ago · Avg_Order_Value"),
        ("SMOTE Balance", "8,337 No → 8,337 Yes"),
    ]
    st.markdown(
        '<div class="card"><div class="card-label">Dataset Information</div>'
        '<div style="display:flex;gap:2rem;flex-wrap:wrap;margin-top:.5rem;">' +
        "".join([
            f'<div style="min-width:180px;">'
            f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:.65rem;'
            f'color:#8892a4;text-transform:uppercase;letter-spacing:.1em;">{l}</div>'
            f'<div style="font-weight:600;color:#0d1117;font-size:.875rem;'
            f'margin-top:4px;">{v}</div></div>'
            for l,v in ds_info
        ]) + "</div></div>", unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">ML Training Pipeline</div>',
                unsafe_allow_html=True)
    steps = [
        ("01","📊","Data Collection","Kaggle + 3 synthetic features"),
        ("02","🔍","EDA","Imbalance, correlations, features"),
        ("03","⚙️","Preprocessing","Encoding, SMOTE, split"),
        ("04","🤖","Model Training","LR, Random Forest, XGBoost"),
        ("05","📈","Evaluation","Accuracy, Recall, ROC-AUC"),
        ("06","🚀","Deployment","Streamlit + Groq AI"),
    ]
    cols6 = st.columns(6)
    for col, (num,ico,title,desc) in zip(cols6, steps):
        with col:
            st.markdown(f"""
            <div style="text-align:center;padding:.75rem .5rem;background:#fff;
                        border:1px solid #e2e5f0;border-radius:12px;
                        box-shadow:0 1px 3px rgba(0,0,0,.04);">
                <div style="font-family:'JetBrains Mono',monospace;font-size:.65rem;
                            color:#8892a4;font-weight:600;letter-spacing:.1em;
                            margin-bottom:.25rem;">{num}</div>
                <div style="font-size:1.3rem;margin-bottom:.3rem;">{ico}</div>
                <div style="font-family:'Syne',sans-serif;font-weight:700;
                            font-size:.82rem;color:#0d1117;margin-bottom:.25rem;">{title}</div>
                <div style="font-size:.72rem;color:#8892a4;line-height:1.45;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="text-align:center;padding:2.5rem 0 0;
                font-family:'JetBrains Mono',monospace;
                font-size:.68rem;color:#8892a4;line-height:2;">
        PurchaseIQ · AI Marketing Intelligence Engine<br>
        Random Forest + LLaMA 3 Generative AI · {datetime.now().year}
    </div>
    """, unsafe_allow_html=True)
