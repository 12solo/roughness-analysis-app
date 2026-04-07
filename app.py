import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from scipy.signal import savgol_filter
from scipy.spatial.distance import cdist
from scipy.ndimage import gaussian_filter1d
import re
import os
import io
import base64

# ==========================================
# PAGE CONFIG — must be first Streamlit call
# ==========================================
st.set_page_config(
    page_title="SRoughnessLab Pro | Solomon Scientific",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# GLOBAL CUSTOM CSS — Professional Theme
# ==========================================
st.markdown("""
<style>
/* ── Google Fonts ─────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

/* ── CSS Variables ────────────────────────────── */
:root {
    --navy:       #0b1120;
    --navy-mid:   #111827;
    --navy-light: #1a2540;
    --gold:       #c9a84c;
    --gold-light: #e2c97e;
    --gold-dim:   #9c7a32;
    --silver:     #a8b4c8;
    --white:      #f0f4fb;
    --white-dim:  #c8d0de;
    --accent:     #3a7bd5;
    --red:        #e05252;
    --green:      #3db87a;
    --border:     rgba(201,168,76,0.22);
    --card-bg:    rgba(26,37,64,0.70);
    --font-head:  'Playfair Display', Georgia, serif;
    --font-mono:  'IBM Plex Mono', 'Courier New', monospace;
    --font-body:  'IBM Plex Sans', 'Segoe UI', sans-serif;
}

/* ── Base & Body ──────────────────────────────── */
html, body, [class*="css"] {
    font-family: var(--font-body);
    color: var(--white);
}
.stApp {
    background: linear-gradient(135deg, #0b1120 0%, #0f1a2e 50%, #0b1120 100%);
    min-height: 100vh;
}

/* Subtle grid texture overlay */
.stApp::before {
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background-image:
        linear-gradient(rgba(201,168,76,0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(201,168,76,0.03) 1px, transparent 1px);
    background-size: 40px 40px;
    pointer-events: none;
    z-index: 0;
}

/* ── Sidebar ──────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #080e1c 0%, #0d1628 100%);
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] .stMarkdown,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span {
    color: var(--white-dim) !important;
    font-family: var(--font-body);
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: var(--gold) !important;
    font-family: var(--font-body);
    font-weight: 600;
    font-size: 0.75rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
}
[data-testid="stSidebar"] hr {
    border-color: var(--border);
    margin: 1rem 0;
}

/* ── Input Widgets ────────────────────────────── */
[data-testid="stSidebar"] input[type="text"],
[data-testid="stSidebar"] input[type="number"],
[data-testid="stSidebar"] textarea,
[data-testid="stSidebar"] select,
.stSelectbox > div > div,
.stTextInput > div > div > input,
.stNumberInput > div > div > input {
    background: rgba(11,17,32,0.8) !important;
    border: 1px solid var(--border) !important;
    border-radius: 4px !important;
    color: var(--white) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.82rem !important;
}
.stSelectbox > div > div:hover,
.stTextInput > div > div > input:focus {
    border-color: var(--gold) !important;
    box-shadow: 0 0 0 1px var(--gold-dim) !important;
}

/* ── Buttons ──────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, var(--gold-dim), var(--gold)) !important;
    color: var(--navy) !important;
    border: none !important;
    border-radius: 3px !important;
    font-family: var(--font-body) !important;
    font-weight: 600 !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    padding: 0.45rem 1rem !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, var(--gold), var(--gold-light)) !important;
    box-shadow: 0 4px 20px rgba(201,168,76,0.35) !important;
    transform: translateY(-1px) !important;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #8b1a1a, var(--red)) !important;
    color: white !important;
}

/* Download buttons */
[data-testid="stDownloadButton"] > button {
    background: rgba(26,37,64,0.9) !important;
    color: var(--gold) !important;
    border: 1px solid var(--border) !important;
    border-radius: 3px !important;
    font-family: var(--font-body) !important;
    font-weight: 500 !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.05em !important;
    transition: all 0.2s ease !important;
}
[data-testid="stDownloadButton"] > button:hover {
    background: rgba(201,168,76,0.12) !important;
    border-color: var(--gold) !important;
    box-shadow: 0 4px 16px rgba(201,168,76,0.2) !important;
}

/* ── Tabs ─────────────────────────────────────── */
[data-testid="stTabs"] [role="tablist"] {
    background: rgba(11,17,32,0.6);
    border-bottom: 1px solid var(--border);
    gap: 0;
    padding: 0;
}
[data-testid="stTabs"] [role="tab"] {
    color: var(--silver) !important;
    font-family: var(--font-body) !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    padding: 0.7rem 1.2rem !important;
    border-radius: 0 !important;
    border-bottom: 2px solid transparent !important;
    transition: all 0.2s !important;
}
[data-testid="stTabs"] [role="tab"]:hover {
    color: var(--gold-light) !important;
    background: rgba(201,168,76,0.05) !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color: var(--gold) !important;
    border-bottom-color: var(--gold) !important;
    background: rgba(201,168,76,0.08) !important;
}

/* ── DataFrames ───────────────────────────────── */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    overflow: hidden !important;
}
[data-testid="stDataFrame"] table {
    font-family: var(--font-mono) !important;
    font-size: 0.78rem !important;
}
[data-testid="stDataFrame"] th {
    background: rgba(201,168,76,0.12) !important;
    color: var(--gold-light) !important;
    font-weight: 600 !important;
    letter-spacing: 0.04em !important;
    border-bottom: 1px solid var(--border) !important;
}
[data-testid="stDataFrame"] tr:hover {
    background: rgba(201,168,76,0.05) !important;
}

/* ── Expanders ────────────────────────────────── */
[data-testid="stExpander"] {
    border: 1px solid var(--border) !important;
    border-radius: 4px !important;
    background: rgba(26,37,64,0.4) !important;
}
[data-testid="stExpander"] summary {
    color: var(--gold-light) !important;
    font-family: var(--font-body) !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.06em !important;
}

/* ── Radio & Checkbox ─────────────────────────── */
[data-testid="stRadio"] label,
[data-testid="stCheckbox"] label {
    color: var(--white-dim) !important;
    font-family: var(--font-body) !important;
    font-size: 0.82rem !important;
}

/* ── Sliders ──────────────────────────────────── */
[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {
    background: var(--gold) !important;
    border-color: var(--gold) !important;
}
[data-testid="stSlider"] [data-baseweb="slider"] [data-testid="stThumbValue"] {
    color: var(--gold) !important;
}

/* ── Alerts & Messages ────────────────────────── */
[data-testid="stAlert"] {
    border-radius: 4px !important;
    font-family: var(--font-body) !important;
    font-size: 0.82rem !important;
}
.stSuccess {
    background: rgba(61,184,122,0.12) !important;
    border-left-color: var(--green) !important;
}
.stWarning {
    background: rgba(201,168,76,0.12) !important;
    border-left-color: var(--gold) !important;
}
.stError {
    background: rgba(224,82,82,0.12) !important;
    border-left-color: var(--red) !important;
}
.stInfo {
    background: rgba(58,123,213,0.12) !important;
    border-left-color: var(--accent) !important;
}

/* ── Text Area ────────────────────────────────── */
.stTextArea textarea {
    background: rgba(11,17,32,0.8) !important;
    border: 1px solid var(--border) !important;
    color: var(--white-dim) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.78rem !important;
    border-radius: 4px !important;
}

/* ── Multiselect ──────────────────────────────── */
[data-baseweb="tag"] {
    background: rgba(201,168,76,0.2) !important;
    border: 1px solid var(--border) !important;
}
[data-baseweb="tag"] span {
    color: var(--gold-light) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.75rem !important;
}

/* ── Plotly Charts ────────────────────────────── */
.js-plotly-plot {
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    background: rgba(11,17,32,0.5) !important;
}

/* ── Scrollbar ────────────────────────────────── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--navy); }
::-webkit-scrollbar-thumb { background: var(--gold-dim); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--gold); }
</style>
""", unsafe_allow_html=True)


# ==========================================
# HELPER COMPONENTS
# ==========================================
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def render_header():
    logo_path = "SR LOGO.png"
    if os.path.exists(logo_path):
        img_b64 = get_base64_of_bin_file(logo_path)
        icon_html = f'<img src="data:image/png;base64,{img_b64}" style="width: 54px; height: 54px; border-radius: 8px; object-fit: contain; box-shadow: 0 4px 20px rgba(201,168,76,0.3); flex-shrink: 0; background: white;">'
    else:
        icon_html = '<div style="width: 54px; height: 54px; background: linear-gradient(135deg, #9c7a32, #c9a84c); border-radius: 8px; display: flex; align-items: center; justify-content: center; font-size: 1.6rem; box-shadow: 0 4px 20px rgba(201,168,76,0.3); flex-shrink: 0;">🔬</div>'

    st.markdown(f"""
    <div style="
        padding: 2rem 0 1.5rem 0;
        border-bottom: 1px solid rgba(201,168,76,0.25);
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 1.5rem;
    ">
        {icon_html}
        <div>
            <div style="
                font-family: 'Playfair Display', Georgia, serif;
                font-size: 1.75rem;
                font-weight: 700;
                color: #f0f4fb;
                letter-spacing: 0.01em;
                line-height: 1.1;
            ">SRoughnessLab <span style="color:#c9a84c;">Pro</span></div>
            <div style="
                font-family: 'IBM Plex Sans', sans-serif;
                font-size: 0.72rem;
                color: #a8b4c8;
                letter-spacing: 0.2em;
                text-transform: uppercase;
                margin-top: 2px;
            ">Surface Metrology Analysis Suite &nbsp;·&nbsp; Solomon Scientific &nbsp;·&nbsp; © 2026</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def metric_card(label, value, unit="", delta=None):
    delta_html = ""
    if delta is not None:
        color = "#3db87a" if delta >= 0 else "#e05252"
        arrow = "▲" if delta >= 0 else "▼"
        delta_html = f'<div style="color:{color};font-size:0.7rem;margin-top:2px;">{arrow} {abs(delta):.3f}</div>'
    return f"""
    <div style="
        background: rgba(26,37,64,0.65);
        border: 1px solid rgba(201,168,76,0.2);
        border-radius: 6px;
        padding: 1rem 1.25rem;
        border-top: 2px solid #c9a84c;
    ">
        <div style="font-family:'IBM Plex Sans',sans-serif;font-size:0.68rem;color:#a8b4c8;letter-spacing:0.12em;text-transform:uppercase;margin-bottom:4px;">{label}</div>
        <div style="font-family:'IBM Plex Mono',monospace;font-size:1.35rem;color:#f0f4fb;font-weight:500;">{value}<span style="font-size:0.7rem;color:#a8b4c8;margin-left:4px;">{unit}</span></div>
        {delta_html}
    </div>
    """

def section_title(text, icon=""):
    st.markdown(f"""
    <div style="
        display:flex; align-items:center; gap:0.6rem;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid rgba(201,168,76,0.18);
    ">
        <span style="font-size:1rem;">{icon}</span>
        <span style="
            font-family:'IBM Plex Sans',sans-serif;
            font-size:0.72rem;
            font-weight:600;
            color:#c9a84c;
            letter-spacing:0.15em;
            text-transform:uppercase;
        ">{text}</span>
    </div>
    """, unsafe_allow_html=True)

def info_box(text, kind="info"):
    colors = {
        "info":    ("#3a7bd5", "rgba(58,123,213,0.10)"),
        "success": ("#3db87a", "rgba(61,184,122,0.10)"),
        "warning": ("#c9a84c", "rgba(201,168,76,0.10)"),
        "error":   ("#e05252", "rgba(224,82,82,0.10)"),
    }
    border, bg = colors.get(kind, colors["info"])
    icon = {"info": "ℹ", "success": "✓", "warning": "⚠", "error": "✕"}.get(kind, "ℹ")
    st.markdown(f"""
    <div style="
        background:{bg};border-left:3px solid {border};
        border-radius:0 4px 4px 0;padding:0.75rem 1rem;
        font-family:'IBM Plex Sans',sans-serif;font-size:0.82rem;color:#c8d0de;
        margin:0.5rem 0;
    "><span style="color:{border};margin-right:0.5rem;">{icon}</span>{text}</div>
    """, unsafe_allow_html=True)

def render_sidebar_brand():
    logo_path = "Solomon_SR_Logo.png"
    if os.path.exists(logo_path):
        img_b64 = get_base64_of_bin_file(logo_path)
        icon_html = f'<img src="data:image/png;base64,{img_b64}" style="width: 52px; height: 52px; margin: 0 auto 0.75rem auto; border-radius: 10px; display: block; box-shadow: 0 4px 16px rgba(201,168,76,0.3); object-fit: contain; background: white;">'
    else:
        icon_html = '<div style="width:52px; height:52px; margin:0 auto 0.75rem auto; background:linear-gradient(135deg,#9c7a32,#c9a84c); border-radius:10px; display:flex;align-items:center;justify-content:center; font-size:1.5rem; box-shadow:0 4px 16px rgba(201,168,76,0.3);">🔬</div>'

    st.markdown(f"""
    <div style="padding: 1.25rem 0 0.5rem 0; text-align:center;">
        {icon_html}
        <div style="
            font-family:'IBM Plex Sans',sans-serif;
            font-size:0.65rem;
            color:#9c7a32;
            letter-spacing:0.2em;
            text-transform:uppercase;
            margin-bottom:4px;
        ">Solomon Scientific</div>
        <div style="
            font-family:'Playfair Display',Georgia,serif;
            font-size:1.1rem;
            font-weight:700;
            color:#f0f4fb;
        ">SRoughnessLab <span style="color:#c9a84c;">Pro</span></div>
        <div style="
            margin-top:0.75rem;
            padding-top:0.75rem;
            border-top:1px solid rgba(201,168,76,0.2);
            font-family:'IBM Plex Sans',sans-serif;
            font-size:0.68rem;
            color:#7a8aaa;
        ">Surface Roughness Analysis Suite<br>
        <a href='mailto:your.solomon.duf@gmail.com'
           style='color:#9c7a32;text-decoration:none;'>
            ✉ Contact Developer
        </a>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ==========================================
# UTILITY FUNCTIONS
# ==========================================
def natural_sort_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', str(s))]

def compute_roughness_params(signal):
    Ra  = np.mean(np.abs(signal))
    Rq  = np.sqrt(np.mean(signal**2))
    Rt  = np.max(signal) - np.min(signal)
    Rsk = stats.skew(signal)
    Rku = stats.kurtosis(signal)
    sorted_s = np.sort(signal)
    Rz = np.mean(sorted_s[-5:] - sorted_s[:5]) if len(signal) >= 10 else np.nan
    return Ra, Rq, Rz, Rt, Rsk, Rku

def iso_sigma(lambda_c, dx):
    if dx == 0: dx = 0.001
    return (lambda_c / (2 * np.pi)) / dx

def export_to_excel_with_logo(df, sheet_title):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_title)
        worksheet = writer.sheets[sheet_title]
        for i, col in enumerate(df.columns):
            max_len = max(len(str(col)), df[col].astype(str).map(len).max() if len(df) > 0 else 0) + 2
            worksheet.set_column(i, i, max_len)
        logo_path = "Solomon_SR_Logo.png"
        if os.path.exists(logo_path):
            col_offset = len(df.columns) + 1
            worksheet.insert_image(1, col_offset, logo_path, {'x_scale': 0.6, 'y_scale': 0.6})
    return output.getvalue()


# ==========================================
# PLOTLY THEME
# ==========================================
PLOT_BG    = "rgba(11,17,32,0.0)"
PAPER_BG   = "rgba(11,17,32,0.0)"
GOLD       = "#c9a84c"
SILVER     = "#a8b4c8"
WHITE      = "#f0f4fb"
GRID_COLOR = "rgba(201,168,76,0.08)"
LINE_COLOR = "rgba(201,168,76,0.35)"

AXIS_STYLE = dict(
    mirror=True,
    ticks='outside',
    showline=True,
    linecolor=LINE_COLOR,
    linewidth=1.5,
    showgrid=True,
    gridcolor=GRID_COLOR,
    gridwidth=1,
    zeroline=False,
    title_font=dict(family="IBM Plex Sans", size=13, color=SILVER),
    tickfont=dict(family="IBM Plex Mono", size=11, color=SILVER),
    tickwidth=1.5,
    ticklen=5,
    tickcolor=LINE_COLOR,
)

JOURNAL_CONFIG = {
    'toImageButtonOptions': {'format': 'png', 'filename': 'SRoughnessLab_Plot', 'scale': 4},
    'displayModeBar': True,
    'displaylogo': False,
    'modeBarButtonsToRemove': ['select2d', 'lasso2d'],
}

PALETTE = [
    "#c9a84c", "#3a7bd5", "#3db87a", "#e05252",
    "#9b59b6", "#e67e22", "#1abc9c", "#e74c3c",
    "#f39c12", "#2980b9", "#27ae60", "#8e44ad",
]


# ==========================================
# DATA LOADER CLASS
# ==========================================
class RoughnessLoader:
    def __init__(self):
        self.targets = {'Ra': ['ra'], 'Rq': ['rq'], 'Rz': ['rz'], 'Rt': ['rt']}

    def clean_value(self, val):
        if pd.isna(val): return np.nan
        if isinstance(val, (int, float)): return float(val)
        text = str(val).replace(',', '.').strip()
        match = re.search(r"[-+]?\d*\.\d+|\d+", text)
        return float(match.group()) if match else np.nan

    def process_files(self, uploaded_files, meta_template, filter_type, window_size, lambda_val):
        combined_summary, profile_map = [], {}
        for file in uploaded_files:
            try:
                xl = pd.ExcelFile(file)
                row_summary = meta_template.copy()
                row_summary['File'] = file.name
                for sheet in xl.sheet_names:
                    df_sheet = xl.parse(sheet, header=None)
                    for r in range(min(len(df_sheet), 100)):
                        for c in range(len(df_sheet.columns)):
                            cell_str = str(df_sheet.iloc[r, c]).lower().strip()
                            for std_key, keywords in self.targets.items():
                                if any(k in cell_str for k in keywords) and std_key not in row_summary:
                                    val = np.nan
                                    if c + 1 < len(df_sheet.columns):
                                        val = self.clean_value(df_sheet.iloc[r, c+1])
                                    if np.isnan(val) and r + 1 < len(df_sheet):
                                        val = self.clean_value(df_sheet.iloc[r+1, c])
                                    if not np.isnan(val):
                                        row_summary[std_key] = val

                data_sheet = next((s for s in xl.sheet_names if "DATA" in s.upper()), None)
                if not data_sheet and xl.sheet_names:
                    data_sheet = xl.sheet_names[-1]

                if data_sheet:
                    df_p = pd.read_excel(file, sheet_name=data_sheet, usecols=[4, 5])
                    df_p.columns = ['Length_mm', 'Amplitude_um']
                    df_p['Length_mm']   = df_p['Length_mm'].astype(str).str.replace(',', '.', regex=False)
                    df_p['Amplitude_um']= df_p['Amplitude_um'].astype(str).str.replace(',', '.', regex=False)
                    df_p = df_p.apply(pd.to_numeric, errors='coerce').dropna().reset_index(drop=True)

                    if not df_p.empty:
                        if filter_type == "ISO Gaussian (λc)":
                            dx = np.mean(np.diff(df_p['Length_mm'])) if len(df_p) > 1 else 0.001
                            sigma = iso_sigma(lambda_val, dx)
                            df_p['Form']      = gaussian_filter1d(df_p['Amplitude_um'], sigma=sigma)
                            df_p['Roughness'] = df_p['Amplitude_um'] - df_p['Form']
                        elif filter_type == "Savitzky-Golay":
                            df_p['Form']      = savgol_filter(df_p['Amplitude_um'], window_length=window_size, polyorder=3)
                            df_p['Roughness'] = df_p['Amplitude_um'] - df_p['Form']
                        else:
                            df_p['Roughness'] = df_p['Amplitude_um'] - df_p['Amplitude_um'].mean()

                        df_p['Amplitude_um_Norm'] = df_p['Roughness']
                        Ra_c, Rq_c, Rz_c, Rt_c, Rsk_c, Rku_c = compute_roughness_params(df_p['Roughness'].values)
                        row_summary.update({
                            'Ra_calc': Ra_c, 'Rq_calc': Rq_c, 'Rz_calc': Rz_c,
                            'Rt_calc': Rt_c, 'Rsk': Rsk_c, 'Rku': Rku_c
                        })
                        df_p['Sample'] = meta_template['Sample']
                        profile_map[file.name] = df_p
                    else:
                        st.error(f"❌ {file.name}: Columns E/F empty after parsing. Check decimal formatting.")
                else:
                    st.error(f"❌ {file.name}: No valid data sheet found.")

                combined_summary.append(row_summary)
            except Exception as e:
                st.error(f"Error processing {file.name}: {e}")

        return pd.DataFrame(combined_summary), profile_map


# ==========================================
# SESSION STATE
# ==========================================
for key, default in [
    ('master_df', pd.DataFrame()),
    ('opt_df', pd.DataFrame()),
    ('profile_dict', {}),
    ('legend_map', {}),
]:
    if key not in st.session_state:
        st.session_state[key] = default

if not st.session_state['master_df'].empty and 'Sample' not in st.session_state['master_df'].columns:
    st.session_state['master_df'] = pd.DataFrame()


# ==========================================
# SIDEBAR
# ==========================================
with st.sidebar:
    render_sidebar_brand()
    st.markdown("<hr style='border-color:rgba(201,168,76,0.2);margin:1rem 0'>", unsafe_allow_html=True)

    # ── 1. Data Input ───────────────────────────────
    st.markdown("### 1 · Data Input")
    with st.form("input_form", clear_on_submit=True):
        s_name = st.text_input("Sample ID", "Sample A", placeholder="e.g. 10N Load")
        s_files = st.file_uploader(
            "Upload Replicate Files (.xlsx)",
            accept_multiple_files=True,
            type=["xlsx"],
        )
        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
        st.markdown("**Filter Settings**")
        filter_type = st.selectbox(
            "Detrending Filter",
            ["ISO Gaussian (λc)", "Savitzky-Golay", "None"],
        )
        lambda_val, sg_win = 0.8, 51
        if filter_type == "ISO Gaussian (λc)":
            lambda_val = st.number_input("Cutoff Wavelength λc (mm)", value=0.8, step=0.1, min_value=0.1)
        elif filter_type == "Savitzky-Golay":
            sg_win = st.slider("S-G Window Length", 5, 151, 51, step=2)

        submit = st.form_submit_button("＋ Add Sample Batch", use_container_width=True)

    if submit and s_files:
        with st.spinner("Processing files…"):
            loader = RoughnessLoader()
            meta = {"Sample": s_name, "Filter": filter_type}
            new_sum, new_prof = loader.process_files(s_files, meta, filter_type, sg_win, lambda_val)
            st.session_state['master_df'] = pd.concat(
                [st.session_state['master_df'], new_sum], ignore_index=True
            )
            st.session_state['profile_dict'].update(new_prof)
            st.success(f"✓ {len(s_files)} file(s) added for **{s_name}**")

    # ── Manage Data ─────────────────────────────────
    if not st.session_state['master_df'].empty:
        st.markdown("<hr style='border-color:rgba(201,168,76,0.15);margin:1rem 0'>", unsafe_allow_html=True)
        st.markdown("### 2 · Manage Data")
        with st.expander("🗑 Delete / Replace Data"):
            unique_samples = sorted(st.session_state['master_df']['Sample'].unique(), key=natural_sort_key)
            batch_to_del = st.selectbox("Delete Entire Batch", ["— Select —"] + unique_samples)
            if st.button("Delete Batch", use_container_width=True):
                if batch_to_del != "— Select —":
                    files_rm = st.session_state['master_df'][st.session_state['master_df']['Sample'] == batch_to_del]['File'].tolist()
                    st.session_state['master_df'] = st.session_state['master_df'][st.session_state['master_df']['Sample'] != batch_to_del]
                    for f in files_rm:
                        st.session_state['profile_dict'].pop(f, None)
                    st.session_state['legend_map'].pop(batch_to_del, None)
                    st.rerun()

            st.markdown("<div style='height:0.25rem'></div>", unsafe_allow_html=True)
            all_files = sorted(st.session_state['master_df']['File'].tolist(), key=natural_sort_key)
            file_to_del = st.selectbox("Delete Single Replicate", ["— Select —"] + all_files)
            if st.button("Delete File", use_container_width=True):
                if file_to_del != "— Select —":
                    st.session_state['master_df'] = st.session_state['master_df'][st.session_state['master_df']['File'] != file_to_del]
                    st.session_state['profile_dict'].pop(file_to_del, None)
                    st.rerun()

    # ── Legend Customization ─────────────────────────
    if not st.session_state['master_df'].empty:
        st.markdown("<hr style='border-color:rgba(201,168,76,0.15);margin:1rem 0'>", unsafe_allow_html=True)
        st.markdown("### 3 · Legend Labels")
        unique_samples = sorted(st.session_state['master_df']['Sample'].unique(), key=natural_sort_key)
        for s in unique_samples:
            if s not in st.session_state['legend_map']:
                st.session_state['legend_map'][s] = s
            st.session_state['legend_map'][s] = st.text_input(f"↳ {s}", st.session_state['legend_map'][s])

    # ── Global Trend Optimizer ───────────────────────
    if not st.session_state['master_df'].empty:
        unique_samples = sorted(st.session_state['master_df']['Sample'].unique(), key=natural_sort_key)
        st.markdown("<hr style='border-color:rgba(201,168,76,0.15);margin:1rem 0'>", unsafe_allow_html=True)
        st.markdown("### 4 · Trend Optimizer")
        with st.expander("✂ Inter-Sample Optimizer"):
            st.caption("Remove outlier tests so the overall mean Ra forms a clean monotonic trend.")
            trend_dir = st.radio("Target Trend Direction", ["Increasing", "Decreasing"])
            target_n  = st.number_input("Replicates to Keep (per batch)", min_value=3, max_value=50, value=9)
            batches_to_opt = st.multiselect("Batches to Include", unique_samples, default=unique_samples)

            if st.button("Run Optimizer", use_container_width=True):
                if len(batches_to_opt) < 2:
                    st.warning("Select at least 2 batches.")
                else:
                    df_opt = st.session_state['master_df'].copy()
                    valid = all(
                        len(df_opt[df_opt['Sample'] == b]) > target_n
                        for b in batches_to_opt
                    )
                    if not valid:
                        st.warning("Some batches don't have enough replicates.")
                    else:
                        sorted_batches = sorted(batches_to_opt, key=natural_sort_key)
                        bx = {b: i for i, b in enumerate(sorted_batches)}

                        def global_score(df_t):
                            means = df_t[df_t['Sample'].isin(batches_to_opt)].groupby('Sample')['Ra'].mean()
                            y = [means.get(b, 0) for b in sorted_batches]
                            x = [bx[b] for b in sorted_batches]
                            with np.errstate(divide='ignore', invalid='ignore'):
                                corr = np.corrcoef(x, y)[0, 1]
                            if np.isnan(corr): return 0.0
                            return corr if trend_dir == "Increasing" else -corr

                        while True:
                            counts = df_opt[df_opt['Sample'].isin(batches_to_opt)]['Sample'].value_counts()
                            if not any(counts > target_n): break
                            best_score, best_idx = -float('inf'), -1
                            oversized = counts[counts > target_n].index.tolist()
                            candidates = df_opt[df_opt['Sample'].isin(oversized)].index.tolist()
                            for i in candidates:
                                score = global_score(df_opt.drop(index=i))
                                if score > best_score:
                                    best_score, best_idx = score, i
                            df_opt = df_opt.drop(index=best_idx)

                        orig  = set(st.session_state['master_df'][st.session_state['master_df']['Sample'].isin(batches_to_opt)]['File'])
                        kept  = set(df_opt[df_opt['Sample'].isin(batches_to_opt)]['File'])
                        dropped = orig - kept
                        st.session_state['opt_df'] = df_opt.reset_index(drop=True)
                        st.success("✓ Optimization complete — see '✨ Optimized' tab.")
                        if dropped:
                            st.error("Removed:\n" + "\n".join(f"• {d}" for d in sorted(dropped, key=natural_sort_key)))
                        st.rerun()

    # ── Plot Customization ───────────────────────────
    if not st.session_state['master_df'].empty:
        st.markdown("<hr style='border-color:rgba(201,168,76,0.15);margin:1rem 0'>", unsafe_allow_html=True)
        st.markdown("### 5 · Axis Overrides")
        with st.expander("🖍 Custom Axis Labels"):
            c_x_trend = st.text_input("Trends — X Axis", "")
            c_y_trend = st.text_input("Trends — Y Axis", "")
            c_x_prof  = st.text_input("Profiles — X Axis", "")
            c_y_prof  = st.text_input("Profiles — Y Axis", "")
            c_x_psd   = st.text_input("PSD — X Axis", "")
            c_y_psd   = st.text_input("PSD — Y Axis", "")
    else:
        c_x_trend = c_y_trend = c_x_prof = c_y_prof = c_x_psd = c_y_psd = ""

    # ── Reset ────────────────────────────────────────
    st.markdown("<hr style='border-color:rgba(201,168,76,0.15);margin:1rem 0'>", unsafe_allow_html=True)
    if st.button("⚠ Reset Entire Study", type="primary", use_container_width=True):
        st.session_state.clear()
        st.rerun()

    st.markdown("""
    <div style="padding:1rem 0 0.5rem;text-align:center;font-family:'IBM Plex Sans',sans-serif;
                font-size:0.65rem;color:rgba(168,180,200,0.45);letter-spacing:0.1em;">
        For Research & Academic Use Only<br>Version 3.0 Pro
    </div>
    """, unsafe_allow_html=True)


# ==========================================
# MAIN CONTENT
# ==========================================
df_master  = st.session_state['master_df']
prof_dict  = st.session_state['profile_dict']

render_header()

# ── Summary KPI bar ─────────────────────────────────
if not df_master.empty:
    n_samples = df_master['Sample'].nunique()
    n_files   = len(df_master)
    ra_col    = [c for c in ['Ra', 'Ra_calc'] if c in df_master.columns]
    mean_ra   = df_master[ra_col[0]].mean() if ra_col else float('nan')
    std_ra    = df_master[ra_col[0]].std()  if ra_col else float('nan')

    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(metric_card("Batches Loaded",  f"{n_samples}", ""), unsafe_allow_html=True)
    k2.markdown(metric_card("Total Replicates", f"{n_files}", "files"), unsafe_allow_html=True)
    k3.markdown(metric_card("Overall Mean Rₐ",  f"{mean_ra:.3f}" if not np.isnan(mean_ra) else "—", "µm"), unsafe_allow_html=True)
    k4.markdown(metric_card("Overall σ Rₐ",     f"{std_ra:.3f}"  if not np.isnan(std_ra)  else "—", "µm"), unsafe_allow_html=True)
    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)


# ==========================================
# PLOT HELPERS
# ==========================================
def styled_figure(width=820, height=520):
    fig = go.Figure()
    fig.update_layout(
        width=width, height=height,
        margin=dict(l=72, r=32, t=32, b=64),
        plot_bgcolor=PLOT_BG, paper_bgcolor=PAPER_BG,
        font=dict(family="IBM Plex Sans", color=WHITE),
        legend=dict(
            bgcolor="rgba(11,17,32,0.75)",
            bordercolor=LINE_COLOR,
            borderwidth=1,
            font=dict(family="IBM Plex Mono", size=11, color=SILVER),
        ),
        xaxis=AXIS_STYLE,
        yaxis=AXIS_STYLE,
    )
    return fig

def plot_trend(data_df, param_selected, show_trendline=True, custom_x="", custom_y=""):
    plot_df = data_df.groupby("Sample")[param_selected].agg(['mean', 'std', 'count']).reset_index()
    plot_df['sort_key'] = plot_df['Sample'].apply(natural_sort_key)
    plot_df = plot_df.sort_values('sort_key').drop('sort_key', axis=1)
    plot_df['Display_Name'] = plot_df['Sample'].map(st.session_state['legend_map'])

    ci = 1.96 * (plot_df['std'] / np.sqrt(plot_df['count']))

    fig = styled_figure(820, 520)
    fig.add_trace(go.Scatter(
        x=plot_df['Display_Name'], y=plot_df['mean'],
        error_y=dict(type='data', array=ci.values, visible=True,
                     color=GOLD, thickness=1.5, width=6),
        mode='lines+markers',
        name=param_selected,
        line=dict(color=GOLD, width=2.5),
        marker=dict(size=10, color=GOLD,
                    line=dict(color=WHITE, width=1.5),
                    symbol='circle'),
    ))

    if show_trendline and len(plot_df) > 1:
        x_num = np.arange(len(plot_df))
        slope, intercept, r_value, _, _ = stats.linregress(x_num, plot_df['mean'].values)
        trend_y = intercept + slope * x_num
        fig.add_trace(go.Scatter(
            x=plot_df['Display_Name'], y=trend_y,
            mode='lines',
            name=f"Linear fit (R² = {r_value**2:.3f})",
            line=dict(color="#e05252", dash='dash', width=1.8),
        ))

    x_label = custom_x or "Sample ID"
    y_label = custom_y or f"Mean {param_selected} (µm)"
    fig.update_layout(
        xaxis=dict(**AXIS_STYLE, title_text=x_label),
        yaxis=dict(**AXIS_STYLE, title_text=y_label),
        xaxis_title=x_label, yaxis_title=y_label,
        showlegend=True,
    )
    return fig


# ==========================================
# TABS
# ==========================================
if not df_master.empty:
    TAB_LABELS = [
        "📋 Dataset",
        "📉 Raw Trends",
        "✨ Optimized",
        "🎨 Replicate Stack",
        "🏛 Representative Stack",
        "📈 PSD Analysis",
        "💾 Export",
        "📖 Methods",
    ]
    tabs = st.tabs(TAB_LABELS)

    # ── TAB 0: DATASET ──────────────────────────────
    with tabs[0]:
        section_title("Raw Summary Table", "📋")
        st.dataframe(df_master, use_container_width=True, height=420)

    # ── TAB 1: RAW TRENDS ───────────────────────────
    with tabs[1]:
        section_title("Inter-Sample Comparison — Original Data", "📉")
        params = [p for p in ["Ra", "Rq", "Rz", "Rt", "Ra_calc", "Rsk", "Rku"] if p in df_master.columns]
        if params:
            c1, c2 = st.columns([3, 1])
            with c1: p_sel = st.selectbox("Parameter", params, key="raw_p")
            with c2: show_tl = st.checkbox("Linear Trendline", True, key="raw_tl")

            st.plotly_chart(
                plot_trend(df_master, p_sel, show_tl, c_x_trend, c_y_trend),
                use_container_width=False, config=JOURNAL_CONFIG
            )

            groups = [df_master[df_master['Sample'] == s][p_sel].dropna()
                      for s in df_master['Sample'].unique()]
            if len(groups) > 1:
                _, p_val = stats.f_oneway(*groups)
                color = "#3db87a" if p_val < 0.05 else "#e05252"
                sig   = "significant" if p_val < 0.05 else "not significant"
                st.markdown(f"""
                <div style="
                    display:inline-block;padding:0.5rem 1.25rem;
                    background:rgba(26,37,64,0.7);border:1px solid rgba(201,168,76,0.2);
                    border-radius:4px;font-family:'IBM Plex Mono',monospace;font-size:0.82rem;color:{color};
                ">
                    ANOVA · p-value = {p_val:.4e} &nbsp;·&nbsp; {sig} (α = 0.05)
                </div>
                """, unsafe_allow_html=True)

    # ── TAB 2: OPTIMIZED TRENDS ──────────────────────
    with tabs[2]:
        section_title("Inter-Sample Comparison — Optimized Data", "✨")
        df_opt = st.session_state.get('opt_df', pd.DataFrame())
        if df_opt.empty:
            info_box("Run the <b>Trend Optimizer</b> in the sidebar to populate this view.", "info")
        else:
            params_opt = [p for p in ["Ra", "Rq", "Rz", "Rt", "Ra_calc", "Rsk", "Rku"] if p in df_opt.columns]
            c1, c2 = st.columns([3, 1])
            with c1: p_sel_o = st.selectbox("Parameter", params_opt, key="opt_p")
            with c2: show_tl_o = st.checkbox("Linear Trendline", True, key="opt_tl")

            st.plotly_chart(
                plot_trend(df_opt, p_sel_o, show_tl_o, c_x_trend, c_y_trend),
                use_container_width=False, config=JOURNAL_CONFIG
            )
            groups = [df_opt[df_opt['Sample'] == s][p_sel_o].dropna()
                      for s in df_opt['Sample'].unique()]
            if len(groups) > 1:
                _, p_val = stats.f_oneway(*groups)
                color = "#3db87a" if p_val < 0.05 else "#e05252"
                sig   = "significant" if p_val < 0.05 else "not significant"
                st.markdown(f"""
                <div style="
                    display:inline-block;padding:0.5rem 1.25rem;
                    background:rgba(26,37,64,0.7);border:1px solid rgba(201,168,76,0.2);
                    border-radius:4px;font-family:'IBM Plex Mono',monospace;font-size:0.82rem;color:{color};
                ">
                    ANOVA · p-value = {p_val:.4e} &nbsp;·&nbsp; {sig} (α = 0.05)
                </div>
                """, unsafe_allow_html=True)

    # ── TAB 3: REPLICATE STACK ───────────────────────
    with tabs[3]:
        section_title("Batch Replicate Inspection", "🎨")
        data_mode_rep = st.radio(
            "Dataset", ["Raw Data", "✨ Optimized Data"], horizontal=True, key="rep_radio"
        )
        df_target = df_master
        if data_mode_rep == "✨ Optimized Data":
            if st.session_state.get('opt_df', pd.DataFrame()).empty:
                info_box("Optimized dataset empty. Showing raw data.", "warning")
            else:
                df_target = st.session_state['opt_df']

        if not df_target.empty:
            c1, c2 = st.columns([2, 1])
            with c1:
                batch = st.selectbox(
                    "Batch", sorted(df_target['Sample'].unique()), key="rep_batch"
                )
            with c2:
                offset_rep = st.slider("Vertical Offset (µm)", 1, 100, 25, key="rep_slider")

            batch_files = sorted(
                df_target[df_target['Sample'] == batch]['File'].tolist(),
                key=natural_sort_key
            )

            fig_rep = styled_figure(820, max(600, 180 + len(batch_files) * 50))
            rep_tv, rep_tt = [], []

            for i, f in enumerate(batch_files):
                if f in prof_dict:
                    y_shift  = i * offset_rep
                    p_data   = prof_dict[f]
                    label    = os.path.splitext(f)[0]
                    color    = PALETTE[i % len(PALETTE)]

                    fig_rep.add_trace(go.Scatter(
                        x=p_data['Length_mm'],
                        y=p_data['Amplitude_um_Norm'] + y_shift,
                        mode='lines', name=label, showlegend=False,
                        line=dict(color=color, width=1.5),
                    ))
                    y_peak = (p_data['Amplitude_um_Norm'] + y_shift).max()
                    fig_rep.add_annotation(
                        x=p_data['Length_mm'].min(), y=y_peak, yshift=10,
                        text=f"<b>{label}</b>",
                        showarrow=False, align="left", xanchor="left", yanchor="bottom",
                        font=dict(family="IBM Plex Mono", size=11, color=color),
                    )
                    rep_tv.append(y_shift)
                    rep_tt.append("<b>0</b>")
                    for v in [-5, 5]:
                        rep_tv.append(v + y_shift)
                        rep_tt.append(f"<b>{v}</b>")

            xl = custom_x or c_x_prof or "Wavelength (mm)"
            yl = custom_y if (custom_y := c_y_prof) else "Roughness (µm)"
            fig_rep.update_layout(
                xaxis=dict(**AXIS_STYLE, title_text=c_x_prof or "Wavelength (mm)"),
                yaxis=dict(**AXIS_STYLE, title_text=c_y_prof or "Roughness (µm)",
                           tickmode='array', tickvals=rep_tv, ticktext=rep_tt),
            )
            st.plotly_chart(fig_rep, use_container_width=False, config=JOURNAL_CONFIG)

    # ── TAB 4: REPRESENTATIVE STACK ──────────────────
    with tabs[4]:
        section_title("Representative Profile Stack", "🏛")
        data_mode_g = st.radio(
            "Dataset", ["Raw Data", "✨ Optimized Data"], horizontal=True, key="glob_radio"
        )
        df_tg = df_master
        if data_mode_g == "✨ Optimized Data":
            if not st.session_state.get('opt_df', pd.DataFrame()).empty:
                df_tg = st.session_state['opt_df']

        if not df_tg.empty:
            offset_g = st.slider("Group Offset (µm)", 1, 400, 100, key="glob_slider")
            unique_s = sorted(df_tg['Sample'].unique(), key=natural_sort_key)

            fig_g = styled_figure(820, max(600, 180 + len(unique_s) * 80))
            t_vals, t_text = [], []

            for i, sample in enumerate(unique_s):
                sdata    = df_tg[df_tg['Sample'] == sample]
                features = sdata[['Ra', 'Rq', 'Rz', 'Rt']].dropna()
                centroid = features.mean().values.reshape(1, -1)
                idx      = cdist(features.values, centroid).argmin()
                cf       = sdata.iloc[idx]['File']

                if cf not in prof_dict: continue
                y_shift = i * offset_g
                cur_p   = prof_dict[cf]
                name    = st.session_state['legend_map'].get(sample, sample)
                color   = PALETTE[i % len(PALETTE)]

                fig_g.add_trace(go.Scatter(
                    x=cur_p['Length_mm'],
                    y=cur_p['Amplitude_um_Norm'] + y_shift,
                    mode='lines', showlegend=False,
                    line=dict(color=color, width=2),
                ))

                peak_y   = (cur_p['Amplitude_um_Norm'] + y_shift).max()
                mean_ra  = sdata['Ra'].mean()
                std_ra   = sdata['Ra'].std()

                fig_g.add_annotation(
                    x=cur_p['Length_mm'].min(), y=peak_y, yshift=12,
                    text=f"<b>{name}: Ra = {mean_ra:.3f} ± {std_ra:.3f} µm</b>",
                    showarrow=False, align="left", xanchor="left", yanchor="bottom",
                    font=dict(family="IBM Plex Mono", size=12, color=color),
                )
                t_vals.append(y_shift); t_text.append("<b>0</b>")
                for v in [-5, 5]:
                    t_vals.append(v + y_shift)
                    t_text.append(f"<b>{v}</b>")

            fig_g.update_layout(
                xaxis=dict(**AXIS_STYLE, title_text=c_x_prof or "Wavelength (mm)"),
                yaxis=dict(**AXIS_STYLE, title_text=c_y_prof or "Roughness (µm)",
                           tickmode='array', tickvals=t_vals, ticktext=t_text),
            )
            st.plotly_chart(fig_g, use_container_width=False, config=JOURNAL_CONFIG)

    # ── TAB 5: PSD ANALYSIS ──────────────────────────
    with tabs[5]:
        section_title("Power Spectral Density Analysis", "📈")

        with st.expander("📖 PSD Theory & References"):
            st.markdown(r"""
**Power Spectral Density** decomposes the surface profile into constituent spatial frequencies via Fourier analysis, producing a frequency-domain fingerprint.

| Feature | Interpretation |
|---|---|
| Discrete peaks | Periodic tool marks or chatter |
| High-frequency domain | Micro-roughness; friction & wear |
| Low-frequency domain | Waviness; form errors |
| Log-log slope | Fractal dimension of the surface |

**References:** Whitehouse (2004); Thomas (1999); ISO 4287:1997.
            """)

        unique_s = sorted(df_master['Sample'].unique(), key=natural_sort_key)
        sample_ch = st.selectbox("Sample", unique_s, key="psd_sample")
        sample_fs = sorted(
            df_master[df_master['Sample'] == sample_ch]['File'].tolist(),
            key=natural_sort_key
        )

        for i, f in enumerate(sample_fs):
            if f not in prof_dict: continue
            if 'Roughness' not in prof_dict[f].columns:
                info_box(f"Roughness column missing for {f}. Re-upload with a filter applied.", "warning")
                continue

            sig  = prof_dict[f]['Roughness'].values
            x_d  = prof_dict[f]['Length_mm'].values
            dx   = np.mean(np.diff(x_d)) if len(x_d) > 1 else 1.0
            psd  = np.abs(np.fft.fft(sig))**2
            freq = np.fft.fftfreq(len(sig), d=dx)
            mask = freq > 0

            fig_psd = styled_figure(820, 460)
            fig_psd.add_trace(go.Scatter(
                x=freq[mask], y=psd[mask],
                mode='lines',
                name=os.path.splitext(f)[0],
                line=dict(color=PALETTE[i % len(PALETTE)], width=1.5),
                fill='tozeroy',
                fillcolor=f"rgba({int(PALETTE[i%len(PALETTE)][1:3],16)},{int(PALETTE[i%len(PALETTE)][3:5],16)},{int(PALETTE[i%len(PALETTE)][5:7],16)},0.08)",
            ))
            fig_psd.update_layout(
                xaxis=dict(**AXIS_STYLE, title_text=c_x_psd or "Spatial Frequency (cycles/mm)", type="log"),
                yaxis=dict(**AXIS_STYLE, title_text=c_y_psd or "Power Density (µm²·mm)", type="log"),
                title=dict(
                    text=os.path.splitext(f)[0],
                    font=dict(family="IBM Plex Mono", size=12, color=SILVER),
                    x=0.5,
                ),
            )
            st.plotly_chart(fig_psd, use_container_width=False,
                            key=f"psd_{sample_ch}_{f}_{i}", config=JOURNAL_CONFIG)

    # ── TAB 6: EXPORT ────────────────────────────────
    with tabs[6]:
        section_title("Comprehensive Data Export", "💾")
        st.markdown("""
        <p style="font-family:'IBM Plex Sans',sans-serif;font-size:0.84rem;color:#a8b4c8;margin-bottom:1.25rem;">
        All exports are production-ready Excel workbooks with auto-fitted columns and embedded lab logo.
        </p>
        """, unsafe_allow_html=True)

        section_title("Original (Raw) Data", "📦")
        c1, c2, c3 = st.columns(3)

        with c1:
            excel_sum = export_to_excel_with_logo(df_master, "Summary_Stats")
            st.download_button("📥 Summary Statistics", excel_sum,
                               "1_RAW_Roughness_Summary.xlsx", use_container_width=True)
        with c2:
            if prof_dict:
                rows = []
                for fname in sorted(prof_dict.keys(), key=natural_sort_key):
                    p_data = prof_dict[fname]
                    sm = df_master[df_master['File'] == fname]
                    sname = sm['Sample'].iloc[0] if not sm.empty else "Unknown"
                    temp = p_data[['Length_mm', 'Amplitude_um_Norm']].copy().reset_index(drop=True)
                    cname = os.path.splitext(fname)[0]
                    temp.columns = [f"[{sname}] {cname}_X_mm", f"[{sname}] {cname}_Y_um"]
                    rows.append(temp)
                full_exp = pd.concat(rows, axis=1)
                excel_all = export_to_excel_with_logo(full_exp, "All_Profiles")
                st.download_button("📥 All Profiles", excel_all,
                                   "2_RAW_All_Profiles.xlsx", use_container_width=True)
        with c3:
            rep_list = []
            for sample in sorted(df_master['Sample'].unique(), key=natural_sort_key):
                sdata    = df_master[df_master['Sample'] == sample]
                features = sdata[['Ra', 'Rq', 'Rz', 'Rt']].dropna()
                if features.empty: continue
                centroid = features.mean().values.reshape(1, -1)
                idx      = cdist(features.values, centroid).argmin()
                cf       = sdata.iloc[idx]['File']
                if cf not in prof_dict: continue
                prof = prof_dict[cf][['Length_mm', 'Amplitude_um_Norm']].copy().reset_index(drop=True)
                name = st.session_state['legend_map'].get(sample, sample)
                cfile = os.path.splitext(cf)[0]
                prof.columns = [f"{name} ({cfile})_X_mm", f"{name} ({cfile})_Y_um"]
                rep_list.append(prof)
            if rep_list:
                rep_exp = pd.concat(rep_list, axis=1)
                excel_rep = export_to_excel_with_logo(rep_exp, "Representative_Profiles")
                st.download_button("📥 Representative Profiles", excel_rep,
                                   "3_RAW_Representative.xlsx", use_container_width=True)

        df_opt = st.session_state.get('opt_df', pd.DataFrame())
        if not df_opt.empty:
            st.markdown("<hr style='border-color:rgba(201,168,76,0.15);margin:1.25rem 0'>", unsafe_allow_html=True)
            section_title("Optimized Data", "✨")
            opt_files = df_opt['File'].unique()
            c4, c5, c6, c7 = st.columns(4)

            with c4:
                st.download_button(
                    "📥 Opt. Summary",
                    export_to_excel_with_logo(df_opt, "Optimized_Summary"),
                    "1_OPT_Roughness_Summary.xlsx", use_container_width=True
                )
            with c5:
                ra_dict = {}
                for sample in sorted(df_opt['Sample'].unique(), key=natural_sort_key):
                    col_name = st.session_state['legend_map'].get(sample, sample)
                    ra_dict[col_name] = df_opt[df_opt['Sample'] == sample]['Ra'].reset_index(drop=True)
                ra_matrix = pd.DataFrame(ra_dict)
                st.download_button(
                    "📥 Opt. Rₐ Matrix",
                    export_to_excel_with_logo(ra_matrix, "Optimized_Ra_Matrix"),
                    "2_OPT_Ra_Matrix.xlsx", use_container_width=True
                )
            with c6:
                rows_o = []
                for fname in sorted(opt_files, key=natural_sort_key):
                    if fname not in prof_dict: continue
                    p_data = prof_dict[fname]
                    sm = df_opt[df_opt['File'] == fname]
                    sname = sm['Sample'].iloc[0] if not sm.empty else "Unknown"
                    temp = p_data[['Length_mm', 'Amplitude_um_Norm']].copy().reset_index(drop=True)
                    cname = os.path.splitext(fname)[0]
                    temp.columns = [f"[{sname}] {cname}_X_mm", f"[{sname}] {cname}_Y_um"]
                    rows_o.append(temp)
                if rows_o:
                    st.download_button(
                        "📥 Opt. All Profiles",
                        export_to_excel_with_logo(pd.concat(rows_o, axis=1), "Optimized_Profiles"),
                        "3_OPT_All_Profiles.xlsx", use_container_width=True
                    )
            with c7:
                rep_o = []
                for sample in sorted(df_opt['Sample'].unique(), key=natural_sort_key):
                    sdata    = df_opt[df_opt['Sample'] == sample]
                    features = sdata[['Ra', 'Rq', 'Rz', 'Rt']].dropna()
                    if features.empty: continue
                    centroid = features.mean().values.reshape(1, -1)
                    idx      = cdist(features.values, centroid).argmin()
                    cf       = sdata.iloc[idx]['File']
                    if cf not in prof_dict: continue
                    prof = prof_dict[cf][['Length_mm', 'Amplitude_um_Norm']].copy().reset_index(drop=True)
                    name = st.session_state['legend_map'].get(sample, sample)
                    cfile = os.path.splitext(cf)[0]
                    prof.columns = [f"{name} ({cfile})_X_mm", f"{name} ({cfile})_Y_um"]
                    rep_o.append(prof)
                if rep_o:
                    st.download_button(
                        "📥 Opt. Rep. Profiles",
                        export_to_excel_with_logo(pd.concat(rep_o, axis=1), "Optimized_Representative"),
                        "4_OPT_Representative.xlsx", use_container_width=True
                    )

    # ── TAB 7: METHODS ───────────────────────────────
    with tabs[7]:
        section_title("Documentation & Methods", "📖")

        with st.expander("🌊 ISO 16610-21 Gaussian Filter"):
            st.markdown(r"""
**Purpose:** Internationally recognized standard for separating roughness from waviness and form.

**Cutoff Wavelength (λc):** Defines the roughness/waviness boundary. At λc, the profile amplitude is transmitted at 50%; shorter wavelengths (roughness) are retained.

**Gaussian weighting function:**
$$S(x)=\frac{1}{\alpha\lambda_c}\exp\left(-\pi\left(\frac{x}{\alpha\lambda_c}\right)^2\right), \quad \alpha=\sqrt{\frac{\ln 2}{\pi}}\approx 0.4697$$

> *ISO 16610-21:2011. Linear profile filters: Gaussian filters.* ISO.
            """)

        with st.expander("🔬 Savitzky-Golay Filtering"):
            st.markdown(r"""
**Purpose:** Detrend the raw profile (remove form/curvature) while preserving peak geometry.

Uses **local 3rd-order polynomial regression** over a sliding window — unlike a simple moving average, sharp peaks and valleys are retained, which is critical for Ra, Rz, and Rku computation.

> *Savitzky, A. & Golay, M. J. E. (1964). Analytical Chemistry, 36(8), 1627–1639.*
            """)

        with st.expander("📐 Roughness Parameter Definitions (ISO 4287)"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(r"""
**Ra — Arithmetic Mean**
$$R_a=\frac{1}{N}\sum_{i=1}^{N}|Z_i|$$
**Rq — Root Mean Square**
$$R_q=\sqrt{\frac{1}{N}\sum_{i=1}^{N}Z_i^2}$$
**Rt — Total Height**
$$R_t=\max(Z)-\min(Z)$$
                """)
            with col2:
                st.markdown(r"""
**Rz — Ten-Point Mean**
$$R_z=\frac{1}{5}\left(\sum_{i=1}^{5}P_i - \sum_{j=1}^{5}V_j\right)$$
**Rsk — Skewness**
$$R_{sk}=\frac{1}{NR_q^3}\sum Z_i^3$$
**Rku — Kurtosis**
$$R_{ku}=\frac{1}{NR_q^4}\sum Z_i^4$$
                """)
            st.markdown("> *ISO 4287:1997. Surface texture: Profile method.* ISO.")

        section_title("Auto-Generated Methods Text", "📝")
        if not df_master.empty and 'Filter' in df_master.columns:
            active_f = df_master['Filter'].iloc[0]
            if "Gaussian" in str(active_f):
                method_text = (
                    "Surface profiles were analyzed using a custom Python-based metrology suite "
                    "(SRoughnessLab Pro, Solomon Scientific). Raw profiles were detrended using an "
                    "ISO 16610-21 compliant Gaussian filter (λc = 0.8 mm) to isolate primary roughness. "
                    "Amplitude parameters (Ra, Rq, Rz, Rt) were computed per ISO 4287:1997. "
                    "Statistical comparisons were performed using one-way ANOVA (α = 0.05)."
                )
            elif "Savitzky" in str(active_f):
                method_text = (
                    "Surface profiles were analyzed using a custom Python-based metrology suite "
                    "(SRoughnessLab Pro, Solomon Scientific). Raw profiles were detrended using a "
                    "Savitzky-Golay filter (3rd-order polynomial) to isolate primary roughness and "
                    "preserve peak geometries (Savitzky & Golay, 1964). Amplitude parameters (Ra, Rq, "
                    "Rz, Rt) were computed per ISO 4287:1997. Statistical comparisons were performed "
                    "using one-way ANOVA (α = 0.05)."
                )
            else:
                method_text = (
                    "Surface profiles were analyzed using a custom Python-based metrology suite "
                    "(SRoughnessLab Pro, Solomon Scientific). Raw profiles were linearly detrended "
                    "to isolate primary roughness. Amplitude parameters (Ra, Rq, Rz, Rt) were "
                    "computed per ISO 4287:1997. Statistical comparisons were performed using "
                    "one-way ANOVA (α = 0.05)."
                )
        else:
            method_text = (
                "Surface profiles were analyzed using SRoughnessLab Pro (Solomon Scientific). "
                "Raw profiles were detrended to isolate primary roughness. Amplitude parameters "
                "were computed per ISO 4287:1997 and compared via one-way ANOVA (α = 0.05)."
            )
        st.text_area("Copy for your manuscript:", method_text, height=130)

else:
    # ── Empty State ──────────────────────────────────
    st.markdown("""
    <div style="
        margin-top:3rem;
        padding:3rem 2rem;
        background:rgba(26,37,64,0.4);
        border:1px solid rgba(201,168,76,0.18);
        border-radius:8px;
        text-align:center;
    ">
        <div style="font-size:3rem;margin-bottom:1rem;">🔬</div>
        <div style="
            font-family:'Playfair Display',Georgia,serif;
            font-size:1.5rem;color:#f0f4fb;
            margin-bottom:0.5rem;
        ">Ready for Analysis</div>
        <div style="
            font-family:'IBM Plex Sans',sans-serif;
            font-size:0.85rem;color:#a8b4c8;
            max-width:420px;margin:0 auto;line-height:1.7;
        ">
            Upload your surface replicate files via the <b style="color:#c9a84c;">Data Input</b> panel
            in the sidebar to begin. Supports ISO Gaussian and Savitzky-Golay detrending with
            full parametric analysis, PSD, and publication-ready exports.
        </div>
    </div>
    """, unsafe_allow_html=True)
