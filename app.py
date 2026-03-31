import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy import stats
import re

# ==========================================
# 1. UNIVERSAL SCIENTIFIC LOADER
# ==========================================
class RoughnessLoader:
    def __init__(self):
        self.targets = {
            'Ra': ['ra', 'arithmetic', 'average roughness'],
            'Rq': ['rq', 'rms'],
            'Rz': ['rz', 'max height'],
            'Rt': ['rt', 'total height']
        }

    def clean_value(self, val):
        if pd.isna(val) or str(val).strip() == "": return np.nan
        if isinstance(val, (int, float)): return float(val)
        text = str(val).replace(',', '.').strip()
        match = re.search(r"[-+]?\d*\.\d+|\d+", text)
        return float(match.group()) if match else np.nan

    def process_files(self, uploaded_files, metadata_list):
        combined_rows = []
        for file, meta in zip(uploaded_files, metadata_list):
            try:
                xl = pd.ExcelFile(file)
                sheet_map = {s.strip().upper(): s for s in xl.sheet_names}
                row_data = meta.copy()
                
                # Pre-initialize scientific columns to prevent KeyErrors
                row_data.update({'Ra_mean': np.nan, 'n_samples': 0, 'std_err': 0})

                # --- PASS 1: REPLICATE EXTRACTION (Sheet: DATA, Col: E, F) ---
                if "DATA" in sheet_map:
                    df_data = pd.read_excel(file, sheet_name=sheet_map["DATA"], usecols="E:F", header=None)
                    points = [self.clean_value(x) for x in df_data.values.flatten() if not pd.isna(self.clean_value(x))]
                    if points:
                        row_data['Ra_mean'] = np.mean(points)
                        row_data['n_samples'] = len(points)
                        row_data['std_err'] = stats.sem(points) if len(points) > 1 else 0

                # --- PASS 2: SUMMARY PARAMETER SCAN (All Sheets) ---
                for sheet in xl.sheet_names:
                    df = xl.parse(sheet, header=None)
                    for r in range(min(len(df), 100)):
                        for c in range(len(df.columns)):
                            cell_str = str(df.iloc[r, c]).lower().strip()
                            for std_name, keywords in self.targets.items():
                                if any(k in cell_str for k in keywords) and std_name not in row_data:
                                    # Check Right, then Down
                                    val = np.nan
                                    if c + 1 < len(df.columns): val = self.clean_value(df.iloc[r, c+1])
                                    if np.isnan(val) and r + 1 < len(df): val = self.clean_value(df.iloc[r+1, c])
                                    
                                    if not np.isnan(val): row_data[std_name] = val
                
                combined_rows.append(row_data)
            except Exception as e:
                st.error(f"Error in {file.name}: {e}")
        return pd.DataFrame(combined_rows)

# ==========================================
# 2. UI & ANALYSIS ENGINE
# ==========================================
st.set_page_config(page_title="Roughness Sci-Lab", layout="wide")
st.title("🔬 Surface Roughness Scientific Analyzer")

st.sidebar.header("1. Data Upload")
uploaded_files = st.sidebar.file_uploader("Upload Excel Files", accept_multiple_files=True, type=['xlsx'])

if uploaded_files:
    mat_id = st.sidebar.text_input("Material/Sample ID", "Sample_01")
    cond = st.sidebar.selectbox("Ageing Condition", ["Control", "Oven", "UV", "Humidity", "Other"])
    day = st.sidebar.number_input("Ageing Day", min_value=0, step=1)

    if st.sidebar.button("Run Batch Analysis", type="primary"):
        loader = RoughnessLoader()
        metas = [{"File": f.name, "Material": mat_id, "Condition": cond, "Day": day} for f in uploaded_files]
        st.session_state['master_df'] = loader.process_files(uploaded_files, metas)

if 'master_df' in st.session_state:
    df = st.session_state['master_df']
    
    t1, t2, t3 = st.tabs(["📊 Data Quality (n≥9)", "📈 Statistics", "📉 Visuals"])

    with t1:
        st.subheader("Scientific Replicate Verification")
        if not df.empty:
            # Color logic for sample size quality
            def color_quality(val):
                return 'color: red; font-weight: bold' if val < 9 else 'color: green'
            
            # Select valid columns for display
            display_cols = [c for c in ['File', 'Day', 'n_samples', 'Ra_mean', 'std_err', 'Ra', 'Rz'] if c in df.columns]
            st.dataframe(df[display_cols].style.applymap(color_quality, subset=['n_samples']), use_container_width=True)
            st.info("💡 Red 'n_samples' indicates fewer than 9 measurements (Scientific threshold).")
        else:
            st.warning("No data extracted. Check your file headers.")

    with t2:
        st.subheader("Comparative Statistics")
        params = [p for p in ["Ra_mean", "Ra", "Rq", "Rz", "Rt"] if p in df.columns]
        if params:
            sel_p = st.selectbox("Parameter", params)
            group = st.radio("Group By", ["Condition", "Day", "Material"])
            
            if len(df[group].unique()) > 1:
                grps = [df[df[group] == g][sel_p].dropna() for g in df[group].unique()]
                if all(len(g) > 0 for g in grps):
                    f, p = stats.f_oneway(*grps)
                    st.metric("ANOVA p-value", f"{p:.4f}")
                    if p < 0.05: st.success("Significant difference detected between groups.")
        else:
            st.error("No numeric parameters found for analysis.")

    with t3:
        if 'sel_p' in locals():
            st.subheader(f"Distribution of {sel_p}")
            fig = px.box(df, x=group, y=sel_p, color="Condition", points="all", notched=True)
            st.plotly_chart(fig, use_container_width=True)
