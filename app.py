import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy import stats
import re

# ==========================================
# 1. DUAL-PASS SCIENTIFIC LOADER
# ==========================================
class RoughnessLoader:
    def __init__(self):
        self.targets = {'Ra': ['ra', 'arithmetic'], 'Rq': ['rq', 'rms'], 
                        'Rz': ['rz', 'max height'], 'Rt': ['rt', 'total height']}

    def clean_value(self, val):
        if pd.isna(val): return np.nan
        if isinstance(val, (int, float)): return float(val)
        text = str(val).replace(',', '.').strip()
        match = re.search(r"[-+]?\d*\.\d+|\d+", text)
        return float(match.group()) if match else np.nan

    def process_files(self, uploaded_files, metadata_list):
        combined_rows = []
        for file, meta in zip(uploaded_files, metadata_list):
            try:
                xl = pd.ExcelFile(file)
                sheet_names = [s.strip().upper() for s in xl.sheet_names]
                
                row_data = meta.copy()
                
                # --- PASS 1: GET REPLICATES FROM 'DATA' SHEET (COL E & F) ---
                replicate_points = []
                if "DATA" in sheet_names:
                    # Usecols [4, 5] corresponds to E and F
                    df_data = pd.read_excel(file, sheet_name=xl.sheet_names[sheet_names.index("DATA")], usecols=[4, 5], header=None)
                    # Flatten columns E and F into one long list of numbers
                    for col in df_data.columns:
                        for val in df_data[col]:
                            c_val = self.clean_value(val)
                            if not np.isnan(c_val):
                                replicate_points.append(c_val)
                
                # Calculate Replicate Stats
                if len(replicate_points) > 0:
                    row_data['Ra_replicates_mean'] = np.mean(replicate_points)
                    row_data['n_replicates'] = len(replicate_points)
                    row_data['std_error'] = stats.sem(replicate_points) if len(replicate_points) > 1 else 0
                else:
                    row_data['n_replicates'] = 0

                # --- PASS 2: GET SUMMARY PARAMS (Ra, Rz, Rt) FROM ANY SHEET ---
                for sheet in xl.sheet_names:
                    df_summary = xl.parse(sheet, header=None)
                    for r in range(min(len(df_summary), 60)): # Search top 60 rows
                        for c in range(len(df_summary.columns)):
                            cell_str = str(df_summary.iloc[r, c]).lower().strip()
                            for std_name, keywords in self.targets.items():
                                if any(k in cell_str for k in keywords) and std_name not in row_data:
                                    if c + 1 < len(df_summary.columns):
                                        val = self.clean_value(df_summary.iloc[r, c+1])
                                        if not np.isnan(val): row_data[std_name] = val
                
                combined_rows.append(row_data)
            except Exception as e:
                st.error(f"Error in {file.name}: {e}")
        return pd.DataFrame(combined_rows)

# ==========================================
# 2. UI & ANALYSIS
# ==========================================
st.set_page_config(page_title="Scientific Roughness Lab", layout="wide")
st.title("🔬 Scientific Roughness Quality Analyzer")

st.sidebar.header("1. Upload Samples")
uploaded_files = st.sidebar.file_uploader("Upload Excel Files", accept_multiple_files=True, type=['xlsx'])

if uploaded_files:
    # Metadata for grouping/comparison
    mat_type = st.sidebar.selectbox("Material Type", ["PLA-Pure", "PLA-Wood", "PLA-Carbon", "Other"])
    condition = st.sidebar.selectbox("Condition", ["Control (0d)", "Oven", "UV Ageing", "Humidity"])
    ageing_days = st.sidebar.number_input("Ageing Time (Days)", min_value=0, value=0)

    if st.sidebar.button("Run Full Analysis", type="primary"):
        loader = RoughnessLoader()
        processed_meta = [{"File": f.name, "Material": mat_type, "Condition": condition, "Days": ageing_days} for f in uploaded_files]
        st.session_state['master_df'] = loader.process_files(uploaded_files, processed_meta)

if 'master_df' in st.session_state:
    df = st.session_state['master_df']
    
    t1, t2, t3 = st.tabs(["📋 Data Quality Check", "📈 Statistical Comparison", "📉 Visual Trends"])

    with t1:
        st.subheader("Replicate Verification (Target n ≥ 9)")
        # Show which files met the scientific quality threshold
        def color_n(val):
            color = 'red' if val < 9 else 'green'
            return f'color: {color}; font-weight: bold'
        
        st.dataframe(df[['File', 'Material', 'Condition', 'n_replicates', 'Ra_replicates_mean', 'std_error']].style.applymap(color_n, subset=['n_replicates']), use_container_width=True)
        st.info("💡 Scientific standard recommends at least 9 repeated measurements per sample to account for surface heterogeneity.")

    with t2:
        st.subheader("Statistical Significance")
        # Compare Different Conditions
        if len(df['Condition'].unique()) > 1 or len(df['Material'].unique()) > 1:
            param = st.selectbox("Parameter to Compare", ["Ra_replicates_mean", "Ra", "Rz", "Rt"])
            group_by = st.radio("Group By:", ["Condition", "Material"])
            
            # Grouping the data
            groups = [df[df[group_by] == g][param].dropna() for g in df[group_by].unique()]
            if len(groups) > 1:
                f_stat, p_val = stats.f_oneway(*groups)
                st.metric("ANOVA p-value", f"{p_val:.4f}")
                if p_val < 0.05:
                    st.success("Significant difference found! The ageing conditions/materials significantly affect surface roughness.")
                else:
                    st.warning("No significant difference found (p > 0.05).")

    with t3:
        st.subheader("Comparative Analysis Visuals")
        # Multi-factor Box Plot
        fig = px.box(df, x="Condition", y="Ra_replicates_mean", color="Material",
                     points="all", notched=True, title="Ra Mean Comparison across Conditions")
        st.plotly_chart(fig, use_container_width=True)
