import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy import stats
import re

# ==========================================
# 1. SCIENTIFIC DATA LOADER (Column E & F)
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
                sheet_names = [s.strip() for s in xl.sheet_names]
                
                # Check for "DATA" sheet for repeated measurements
                data_points = []
                if "DATA" in sheet_names:
                    # Load columns E and F (indices 4 and 5)
                    df_data = pd.read_excel(file, sheet_name="DATA", usecols=[4, 5], header=None)
                    # Flatten both columns and clean
                    raw_points = df_data.values.flatten()
                    data_points = [self.clean_value(x) for x in raw_points if not pd.isna(self.clean_value(x))]

                # Also search for summary params (Ra, Rz, etc.) across all sheets
                summary_params = {}
                for sheet in xl.sheet_names:
                    df = xl.parse(sheet, header=None)
                    for r in range(min(len(df), 50)): # Scan top 50 rows
                        for c in range(len(df.columns)):
                            cell_str = str(df.iloc[r, c]).lower().strip()
                            for std_name, keywords in self.targets.items():
                                if any(k in cell_str for k in keywords) and std_name not in summary_params:
                                    if c + 1 < len(df.columns):
                                        val = self.clean_value(df.iloc[r, c+1])
                                        if not np.isnan(val): summary_params[std_name] = val

                # If we found repeated data points, calculate the mean for Ra
                if len(data_points) >= 1:
                    summary_params['Ra_mean'] = np.mean(data_points)
                    summary_params['n_samples'] = len(data_points)
                    summary_params['std_err'] = stats.sem(data_points)
                    summary_params['raw_data'] = data_points
                else:
                    summary_params['n_samples'] = 1 # Single point from summary

                row_data = {**meta, **summary_params}
                combined_rows.append(row_data)
            except Exception as e:
                st.error(f"Error in {file.name}: {e}")
        return pd.DataFrame(combined_rows)

# ==========================================
# 2. ADVANCED STATISTICAL ENGINE
# ==========================================
def run_comparative_stats(df, param, group_col):
    results = {}
    groups = df[group_col].unique()
    
    # Check Sample Size Quality (n >= 9)
    quality_check = df.groupby(group_col)['n_samples'].sum()
    results['quality'] = quality_check
    
    # 1. ANOVA (Between multiple conditions)
    if len(groups) > 1:
        data_groups = [df[df[group_col] == g][param].dropna() for g in groups]
        f_stat, p_val = stats.f_oneway(*data_groups)
        results['anova_p'] = p_val
        
    return results

# ==========================================
# 3. UI LAYOUT
# ==========================================
st.set_page_config(page_title="Scientific Roughness Lab", layout="wide")
st.title("🔬 Advanced Roughness Quality Analyzer")

st.sidebar.header("1. Upload Samples")
uploaded_files = st.sidebar.file_uploader("Upload .xlsx (Bulk)", accept_multiple_files=True, type=['xlsx'])

if uploaded_files:
    # Metadata for comparison
    mat_type = st.sidebar.text_input("Sample ID / Material", "Polymer_A")
    condition = st.sidebar.selectbox("Ageing Environment", ["Control", "Oven", "UV", "Humidity"])
    
    if st.sidebar.button("Run Scientific Analysis"):
        loader = RoughnessLoader()
        processed_meta = [{"File": f.name, "Material": mat_type, "Condition": condition} for f in uploaded_files]
        st.session_state['master_df'] = loader.process_files(uploaded_files, processed_meta)

if 'master_df' in st.session_state:
    df = st.session_state['master_df']
    tab1, tab2, tab3 = st.tabs(["📋 Data Quality", "📊 Statistical Comparison", "📉 Distributions"])

    with tab1:
        st.subheader("Sample Replicates & Quality Check")
        # Highlight rows where n < 9
        def highlight_low_n(s):
            return ['background-color: #ffcccc' if v < 9 else '' for v in s]
        
        display_df = df[['File', 'Condition', 'Ra_mean', 'n_samples', 'std_err']]
        st.dataframe(display_df.style.apply(highlight_low_n, subset=['n_samples']), use_container_width=True)
        st.caption("Rows in red have fewer than the recommended 9 data points.")

    with tab2:
        st.subheader("Comparing Conditions")
        param_to_test = st.selectbox("Parameter for Comparison", ["Ra_mean", "Rz", "Rq"])
        compare_by = st.radio("Compare across:", ["Condition", "Material"])
        
        stats_res = run_comparative_stats(df, param_to_test, compare_by)
        
        col1, col2 = st.columns(2)
        if 'anova_p' in stats_res:
            col1.metric("ANOVA p-value", f"{stats_res['anova_p']:.4f}")
            if stats_res['anova_p'] < 0.05:
                col1.success("Significant difference between conditions detected.")
            else:
                col1.info("No significant variation found between conditions.")

    with tab3:
        st.subheader("Inter-Sample Variance")
        fig = px.box(df, x=compare_by, y=param_to_test, color="Condition",
                     points="all", notched=True, title=f"Comparison of {param_to_test}")
        st.plotly_chart(fig, use_container_width=True)
