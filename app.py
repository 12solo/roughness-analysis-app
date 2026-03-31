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
                # Standardize sheet names to uppercase for comparison
                sheet_map = {s.strip().upper(): s for s in xl.sheet_names}
                
                row_data = meta.copy()
                
                # Initialize empty columns to prevent KeyError later
                row_data.update({'Ra_replicates_mean': np.nan, 'n_replicates': 0, 'std_error': 0})
                
                # --- PASS 1: DATA SHEET (REPLICATES) ---
                if "DATA" in sheet_map:
                    original_name = sheet_map["DATA"]
                    # Load E and F (indices 4, 5)
                    df_data = pd.read_excel(file, sheet_name=original_name, usecols=[4, 5], header=None)
                    replicate_points = []
                    for col in df_data.columns:
                        for val in df_data[col]:
                            c_val = self.clean_value(val)
                            if not np.isnan(c_val):
                                replicate_points.append(c_val)
                    
                    if replicate_points:
                        row_data['Ra_replicates_mean'] = np.mean(replicate_points)
                        row_data['n_replicates'] = len(replicate_points)
                        row_data['std_error'] = stats.sem(replicate_points) if len(replicate_points) > 1 else 0

                # --- PASS 2: SUMMARY PARAMS (Ra, Rz, etc.) ---
                for sheet in xl.sheet_names:
                    df_summary = xl.parse(sheet, header=None)
                    for r in range(min(len(df_summary), 60)):
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
# 3. UI & ANALYSIS
# ==========================================
st.set_page_config(page_title="Scientific Roughness Lab", layout="wide")
st.title("🔬 Scientific Roughness Quality Analyzer")

st.sidebar.header("1. Upload Samples")
uploaded_files = st.sidebar.file_uploader("Upload Excel Files", accept_multiple_files=True, type=['xlsx'])

if uploaded_files:
    mat_type = st.sidebar.text_input("Material ID", "PLA_Comp_01")
    condition = st.sidebar.selectbox("Condition", ["Control", "Oven", "UV Ageing", "Humidity"])
    ageing_days = st.sidebar.number_input("Days", min_value=0, value=0)

    if st.sidebar.button("Run Full Analysis", type="primary"):
        loader = RoughnessLoader()
        processed_meta = [{"File": f.name, "Material": mat_type, "Condition": condition, "Days": ageing_days} for f in uploaded_files]
        st.session_state['master_df'] = loader.process_files(uploaded_files, processed_meta)

if 'master_df' in st.session_state:
    df = st.session_state['master_df']
    
    t1, t2, t3 = st.tabs(["📋 Data Quality Check", "📈 Statistical Comparison", "📉 Visual Trends"])

    with t1:
        st.subheader("Replicate Verification (Target n ≥ 9)")
        
        # Check if we successfully found any replicates
        if 'n_replicates' in df.columns and df['n_replicates'].sum() > 0:
            def color_n(val):
                return 'color: red; font-weight: bold' if val < 9 else 'color: green'
            
            # List of columns we WANT to show
            cols_to_show = ['File', 'Material', 'Condition', 'n_replicates', 'Ra_replicates_mean', 'std_error']
            # Only show columns that actually exist in the dataframe
            existing_cols = [c for c in cols_to_show if c in df.columns]
            
            st.dataframe(df[existing_cols].style.applymap(color_n, subset=['n_replicates'] if 'n_replicates' in existing_cols else []), use_container_width=True)
        else:
            st.warning("⚠️ No 'DATA' sheet or replicate values in Columns E/F were found. Please check your Excel file structure.")
            st.write("Current Columns Extracted:", list(df.columns))

    with t2:
        st.subheader("Statistical Significance")
        # Find which numeric parameters were extracted
        numeric_params = [p for p in ["Ra_replicates_mean", "Ra", "Rz", "Rt"] if p in df.columns]
        
        if numeric_params:
            param = st.selectbox("Parameter to Compare", numeric_params)
            group_by = st.radio("Group By:", ["Condition", "Material"])
            
            if len(df[group_by].unique()) > 1:
                groups = [df[df[group_by] == g][param].dropna() for g in df[group_by].unique()]
                if all(len(g) > 0 for g in groups):
                    f_stat, p_val = stats.f_oneway(*groups)
                    st.metric("ANOVA p-value", f"{p_val:.4f}")
                    if p_val < 0.05: st.success("Significant difference found!")
                    else: st.info("No significant difference detected.")
        else:
            st.error("No numeric parameters found to compare.")

    with t3:
        st.subheader("Visual Analysis")
        if 'param' in locals() and not df.empty:
            fig = px.box(df, x=group_by, y=param, color="Condition", points="all", notched=True)
            st.plotly_chart(fig, use_container_width=True)
