import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy import stats
import re
import io

# ==========================================
# 1. IMPROVED DATA LOADER (More Flexible)
# ==========================================
class RoughnessLoader:
    def __init__(self):
        # We look for these keywords anywhere in your Excel headers
        self.targets = {
            'Ra': ['ra', 'average', 'arithmetic'],
            'Rq': ['rq', 'rms', 'root'],
            'Rz': ['rz', 'max height', 'ten-point'],
            'Rt': ['rt', 'total height']
        }

    def process_files(self, uploaded_files, metadata_list):
        combined_data = []
        for file, meta in zip(uploaded_files, metadata_list):
            try:
                df_params = pd.read_excel(file, sheet_name=0)
                # Clean up column names: lowercase and remove spaces/units
                df_params.columns = [str(c).lower().strip() for c in df_params.columns]
                
                row_data = meta.copy()
                found_any = False

                for standard_name, keywords in self.targets.items():
                    # Check if any keyword exists in any of the Excel columns
                    for col in df_params.columns:
                        if any(k in col for k in keywords):
                            row_data[standard_name] = pd.to_numeric(df_params[col].iloc[0], errors='coerce')
                            found_any = True
                            break
                
                if found_any:
                    combined_data.append(row_data)
                else:
                    st.warning(f"⚠️ No roughness parameters found in {file.name}. Check your headers!")
            except Exception as e:
                st.error(f"❌ Error processing {file.name}: {e}")
                
        return pd.DataFrame(combined_data)

# ... [Keep your stats functions as they are] ...

# ==========================================
# 3. MAIN APP UI (With Safety Checks)
# ==========================================
# [Sidebar code stays the same]

if 'master_df' in st.session_state:
    df = st.session_state['master_df']
    
    if df.empty:
        st.error("The combined dataset is empty. Please check your Excel file headers.")
    else:
        tab1, tab2, tab3 = st.tabs(["📊 Data", "📈 Stats", "📉 Plots"])
        
        with tab1:
            st.write("Found the following data columns:")
            st.dataframe(df)
            
        with tab2:
            # ONLY show parameters that actually exist in the data
            available_params = [p for p in ["Ra", "Rq", "Rz", "Rt"] if p in df.columns]
            
            if not available_params:
                st.error("No valid roughness parameters (Ra, Rq, etc.) were identified in the files.")
            else:
                param = st.selectbox("Parameter", available_params)
                group = st.selectbox("Group By", ["Day", "Condition"])
                
                summary = get_stats_summary(df, group, param)
                st.table(summary)
                
                if len(df[group].unique()) > 1:
                    f, p = perform_anova(df, param, group)
                    st.metric("ANOVA p-value", f"{p:.4f}")

# ==========================================
# 2. ANALYSIS LOGIC (Internal)
# ==========================================
def get_stats_summary(df, group_col, value_col):
    summary = df.groupby(group_col)[value_col].agg(['mean', 'std', 'count']).reset_index()
    summary['ci_95'] = 1.96 * (summary['std'] / np.sqrt(summary['count']))
    return summary

def perform_anova(df, value_col, group_col):
    groups = [group[value_col].values for name, group in df.groupby(group_col)]
    return stats.f_oneway(*groups)

# ==========================================
# 3. MAIN APP UI
# ==========================================
st.set_page_config(page_title="BioMaterial Roughness Analyzer", layout="wide")
st.title("🧪 Surface Roughness Analysis Dashboard")

# SIDEBAR
st.sidebar.header("1. Data Input")
uploaded_files = st.sidebar.file_uploader("Upload Excel Files", accept_multiple_files=True, type=['xlsx'])

if uploaded_files:
    mat_type = st.sidebar.text_input("Material Type", "PLA-Composite")
    ageing_cond = st.sidebar.selectbox("Ageing Condition", ["Oven", "Xenon UV", "Control"])
    
    processed_list = []
    for f in uploaded_files:
        day_match = re.search(r'\d+', f.name)
        day = int(day_match.group()) if day_match else 0
        processed_list.append({"File": f.name, "Material": mat_type, "Condition": ageing_cond, "Day": day})

    if st.sidebar.button("Process & Merge Data"):
        loader = RoughnessLoader()
        st.session_state['master_df'] = loader.process_files(uploaded_files, processed_list)

# MAIN PANEL
if 'master_df' in st.session_state:
    df = st.session_state['master_df']
    tab1, tab2, tab3 = st.tabs(["📊 Data", "📈 Stats", "📉 Plots"])
    
    with tab1:
        st.dataframe(df)
        
    with tab2:
        param = st.selectbox("Parameter", ["Ra", "Rq", "Rz", "Rt"])
        group = st.selectbox("Group By", ["Day", "Condition"])
        summary = get_stats_summary(df, group, param)
        st.table(summary)
        
        if len(df[group].unique()) > 1:
            f, p = perform_anova(df, param, group)
            st.metric("ANOVA p-value", f"{p:.4f}")

    with tab3:
        fig = px.box(df, x=group, y=param, color="Condition", points="all")
        st.plotly_chart(fig, use_container_width=True)
