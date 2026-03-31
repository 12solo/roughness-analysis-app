import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy import stats
import re
import io

# ==========================================
# 1. DEEP-SCAN DATA LOADER
# ==========================================
class RoughnessLoader:
    def __init__(self):
        # Keywords to identify the parameters
        self.targets = {
            'Ra': ['ra', 'arithmetic', 'average roughness'],
            'Rq': ['rq', 'rms', 'root mean'],
            'Rz': ['rz', 'max height'],
            'Rt': ['rt', 'total height']
        }

    def clean_value(self, val):
        """Extracts the first number found in a cell (handles units like um)"""
        if pd.isna(val): return np.nan
        if isinstance(val, (int, float)): return float(val)
        # Regex to find numbers like 0.452 or 1,23
        match = re.search(r"[-+]?\d*[.,]\d+|\d+", str(val).replace(',', '.'))
        return float(match.group()) if match else np.nan

    def process_files(self, uploaded_files, metadata_list):
        combined_data = []
        for file, meta in zip(uploaded_files, metadata_list):
            try:
                # Get all sheet names to find 'DATA' or 'Certificate'
                xl = pd.ExcelFile(file)
                sheet_names = [s.upper() for s in xl.sheet_names]
                
                # Priority: 1. DATA, 2. CERTIFICATE, 3. First sheet
                target_sheet = 0
                if 'DATA' in sheet_names:
                    target_sheet = xl.sheet_names[sheet_names.index('DATA')]
                elif 'CERTIFICATE' in sheet_names:
                    target_sheet = xl.sheet_names[sheet_names.index('CERTIFICATE')]

                df_raw = pd.read_excel(file, sheet_name=target_sheet, header=None)
                
                row_data = meta.copy()
                found_params = False

                # Scan every cell in the chosen sheet
                for r in range(len(df_raw)):
                    for c in range(len(df_raw.columns)):
                        cell_content = str(df_raw.iloc[r, c]).lower().strip()
                        
                        for standard_name, keywords in self.targets.items():
                            if any(k == cell_content or k + ":" == cell_content for k in keywords):
                                # Found the label! Look at the cell to the right (c+1)
                                try:
                                    val_to_clean = df_raw.iloc[r, c + 1]
                                    row_data[standard_name] = self.clean_value(val_to_clean)
                                    found_params = True
                                except:
                                    continue
                
                if found_params:
                    combined_data.append(row_data)
                else:
                    st.warning(f"⚠️ '{file.name}': Could not find labels Ra/Rq/Rz/Rt in sheet '{target_sheet}'.")
            
            except Exception as e:
                st.error(f"❌ Error in {file.name}: {e}")
                
        return pd.DataFrame(combined_data)

# ==========================================
# 2. STATISTICAL FUNCTIONS
# ==========================================
def get_stats_summary(df, group_col, value_col):
    if value_col not in df.columns or df.empty:
        return pd.DataFrame()
    
    temp_df = df.dropna(subset=[value_col])
    if temp_df.empty: return pd.DataFrame()

    summary = temp_df.groupby(group_col)[value_col].agg(['mean', 'std', 'count']).reset_index()
    summary['std'] = summary['std'].fillna(0)
    summary['ci_95'] = 1.96 * (summary['std'] / np.sqrt(summary['count'].replace(0, 1)))
    return summary

def perform_anova(df, value_col, group_col):
    temp_df = df.dropna(subset=[value_col])
    groups = [group[value_col].values for name, group in temp_df.groupby(group_col)]
    if len(groups) < 2: return np.nan, np.nan
    return stats.f_oneway(*groups)

# ==========================================
# 3. STREAMLIT UI
# ==========================================
st.set_page_config(page_title="Roughness Analyzer Pro", layout="wide")
st.title("🧪 Surface Roughness Analysis Dashboard")

# --- SIDEBAR ---
st.sidebar.header("1. Upload Data")
uploaded_files = st.sidebar.file_uploader("Upload Excel Files", accept_multiple_files=True, type=['xlsx'])

if uploaded_files:
    st.sidebar.subheader("2. Metadata")
    mat_type = st.sidebar.text_input("Material Type", "Biocomposite")
    ageing_cond = st.sidebar.selectbox("Condition", ["Oven", "Xenon UV", "Control", "Other"])
    
    processed_list = []
    for f in uploaded_files:
        day_match = re.search(r'\d+', f.name)
        day = int(day_match.group()) if day_match else 0
        processed_list.append({"File": f.name, "Material": mat_type, "Condition": ageing_cond, "Day": day})

    if st.sidebar.button("Process Files", type="primary"):
        loader = RoughnessLoader()
        with st.spinner("Scanning DATA sheets..."):
            result_df = loader.process_files(uploaded_files, processed_list)
            st.session_state['master_df'] = result_df

# --- MAIN PANEL ---
if 'master_df' in st.session_state:
    df = st.session_state['master_df']
    
    if df.empty:
        st.error("No data found. Ensure the cells contain the exact text 'Ra', 'Rq', 'Rz', or 'Rt'.")
    else:
        tab1, tab2, tab3 = st.tabs(["📊 Data", "📈 Statistics", "📉 Visualizations"])

        with tab1:
            st.dataframe(df, use_container_width=True)

        with tab2:
            found_params = [p for p in ["Ra", "Rq", "Rz", "Rt"] if p in df.columns]
            if found_params:
                p_sel = st.selectbox("Parameter", found_params)
                g_sel = st.selectbox("Group By", ["Day", "Condition"])
                
                summary = get_stats_summary(df, g_sel, p_sel)
                st.table(summary)
                
                if len(df[g_sel].unique()) > 1:
                    f, p = perform_anova(df, p_sel, g_sel)
                    st.metric("ANOVA p-value", f"{p:.4f}")

        with tab3:
            if found_params:
                fig = px.box(df, x=g_sel, y=p_sel, color="Condition", points="all", template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Upload files to begin.")
