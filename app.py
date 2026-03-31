import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy import stats
import re
import io

# ==========================================
# 1. SMART SCAN DATA LOADER (Indestructible)
# ==========================================
class RoughnessLoader:
    def __init__(self):
        # Fuzzy matching: looks for these strings anywhere in a cell
        self.targets = {
            'Ra': ['ra', 'arithmetic', 'average roughness'],
            'Rq': ['rq', 'rms', 'root mean'],
            'Rz': ['rz', 'max height'],
            'Rt': ['rt', 'total height']
        }

    def clean_value(self, val):
        """Extracts the first number found in a cell (handles units and commas)"""
        if pd.isna(val): return np.nan
        if isinstance(val, (int, float)): return float(val)
        # Convert comma to dot (European format) and strip spaces
        text = str(val).replace(',', '.').strip()
        # Regex to find numbers like 0.452, 12, or -1.5
        match = re.search(r"[-+]?\d*\.\d+|\d+", text)
        return float(match.group()) if match else np.nan

    def process_files(self, uploaded_files, metadata_list):
        combined_data = []
        for file, meta in zip(uploaded_files, metadata_list):
            try:
                xl = pd.ExcelFile(file)
                row_data = meta.copy()
                found_params = {}

                # Scan ALL sheets in the Excel file
                for sheet in xl.sheet_names:
                    df = xl.parse(sheet, header=None)
                    
                    for r in range(len(df)):
                        for c in range(len(df.columns)):
                            cell_str = str(df.iloc[r, c]).lower().strip()
                            
                            for standard_name, keywords in self.targets.items():
                                # Check if cell contains the keyword and we haven't found it yet
                                if any(k in cell_str for k in keywords) and standard_name not in found_params:
                                    
                                    # SEARCH NEIGHBORS: Right, then Down, then Left
                                    potential_values = []
                                    if c + 1 < len(df.columns): potential_values.append(df.iloc[r, c + 1]) # Right
                                    if r + 1 < len(df): potential_values.append(df.iloc[r + 1, c])         # Down
                                    if c - 1 >= 0: potential_values.append(df.iloc[r, c - 1])              # Left
                                    
                                    for val_candidate in potential_values:
                                        numeric_val = self.clean_value(val_candidate)
                                        if not np.isnan(numeric_val):
                                            found_params[standard_name] = numeric_val
                                            break
                
                if found_params:
                    row_data.update(found_params)
                    combined_data.append(row_data)
                else:
                    st.warning(f"⚠️ '{file.name}': No numeric values found near Ra/Rq/Rz/Rt labels.")
            
            except Exception as e:
                st.error(f"❌ Critical error in {file.name}: {e}")
                
        return pd.DataFrame(combined_data)

# ==========================================
# 2. STATISTICAL FUNCTIONS
# ==========================================
def get_stats_summary(df, group_col, value_col):
    if value_col not in df.columns or df.empty:
        return pd.DataFrame()
    
    # Drop empty values to prevent stats errors
    temp_df = df.dropna(subset=[value_col])
    if temp_df.empty: return pd.DataFrame()

    summary = temp_df.groupby(group_col)[value_col].agg(['mean', 'std', 'count']).reset_index()
    summary['std'] = summary['std'].fillna(0)
    # 95% Confidence Interval
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
st.set_page_config(page_title="Bio-Surface Pro Analyzer", layout="wide")
st.title("🧪 Surface Roughness Analysis Dashboard")
st.markdown("Automated batch analysis for bioplastic and biocomposite degradation studies.")

# --- SIDEBAR ---
st.sidebar.header("1. Upload Files")
uploaded_files = st.sidebar.file_uploader("Bulk Upload (.xlsx)", accept_multiple_files=True, type=['xlsx'])

if uploaded_files:
    st.sidebar.subheader("2. Metadata Settings")
    mat_type = st.sidebar.text_input("Material Type", "PLA-Biocomposite")
    ageing_cond = st.sidebar.selectbox("Ageing Condition", ["Oven", "Xenon UV", "Humidity Chamber", "Control", "Other"])
    
    # Auto-extracting "Day" from filename
    processed_list = []
    for f in uploaded_files:
        day_match = re.search(r'\d+', f.name)
        day = int(day_match.group()) if day_match else 0
        processed_list.append({
            "File": f.name, "Material": mat_type, 
            "Condition": ageing_cond, "Day": day
        })

    if st.sidebar.button("Process & Analyze", type="primary"):
        loader = RoughnessLoader()
        with st.spinner("Deep-scanning all Excel sheets..."):
            result_df = loader.process_files(uploaded_files, processed_list)
            st.session_state['master_df'] = result_df

# --- MAIN DASHBOARD ---
if 'master_df' in st.session_state:
    df = st.session_state['master_df']
    
    if df.empty:
        st.error("No data found. Please ensure the Excel files contain text labels like 'Ra' or 'Rz' with numbers next to them.")
    else:
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Dataset", "📈 Statistics", "📉 Visualizations", "💾 Export"])

        with tab1:
            st.subheader("Extracted Parameters")
            st.dataframe(df, use_container_width=True)

        with tab2:
            st.subheader("Statistical Summary")
            found_params = [p for p in ["Ra", "Rq", "Rz", "Rt"] if p in df.columns]
            
            if found_params:
                c1, c2 = st.columns(2)
                with c1:
                    param = st.selectbox("Parameter", found_params)
                with c2:
                    group = st.selectbox("Group By", ["Day", "Condition"])

                summary = get_stats_summary(df, group, param)
                if not summary.empty:
                    st.table(summary)

                    if len(df[group].unique()) > 1:
                        f_val, p_val = perform_anova(df, param, group)
                        st.metric("ANOVA p-value", f"{p_val:.4f}")
                        if p_val < 0.05:
                            st.success("Result: Statistically Significant difference.")
                        else:
                            st.warning("Result: No significant difference detected.")
            else:
                st.warning("No valid roughness parameters found in the data.")

        with tab3:
            if found_params:
                st.subheader(f"Distribution of {param}")
                fig = px.box(df, x=group, y=param, color="Condition", 
                             points="all", title=f"{param} across {group}",
                             template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)
                
                st.subheader("Degradation Trend")
                trend_df = get_stats_summary(df, "Day", param)
                if not trend_df.empty:
                    fig_line = px.line(trend_df, x="Day", y="mean", error_y="ci_95",
                                       markers=True, title=f"Mean {param} vs Ageing Time")
                    st.plotly_chart(fig_line, use_container_width=True)

        with tab4:
            st.subheader("Export Analysis")
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download CSV Dataset", data=csv, 
                               file_name="roughness_analysis_results.csv", mime="text/csv")
else:
    st.info("👋 To start, upload your Excel files in the sidebar and click 'Process'.")
