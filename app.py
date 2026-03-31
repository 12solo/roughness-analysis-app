import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy import stats
import re
import io

# ==========================================
# 1. ULTRA-ROBUST DATA LOADER
# ==========================================
class RoughnessLoader:
    def __init__(self):
        # We search for these strings inside your column names
        self.targets = {
            'Ra': ['ra', 'arithmetic', 'average roughness'],
            'Rq': ['rq', 'rms', 'root mean'],
            'Rz': ['rz', 'max height'],
            'Rt': ['rt', 'total height']
        }

    def clean_value(self, val):
        """Extracts just the number from a string like '5.2 um'"""
        if pd.isna(val): return np.nan
        if isinstance(val, (int, float)): return float(val)
        # Use regex to find the first number (integer or decimal) in the string
        match = re.search(r"[-+]?\d*\.\d+|\d+", str(val))
        return float(match.group()) if match else np.nan

    def process_files(self, uploaded_files, metadata_list):
        combined_data = []
        for file, meta in zip(uploaded_files, metadata_list):
            try:
                # Read the first sheet of the Excel file
                df_raw = pd.read_excel(file)
                
                # Clean column names for searching (lowercase, no spaces)
                cols = [str(c).lower().strip() for c in df_raw.columns]
                
                row_data = meta.copy()
                found_params = False

                for standard_name, keywords in self.targets.items():
                    for i, col_name in enumerate(cols):
                        if any(k in col_name for k in keywords):
                            # Get the value from the first data row of that column
                            raw_val = df_raw.iloc[0, i]
                            row_data[standard_name] = self.clean_value(raw_val)
                            found_params = True
                            break
                
                if found_params:
                    combined_data.append(row_data)
                else:
                    st.warning(f"⚠️ '{file.name}' loaded, but no Ra/Rq/Rz/Rt columns found.")
            
            except Exception as e:
                st.error(f"❌ Error in {file.name}: {e}")
                
        return pd.DataFrame(combined_data)

# ==========================================
# 2. STATISTICAL FUNCTIONS
# ==========================================
def get_stats_summary(df, group_col, value_col):
    if value_col not in df.columns or df.empty:
        return pd.DataFrame()
    
    # Clean data for stats
    temp_df = df.dropna(subset=[value_col])
    summary = temp_df.groupby(group_col)[value_col].agg(['mean', 'std', 'count']).reset_index()
    
    # Handle CI calculation
    summary['std'] = summary['std'].fillna(0)
    summary['ci_95'] = 1.96 * (summary['std'] / np.sqrt(summary['count'].replace(0, 1)))
    return summary

def perform_anova(df, value_col, group_col):
    temp_df = df.dropna(subset=[value_col])
    groups = [group[value_col].values for name, group in temp_df.groupby(group_col)]
    if len(groups) < 2:
        return np.nan, np.nan
    return stats.f_oneway(*groups)

# ==========================================
# 3. STREAMLIT UI
# ==========================================
st.set_page_config(page_title="BioMaterial Roughness Analyzer", layout="wide")

st.title("🧪 Surface Roughness Analysis Dashboard")
st.markdown("Automated analysis for Bioplastic and Biocomposite degradation.")

# --- SIDEBAR ---
st.sidebar.header("1. Upload Data")
uploaded_files = st.sidebar.file_uploader("Bulk Upload (.xlsx)", accept_multiple_files=True, type=['xlsx'])

if uploaded_files:
    st.sidebar.subheader("2. Set Metadata")
    mat_type = st.sidebar.text_input("Material Composition", "PLA-Biocomposite")
    ageing_cond = st.sidebar.selectbox("Ageing Condition", ["Oven", "Xenon UV", "Control", "Other"])
    
    # Logic to auto-extract "Day" from filename
    processed_list = []
    for f in uploaded_files:
        day_match = re.search(r'\d+', f.name)
        day = int(day_match.group()) if day_match else 0
        processed_list.append({
            "File": f.name, "Material": mat_type, 
            "Condition": ageing_cond, "Day": day
        })

    if st.sidebar.button("Analyze All Files", type="primary"):
        loader = RoughnessLoader()
        with st.spinner("Processing..."):
            result_df = loader.process_files(uploaded_files, processed_list)
            st.session_state['master_df'] = result_df

# --- MAIN DASHBOARD ---
if 'master_df' in st.session_state:
    df = st.session_state['master_df']
    
    if df.empty:
        st.error("No valid data could be extracted. Check your Excel column headers (must contain Ra, Rq, Rz, or Rt).")
    else:
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Dataset", "📈 Statistics", "📉 Visualizations", "💾 Export"])

        with tab1:
            st.subheader("Combined Data Preview")
            st.dataframe(df, use_container_width=True)

        with tab2:
            st.subheader("Grouped Summary")
            # Only show params that were found in the data
            found_params = [p for p in ["Ra", "Rq", "Rz", "Rt"] if p in df.columns]
            
            if found_params:
                col1, col2 = st.columns(2)
                with col1:
                    param = st.selectbox("Select Parameter", found_params)
                with col2:
                    group = st.selectbox("Group By", ["Day", "Condition"])

                summary = get_stats_summary(df, group, param)
                st.dataframe(summary, use_container_width=True)

                # ANOVA
                if len(df[group].unique()) > 1:
                    f_val, p_val = perform_anova(df, param, group)
                    st.metric("ANOVA p-value", f"{p_val:.4f}")
                    if p_val < 0.05:
                        st.success("Statistically significant difference detected!")
            else:
                st.warning("No roughness parameters found to analyze.")

        with tab3:
            st.subheader("Interactive Distribution Plot")
            if found_params:
                fig = px.box(df, x=group, y=param, color="Condition", 
                             points="all", notched=True, 
                             title=f"{param} Distribution by {group}",
                             template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)
                
                # Trend line
                st.subheader("Ageing Trend")
                trend_summary = get_stats_summary(df, "Day", param)
                fig_trend = px.line(trend_summary, x="Day", y="mean", error_y="ci_95",
                                    markers=True, title=f"Mean {param} Evolution")
                st.plotly_chart(fig_trend, use_container_width=True)

        with tab4:
            st.subheader("Download Results")
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download CSV Report", data=csv, 
                               file_name="roughness_report.csv", mime="text/csv")

else:
    st.info("👋 Upload Excel files on the left to begin your analysis.")
