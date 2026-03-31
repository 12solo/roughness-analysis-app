import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy import stats
import re
import io

# ==========================================
# 1. SMART SCAN DATA LOADER (With Profile Extraction)
# ==========================================
class RoughnessLoader:
    def __init__(self):
        self.targets = {
            'Ra': ['ra', 'arithmetic', 'average roughness'],
            'Rq': ['rq', 'rms', 'root mean'],
            'Rz': ['rz', 'max height'],
            'Rt': ['rt', 'total height']
        }

    def clean_value(self, val):
        if pd.isna(val): return np.nan
        if isinstance(val, (int, float)): return float(val)
        text = str(val).replace(',', '.').strip()
        match = re.search(r"[-+]?\d*\.\d+|\d+", text)
        return float(match.group()) if match else np.nan

    def process_files(self, uploaded_files, metadata_list):
        combined_data = []
        all_profiles = {} # Dictionary to store E & F data per file

        for file, meta in zip(uploaded_files, metadata_list):
            try:
                xl = pd.ExcelFile(file)
                row_data = meta.copy()
                found_params = {}

                # 1. EXTRACT SUMMARY PARAMETERS (Ra, Rz, etc.)
                for sheet in xl.sheet_names:
                    df = xl.parse(sheet, header=None)
                    for r in range(min(len(df), 100)): # Scan top rows
                        for c in range(len(df.columns)):
                            cell_str = str(df.iloc[r, c]).lower().strip()
                            for standard_name, keywords in self.targets.items():
                                if any(k in cell_str for k in keywords) and standard_name not in found_params:
                                    potential_values = []
                                    if c + 1 < len(df.columns): potential_values.append(df.iloc[r, c + 1])
                                    if r + 1 < len(df): potential_values.append(df.iloc[r + 1, c])
                                    
                                    for val_candidate in potential_values:
                                        numeric_val = self.clean_value(val_candidate)
                                        if not np.isnan(numeric_val):
                                            found_params[standard_name] = numeric_val
                                            break
                
                # 2. EXTRACT PROFILE DATA (Columns E and F from "DATA" sheet)
                # Note: 'E' is index 4, 'F' is index 5
                profile_df = pd.DataFrame()
                data_sheet_name = next((s for s in xl.sheet_names if "DATA" in s.upper()), None)
                
                if data_sheet_name:
                    # Read only columns E and F
                    df_profile = pd.read_excel(file, sheet_name=data_sheet_name, usecols="E:F")
                    df_profile.columns = ['Length_mm', 'Amplitude_um'] # Renaming for clarity
                    # Clean data: drop non-numeric rows
                    df_profile = df_profile.apply(pd.to_numeric, errors='coerce').dropna()
                    all_profiles[file.name] = df_profile

                if found_params:
                    row_data.update(found_params)
                    combined_data.append(row_data)
                else:
                    st.warning(f"⚠️ '{file.name}': No numeric values found near Ra/Rq/Rz/Rt labels.")
            
            except Exception as e:
                st.error(f"❌ Error in {file.name}: {e}")
                
        return pd.DataFrame(combined_data), all_profiles

# ==========================================
# 2. STATISTICAL FUNCTIONS
# ==========================================
def get_stats_summary(df, group_col, value_col):
    if value_col not in df.columns or df.empty: return pd.DataFrame()
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
st.set_page_config(page_title="Bio-Surface Pro Analyzer", layout="wide")
st.title("🧪 Surface Roughness Analysis Dashboard")

# --- SIDEBAR ---
st.sidebar.header("1. Upload Files")
uploaded_files = st.sidebar.file_uploader("Bulk Upload (.xlsx)", accept_multiple_files=True, type=['xlsx'])

if uploaded_files:
    st.sidebar.subheader("2. Metadata Settings")
    mat_type = st.sidebar.text_input("Material Type", "PLA-Biocomposite")
    ageing_cond = st.sidebar.selectbox("Ageing Condition", ["Oven", "Xenon UV", "Humidity Chamber", "Control", "Other"])
    
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
        with st.spinner("Deep-scanning sheets and extracting profiles..."):
            res_df, res_profiles = loader.process_files(uploaded_files, processed_list)
            st.session_state['master_df'] = res_df
            st.session_state['profile_data'] = res_profiles

# --- MAIN DASHBOARD ---
if 'master_df' in st.session_state:
    df = st.session_state['master_df']
    profiles = st.session_state.get('profile_data', {})
    
    if df.empty:
        st.error("No data found. Ensure labels like 'Ra' or 'Rz' are present.")
    else:
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Dataset", "📉 Profile Plots", "📈 Statistics", "🎨 Visualizations", "💾 Export"])

        with tab1:
            st.subheader("Extracted Summary Parameters")
            st.dataframe(df, use_container_width=True)

        with tab2:
            st.subheader("Surface Roughness Profiles (Col E & F)")
            if not profiles:
                st.info("No 'DATA' sheet profile data found in uploaded files.")
            else:
                selected_file = st.selectbox("Select File to View Profile", list(profiles.keys()))
                prof_df = profiles[selected_file]
                
                fig_prof = px.line(prof_df, x='Length_mm', y='Amplitude_um', 
                                  title=f"Roughness Profile: {selected_file}",
                                  labels={'Length_mm': 'Travel Length (mm)', 'Amplitude_um': 'Amplitude (µm)'},
                                  template="plotly_white")
                fig_prof.update_traces(line_color='#2E86C1', line_width=1)
                st.plotly_chart(fig_prof, use_container_width=True)
                
                st.write(f"Showing {len(prof_df)} data points from columns E (Length) and F (Amplitude).")

        with tab3:
            st.subheader("Statistical Summary")
            found_params = [p for p in ["Ra", "Rq", "Rz", "Rt"] if p in df.columns]
            if found_params:
                c1, c2 = st.columns(2)
                with c1: param = st.selectbox("Parameter", found_params)
                with c2: group = st.selectbox("Group By", ["Day", "Condition"])
                summary = get_stats_summary(df, group, param)
                if not summary.empty:
                    st.table(summary)
                    if len(df[group].unique()) > 1:
                        f_val, p_val = perform_anova(df, param, group)
                        st.metric("ANOVA p-value", f"{p_val:.4f}")

        with tab4:
            if found_params:
                fig = px.box(df, x=group, y=param, color="Condition", points="all", template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)

        with tab5:
            st.subheader("Export Analysis")
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download CSV Results", data=csv, file_name="roughness_summary.csv", mime="text/csv")
else:
    st.info("👋 Upload Excel files to begin analysis.")
