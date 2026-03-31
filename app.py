import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy import stats
import re
import io

# ==========================================
# 1. SMART SCAN DATA LOADER (Full Extraction)
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
        if pd.isna(val): return np.nan
        if isinstance(val, (int, float)): return float(val)
        text = str(val).replace(',', '.').strip()
        match = re.search(r"[-+]?\d*\.\d+|\d+", text)
        return float(match.group()) if match else np.nan

    def process_files(self, uploaded_files, metadata_list):
        combined_summary = []
        profile_list = [] # List to hold dataframes for the Master Profile Sheet

        for file, meta in zip(uploaded_files, metadata_list):
            try:
                xl = pd.ExcelFile(file)
                row_summary = meta.copy()
                found_params = {}

                # 1. SUMMARY PARAMETER SCAN
                for sheet in xl.sheet_names:
                    df_sheet = xl.parse(sheet, header=None)
                    for r in range(min(len(df_sheet), 100)):
                        for c in range(len(df_sheet.columns)):
                            cell_str = str(df_sheet.iloc[r, c]).lower().strip()
                            for std_name, keywords in self.targets.items():
                                if any(k in cell_str for k in keywords) and std_name not in found_params:
                                    # Check Right or Down for the value
                                    val = np.nan
                                    if c + 1 < len(df_sheet.columns): val = self.clean_value(df_sheet.iloc[r, c+1])
                                    if np.isnan(val) and r + 1 < len(df_sheet): val = self.clean_value(df_sheet.iloc[r+1, c])
                                    
                                    if not np.isnan(val):
                                        found_params[standard_name] = val
                
                # 2. PROFILE DATA EXTRACTION (Col E & F)
                data_sheet = next((s for s in xl.sheet_names if "DATA" in s.upper()), None)
                if data_sheet:
                    # Read E (4) and F (5)
                    df_p = pd.read_excel(file, sheet_name=data_sheet, usecols=[4, 5])
                    df_p.columns = ['Length_mm', 'Amplitude_um']
                    df_p = df_p.apply(pd.to_numeric, errors='coerce').dropna()
                    
                    # Add metadata tags so we know which file this data belongs to
                    df_p['File'] = file.name
                    df_p['Material'] = meta['Material']
                    df_p['Condition'] = meta['Condition']
                    df_p['Day'] = meta['Day']
                    profile_list.append(df_p)

                if found_params:
                    row_summary.update(found_params)
                combined_summary.append(row_summary)

            except Exception as e:
                st.error(f"❌ Error in {file.name}: {e}")
                
        # Combine all profiles into one big table
        master_profiles = pd.concat(profile_list, ignore_index=True) if profile_list else pd.DataFrame()
        return pd.DataFrame(combined_summary), master_profiles

# ==========================================
# 2. UI & DASHBOARD
# ==========================================
st.set_page_config(page_title="Scientific Roughness Lab", layout="wide")
st.title("🔬 Surface Roughness Scientific Analyzer")

st.sidebar.header("1. Upload Samples")
uploaded_files = st.sidebar.file_uploader("Upload .xlsx Files", accept_multiple_files=True, type=['xlsx'])

if uploaded_files:
    m_id = st.sidebar.text_input("Material ID", "Biopolymer_A")
    cond = st.sidebar.selectbox("Condition", ["Control", "Oven", "UV", "Humidity"])
    
    if st.sidebar.button("Process & Extract All", type="primary"):
        loader = RoughnessLoader()
        metas = []
        for f in uploaded_files:
            day_match = re.search(r'\d+', f.name)
            metas.append({"File": f.name, "Material": m_id, "Condition": cond, "Day": int(day_match.group()) if day_match else 0})
        
        with st.spinner("Processing files and combining profiles..."):
            sum_df, prof_df = loader.process_files(uploaded_files, metas)
            st.session_state['summary_df'] = sum_df
            st.session_state['master_profiles'] = prof_df

if 'summary_df' in st.session_state:
    sum_df = st.session_state['summary_df']
    prof_df = st.session_state['master_profiles']
    
    tabs = st.tabs(["📊 Summary Dataset", "📈 Profile Viewer", "📉 Visual Analysis", "💾 Bulk Export"])

    with tabs[0]:
        st.subheader("Extracted Summary Parameters (Ra, Rq, Rz, Rt)")
        st.dataframe(sum_df, use_container_width=True)

    with tabs[1]:
        st.subheader("Individual Profile Visualizer")
        if not prof_df.empty:
            f_select = st.selectbox("Select File", prof_df['File'].unique())
            subset = prof_df[prof_df['File'] == f_select]
            fig = px.line(subset, x='Length_mm', y='Amplitude_um', title=f"Profile: {f_select}")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No profile data found in 'DATA' sheets.")

    with tabs[2]:
        st.subheader("Comparative Boxplots")
        params = [p for p in ["Ra", "Rq", "Rz", "Rt"] if p in sum_df.columns]
        if params:
            p_sel = st.selectbox("Choose Parameter", params)
            fig_box = px.box(sum_df, x="Day", y=p_sel, color="Condition", points="all", notched=True)
            st.plotly_chart(fig_box, use_container_width=True)

    with tabs[3]:
        st.subheader("Download Unified Data")
        
        c1, c2 = st.columns(2)
        
        # 1. Download Summary
        with c1:
            st.write("### 1. Summary Report")
            st.caption("Contains Ra, Rq, etc. for all files (One row per file)")
            csv_sum = sum_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Summary CSV", csv_sum, "roughness_summary.csv", "text/csv")

        # 2. Download Master Profiles
        with c2:
            st.write("### 2. Master Profile Data")
            st.caption("Contains ALL data points from Col E & F for ALL files (Long format)")
            if not prof_df.empty:
                csv_prof = prof_df.to_csv(index=False).encode('utf-8')
                st.download_button("Download Master Profiles CSV", csv_prof, "master_profiles_E_F.csv", "text/csv")
            else:
                st.error("No profile data available to download.")
