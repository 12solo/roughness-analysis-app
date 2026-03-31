import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy import stats
import re
import io

# ==========================================
# 1. SMART SCAN DATA LOADER
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
        profile_map = {} 

        for file, meta in zip(uploaded_files, metadata_list):
            try:
                xl = pd.ExcelFile(file)
                row_summary = meta.copy()
                found_params = {}

                # --- PASS 1: SUMMARY PARAMETERS ---
                for sheet in xl.sheet_names:
                    df_sheet = xl.parse(sheet, header=None)
                    for r in range(min(len(df_sheet), 100)):
                        for c in range(len(df_sheet.columns)):
                            cell_str = str(df_sheet.iloc[r, c]).lower().strip()
                            for std_key, keywords in self.targets.items():
                                if any(k in cell_str for k in keywords) and std_key not in found_params:
                                    val = np.nan
                                    if c + 1 < len(df_sheet.columns): 
                                        val = self.clean_value(df_sheet.iloc[r, c+1])
                                    if np.isnan(val) and r + 1 < len(df_sheet): 
                                        val = self.clean_value(df_sheet.iloc[r+1, c])
                                    if not np.isnan(val): found_params[std_key] = val
                
                row_summary.update(found_params)
                combined_summary.append(row_summary)

                # --- PASS 2: PROFILE DATA (Col E & F) ---
                data_sheet_name = next((s for s in xl.sheet_names if "DATA" in s.upper()), None)
                if data_sheet_name:
                    df_p = pd.read_excel(file, sheet_name=data_sheet_name, usecols=[4, 5])
                    df_p.columns = ['Length_mm', 'Amplitude_um']
                    df_p = df_p.apply(pd.to_numeric, errors='coerce').dropna().reset_index(drop=True)
                    
                    # Store profile with meta info for header naming
                    profile_map[file.name] = {
                        'data': df_p,
                        'meta': meta
                    }

            except Exception as e:
                st.error(f"❌ Error in {file.name}: {e}")
                
        return pd.DataFrame(combined_summary), profile_map

# ==========================================
# 2. UI & DASHBOARD
# ==========================================
st.set_page_config(page_title="Scientific Roughness Lab", layout="wide")
st.title("🔬 Surface Roughness Scientific Analyzer")

# SIDEBAR
st.sidebar.header("1. Upload Samples")
uploaded_files = st.sidebar.file_uploader("Upload .xlsx Files", accept_multiple_files=True, type=['xlsx'])

if uploaded_files:
    m_id = st.sidebar.text_input("Material ID", "PLA_Composite")
    cond = st.sidebar.selectbox("Condition", ["Control", "Oven", "UV", "Humidity"])
    
    if st.sidebar.button("Process & Extract All", type="primary"):
        loader = RoughnessLoader()
        metas = []
        for f in uploaded_files:
            day_match = re.search(r'\d+', f.name)
            metas.append({
                "File": f.name, "Material": m_id, "Condition": cond, 
                "Day": int(day_match.group()) if day_match else 0
            })
        
        sum_df, prof_dict = loader.process_files(uploaded_files, metas)
        st.session_state['summary_df'] = sum_df
        st.session_state['profile_dict'] = prof_dict

# MAIN TABS
if 'summary_df' in st.session_state:
    sum_df = st.session_state['summary_df']
    prof_dict = st.session_state['profile_dict']
    
    tabs = st.tabs(["📊 Summary Dataset", "📈 Profile Viewer", "📉 Visual Analysis", "💾 Columnar Export"])

    with tabs[0]:
        st.dataframe(sum_df, use_container_width=True)

    with tabs[1]:
        if prof_dict:
            selected_f = st.selectbox("Select File to View", list(prof_dict.keys()))
            subset = prof_dict[selected_f]['data']
            fig = px.line(subset, x='Length_mm', y='Amplitude_um', title=f"Profile: {selected_f}")
            st.plotly_chart(fig, use_container_width=True)

    with tabs[2]:
        params = [p for p in ["Ra", "Rq", "Rz", "Rt"] if p in sum_df.columns]
        if params:
            p_sel = st.selectbox("Choose Parameter", params)
            fig_box = px.box(sum_df, x="Day", y=p_sel, color="Condition", points="all", notched=True)
            st.plotly_chart(fig_box, use_container_width=True)

    with tabs[3]:
        st.subheader("Bulk Export (Columnar Arrangement)")
        
        if prof_dict:
            naming_opt = st.radio("Naming Convention for Columns:", 
                                  ["File Name", "Material_Day", "Condition_Day"], horizontal=True)
            
            # --- CONSTRUCT WIDE DATAFRAME ---
            wide_dfs = []
            for fname, content in prof_dict.items():
                temp_df = content['data'].copy()
                m = content['meta']
                
                # Define Header based on user selection
                if naming_opt == "File Name": header = fname
                elif naming_opt == "Material_Day": header = f"{m['Material']}_Day{m['Day']}"
                else: header = f"{m['Condition']}_Day{m['Day']}"
                
                # Rename columns to be unique for this file
                temp_df.columns = [f"{header}_Length(mm)", f"{header}_Amplitude(um)"]
                wide_dfs.append(temp_df)
            
            # Join all dataframes side-by-side
            master_wide_df = pd.concat(wide_dfs, axis=1)
            
            st.write(f"Preview of Columnar Export ({len(wide_dfs)} samples):")
            st.dataframe(master_wide_df.head(50), use_container_width=True)
            
            csv_wide = master_wide_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Columnar Profile Data (CSV)", csv_wide, "columnar_profiles.csv", "text/csv")
        else:
            st.warning("No profile data found in 'DATA' sheets to export.")
