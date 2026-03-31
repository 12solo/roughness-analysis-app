import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import re

# ==========================================
# 1. INITIALIZE SESSION STATE
# ==========================================
if 'master_df' not in st.session_state:
    st.session_state['master_df'] = pd.DataFrame()
if 'profile_dict' not in st.session_state:
    st.session_state['profile_dict'] = {}
if 'legend_map' not in st.session_state:
    st.session_state['legend_map'] = {}

# ==========================================
# 2. SMART SCAN DATA LOADER
# ==========================================
class RoughnessLoader:
    def __init__(self):
        self.targets = {'Ra': ['ra'], 'Rq': ['rq'], 'Rz': ['rz'], 'Rt': ['rt']}

    def clean_value(self, val):
        if pd.isna(val): return np.nan
        if isinstance(val, (int, float)): return float(val)
        text = str(val).replace(',', '.').strip()
        match = re.search(r"[-+]?\d*\.\d+|\d+", text)
        return float(match.group()) if match else np.nan

    def process_files(self, uploaded_files, meta_template):
        combined_summary = []
        profile_map = {} 

        for file in uploaded_files:
            try:
                xl = pd.ExcelFile(file)
                row_summary = meta_template.copy()
                row_summary['File'] = file.name
                
                for sheet in xl.sheet_names:
                    df_sheet = xl.parse(sheet, header=None)
                    for r in range(min(len(df_sheet), 100)):
                        for c in range(len(df_sheet.columns)):
                            cell_str = str(df_sheet.iloc[r, c]).lower().strip()
                            for std_key, keywords in self.targets.items():
                                if any(k in cell_str for k in keywords) and std_key not in row_summary:
                                    val = np.nan
                                    if c + 1 < len(df_sheet.columns): val = self.clean_value(df_sheet.iloc[r, c+1])
                                    if np.isnan(val) and r + 1 < len(df_sheet): val = self.clean_value(df_sheet.iloc[r+1, c])
                                    if not np.isnan(val): row_summary[std_key] = val
                combined_summary.append(row_summary)

                data_sheet = next((s for s in xl.sheet_names if "DATA" in s.upper()), None)
                if data_sheet:
                    df_p = pd.read_excel(file, sheet_name=data_sheet, usecols=[4, 5])
                    df_p.columns = ['Length_mm', 'Amplitude_um']
                    df_p = df_p.apply(pd.to_numeric, errors='coerce').dropna().reset_index(drop=True)
                    if not df_p.empty:
                        # Normalize to zero-mean
                        df_p['Amplitude_um_Norm'] = df_p['Amplitude_um'] - df_p['Amplitude_um'].mean()
                        df_p['Sample'] = meta_template['Sample']
                        profile_map[file.name] = df_p
            except Exception as e:
                st.error(f"Error: {e}")
        return pd.DataFrame(combined_summary), profile_map

# ==========================================
# 3. UI & SIDEBAR
# ==========================================
st.set_page_config(page_title="Scientific Roughness Lab", layout="wide")
st.title("🔬 Scientific Roughness Analyzer")

with st.sidebar:
    st.header("1. Data Input")
    with st.form("input_form", clear_on_submit=True):
        s_name = st.text_input("Sample ID", "Sample A")
        s_files = st.file_uploader("Upload Replicates (.xlsx)", accept_multiple_files=True)
        submit = st.form_submit_button("Add Batch")

    if submit and s_files:
        loader = RoughnessLoader()
        meta = {"Sample": s_name, "Condition": "Standard", "Day": 0}
        new_sum, new_prof = loader.process_files(s_files, meta)
        st.session_state['master_df'] = pd.concat([st.session_state['master_df'], new_sum], ignore_index=True)
        st.session_state['profile_dict'].update(new_prof)

    st.header("2. Legend Customization")
    if not st.session_state['master_df'].empty:
        for s in sorted(st.session_state['master_df']['Sample'].unique()):
            st.session_state['legend_map'][s] = st.text_input(f"Label '{s}':", s)

    if st.button("Reset Study", type="primary"):
        st.session_state['master_df'] = pd.DataFrame()
        st.session_state['profile_dict'] = {}
        st.session_state['legend_map'] = {}
        st.rerun()

# ==========================================
# 4. DASHBOARD
# ==========================================
df = st.session_state['master_df']
profiles = st.session_state['profile_dict']

if not df.empty:
    tabs = st.tabs(["📊 Dataset", "📉 Trends", "🎨 Batch Stack", "🏛️ Representative Stack", "💾 Export"])

    with tabs[2]:
        st.subheader("Batch Replicate Stack (Single Main Axis)")
        batch_to_check = st.selectbox("Select Batch:", sorted(df['Sample'].unique()))
        batch_files = sorted(df[df['Sample'] == batch_to_check]['File'].tolist())
        offset_val = st.slider("Vertical Offset (µm)", 1, 100, 20, key="rep_off")
        
        fig_rep = go.Figure()
        for i, f in enumerate(batch_files):
            p_data = profiles[f]
            fig_rep.add_trace(go.Scatter(
                x=p_data['Length_mm'], 
                y=p_data['Amplitude_um_Norm'] + (i * offset_val),
                mode='lines', name=f"Rep {i+1}"
            ))
        
        fig_rep.update_layout(
            template="simple_white",
            xaxis_title="<b>Travel Length (mm)</b>",
            yaxis_title="<b>Real Amplitude + Offset (µm)</b>",
            xaxis=dict(mirror=True, ticks='outside', showline=True, linecolor='black', linewidth=2.5),
            yaxis=dict(mirror=True, ticks='outside', showline=True, linecolor='black', linewidth=2.5, autorange=True),
            font=dict(family="Arial", size=14, color="black")
        )
        st.plotly_chart(fig_rep, use_container_width=True)

    with tabs[3]:
        st.subheader("Representative Profile Stack (Single Main Axis)")
        offset_global = st.slider("Group Vertical Offset (µm)", 1, 200, 50, key="glob_off")
        fig_glob = go.Figure()
        
        unique_samples = sorted(df['Sample'].unique())
        for i, sample in enumerate(unique_samples):
            # Select Representative (Closest to Mean Ra)
            sample_data = df[df['Sample'] == sample]
            mean_ra = sample_data['Ra'].mean()
            closest_file = sample_data.iloc[(sample_data['Ra'] - mean_ra).abs().argsort()[:1]]['File'].values[0]
            
            p_data = profiles[closest_file]
            name = st.session_state['legend_map'].get(sample, sample)
            
            fig_glob.add_trace(go.Scatter(
                x=p_data['Length_mm'], 
                y=p_data['Amplitude_um_Norm'] + (i * offset_global), 
                mode='lines', name=name, line=dict(width=2.5)
            ))

        fig_glob.update_layout(
            template="simple_white",
            xaxis_title="<b>Travel Length (mm)</b>",
            yaxis_title="<b>Real Amplitude + Group Offset (µm)</b>",
            xaxis=dict(mirror=True, ticks='outside', showline=True, linecolor='black', linewidth=2.5),
            yaxis=dict(mirror=True, ticks='outside', showline=True, linecolor='black', linewidth=2.5, autorange=True),
            font=dict(family="Arial", size=14, color="black"),
            legend=dict(bordercolor="black", borderwidth=1),
            height=700
        )
        st.plotly_chart(fig_glob, use_container_width=True)
        st.info("💡 Each profile shows its actual topography centered at zero and offset vertically for group comparison.")

else:
    st.info("👋 Upload data batches to begin.")
