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
        s_name = st.text_input("Sample/Batch ID", "Sample A")
        s_files = st.file_uploader("Upload Replicate Files (.xlsx)", accept_multiple_files=True)
        submit = st.form_submit_button("Add Sample Batch")

    if submit and s_files:
        loader = RoughnessLoader()
        meta = {"Sample": s_name, "Condition": "Standard", "Day": 0}
        new_sum, new_prof = loader.process_files(s_files, meta)
        st.session_state['master_df'] = pd.concat([st.session_state['master_df'], new_sum], ignore_index=True)
        st.session_state['profile_dict'].update(new_prof)

    st.markdown("---")
    st.header("2. Legend Customization")
    if not st.session_state['master_df'].empty:
        unique_samples = sorted(st.session_state['master_df']['Sample'].unique())
        for s in unique_samples:
            st.session_state['legend_map'][s] = st.text_input(f"Rename '{s}':", s)

    if st.button("Reset Entire Study", type="primary"):
        st.session_state['master_df'] = pd.DataFrame()
        st.session_state['profile_dict'] = {}
        st.session_state['legend_map'] = {}
        st.rerun()

# ==========================================
# 4. DASHBOARD TABS
# ==========================================
df = st.session_state['master_df']
profiles = st.session_state['profile_dict']

# Shared Layout Settings
SHARED_LAYOUT = dict(
    template="simple_white",
    height=850,
    font=dict(family="Arial", size=14, color="black"),
    margin=dict(l=80, r=40, t=60, b=80),
    xaxis=dict(mirror=True, ticks='outside', showline=True, linecolor='black', linewidth=2.5, title_font=dict(size=16)),
    yaxis=dict(mirror=True, ticks='outside', showline=True, linecolor='black', linewidth=2.5, title_font=dict(size=16), autorange=True),
)

if not df.empty:
    tabs = st.tabs(["📊 Dataset", "📉 Trends", "🌊 Individual Ra Profile", "🎨 Batch Replicate Stack", "🏛️ Representative Stack", "💾 Export"])

    with tabs[3]:
        st.subheader("Batch Replicate Inspection (Direct Labels)")
        batch_to_check = st.selectbox("Select Batch:", sorted(df['Sample'].unique()))
        batch_files = sorted(df[df['Sample'] == batch_to_check]['File'].tolist())
        offset_rep = st.slider("Vertical Offset (µm)", 1, 50, 15, key="rep_off")
        
        fig_rep = go.Figure()
        tick_vals, tick_text = [], []
        
        for i, f in enumerate(batch_files):
            if f in profiles:
                y_shift = i * offset_rep
                p_data = profiles[f]
                # Add Profile Line
                fig_rep.add_trace(go.Scatter(x=p_data['Length_mm'], y=p_data['Amplitude_um_Norm'] + y_shift, mode='lines', name=f"Rep {i+1}", showlegend=False))
                
                # Add In-Plot Legend (File name)
                fig_rep.add_annotation(x=p_data['Length_mm'].mean(), y=y_shift + p_data['Amplitude_um_Norm'].max() + 2,
                                     text=f"<b>Rep {i+1} ({f})</b>", showarrow=False, font=dict(size=11))
                
                for t in [-5, 0, 5]:
                    tick_vals.append(t + y_shift); tick_text.append(str(t))
        
        fig_rep.update_layout(**SHARED_LAYOUT)
        fig_rep.update_layout(xaxis_title="<b>Travel Length (mm)</b>", yaxis_title="<b>Amplitude (µm)</b>",
                            yaxis=dict(tickmode='array', tickvals=tick_vals, ticktext=tick_text))
        st.plotly_chart(fig_rep, use_container_width=True)

    with tabs[4]:
        st.subheader("Representative Profile Comparison (Mean ± SD Labels)")
        offset_global = st.slider("Group Vertical Offset (µm)", 1, 150, 40, key="glob_off")
        fig_glob = go.Figure()
        t_vals, t_text = [], []
        
        unique_samples = sorted(df['Sample'].unique())
        for i, sample in enumerate(unique_samples):
            sample_data = df[df['Sample'] == sample]
            m_ra = sample_data['Ra'].mean()
            s_ra = sample_data['Ra'].std()
            
            # Select Real Representative
            closest_file = sample_data.iloc[(sample_data['Ra'] - m_ra).abs().argsort()[:1]]['File'].values[0]
            
            if closest_file in profiles:
                p_data = profiles[closest_file]
                y_shift = i * offset_global
                name = st.session_state['legend_map'].get(sample, sample)
                
                # Add Profile Line
                fig_glob.add_trace(go.Scatter(x=p_data['Length_mm'], y=p_data['Amplitude_um_Norm'] + y_shift, mode='lines', name=name, line=dict(width=2.5), showlegend=False))
                
                # Add In-Plot Legend with Mean + SD
                label_text = f"<b>{name}</b><br>Ra: {m_ra:.3f} ± {s_ra:.3f} µm"
                fig_glob.add_annotation(x=p_data['Length_mm'].min(), y=y_shift + offset_global/3,
                                      text=label_text, showarrow=False, align="left", xanchor="left",
                                      font=dict(size=13, color="black"), bgcolor="rgba(255,255,255,0.7)")
                
                for t in [-10, 0, 10]:
                    t_vals.append(t + y_shift); t_text.append(str(t))
        
        fig_glob.update_layout(**SHARED_LAYOUT)
        fig_glob.update_layout(
            xaxis_title="<b>Travel Length (mm)</b>", yaxis_title="<b>Amplitude (µm)</b>",
            yaxis=dict(tickmode='array', tickvals=t_vals, ticktext=t_text),
            margin=dict(t=80) # Extra space for top labels
        )
        st.plotly_chart(fig_glob, use_container_width=True)
        st.info("💡 Labels are placed directly on the plot. Ra values represent the Mean ± SD of the entire batch.")

    with tabs[5]:
        wide_list = []
        for fname, p_data in profiles.items():
            meta = df[df['File'] == fname].iloc[0]
            header = f"{st.session_state['legend_map'].get(meta['Sample'], meta['Sample'])}_{fname}"
            temp = p_data[['Length_mm', 'Amplitude_um']].copy()
            temp.columns = [f"{header}_L", f"{header}_Amp"]
            wide_list.append(temp)
        if wide_list:
            st.download_button("Download CSV", pd.concat(wide_list, axis=1).to_csv(index=False).encode('utf-8'), "export.csv")
else:
    st.info("👋 Upload your sample batches to begin.")
