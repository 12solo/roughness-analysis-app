import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import re
import os

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
        st.success(f"Added replicates for {s_name}")

    if not st.session_state['master_df'].empty:
        st.header("2. Legend Customization")
        unique_samples = sorted(st.session_state['master_df']['Sample'].unique())
        for s in unique_samples:
            if s not in st.session_state['legend_map']:
                st.session_state['legend_map'][s] = s
            st.session_state['legend_map'][s] = st.text_input(f"Rename '{s}':", st.session_state['legend_map'][s])

    if st.button("Reset Entire Study", type="primary"):
        st.session_state['master_df'] = pd.DataFrame()
        st.session_state['profile_dict'] = {}
        st.session_state['legend_map'] = {}
        st.rerun()

# ==========================================
# 4. DASHBOARD TABS
# ==========================================
df_master = st.session_state['master_df']
prof_dict = st.session_state['profile_dict']

# JOURNAL STYLE CONFIGURATION
AXIS_STYLE = dict(
    mirror=True, ticks='outside', showline=True, 
    linecolor='black', linewidth=2.5,
    title_font=dict(family="Times New Roman", size=20, color="black"),
    tickfont=dict(family="Times New Roman", size=16, color="black")
)

if not df_master.empty:
    tabs = st.tabs(["📊 Dataset", "📉 Trends", "🎨 Replicate Stack", "🏛️ Representative Stack", "💾 Export"])

    with tabs[0]:
        st.subheader("Summary Table")
        st.dataframe(df_master, use_container_width=True)

    with tabs[1]:
        st.subheader("Inter-Sample Trends")
        params = [p for p in ["Ra", "Rq", "Rz", "Rt"] if p in df_master.columns]
        if params:
            p_sel = st.selectbox("Select Parameter", params)
            plot_df = df_master.groupby(["Sample"])[p_sel].agg(['mean', 'std', 'count']).reset_index()
            plot_df['Display_Sample'] = plot_df['Sample'].map(st.session_state['legend_map'])
            
            fig_trend = px.line(plot_df, x="Display_Sample", y="mean", 
                                error_y=1.96*(plot_df['std']/np.sqrt(plot_df['count'])), 
                                markers=True, template="simple_white")
            fig_trend.update_layout(
                xaxis_title="<b>Sample ID</b>", 
                yaxis_title=f"<b>Mean {p_sel} (µm)</b>", 
                xaxis=AXIS_STYLE, yaxis=AXIS_STYLE,
                font=dict(family="Times New Roman")
            )
            st.plotly_chart(fig_trend, use_container_width=True)

    with tabs[2]:
        st.subheader("Batch Replicate Stack")
        batch_to_check = st.selectbox("Select Batch:", sorted(df_master['Sample'].unique()))
        batch_files = sorted(df_master[df_master['Sample'] == batch_to_check]['File'].tolist())
        offset_rep = st.slider("Vertical Offset (µm)", 1, 100, 20)
        
        fig_rep = go.Figure()
        tick_vals, tick_text = [], []
        for i, f in enumerate(batch_files):
            y_shift = i * offset_rep
            clean_name = os.path.splitext(f)[0]
            fig_rep.add_trace(go.Scatter(x=prof_dict[f]['Length_mm'], y=prof_dict[f]['Amplitude_um_Norm'] + y_shift, mode='lines', name=f"Rep {i+1}", showlegend=False))
            
            fig_rep.add_annotation(
                x=prof_dict[f]['Length_mm'].mean(), 
                y=y_shift + prof_dict[f]['Amplitude_um_Norm'].max() + 3, 
                text=f"<b>Rep {i+1} ({clean_name})</b>", 
                showarrow=False, 
                font=dict(family="Times New Roman", size=14, color="black"),
                bgcolor="rgba(255,255,255,0.9)", borderwidth=0
            )
            for t in [-5, 0, 5]:
                tick_vals.append(t + y_shift); tick_text.append(f"<b>{t}</b>")
        
        fig_rep.update_layout(template="simple_white", height=850, xaxis_title="<b>Travel Length (mm)</b>", yaxis_title="<b>Amplitude (µm)</b>", xaxis=AXIS_STYLE, yaxis=dict(tickmode='array', tickvals=tick_vals, ticktext=tick_text, **AXIS_STYLE))
        st.plotly_chart(fig_rep, use_container_width=True)

    with tabs[3]:
        st.subheader("Representative Stack")
        offset_global = st.slider("Group Offset (µm)", 1, 200, 50)
        fig_glob = go.Figure()
        t_vals, t_text = [], []
        unique_samples = sorted(df_master['Sample'].unique())
        for i, sample in enumerate(unique_samples):
            sample_data = df_master[df_master['Sample'] == sample]
            mean_ra = sample_data['Ra'].mean()
            std_ra = sample_data['Ra'].std()
            closest_file = sample_data.iloc[(sample_data['Ra'] - mean_ra).abs().argsort()[:1]]['File'].values[0]
            y_shift = i * offset_global
            name = st.session_state['legend_map'].get(sample, sample)
            
            fig_glob.add_trace(go.Scatter(x=prof_dict[closest_file]['Length_mm'], y=prof_dict[closest_file]['Amplitude_um_Norm'] + y_shift, mode='lines', name=name, showlegend=False))
            
            fig_glob.add_annotation(
                x=prof_dict[closest_file]['Length_mm'].min(), 
                y=y_shift + offset_global/3, 
                text=f"<b>{name}</b><br><b>Ra: {mean_ra:.3f} ± {std_ra:.3f} µm</b>", 
                showarrow=False, align="left", xanchor="left", 
                font=dict(family="Times New Roman", size=16, color="black"), 
                bgcolor="rgba(255,255,255,0.85)", borderwidth=0
            )
            for t in [-10, 0, 10]:
                t_vals.append(t + y_shift); t_text.append(f"<b>{t}</b>")
        
        fig_glob.update_layout(template="simple_white", height=850, xaxis_title="<b>Travel Length (mm)</b>", yaxis_title="<b>Amplitude (µm)</b>", xaxis=AXIS_STYLE, yaxis=dict(tickmode='array', tickvals=t_vals, ticktext=t_text, **AXIS_STYLE))
        st.plotly_chart(fig_glob, use_container_width=True)

    with tabs[4]:
        st.subheader("Export Dataset")
        wide_list = []
        for fname, p_data in prof_dict.items():
            meta = df_master[df_master['File'] == fname].iloc[0]
            header = f"{st.session_state['legend_map'].get(meta['Sample'], meta['Sample'])}_{fname}"
            temp = p_data[['Length_mm', 'Amplitude_um']].copy()
            temp.columns = [f"{header}_L", f"{header}_Amp"]
            wide_list.append(temp)
        if wide_list:
            st.download_button("Download CSV", pd.concat(wide_list, axis=1).to_csv(index=False).encode('utf-8'), "scientific_export.csv")
else:
    st.info("👋 Use the sidebar to upload your sample replicates.")
