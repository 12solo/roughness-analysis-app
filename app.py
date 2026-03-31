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
                        df_p['Condition'] = meta_template['Condition']
                        df_p['Day'] = meta_template['Day']
                        profile_map[file.name] = df_p

            except Exception as e:
                st.error(f"Error in {file.name}: {e}")
                
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
        st.success(f"Added {len(s_files)} replicates for {s_name}")

    st.markdown("---")
    st.header("2. Legend Customization")
    if not st.session_state['master_df'].empty:
        unique_samples = st.session_state['master_df']['Sample'].unique()
        for s in unique_samples:
            st.session_state['legend_map'][s] = st.text_input(f"Rename '{s}' to:", s)

    if st.button("Reset Entire Study", type="primary"):
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
    tabs = st.tabs(["📋 Dataset", "📈 Stats", "📉 Trends", "🌊 Individual Batch Profiles", "🎨 Master Stacked Plot", "💾 Export"])

    with tabs[0]:
        st.dataframe(df, use_container_width=True)

    with tabs[1]:
        params = [p for p in ["Ra", "Rq", "Rz", "Rt"] if p in df.columns]
        if params:
            p_sel = st.selectbox("Select Parameter", params)
            stats_df = df.groupby(["Sample", "Condition", "Day"])[p_sel].agg(['mean', 'std', 'count']).reset_index()
            st.dataframe(stats_df.style.format(precision=4), use_container_width=True)

    with tabs[2]:
        if 'p_sel' in locals():
            plot_df = df.groupby(["Sample", "Condition", "Day"])[p_sel].agg(['mean', 'std', 'count']).reset_index()
            # Map legend names
            plot_df['Sample'] = plot_df['Sample'].map(st.session_state['legend_map'])
            fig_trend = px.line(plot_df, x="Sample", y="mean", markers=True, template="simple_white")
            fig_trend.update_layout(showlegend=False)
            st.plotly_chart(fig_trend, use_container_width=True)

    with tabs[3]:
        st.subheader("Batch Replicate Inspection")
        if profiles:
            unique_batches = df['Sample'].unique()
            selected_s = st.selectbox("Select Batch to View:", unique_batches)
            batch_files = df[df['Sample'] == selected_s]['File'].tolist()
            
            offset_val = st.slider("Vertical Offset (µm)", 1, 50, 10, key="batch_offset")
            fig_batch = go.Figure()
            
            for i, fname in enumerate(sorted(batch_files)):
                if fname in profiles:
                    p_data = profiles[fname]
                    fig_batch.add_trace(go.Scatter(
                        x=p_data['Length_mm'],
                        y=p_data['Amplitude_um_Norm'] + (i * offset_val),
                        mode='lines', name=f"Rep {i+1}", line=dict(width=1)
                    ))
            
            fig_batch.update_layout(template="simple_white", showlegend=True, 
                                    xaxis=dict(mirror=True, ticks='outside', showline=True, linecolor='black'),
                                    yaxis=dict(mirror=True, ticks='outside', showline=True, linecolor='black'))
            st.plotly_chart(fig_batch, use_container_width=True)

    with tabs[4]:
        st.subheader("Global Comparison (Mean Profiles)")
        if profiles:
            global_offset = st.slider("Vertical Offset (µm)", 1, 100, 25, key="global_offset")
            fig_global = go.Figure()
            
            unique_samples = sorted(df['Sample'].unique())
            for i, sample in enumerate(unique_samples):
                sample_files = df[df['Sample'] == sample]['File'].tolist()
                relevant_profs = [profiles[f] for f in sample_files if f in profiles]
                
                if relevant_profs:
                    combined = pd.concat(relevant_profs)
                    combined['L_grp'] = combined['Length_mm'].round(5)
                    mean_prof = combined.groupby('L_grp')['Amplitude_um_Norm'].mean().reset_index()
                    
                    # Use the Custom Legend Name
                    display_name = st.session_state['legend_map'].get(sample, sample)
                    
                    fig_global.add_trace(go.Scatter(
                        x=mean_prof['L_grp'],
                        y=mean_prof['Amplitude_um_Norm'] + (i * global_offset),
                        mode='lines', name=display_name, line=dict(width=2.5)
                    ))
            
            fig_global.update_layout(
                template="simple_white", 
                xaxis_title="<b>Travel Length (mm)</b>", 
                yaxis_title="<b>Amplitude + Offset (µm)</b>", 
                height=700,
                font=dict(family="Arial", size=14, color="black"),
                # AXIS BORDER & TICKS
                xaxis=dict(mirror=True, ticks='outside', showline=True, linecolor='black', gridcolor='lightgrey'),
                yaxis=dict(mirror=True, ticks='outside', showline=True, linecolor='black', gridcolor='lightgrey'),
                legend=dict(bordercolor="black", borderwidth=1)
            )
            st.plotly_chart(fig_global, use_container_width=True)

    with tabs[5]:
        wide_list = []
        for fname, p_data in profiles.items():
            meta = df[df['File'] == fname].iloc[0]
            header = st.session_state['legend_map'].get(meta['Sample'], meta['Sample']) + "_" + fname
            temp = p_data[['Length_mm', 'Amplitude_um']].copy()
            temp.columns = [f"{header}_Length", f"{header}_Amplitude"]
            wide_list.append(temp)
        if wide_list:
            st.download_button("Download CSV", pd.concat(wide_list, axis=1).to_csv(index=False).encode('utf-8'), "profiles.csv")
else:
    st.info("👋 Upload data for your sample batches to begin.")
