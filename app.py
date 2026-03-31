import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import re
import io

# ==========================================
# 1. INITIALIZE SESSION STATE (Persistent)
# ==========================================
if 'master_df' not in st.session_state:
    st.session_state['master_df'] = pd.DataFrame()
if 'profile_dict' not in st.session_state:
    st.session_state['profile_dict'] = {}

# ==========================================
# 2. UNIVERSAL SCIENTIFIC LOADER
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
                
                # Summary Extraction (Pass 1)
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

                # Profile Extraction & Normalization (Pass 2)
                data_sheet = next((s for s in xl.sheet_names if "DATA" in s.upper()), None)
                if data_sheet:
                    df_p = pd.read_excel(file, sheet_name=data_sheet, usecols=[4, 5])
                    df_p.columns = ['Length_mm', 'Amplitude_um']
                    df_p = df_p.apply(pd.to_numeric, errors='coerce').dropna().reset_index(drop=True)
                    # SCIENTIFIC NORMALIZATION: Zero-Mean Centering
                    df_p['Amplitude_um_Norm'] = df_p['Amplitude_um'] - df_p['Amplitude_um'].mean()
                    profile_map[file.name] = df_p

            except Exception as e:
                st.error(f"Error in {file.name}: {e}")
                
        return pd.DataFrame(combined_summary), profile_map

# ==========================================
# 3. UI & SIDEBAR (Multi-Batch Input)
# ==========================================
st.set_page_config(page_title="Scientific Roughness Lab", layout="wide")
st.title("🔬 Scientific Surface Roughness Analyzer")

with st.sidebar:
    st.header("1. Sample Batch Input")
    st.info("Upload replicates for Sample A, process, then upload Sample B, etc.")
    
    with st.form("input_form", clear_on_submit=True):
        s_name = st.text_input("Sample/Specimen ID", "Sample A")
        s_cond = st.selectbox("Condition", ["Control", "Oven", "UV", "Humidity"])
        s_day = st.number_input("Ageing Day", min_value=0, step=1)
        s_files = st.file_uploader("Upload Replicate Files (.xlsx)", accept_multiple_files=True)
        submit = st.form_submit_button("Add to Analysis Batch")

    if submit and s_files:
        loader = RoughnessLoader()
        meta = {"Sample": s_name, "Condition": s_cond, "Day": s_day}
        new_sum, new_prof = loader.process_files(s_files, meta)
        st.session_state['master_df'] = pd.concat([st.session_state['master_df'], new_sum], ignore_index=True)
        st.session_state['profile_dict'].update(new_prof)
        st.success(f"Added {len(s_files)} replicates for {s_name}")

    if st.button("Reset All Data"):
        st.session_state['master_df'] = pd.DataFrame()
        st.session_state['profile_dict'] = {}
        st.rerun()

# ==========================================
# 4. SCIENTIFIC DASHBOARD
# ==========================================
df = st.session_state['master_df']

if not df.empty:
    tabs = st.tabs(["📋 Dataset", "📈 Grouped Stats", "📉 Comparative Trends", "🌊 Ra Profile Plot", "💾 Export"])

    with tabs[0]:
        st.dataframe(df, use_container_width=True)

    with tabs[1]:
        st.subheader("Scenario 1: Replicate Precision (Mean, SD, CV%)")
        params = [p for p in ["Ra", "Rq", "Rz", "Rt"] if p in df.columns]
        if params:
            p_sel = st.selectbox("Select Parameter", params)
            # Grouping by sample/condition/day to analyze the replicates
            stats_df = df.groupby(["Sample", "Condition", "Day"])[p_sel].agg(['mean', 'std', 'count']).reset_index()
            stats_df['CV%'] = (stats_df['std'] / stats_df['mean']) * 100
            st.dataframe(stats_df.style.format(precision=4), use_container_width=True)
            st.info("💡 **Scientific Note:** CV% (Coefficient of Variation) < 10% indicates excellent test repeatability.")

    with tabs[2]:
        st.subheader("Scenario 2: Inter-Sample Comparison (X-Axis: Samples)")
        # Scientific Trend Plotting with Error Bars
        plot_df = df.groupby(["Sample", "Condition", "Day"])[p_sel].agg(['mean', 'std', 'count']).reset_index()
        plot_df['CI95'] = 1.96 * (plot_df['std'] / np.sqrt(plot_df['count']))
        
        fig_trend = go.Figure()
        for label, group_data in plot_df.groupby(["Condition", "Day"]):
            fig_trend.add_trace(go.Scatter(
                x=group_data['Sample'], 
                y=group_data['mean'],
                error_y=dict(type='data', array=group_data['CI95'], visible=True),
                name=f"{label[0]} (Day {label[1]})", mode='lines+markers'
            ))
        fig_trend.update_layout(xaxis_title="Specimens (Sample A, B, C...)", yaxis_title=f"Mean {p_sel} (µm)", template="plotly_white")
        st.plotly_chart(fig_trend, use_container_width=True)
        

    with tabs[3]:
        st.subheader("Normalized Surface Profile Plot")
        f_list = list(st.session_state['profile_dict'].keys())
        sel_f = st.selectbox("View Profile for File:", f_list)
        prof_df = st.session_state['profile_dict'][sel_f]
        
        fig_prof = px.line(prof_df, x='Length_mm', y='Amplitude_um_Norm', title=f"Centered Profile: {sel_f}")
        fig_prof.add_hline(y=0, line_dash="dash", line_color="red")
        fig_prof.update_layout(xaxis_title="Length (mm)", yaxis_title="Amplitude (µm) [Mean-Centered]")
        st.plotly_chart(fig_prof, use_container_width=True)
        

    with tabs[4]:
        st.subheader("Bulk Columnar Export (Side-by-Side)")
        wide_list = []
        for fname, p_data in st.session_state['profile_dict'].items():
            meta = df[df['File'] == fname].iloc[0]
            header = f"{meta['Sample']}_{meta['Condition']}_D{meta['Day']}_{fname}"
            temp = p_data[['Length_mm', 'Amplitude_um']].copy()
            temp.columns = [f"{header}_Length", f"{header}_Amplitude"]
            wide_list.append(temp)
        
        if wide_list:
            final_csv = pd.concat(wide_list, axis=1).to_csv(index=False).encode('utf-8')
            st.download_button("Download Wide-Format CSV", final_csv, "scientific_profiles_wide.csv")
else:
    st.info("👋 Upload your first batch (e.g., Sample A, 15 files) to begin.")
