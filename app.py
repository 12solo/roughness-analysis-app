import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import re

# ==========================================
# 1. SESSION STATE FOR DATA STORAGE
# ==========================================
if 'master_df' not in st.session_state:
    st.session_state['master_df'] = pd.DataFrame()
if 'profile_dict' not in st.session_state:
    st.session_state['profile_dict'] = {}

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
                    profile_map[file.name] = df_p

            except Exception as e:
                st.error(f"Error in {file.name}: {e}")
                
        return pd.DataFrame(combined_summary), profile_map

# ==========================================
# 3. UI & SIDEBAR INPUT
# ==========================================
st.set_page_config(page_title="Multi-Sample Roughness Lab", layout="wide")
st.title("🔬 Multi-Sample Scientific Roughness Analyzer")

with st.sidebar:
    st.header("1. Data Input")
    with st.form("sample_upload_form", clear_on_submit=True):
        sample_name = st.text_input("Sample/Specimen Name (e.g., Sample A)", "Sample A")
        condition = st.selectbox("Experimental Condition", ["Control", "Oven", "UV", "Humidity"])
        ageing_day = st.number_input("Ageing Day", min_value=0, step=1)
        files = st.file_uploader("Upload Replicates for this Sample (.xlsx)", accept_multiple_files=True)
        submit = st.form_submit_button("Add Sample Group to Analysis")

    if submit and files:
        loader = RoughnessLoader()
        meta = {"Sample": sample_name, "Condition": condition, "Day": ageing_day}
        new_summary, new_profiles = loader.process_files(files, meta)
        st.session_state['master_df'] = pd.concat([st.session_state['master_df'], new_summary], ignore_index=True)
        st.session_state['profile_dict'].update(new_profiles)
        st.success(f"Added {len(files)} replicates for {sample_name}")

    if st.button("Reset Entire Study"):
        st.session_state['master_df'] = pd.DataFrame()
        st.session_state['profile_dict'] = {}
        st.rerun()

# ==========================================
# 4. SCIENTIFIC ANALYSIS
# ==========================================
df = st.session_state['master_df']

if not df.empty:
    tabs = st.tabs(["📊 Master Dataset", "📈 Grouped Statistics", "📉 Inter-Sample Comparison", "💾 Columnar Export"])

    with tabs[0]:
        st.dataframe(df, use_container_width=True)

    with tabs[1]:
        params = [p for p in ["Ra", "Rq", "Rz", "Rt"] if p in df.columns]
        if params:
            p_sel = st.selectbox("Parameter for Stats", params)
            stats_df = df.groupby(["Sample", "Condition", "Day"])[p_sel].agg(['mean', 'std', 'count']).reset_index()
            stats_df['CV%'] = (stats_df['std'] / stats_df['mean']) * 100
            st.write("### Summary Statistics Table")
            st.dataframe(stats_df.style.format(precision=4), use_container_width=True)

    with tabs[2]:
        st.subheader("Inter-Sample Comparison")
        if len(df['Sample'].unique()) > 0:
            # Calculate 95% CI for the plot
            plot_data = df.groupby(["Sample", "Condition", "Day"])[p_sel].agg(['mean', 'std', 'count']).reset_index()
            plot_data['CI95'] = 1.96 * (plot_data['std'] / np.sqrt(plot_data['count']))
            
            # SORT by Sample name for horizontal axis
            plot_data = plot_data.sort_values("Sample")

            st.write(f"#### Comparison of {p_sel} across Samples")
            fig_comp = go.Figure()

            # Plot each Condition/Day combination as a separate trace
            for group_label, group_df in plot_data.groupby(["Condition", "Day"]):
                legend_name = f"{group_label[0]} (Day {group_label[1]})"
                
                fig_comp.add_trace(go.Scatter(
                    x=group_df['Sample'], # HORIZONTAL AXIS IS SAMPLES
                    y=group_df['mean'],
                    error_y=dict(type='data', array=group_df['CI95'], visible=True),
                    name=legend_name,
                    mode='lines+markers'
                ))
            
            fig_comp.update_layout(
                xaxis_title="Specimen / Sample Name",
                yaxis_title=f"Mean {p_sel} (µm)",
                template="plotly_white",
                xaxis={'categoryorder':'array', 'categoryarray': sorted(plot_data['Sample'].unique())}
            )
            st.plotly_chart(fig_comp, use_container_width=True)
            
            st.info("💡 The horizontal axis now represents your different samples (A, B, C...). Lines connect the same condition/ageing period across these samples.")

    with tabs[3]:
        # Export logic remains same
        st.subheader("Columnar Export")
        wide_list = []
        for file_name, p_data in st.session_state['profile_dict'].items():
            file_meta = df[df['File'] == file_name].iloc[0]
            header = f"{file_meta['Sample']}_{file_meta['Condition']}_D{file_meta['Day']}_{file_name}"
            p_temp = p_data.copy()
            p_temp.columns = [f"{header}_Length", f"{header}_Amplitude"]
            wide_list.append(p_temp)
        if wide_list:
            wide_df = pd.concat(wide_list, axis=1)
            csv = wide_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download CSV", csv, "master_profiles.csv", "text/csv")
else:
    st.info("👋 Use the sidebar to add your sample groups (Sample A, Sample B...) one by one.")
