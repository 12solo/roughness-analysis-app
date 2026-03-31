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
                
                # Summary extraction logic (Pass 1)
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

                # Profile extraction and Normalization (Pass 2)
                data_sheet = next((s for s in xl.sheet_names if "DATA" in s.upper()), None)
                if data_sheet:
                    df_p = pd.read_excel(file, sheet_name=data_sheet, usecols=[4, 5])
                    df_p.columns = ['Length_mm', 'Amplitude_um']
                    df_p = df_p.apply(pd.to_numeric, errors='coerce').dropna().reset_index(drop=True)
                    
                    # --- SCIENTIFIC NORMALIZATION ---
                    # 1. Zero-Mean Centering: Subtract mean from Amplitude to center at 0
                    df_p['Amplitude_um_Norm'] = df_p['Amplitude_um'] - df_p['Amplitude_um'].mean()
                    
                    profile_map[file.name] = df_p

            except Exception as e:
                st.error(f"Error in {file.name}: {e}")
                
        return pd.DataFrame(combined_summary), profile_map

# ==========================================
# 3. UI & SIDEBAR INPUT
# ==========================================
st.set_page_config(page_title="Scientific Roughness Lab", layout="wide")
st.title("🔬 Scientific Roughness Analyzer & Profiler")

with st.sidebar:
    st.header("1. Data Input")
    with st.form("sample_upload_form", clear_on_submit=True):
        sample_name = st.text_input("Sample/Specimen Name", "Sample A")
        condition = st.selectbox("Experimental Condition", ["Control", "Oven", "UV", "Humidity"])
        ageing_day = st.number_input("Ageing Day", min_value=0, step=1)
        files = st.file_uploader("Upload Replicates (.xlsx)", accept_multiple_files=True)
        submit = st.form_submit_button("Add Sample Group to Analysis")

    if submit and files:
        loader = RoughnessLoader()
        meta = {"Sample": sample_name, "Condition": condition, "Day": ageing_day}
        new_summary, new_profiles = loader.process_files(files, meta)
        st.session_state['master_df'] = pd.concat([st.session_state['master_df'], new_summary], ignore_index=True)
        st.session_state['profile_dict'].update(new_profiles)
        st.success(f"Added {len(files)} replicates for {sample_name}")

    if st.button("Reset Study"):
        st.session_state['master_df'] = pd.DataFrame()
        st.session_state['profile_dict'] = {}
        st.rerun()

# ==========================================
# 4. ANALYSIS & PLOTTING
# ==========================================
df = st.session_state['master_df']

if not df.empty:
    tabs = st.tabs(["📊 Dataset", "📉 Ra Profile Plot", "📈 Statistics", "🔬 Inter-Sample Comparison"])

    with tabs[0]:
        st.dataframe(df, use_container_width=True)

    with tabs[1]:
        st.subheader("Normalized Surface Roughness Profile")
        if st.session_state['profile_dict']:
            # Select which file to plot
            f_names = list(st.session_state['profile_dict'].keys())
            selected_f = st.selectbox("Select File to Visualize", f_names)
            p_df = st.session_state['profile_dict'][selected_f]

            # Create the Plot
            fig_p = go.Figure()
            
            # Zero Line Reference
            fig_p.add_shape(type="line", x0=p_df['Length_mm'].min(), y0=0, x1=p_df['Length_mm'].max(), y1=0,
                            line=dict(color="Red", width=1, dash="dash"))

            # Normalized Profile
            fig_p.add_trace(go.Scatter(x=p_df['Length_mm'], y=p_df['Amplitude_um_Norm'],
                                       mode='lines', name='Normalized Ra Profile',
                                       line=dict(color='#1f77b4', width=1)))

            fig_p.update_layout(title=f"Normalized Profile: {selected_f}",
                                xaxis_title="Travel Length (mm)",
                                yaxis_title="Amplitude (µm)",
                                template="plotly_white",
                                yaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor='Black'))
            
            st.plotly_chart(fig_p, use_container_width=True)
            
            st.info("💡 **Scientific Normalization:** The Y-axis has been centered to a mean of zero. This represents the 'Ra' profile by removing the vertical positioning offset of the stylus.")

    with tabs[2]:
        # Statistics logic (Mean, SD, CV%)
        params = [p for p in ["Ra", "Rq", "Rz", "Rt"] if p in df.columns]
        p_sel = st.selectbox("Parameter for Stats", params)
        stats_df = df.groupby(["Sample", "Condition", "Day"])[p_sel].agg(['mean', 'std', 'count']).reset_index()
        stats_df['CV%'] = (stats_df['std'] / stats_df['mean']) * 100
        st.write("### Grouped Summary Table")
        st.dataframe(stats_df.style.format(precision=4), use_container_width=True)

    with tabs[3]:
        st.subheader("Inter-Sample Comparison")
        plot_data = df.groupby(["Sample", "Condition", "Day"])[p_sel].agg(['mean', 'std', 'count']).reset_index()
        plot_data['CI95'] = 1.96 * (plot_data['std'] / np.sqrt(plot_data['count']))
        
        fig_comp = px.line(plot_data, x="Sample", y="mean", color="Condition", 
                           error_y="CI95", markers=True, title=f"Comparison of {p_sel} across Samples")
        st.plotly_chart(fig_comp, use_container_width=True)

else:
    st.info("👋 Use the sidebar to upload your sample replicates.")
