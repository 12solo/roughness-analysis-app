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
                
                # Pass 1: Summary Stats extraction
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

                # Pass 2: Profile Normalization
                data_sheet = next((s for s in xl.sheet_names if "DATA" in s.upper()), None)
                if data_sheet:
                    df_p = pd.read_excel(file, sheet_name=data_sheet, usecols=[4, 5])
                    df_p.columns = ['Length_mm', 'Amplitude_um']
                    df_p = df_p.apply(pd.to_numeric, errors='coerce').dropna().reset_index(drop=True)
                    
                    if not df_p.empty:
                        # Scientific Normalization: Mean-centering
                        df_p['Amplitude_um_Norm'] = df_p['Amplitude_um'] - df_p['Amplitude_um'].mean()
                        df_p['Sample_Label'] = f"{meta_template['Sample']} ({meta_template['Condition']})"
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
        s_name = st.text_input("Sample Name", "Sample A")
        s_cond = st.selectbox("Condition", ["Control", "Oven", "UV", "Humidity"])
        s_day = st.number_input("Ageing Day", min_value=0, step=1)
        s_files = st.file_uploader("Upload Replicate Files (.xlsx)", accept_multiple_files=True)
        submit = st.form_submit_button("Add Sample Batch")

    if submit and s_files:
        loader = RoughnessLoader()
        meta = {"Sample": s_name, "Condition": s_cond, "Day": s_day}
        new_sum, new_prof = loader.process_files(s_files, meta)
        st.session_state['master_df'] = pd.concat([st.session_state['master_df'], new_sum], ignore_index=True)
        st.session_state['profile_dict'].update(new_prof)
        st.success(f"Added {len(s_files)} replicates for {s_name}")

    st.markdown("---")
    if st.button("Reset Entire Study", type="primary"):
        st.session_state['master_df'] = pd.DataFrame()
        st.session_state['profile_dict'] = {}
        st.rerun()

# ==========================================
# 4. DASHBOARD
# ==========================================
df = st.session_state['master_df']
profiles = st.session_state['profile_dict']

if not df.empty:
    tabs = st.tabs(["📋 Dataset", "📈 Statistics", "📉 Comparative Trends", "🌊 Individual Ra Profile", "🎨 Stacked Comparison", "💾 Export"])

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
        if 'p_sel' in locals():
            plot_df = df.groupby(["Sample", "Condition", "Day"])[p_sel].agg(['mean', 'std', 'count']).reset_index()
            plot_df['CI95'] = 1.96 * (plot_df['std'] / np.sqrt(plot_df['count']))
            fig_trend = px.line(plot_df, x="Sample", y="mean", color="Condition", error_y="CI95", markers=True, 
                                title=f"Comparison of {p_sel} Across Samples")
            st.plotly_chart(fig_trend, use_container_width=True)

    with tabs[3]:
        st.subheader("Individual Normalized Ra Profile")
        if profiles:
            sel_f = st.selectbox("Select Test File to View Profile:", list(profiles.keys()))
            p_data = profiles[sel_f]
            if not p_data.empty:
                fig_indiv = go.Figure()
                fig_indiv.add_trace(go.Scatter(x=p_data['Length_mm'], y=p_data['Amplitude_um_Norm'], mode='lines', name='Profile'))
                fig_indiv.add_hline(y=0, line_dash="dash", line_color="red")
                fig_indiv.update_layout(xaxis_title="Length (mm)", yaxis_title="Amplitude (µm)", template="simple_white")
                st.plotly_chart(fig_indiv, use_container_width=True)

    with tabs[4]:
        st.subheader("Stacked Profile Plot (Scientific Visualization)")
        if profiles:
            unique_samples = df['Sample'].unique()
            offset_step = st.slider("Vertical Offset (µm)", 1, 100, 20)
            
            fig_stack = go.Figure()
            
            # CASE A: MULTIPLE BATCHES -> Plot Mean Profile of each Sample
            if len(unique_samples) > 1:
                st.info("Multiple samples detected: Plotting the **Mean Profile** of each batch.")
                for i, sample in enumerate(sorted(unique_samples)):
                    # Get all files belonging to this sample
                    sample_files = df[df['Sample'] == sample]['File'].tolist()
                    sample_profiles = [profiles[f] for f in sample_files if f in profiles]
                    
                    if sample_profiles:
                        # Combine and average profiles (grouped by Length)
                        combined = pd.concat(sample_profiles)
                        # Round Length to handle small floating point differences in sampling
                        combined['Length_mm_grp'] = combined['Length_mm'].round(6)
                        mean_profile = combined.groupby('Length_mm_grp')['Amplitude_um_Norm'].mean().reset_index()
                        
                        y_offset = i * offset_step
                        fig_stack.add_trace(go.Scatter(
                            x=mean_profile['Length_mm_grp'],
                            y=mean_profile['Amplitude_um_Norm'] + y_offset,
                            mode='lines',
                            name=f"Mean: {sample}",
                            line=dict(width=2)
                        ))
            
            # CASE B: ONLY ONE BATCH -> Plot All Tests for that Sample
            else:
                st.info("Single batch detected: Plotting **all individual tests** stacked.")
                sorted_files = sorted(profiles.keys())
                for i, fname in enumerate(sorted_files):
                    p_data = profiles[fname]
                    y_offset = i * offset_step
                    fig_stack.add_trace(go.Scatter(
                        x=p_data['Length_mm'],
                        y=p_data['Amplitude_um_Norm'] + y_offset,
                        mode='lines',
                        name=fname,
                        line=dict(width=1)
                    ))
            
            fig_stack.update_layout(
                template="simple_white",
                xaxis_title="<b>Travel Length (mm)</b>",
                yaxis_title="<b>Amplitude + Vertical Offset (µm)</b>",
                height=700,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_stack, use_container_width=True)
            st.caption("Profiles are mean-centered (normalized) and then offset vertically for clarity.")

    with tabs[5]:
        st.subheader("Bulk Export (Columnar Wide Format)")
        wide_list = []
        for fname, p_data in profiles.items():
            meta = df[df['File'] == fname].iloc[0]
            header = f"{meta['Sample']}_{meta['Condition']}_D{meta['Day']}_{fname}"
            temp = p_data[['Length_mm', 'Amplitude_um']].copy()
            temp.columns = [f"{header}_Length", f"{header}_Amplitude"]
            wide_list.append(temp)
        if wide_list:
            final_wide = pd.concat(wide_list, axis=1)
            st.download_button("Download CSV", final_wide.to_csv(index=False).encode('utf-8'), "scientific_profiles_wide.csv")
else:
    st.info("👋 Use the sidebar to upload your sample replicates. The app will automatically stack individual tests for single batches or mean profiles for comparative studies.")
