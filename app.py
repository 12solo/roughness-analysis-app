import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import re
import io

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

                # Profile Extraction & Normalization
                data_sheet = next((s for s in xl.sheet_names if "DATA" in s.upper()), None)
                if data_sheet:
                    df_p = pd.read_excel(file, sheet_name=data_sheet, usecols=[4, 5])
                    df_p.columns = ['Length_mm', 'Amplitude_um']
                    df_p = df_p.apply(pd.to_numeric, errors='coerce').dropna().reset_index(drop=True)
                    
                    if not df_p.empty:
                        # Scientific Normalization
                        df_p['Amplitude_um_Norm'] = df_p['Amplitude_um'] - df_p['Amplitude_um'].mean()
                        # Tag with metadata for overlay plotting
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

# ==========================================
# 4. DASHBOARD
# ==========================================
df = st.session_state['master_df']
profiles = st.session_state['profile_dict']

if not df.empty:
    tabs = st.tabs(["📋 Dataset", "📈 Stats", "📉 Comparative Trends", "🌊 Ra Profile Plot", "🎭 Overlay Plot", "💾 Export"])

    with tabs[0]:
        st.dataframe(df, use_container_width=True)

    with tabs[1]:
        # Statistics logic (Mean, SD, CV%)
        params = [p for p in ["Ra", "Rq", "Rz", "Rt"] if p in df.columns]
        p_sel = st.selectbox("Select Parameter", params)
        stats_df = df.groupby(["Sample", "Condition", "Day"])[p_sel].agg(['mean', 'std', 'count']).reset_index()
        stats_df['CV%'] = (stats_df['std'] / stats_df['mean']) * 100
        st.dataframe(stats_df.style.format(precision=4), use_container_width=True)

    with tabs[2]:
        # Trend comparison with 95% CI
        plot_df = df.groupby(["Sample", "Condition", "Day"])[p_sel].agg(['mean', 'std', 'count']).reset_index()
        plot_df['CI95'] = 1.96 * (plot_df['std'] / np.sqrt(plot_df['count']))
        fig_trend = px.line(plot_df, x="Sample", y="mean", color="Condition", error_y="CI95", markers=True)
        st.plotly_chart(fig_trend, use_container_width=True)
        

    with tabs[3]:
        st.subheader("Individual Normalized Profile")
        if profiles:
            sel_f = st.selectbox("View Profile for:", list(profiles.keys()))
            p_data = profiles[sel_f]
            if not p_data.empty:
                fig_prof = px.line(p_data, x='Length_mm', y='Amplitude_um_Norm', title=f"Centered Profile: {sel_f}")
                fig_prof.add_hline(y=0, line_dash="dash", line_color="red")
                st.plotly_chart(fig_prof, use_container_width=True)
                

    with tabs[4]:
        st.subheader("Scientific Overlay Plot (Journal Ready)")
        if profiles:
            # Combine profiles for overlay
            overlay_df = pd.concat(profiles.values(), ignore_index=True)
            
            # Create a high-quality journal-ready plot
            fig_ov = go.Figure()
            
            # Color palette for distinct samples
            colors = px.colors.qualitative.Bold
            
            for i, (label, group) in enumerate(overlay_df.groupby("Sample_Label")):
                fig_ov.add_trace(go.Scatter(
                    x=group['Length_mm'], 
                    y=group['Amplitude_um_Norm'],
                    mode='lines',
                    name=label,
                    line=dict(width=1, color=colors[i % len(colors)]),
                    opacity=0.8
                ))
            
            fig_ov.update_layout(
                template="simple_white",
                xaxis_title="<b>Travel Length (mm)</b>",
                yaxis_title="<b>Amplitude (µm)</b>",
                legend_title="Samples",
                font=dict(family="Arial", size=12, color="black"),
                hovermode="x unified"
            )
            st.plotly_chart(fig_ov, use_container_width=True)
            st.info("💡 **Publication Tip:** This plot uses 'Simple White' template and Arial fonts, which are preferred by most scientific journals (Elsevier, Springer, etc.).")

    with tabs[5]:
        # Wide-Format Export logic
        wide_list = []
        for fname, p_data in profiles.items():
            meta = df[df['File'] == fname].iloc[0]
            header = f"{meta['Sample']}_{meta['Condition']}_D{meta['Day']}_{fname}"
            temp = p_data[['Length_mm', 'Amplitude_um']].copy()
            temp.columns = [f"{header}_Length", f"{header}_Amplitude"]
            wide_list.append(temp)
        if wide_list:
            st.download_button("Download CSV", pd.concat(wide_list, axis=1).to_csv(index=False).encode('utf-8'), "wide_profiles.csv")
else:
    st.info("👋 Upload data for Sample A, B, etc. in the sidebar to begin.")
