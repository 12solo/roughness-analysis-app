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
if 'summary_df' not in st.session_state:
    st.session_state['summary_df'] = pd.DataFrame()
if 'profile_dict' not in st.session_state:
    st.session_state['profile_dict'] = {}

# ==========================================
# 2. SMART SCAN DATA LOADER
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

                # PASS 1: SUMMARY PARAMETERS
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

                # PASS 2: PROFILE DATA (Col E & F)
                data_sheet_name = next((s for s in xl.sheet_names if "DATA" in s.upper()), None)
                if data_sheet_name:
                    df_p = pd.read_excel(file, sheet_name=data_sheet_name, usecols=[4, 5])
                    df_p.columns = ['Length_mm', 'Amplitude_um']
                    df_p = df_p.apply(pd.to_numeric, errors='coerce').dropna().reset_index(drop=True)
                    profile_map[file.name] = {'data': df_p, 'meta': meta}

            except Exception as e:
                st.error(f"❌ Error in {file.name}: {e}")
                
        return pd.DataFrame(combined_summary), profile_map

# ==========================================
# 3. SCIENTIFIC STATS ENGINE
# ==========================================
def get_advanced_stats(df, group_cols, value_col):
    if value_col not in df.columns or df.empty:
        return pd.DataFrame()
    
    # Calculate Mean, SD, Count
    stats_df = df.groupby(group_cols)[value_col].agg(['mean', 'std', 'count']).reset_index()
    
    # Calculate Coefficient of Variation (CV%) - Important for Scenario 1 (Replicates)
    stats_df['CV%'] = (stats_df['std'] / stats_df['mean']) * 100
    
    # Calculate 95% Confidence Interval
    stats_df['SEM'] = stats_df['std'] / np.sqrt(stats_df['count'])
    stats_df['CI_95'] = 1.96 * stats_df['SEM']
    
    return stats_df

# ==========================================
# 4. UI LAYOUT
# ==========================================
st.set_page_config(page_title="Scientific Roughness Lab", layout="wide")
st.title("🔬 Advanced Roughness Analysis Dashboard")

# SIDEBAR
st.sidebar.header("1. Upload Samples")
uploaded_files = st.sidebar.file_uploader("Upload .xlsx Files", accept_multiple_files=True, type=['xlsx'])

if uploaded_files:
    st.sidebar.subheader("2. Sample Metadata")
    m_id = st.sidebar.text_input("Material/Specimen Name", "Sample_A")
    cond = st.sidebar.selectbox("Condition/Formulation", ["Control", "Oven 70C", "UV Exposure", "Humidity"])
    
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
        st.success(f"Processed {len(uploaded_files)} files successfully!")

# MAIN DASHBOARD
if not st.session_state['summary_df'].empty:
    df = st.session_state['summary_df']
    prof_dict = st.session_state['profile_dict']
    
    tabs = st.tabs(["📊 Dataset", "📉 Profile Viewer", "📈 Scientific Stats", "🔬 Trend Analysis", "💾 Columnar Export"])

    with tabs[0]:
        st.subheader("Combined Results Table")
        st.dataframe(df, use_container_width=True)

    with tabs[1]:
        if prof_dict:
            selected_f = st.selectbox("Select File to View Profile", list(prof_dict.keys()))
            subset = prof_dict[selected_f]['data']
            fig = px.line(subset, x='Length_mm', y='Amplitude_um', title=f"Profile: {selected_f}", template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

    with tabs[2]:
        st.subheader("Scientific Grouped Analysis")
        params = [p for p in ["Ra", "Rq", "Rz", "Rt"] if p in df.columns]
        if params:
            col1, col2 = st.columns(2)
            with col1:
                p_sel = st.selectbox("Analysis Parameter", params)
            with col2:
                g_sel = st.multiselect("Group By (Select 1 or more)", ["Material", "Condition", "Day"], default=["Condition", "Day"])
            
            if g_sel:
                res_stats = get_advanced_stats(df, g_sel, p_sel)
                st.write("### Summary Statistics Table")
                st.dataframe(res_stats.style.format(precision=4), use_container_width=True)
                
                # Significance Testing
                if len(df[g_sel[0]].unique()) > 1:
                    st.write("### Statistical Significance (ANOVA)")
                    groups = [df[df[g_sel[0]] == val][p_sel].dropna() for val in df[g_sel[0]].unique()]
                    f_val, p_val = stats.f_oneway(*groups)
                    st.metric("ANOVA p-value", f"{p_val:.4f}")
                    if p_val < 0.05:
                        st.success("Significant difference detected between groups!")
                    else:
                        st.info("No statistically significant difference found.")

    with tabs[3]:
        st.subheader("Degradation & Comparison Trends")
        if params:
            # SCENARIO 1: Box Plot for Distribution (Great for the 15 replicates)
            st.write("#### Distribution of Measurements (Scenario 1: Replicates)")
            fig_box = px.box(df, x="Condition", y=p_sel, color="Material", points="all", 
                             title=f"Variance in {p_sel} Measurements", notched=True)
            st.plotly_chart(fig_box, use_container_width=True)
            
            # SCENARIO 2: Trend Plot with Error Bars (Great for different days/conditions)
            st.write("#### Trend Analysis over Time (Scenario 2: Trends)")
            trend_df = get_advanced_stats(df, ["Condition", "Day"], p_sel)
            
            fig_trend = go.Figure()
            for condition in trend_df['Condition'].unique():
                c_data = trend_df[trend_df['Condition'] == condition].sort_values('Day')
                fig_trend.add_trace(go.Scatter(
                    x=c_data['Day'], y=c_data['mean'],
                    error_y=dict(type='data', array=c_data['CI_95'], visible=True),
                    name=condition, mode='lines+markers'
                ))
            
            fig_trend.update_layout(title=f"Mean {p_sel} Evolution (with 95% CI)",
                                    xaxis_title="Days", yaxis_title=f"Mean {p_sel} (µm)",
                                    template="plotly_white")
            st.plotly_chart(fig_trend, use_container_width=True)

    with tabs[4]:
        st.subheader("Bulk Export (Columnar Arrangement)")
        # ... [Previous Columnar Export Code remains the same] ...
        if prof_dict:
            naming_opt = st.radio("Label Columns by:", ["File Name", "Material_Day", "Condition_Day"], horizontal=True)
            wide_dfs = []
            for fname, content in prof_dict.items():
                temp_df = content['data'].copy()
                m = content['meta']
                header = fname if naming_opt == "File Name" else (f"{m['Material']}_Day{m['Day']}" if naming_opt == "Material_Day" else f"{m['Condition']}_Day{m['Day']}")
                
                # Rename columns and handle repeats
                base_header = header
                counter = 1
                while f"{header}_Length(mm)" in [col for d in wide_dfs for col in d.columns]:
                    header = f"{base_header}_Repeat{counter}"
                    counter += 1
                temp_df.columns = [f"{header}_Length(mm)", f"{header}_Amplitude(um)"]
                wide_dfs.append(temp_df)
            
            master_wide_df = pd.concat(wide_dfs, axis=1)
            csv_wide = master_wide_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Columnar Profile Data (CSV)", csv_wide, "columnar_profiles.csv", "text/csv")
else:
    st.info("👋 Upload your files in the sidebar and click 'Process' to begin scientific analysis.")
