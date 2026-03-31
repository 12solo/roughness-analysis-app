import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import re
import io

# ==========================================
# 1. INITIALIZE SESSION STATE (Prevents KeyErrors)
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

                # PASS 1: EXTRACT SUMMARY PARAMETERS
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
# 3. UI LAYOUT
# ==========================================
st.set_page_config(page_title="Scientific Roughness Lab", layout="wide")
st.title("🔬 Advanced Surface Roughness Dashboard")

# SIDEBAR
st.sidebar.header("1. Data Input")
uploaded_files = st.sidebar.file_uploader("Upload .xlsx Files", accept_multiple_files=True, type=['xlsx'])

if uploaded_files:
    st.sidebar.subheader("2. Research Scenario")
    # This choice determines the statistical output
    scenario = st.sidebar.radio(
        "Select Scenario:",
        ["Replicate Tests (Same Sample/Batch)", "Comparative Study (Different Samples/Dates)"]
    )
    
    m_id = st.sidebar.text_input("Material / Specimen Name", "Specimen_01")
    
    if scenario == "Comparative Study (Different Samples/Dates)":
        cond = st.sidebar.selectbox("Condition / Formulation", ["Control", "Oven Ageing", "UV Exposure", "Humidity"])
        day = st.sidebar.number_input("Ageing Day", min_value=0, step=1)
    else:
        cond = "Single Batch"
        day = 0

    if st.sidebar.button("Run Scientific Analysis", type="primary"):
        loader = RoughnessLoader()
        metas = []
        for f in uploaded_files:
            # Auto-detect day from filename if comparative
            day_val = day
            if scenario == "Comparative Study (Different Samples/Dates)":
                day_match = re.search(r'\d+', f.name)
                if day_match and day == 0: day_val = int(day_match.group())

            metas.append({
                "File": f.name, "Material": m_id, 
                "Condition": cond, "Day": day_val,
                "Study_Type": scenario # Saved as 'Study_Type' to avoid KeyErrors
            })
        
        sum_df, prof_dict = loader.process_files(uploaded_files, metas)
        st.session_state['summary_df'] = sum_df
        st.session_state['profile_dict'] = prof_dict

# MAIN DASHBOARD
if not st.session_state['summary_df'].empty:
    df = st.session_state['summary_df']
    prof_dict = st.session_state['profile_dict']
    study_mode = df['Study_Type'].iloc[0]
    
    tabs = st.tabs(["📋 Dataset", "📈 Scientific Stats", "📉 Profile Analysis", "💾 Columnar Export"])

    with tabs[0]:
        st.subheader("Extracted Summary Table")
        st.dataframe(df, use_container_width=True)

    with tabs[1]:
        params = [p for p in ["Ra", "Rq", "Rz", "Rt"] if p in df.columns]
        if params:
            p_sel = st.selectbox("Select Parameter", params)
            
            if "Replicate" in study_mode:
                st.subheader(f"Statistical Reliability for {m_id}")
                
                # Scientific Replicate Analysis
                mean_val = df[p_sel].mean()
                std_val = df[p_sel].std()
                cv_val = (std_val / mean_val * 100) if mean_val != 0 else 0
                
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Mean (µm)", f"{mean_val:.4f}")
                c2.metric("Std Dev (σ)", f"{std_val:.4f}")
                c3.metric("Tests (n)", len(df))
                c4.metric("CV (%)", f"{cv_val:.2f}%")
                
                st.write("### Histogram of Measurements")
                fig_hist = px.histogram(df, x=p_sel, marginal="box", title=f"Replicate Spread: {p_sel}")
                st.plotly_chart(fig_hist, use_container_width=True)
                
                st.info("💡 **Scientific Note:** A Coefficient of Variation (CV%) below 10% is the target for high-precision surface analysis.")

            else:
                st.subheader("Comparative Trend & Significance")
                
                # Trend calculation with Error Bars
                grp = df.groupby(["Condition", "Day"])[p_sel].agg(['mean', 'std', 'count']).reset_index()
                grp['CI_95'] = 1.96 * (grp['std'] / np.sqrt(grp['count']))
                
                st.write("### Mean Values per Condition/Day")
                st.dataframe(grp.style.format(precision=4), use_container_width=True)
                
                st.write("### Degradation Trend Plot")
                fig_trend = go.Figure()
                for c in grp['Condition'].unique():
                    subset = grp[grp['Condition'] == c].sort_values('Day')
                    fig_trend.add_trace(go.Scatter(
                        x=subset['Day'], y=subset['mean'],
                        error_y=dict(type='data', array=subset['CI_95'], visible=True),
                        name=c, mode='lines+markers'
                    ))
                fig_trend.update_layout(xaxis_title="Ageing Days", yaxis_title=f"Mean {p_sel} (µm)", template="plotly_white")
                st.plotly_chart(fig_trend, use_container_width=True)
                

    with tabs[2]:
        st.subheader("Surface Roughness Profiles (Col E & F)")
        if prof_dict:
            f_sel = st.selectbox("Select Test File", list(prof_dict.keys()))
            subset_p = prof_dict[f_sel]['data']
            fig_p = px.line(subset_p, x='Length_mm', y='Amplitude_um', title=f"Surface Topography: {f_sel}")
            st.plotly_chart(fig_p, use_container_width=True)

    with tabs[3]:
        st.subheader("Bulk Export (Columnar Wide Format)")
        naming = st.radio("Label Columns By:", ["File Name", "Material_Condition", "Study_Type"], horizontal=True)
        
        wide_list = []
        for fname, content in prof_dict.items():
            temp_df = content['data'].copy()
            m = content['meta']
            
            header = fname if naming == "File Name" else (f"{m['Material']}_{m['Condition']}" if naming == "Material_Condition" else m['Study_Type'])

            # Handle duplicates in headers
            base_h = header
            counter = 1
            while f"{header}_Length(mm)" in [col for d in wide_list for col in d.columns]:
                header = f"{base_h}_Rep{counter}"
                counter += 1
            
            temp_df.columns = [f"{header}_Length(mm)", f"{header}_Amplitude(um)"]
            wide_list.append(temp_df)
        
        if wide_list:
            master_wide = pd.concat(wide_list, axis=1)
            st.dataframe(master_wide.head(10))
            csv = master_wide.to_csv(index=False).encode('utf-8')
            st.download_button("Download Unified CSV", csv, "roughness_profiles_wide.csv", "text/csv")
else:
    st.info("👋 Select your scenario and upload files to begin.")
