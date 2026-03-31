import streamlit as st
import pandas as pd
import sys
import os
import re

# --- FIX FOR MODULENOTFOUNDERROR ---
# This ensures the cloud environment looks in the current folder for your files
sys.path.append(os.path.dirname(__file__))

try:
    from data_loader import RoughnessLoader
    import analysis as ana
    import visualization as viz
except ImportError as e:
    st.error(f"Error importing local modules: {e}")
    st.stop()

# --- CONFIG ---
st.set_page_config(page_title="BioMaterial Roughness Analyzer", layout="wide")

st.title("🧪 Surface Roughness Analysis Dashboard")
st.markdown("---")

# --- SIDEBAR: UPLOAD & METADATA ---
st.sidebar.header("1. Data Input")
uploaded_files = st.sidebar.file_uploader(
    "Upload Excel Files", 
    accept_multiple_files=True, 
    type=['xlsx'],
    key="file_uploader"
)

if uploaded_files:
    st.sidebar.subheader("2. Metadata Assignment")
    mat_type = st.sidebar.text_input("Material Type", "PLA-Composite")
    ageing_cond = st.sidebar.selectbox(
        "Ageing Condition", 
        ["Oven", "Xenon UV", "Humidity Chamber", "Control", "Other"]
    )
    
    if ageing_cond == "Other":
        ageing_cond = st.sidebar.text_input("Specify Condition", "Custom")

    # Metadata mapping
    processed_list = []
    for f in uploaded_files:
        # Extract numeric day from filename (e.g., 'sample_day14.xlsx' -> 14)
        day_match = re.search(r'\d+', f.name)
        day = int(day_match.group()) if day_match else 0
        
        processed_list.append({
            "File": f.name,
            "Material": mat_type,
            "Condition": ageing_cond,
            "Day": day
        })

    if st.sidebar.button("Process & Merge Data", type="primary"):
        loader = RoughnessLoader()
        with st.spinner("Processing files..."):
            final_df = loader.process_files(uploaded_files, processed_list)
            st.session_state['master_df'] = final_df
            st.success("Data processed successfully!")

# --- MAIN PANEL ---
if 'master_df' in st.session_state:
    df = st.session_state['master_df']
    
    tabs = st.tabs(["📊 Data Preview", "📈 Statistical Analysis", "📉 Visualizations", "💾 Export"])
    
    with tabs[0]:
        st.subheader("Unified Dataset")
        st.dataframe(df, use_container_width=True)
        
    with tabs[1]:
        st.subheader("Statistical Summary")
        col_p, col_g = st.columns(2)
        with col_p:
            param = st.selectbox("Select Parameter", ["Ra", "Rq", "Rz", "Rt"])
        with col_g:
            grouping = st.selectbox("Group By", ["Day", "Condition", "Material"])
        
        summary = ana.get_stats_summary(df, grouping, param)
        st.table(summary)
        
        # Significance Testing
        if len(df[grouping].unique()) > 1:
            st.markdown("### Significance Testing (ANOVA)")
            f_stat, p_val = ana.perform_anova(df, param, grouping)
            
            c1, c2 = st.columns(2)
            c1.metric("F-Statistic", f"{f_stat:.2f}")
            c2.metric("p-value", f"{p_val:.4f}")
            
            if p_val < 0.05:
                st.success(f"Result: There is a statistically significant difference in {param} across {grouping} groups.")
            else:
                st.warning(f"Result: No significant difference found in {param} across {grouping} groups (p > 0.05).")

    with tabs[2]:
        st.subheader("Visual Analysis")
        viz_col1, viz_col2 = st.columns(2)
        
        with viz_col1:
            st.plotly_chart(
                viz.plot_box_distribution(df, param, grouping, "Condition"), 
                use_container_width=True
            )
            
        with viz_col2:
            summary_trend = ana.get_stats_summary(df, "Day", param)
            st.plotly_chart(
                viz.plot_ageing_trend(summary_trend, "mean", "Day", None), 
                use_container_width=True
            )

    with tabs[3]:
        st.subheader("Download Processed Data")
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Results as CSV",
            data=csv,
            file_name=f"roughness_analysis_{mat_type}.csv",
            mime='text/csv',
        )
else:
    st.info("👋 Welcome! Please upload your Excel files in the sidebar and click 'Process' to begin.")
