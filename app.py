import streamlit as st
import pandas as pd
from data_loader import RoughnessLoader
import analysis as ana
import visualization as viz

st.set_page_config(page_title="BioMaterial Roughness Analyzer", layout="wide")

st.title("🧪 Surface Roughness Analysis Dashboard")
st.markdown("---")

# --- SIDEBAR: UPLOAD & METADATA ---
st.sidebar.header("1. Data Input")
uploaded_files = st.sidebar.file_uploader("Upload Excel Files", accept_multiple_files=True, type=['xlsx'])

all_data = []

if uploaded_files:
    st.sidebar.subheader("2. Metadata Assignment")
    mat_type = st.sidebar.text_input("Material Type", "PLA-Composite")
    ageing_cond = st.sidebar.selectbox("Ageing Condition", ["Oven", "Xenon UV", "Humidity Chamber", "Control"])
    
    # Bulk metadata setup
    st.info(f"Loaded {len(uploaded_files)} files. Assigning metadata...")
    
    loader = RoughnessLoader()
    
    # In a real scenario, we'd map filenames to specific ageing days
    # Here, we assume filename contains the day or allow manual mapping
    processed_list = []
    for f in uploaded_files:
        # Simple logic: try to find a number in filename for 'Day'
        import re
        day_match = re.search(r'\d+', f.name)
        day = int(day_match.group()) if day_match else 0
        
        processed_list.append({
            "File": f.name,
            "Material": mat_type,
            "Condition": ageing_cond,
            "Day": day
        })

    if st.sidebar.button("Process & Merge Data"):
        final_df = loader.process_files(uploaded_files, processed_list)
        st.session_state['master_df'] = final_df

# --- MAIN PANEL ---
if 'master_df' in st.session_state:
    df = st.session_state['master_df']
    
    tabs = st.tabs(["📊 Data Preview", "📈 Statistical Analysis", "📉 Visualizations", "💾 Export"])
    
    with tabs[0]:
        st.dataframe(df, use_container_width=True)
        
    with tabs[1]:
        st.subheader("Statistical Summary")
        param = st.selectbox("Select Parameter", ["Ra", "Rq", "Rz", "Rt"])
        grouping = st.selectbox("Group By", ["Day", "Condition", "Material"])
        
        summary = ana.get_stats_summary(df, grouping, param)
        st.table(summary)
        
        if len(df[grouping].unique()) > 1:
            f_stat, p_val = ana.perform_anova(df, param, grouping)
            st.metric("ANOVA p-value", f"{p_val:.4f}")
            if p_val < 0.05:
                st.success("Significant difference detected between groups!")
            else:
                st.warning("No significant difference detected.")

    with tabs[2]:
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(viz.plot_box_distribution(df, param, grouping, "Condition"), use_container_width=True)
        with col2:
            summary_trend = ana.get_stats_summary(df, "Day", param)
            st.plotly_chart(viz.plot_ageing_trend(summary_trend, "mean", "Day", None), use_container_width=True)

    with tabs[3]:
        st.subheader("Export Results")
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Cleaned CSV", data=csv, file_name="roughness_results.csv", mime='text/csv')
else:
    st.write("Please upload and process files in the sidebar to begin.")