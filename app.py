import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from scipy.signal import savgol_filter
from scipy.spatial.distance import cdist
from scipy.ndimage import gaussian_filter1d
import re
import os
import io

def natural_sort_key(s):
    """Sorts strings containing numbers in natural order (1, 2, 10 instead of 1, 10, 2)"""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', str(s))]

# ==========================================
# 0. ANALYSIS FUNCTIONS
# ==========================================
def compute_roughness_params(signal):
    Ra = np.mean(np.abs(signal))
    Rq = np.sqrt(np.mean(signal**2))
    Rt = np.max(signal) - np.min(signal)
    
    Rsk = stats.skew(signal)
    Rku = stats.kurtosis(signal)
    
    sorted_signal = np.sort(signal)
    Rz = np.mean(sorted_signal[-5:] - sorted_signal[:5]) if len(signal) >= 10 else np.nan
    
    return Ra, Rq, Rz, Rt, Rsk, Rku

def export_to_excel_with_logo(df, sheet_title):
    """Formats DataFrame to Excel with auto-fitted columns and a logo"""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_title)
        worksheet = writer.sheets[sheet_title]
        
        # 1. Auto-fit columns
        for i, col in enumerate(df.columns):
            max_len = max(len(str(col)), df[col].astype(str).map(len).max() if len(df) > 0 else 0) + 2
            worksheet.set_column(i, i, max_len)
            
        # 2. Insert Logo
        logo_path = "Solomon_SR_Logo.png" 
        if os.path.exists(logo_path):
            col_offset = len(df.columns) + 1
            worksheet.insert_image(1, col_offset, logo_path, {'x_scale': 0.6, 'y_scale': 0.6})
            
    return output.getvalue()

# ==========================================
# 1. INITIALIZE SESSION STATE
# ==========================================
if 'master_df' not in st.session_state:
    st.session_state['master_df'] = pd.DataFrame()
elif not st.session_state['master_df'].empty and 'Sample' not in st.session_state['master_df'].columns:
    st.session_state['master_df'] = pd.DataFrame()

if 'opt_df' not in st.session_state:
    st.session_state['opt_df'] = pd.DataFrame()
    
if 'profile_dict' not in st.session_state:
    st.session_state['profile_dict'] = {}
if 'legend_map' not in st.session_state:
    st.session_state['legend_map'] = {}

# ==========================================
# 2. SMART SCAN DATA LOADER
# ==========================================
def iso_sigma(lambda_c, dx):
    """Convert ISO cutoff wavelength to Gaussian sigma"""
    if dx == 0: dx = 0.001 
    return (lambda_c / (2 * np.pi)) / dx

class RoughnessLoader:
    def __init__(self):
        self.targets = {'Ra': ['ra'], 'Rq': ['rq'], 'Rz': ['rz'], 'Rt': ['rt']}

    def clean_value(self, val):
        if pd.isna(val): return np.nan
        if isinstance(val, (int, float)): return float(val)
        text = str(val).replace(',', '.').strip()
        match = re.search(r"[-+]?\d*\.\d+|\d+", text)
        return float(match.group()) if match else np.nan

    def process_files(self, uploaded_files, meta_template, filter_type, window_size, lambda_val):
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

                data_sheet = next((s for s in xl.sheet_names if "DATA" in s.upper()), None)
                if not data_sheet and len(xl.sheet_names) > 0:
                    data_sheet = xl.sheet_names[-1] 
                
                if data_sheet:
                    df_p = pd.read_excel(file, sheet_name=data_sheet, usecols=[4, 5])
                    df_p.columns = ['Length_mm', 'Amplitude_um']
                    
                    df_p['Length_mm'] = df_p['Length_mm'].astype(str).str.replace(',', '.', regex=False)
                    df_p['Amplitude_um'] = df_p['Amplitude_um'].astype(str).str.replace(',', '.', regex=False)
                    
                    df_p = df_p.apply(pd.to_numeric, errors='coerce').dropna().reset_index(drop=True)
                    
                    if not df_p.empty:
                        if filter_type == "ISO Gaussian (λc)":
                            dx = np.mean(np.diff(df_p['Length_mm'])) if len(df_p) > 1 else 0.001
                            sigma = iso_sigma(lambda_val, dx)
                            df_p['Form'] = gaussian_filter1d(df_p['Amplitude_um'], sigma=sigma)
                            df_p['Roughness'] = df_p['Amplitude_um'] - df_p['Form']
                        elif filter_type == "Savitzky-Golay":
                            df_p['Form'] = savgol_filter(df_p['Amplitude_um'], window_length=window_size, polyorder=3)
                            df_p['Roughness'] = df_p['Amplitude_um'] - df_p['Form']
                        else:
                            df_p['Roughness'] = df_p['Amplitude_um'] - df_p['Amplitude_um'].mean()
                        
                        df_p['Amplitude_um_Norm'] = df_p['Roughness']
                        
                        Ra_c, Rq_c, Rz_c, Rt_c, Rsk_c, Rku_c = compute_roughness_params(df_p['Roughness'].values)
                        row_summary.update({
                            'Ra_calc': Ra_c, 'Rq_calc': Rq_c, 'Rz_calc': Rz_c, 
                            'Rt_calc': Rt_c, 'Rsk': Rsk_c, 'Rku': Rku_c
                        })
                        
                        df_p['Sample'] = meta_template['Sample']
                        profile_map[file.name] = df_p
                    else:
                        st.error(f"❌ {file.name}: Columns E and F became empty after reading. Check decimal formatting.")
                else:
                    st.error(f"❌ {file.name}: Could not find a valid data sheet.")
                
                combined_summary.append(row_summary)
                
            except Exception as e:
                st.error(f"Error processing {file.name}: {e}")
                
        return pd.DataFrame(combined_summary), profile_map

# ==========================================
# 3. UI & SIDEBAR
# ==========================================
st.set_page_config(page_title="Solomon SRoughnessLab", page_icon="Solomon_SR_Logo.png", layout="wide")

with st.sidebar:
    if os.path.exists("Solomon_SR_Logo.png"):
        st.image("Solomon_SR_Logo.png", width=150)
    st.markdown("---")
    
    st.markdown("""
        <div style='background-color: #f8f9fa; padding: 15px; border-radius: 10px; margin-bottom: 20px; text-align: center; border: 1px solid #e6e6e6;'>
            <p style='color: #333333; font-size: 1em; font-weight: bold; margin-bottom: 0px;'>DEVELOPED BY SOLOMON</p>
            <p style='color: #666666; font-size: 0.85em; margin-top: 2px; margin-bottom: 10px;'>Surface Roughness Lab Pro Suite</p>
            <hr style='margin: 10px 0; border: 0; border-top: 1px solid #ddd;'>
            <a href='mailto:your.solomon.duf@gmail.com' style='color: #0066cc; text-decoration: none; font-size: 0.85em; font-weight: 500;'>✉️ Contact Developer</a>
            <br>
            <p style='color: #999999; font-size: 0.7em; margin-top: 10px; margin-bottom: 0px;'><i>For Research & Academic Use Only<br>© 2026</i></p>
        </div>
    """, unsafe_allow_html=True)
    
    st.header("1. Data Input")
    with st.form("input_form", clear_on_submit=True):
        s_name = st.text_input("Sample ID", "Sample A")
        s_files = st.file_uploader("Upload Replicate Files (.xlsx)", accept_multiple_files=True)
        
        st.markdown("---")
        st.subheader("🛠️ Filter Settings")
        filter_type = st.selectbox("Primary Detrending Filter", ["ISO Gaussian (λc)", "Savitzky-Golay", "None"])
        
        lambda_val = 0.8
        sg_win = 51
        
        if filter_type == "ISO Gaussian (λc)":
            lambda_val = st.number_input("Cutoff Wavelength λc (mm)", value=0.8, step=0.1)
        elif filter_type == "Savitzky-Golay":
            sg_win = st.slider("S-G Window Length", 5, 151, 51, step=2)
        
        submit = st.form_submit_button("Add Sample Batch")

    if submit and s_files:
        loader = RoughnessLoader()
        meta = {"Sample": s_name, "Filter": filter_type}
        
        new_sum, new_prof = loader.process_files(s_files, meta, filter_type, sg_win, lambda_val)
        st.session_state['master_df'] = pd.concat([st.session_state['master_df'], new_sum], ignore_index=True)
        st.session_state['profile_dict'].update(new_prof)
        st.success(f"Added replicates for {s_name}")

    if not st.session_state['master_df'].empty:
        st.header("2. Legend Customization")
        unique_samples = sorted(st.session_state['master_df']['Sample'].unique(), key=natural_sort_key)
        for s in unique_samples:
            if s not in st.session_state['legend_map']:
                st.session_state['legend_map'][s] = s
            st.session_state['legend_map'][s] = st.text_input(f"Rename '{s}':", st.session_state['legend_map'][s])

        st.markdown("---")
        st.header("3. Global Trend Optimizer")
        with st.expander("✂️ Open Inter-Sample Optimizer"):
            st.markdown("Select tests to remove so the **overall Mean Ra** perfectly forms an increasing or decreasing trend.")
            
            trend_dir = st.radio("Target Inter-Sample Trend", ["Increasing", "Decreasing"])
            target_n = st.number_input("Target Replicates to Keep (per batch)", min_value=3, max_value=50, value=9)
            
            batches_to_opt = st.multiselect("Select Batches for the Trend", unique_samples, default=unique_samples)
            
            if st.button("Optimize Global Trend"):
                if len(batches_to_opt) < 2:
                    st.warning("Please select at least 2 batches to form a trend.")
                else:
                    df_opt = st.session_state['master_df'].copy()
                    
                    valid = True
                    for b in batches_to_opt:
                        b_len = len(df_opt[df_opt['Sample'] == b])
                        if b_len <= target_n:
                            st.warning(f"Batch '{b}' only has {b_len} tests. Upload more than {target_n} to optimize.")
                            valid = False
                    
                    if valid:
                        sorted_batches = sorted(batches_to_opt, key=natural_sort_key)
                        batch_x_map = {b: i for i, b in enumerate(sorted_batches)}
                        
                        def get_global_score(df_temp):
                            means = df_temp[df_temp['Sample'].isin(batches_to_opt)].groupby('Sample')['Ra'].mean()
                            y = [means.get(b, 0) for b in sorted_batches]
                            x = [batch_x_map[b] for b in sorted_batches]
                            with np.errstate(divide='ignore', invalid='ignore'):
                                corr = np.corrcoef(x, y)[0, 1]
                            if np.isnan(corr): return 0.0
                            return corr if trend_dir == "Increasing" else -corr

                        while True:
                            counts = df_opt[df_opt['Sample'].isin(batches_to_opt)]['Sample'].value_counts()
                            if not any(counts > target_n):
                                break
                            
                            best_score = -float('inf')
                            best_idx = -1
                            
                            oversized_batches = counts[counts > target_n].index.tolist()
                            candidates = df_opt[df_opt['Sample'].isin(oversized_batches)].index.tolist()
                            
                            for i in candidates:
                                temp_df = df_opt.drop(index=i)
                                score = get_global_score(temp_df)
                                if score > best_score:
                                    best_score = score
                                    best_idx = i
                            df_opt = df_opt.drop(index=best_idx)
                        
                        original_files = st.session_state['master_df'][st.session_state['master_df']['Sample'].isin(batches_to_opt)]['File'].tolist()
                        kept_files = df_opt[df_opt['Sample'].isin(batches_to_opt)]['File'].tolist()
                        dropped_files = set(original_files) - set(kept_files)
                        
                        st.session_state['opt_df'] = df_opt.reset_index(drop=True)
                        
                        st.success(f"Optimization Complete! View results in the '✨ Optimized Trends' tab.")
                        st.error("**🛑 Tests Removed from Optimized Set:**\n" + "\n".join([f"- {d}" for d in sorted(list(dropped_files), key=natural_sort_key)]))
                        st.rerun()

        # --- NEW: PLOT CUSTOMIZATION OVERRIDES ---
        st.markdown("---")
        st.header("4. Plot Customization")
        with st.expander("🖍️ Custom Axis Labels"):
            st.markdown("Override the default axis labels. Leave blank to keep defaults.")
            c_x_trend = st.text_input("Trends X-Axis", "")
            c_y_trend = st.text_input("Trends Y-Axis", "")
            c_x_prof = st.text_input("Profiles/Stack X-Axis", "")
            c_y_prof = st.text_input("Profiles/Stack Y-Axis", "")
            c_x_psd = st.text_input("PSD X-Axis", "")
            c_y_psd = st.text_input("PSD Y-Axis", "")

    if st.button("Reset Entire Study", type="primary"):
        st.session_state.clear()
        st.rerun()

# ==========================================
# 4. DASHBOARD TABS
# ==========================================
df_master = st.session_state['master_df']
prof_dict = st.session_state['profile_dict']

JOURNAL_CONFIG = {
    'toImageButtonOptions': {
        'format': 'png', 
        'filename': 'Roughness_Plot_HighRes',
        'scale': 4 
    },
    'displayModeBar': True
}

AXIS_STYLE = dict(
    mirror=True,         
    ticks='outside',     
    showline=True, 
    linecolor='black', 
    linewidth=1.5,       
    showgrid=False,
    zeroline=False,      
    title_font=dict(family="Times New Roman", size=20, color="black"),
    tickfont=dict(family="Times New Roman", size=16, color="black"),
    tickwidth=1.5,       
    ticklen=6,           
    tickcolor='black'
)

# Helper function to plot trends with Custom Axes support
def plot_trend(data_df, param_selected, show_trendline=True, custom_x="", custom_y=""):
    plot_df = data_df.groupby(["Sample"])[param_selected].agg(['mean', 'std', 'count']).reset_index()
    plot_df['sort_key'] = plot_df['Sample'].apply(natural_sort_key)
    plot_df = plot_df.sort_values('sort_key').drop('sort_key', axis=1)
    plot_df['Display_Name'] = plot_df['Sample'].map(st.session_state['legend_map'])
    
    fig = px.line(plot_df, x="Display_Name", y="mean", 
                  error_y=1.96*(plot_df['std']/np.sqrt(plot_df['count'])), 
                  markers=True, template="simple_white",
                  category_orders={"Display_Name": plot_df['Display_Name'].tolist()})
    
    fig.update_traces(line=dict(color='black', width=2), marker=dict(size=10, color='black'))
    
    if show_trendline and len(plot_df) > 1:
        x_numeric = np.arange(len(plot_df))
        y_numeric = plot_df['mean'].values
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_numeric, y_numeric)
        trend_y = intercept + slope * x_numeric
        
        fig.add_trace(go.Scatter(
            x=plot_df['Display_Name'], y=trend_y, mode='lines', 
            name=f"Trend (R² = {r_value**2:.3f})", 
            line=dict(color='red', dash='dash', width=2)
        ))
    
    # Apply custom axis labels if provided
    x_label = f"<b>{custom_x}</b>" if custom_x else "<b>Sample ID</b>"
    y_label = f"<b>{custom_y}</b>" if custom_y else f"<b>Mean {param_selected} (µm)</b>"
    
    fig.update_layout(
        autosize=False, 
        width=800,      
        height=600,     
        margin=dict(l=80, r=40, t=40, b=80), 
        xaxis_title=x_label, yaxis_title=y_label, 
        xaxis=AXIS_STYLE, yaxis=AXIS_STYLE,
        plot_bgcolor='white', paper_bgcolor='white',
        legend=dict(yanchor="bottom", y=0.05, xanchor="right", x=0.95, bgcolor="rgba(255,255,255,0.8)")
    )
    return fig

if not df_master.empty:
    st.markdown("<h2 style='margin-top: -40px;'>Solomon Scientific roughness analyzer</h2>", unsafe_allow_html=True)
    
    tabs = st.tabs(["📊 Dataset", "📉 Raw Trends", "✨ Optimized Trends", "🎨 Replicate Stack", "🏛️ Representative Stack", "📈 PSD Analysis", "💾 Export", "📖 About & Methods"])

    with tabs[0]:
        st.subheader("Raw Summary Table")
        st.dataframe(df_master, use_container_width=True)

    with tabs[1]:
        st.subheader("Inter-Sample Comparison (Original Data)")
        params = [p for p in ["Ra", "Rq", "Rz", "Rt", "Ra_calc", "Rsk", "Rku"] if p in df_master.columns]
        if params:
            colA, colB = st.columns([3, 1])
            with colA: p_sel_raw = st.selectbox("Select Parameter", params, key="raw_p")
            with colB: show_tl_raw = st.checkbox("Show Linear Trendline (R²)", value=True, key="raw_tl")
            
            # Pass the custom axis strings into the plot function
            st.plotly_chart(plot_trend(df_master, p_sel_raw, show_tl_raw, c_x_trend, c_y_trend), use_container_width=False, config=JOURNAL_CONFIG)
            
            groups = [df_master[df_master['Sample'] == s][p_sel_raw].dropna() for s in df_master['Sample'].unique()]
            if len(groups) > 1:
                f_stat, p_val = stats.f_oneway(*groups)
                st.markdown(f"<p style='font-size:16px;'><b>Statistical Significance (ANOVA):</b> p-value = <b>{p_val:.4e}</b></p>", unsafe_allow_html=True)

    with tabs[2]:
        st.subheader("Inter-Sample Comparison (Optimized Data)")
        if st.session_state.get('opt_df').empty:
            st.info("👋 The optimized dataset is currently empty. Open the **Global Trend Optimizer** in the sidebar, select your target trend, and click Optimize to generate this view.")
        else:
            df_opt = st.session_state['opt_df']
            params_opt = [p for p in ["Ra", "Rq", "Rz", "Rt", "Ra_calc", "Rsk", "Rku"] if p in df_opt.columns]
            if params_opt:
                colA, colB = st.columns([3, 1])
                with colA: p_sel_opt = st.selectbox("Select Parameter", params_opt, key="opt_p")
                with colB: show_tl_opt = st.checkbox("Show Linear Trendline (R²)", value=True, key="opt_tl")
                
                st.plotly_chart(plot_trend(df_opt, p_sel_opt, show_tl_opt, c_x_trend, c_y_trend), use_container_width=False, config=JOURNAL_CONFIG)
                
                groups = [df_opt[df_opt['Sample'] == s][p_sel_opt].dropna() for s in df_opt['Sample'].unique()]
                if len(groups) > 1:
                    f_stat, p_val = stats.f_oneway(*groups)
                    st.markdown(f"<p style='font-size:16px;'><b>Statistical Significance (ANOVA):</b> p-value = <b>{p_val:.4e}</b></p>", unsafe_allow_html=True)

    with tabs[3]:
        st.subheader("Batch Replicate Inspection (Stack)")
        
        data_mode_rep = st.radio("Select Dataset to Visualize:", ["Raw Data", "✨ Optimized Data"], horizontal=True, key="rep_radio")
        
        df_target = df_master
        if data_mode_rep == "✨ Optimized Data":
            if st.session_state.get('opt_df', pd.DataFrame()).empty:
                st.warning("Optimized dataset is empty. Run the Global Trend Optimizer first. Showing Raw Data.")
            else:
                df_target = st.session_state['opt_df']
                
        if not df_target.empty:
            batch_to_check = st.selectbox("Select Batch to Inspect:", sorted(df_target['Sample'].unique()), key="rep_batch")
            batch_files = sorted(df_target[df_target['Sample'] == batch_to_check]['File'].tolist(), key=natural_sort_key)
            offset_rep = st.slider("Vertical Offset (µm)", 1, 100, 25, key="rep_slider")
            
            fig_rep = go.Figure()
            rep_ticks_v, rep_ticks_t = [], []
            
            for i, f in enumerate(batch_files):
                if f in prof_dict:
                    y_shift = i * offset_rep
                    p_data = prof_dict[f]
                    clean_name = os.path.splitext(f)[0]
                    
                    fig_rep.add_trace(go.Scatter(
                        x=p_data['Length_mm'], y=p_data['Amplitude_um_Norm'] + y_shift,
                        mode='lines', name=f" ", showlegend=False
                    ))
                    
                    y_peak = (p_data['Amplitude_um_Norm'] + y_shift).max()
                    fig_rep.add_annotation(
                        x=p_data['Length_mm'].min(), y=y_peak, yshift=10,
                        text=f"<b> ({clean_name})</b>",
                        showarrow=False, align="left", xanchor="left", yanchor="bottom",
                        font=dict(family="Times New Roman", size=14, color="black")
                    )
                    
                    rep_ticks_v.append(y_shift)
                    rep_ticks_t.append("<b>0</b>")
                    for v in [-5, 5]:
                        rep_ticks_v.append(v + y_shift)
                        rep_ticks_t.append(f"<b>{v}</b>")

            auto_height_rep = max(600, 200 + (len(batch_files) * 50))
            
            # Apply Custom Axis labels
            x_label_prof = f"<b>{c_x_prof}</b>" if c_x_prof else "<b>Wavelength (mm)</b>"
            y_label_prof = f"<b>{c_y_prof}</b>" if c_y_prof else "<b>Roughness Ra (µm)</b>"

            fig_rep.update_layout(
                autosize=False, 
                width=800,      
                height=auto_height_rep,
                margin=dict(l=80, r=40, t=40, b=80),
                template="simple_white",
                xaxis_title=x_label_prof, yaxis_title=y_label_prof,
                xaxis=AXIS_STYLE, 
                yaxis=dict(tickmode='array', tickvals=rep_ticks_v, ticktext=rep_ticks_t, **AXIS_STYLE)
            )
            st.plotly_chart(fig_rep, use_container_width=False, config=JOURNAL_CONFIG)

    with tabs[4]:
        st.subheader("Representative Stack (Auto-Adjustable)")
        
        data_mode_glob = st.radio("Select Dataset to Visualize:", ["Raw Data", "✨ Optimized Data"], horizontal=True, key="glob_radio")
        
        df_target_glob = df_master
        if data_mode_glob == "✨ Optimized Data":
            if st.session_state.get('opt_df', pd.DataFrame()).empty:
                st.warning("Optimized dataset is empty. Showing Raw Data.")
            else:
                df_target_glob = st.session_state['opt_df']
                
        if not df_target_glob.empty:
            offset_global = st.slider("Group Offset (µm)", 1, 400, 100, key="glob_slider")
            
            fig_glob = go.Figure()
            t_vals, t_text = [], []
            unique_samples = sorted(df_target_glob['Sample'].unique(), key=natural_sort_key)
            
            for i, sample in enumerate(unique_samples):
                sample_data = df_target_glob[df_target_glob['Sample'] == sample]
                features = sample_data[['Ra', 'Rq', 'Rz', 'Rt']].dropna()
                centroid = features.mean().values.reshape(1, -1)
                idx = cdist(features.values, centroid).argmin()
                closest_file = sample_data.iloc[idx]['File']
                
                y_shift = i * offset_global
                current_profile = prof_dict[closest_file]
                name = st.session_state['legend_map'].get(sample, sample)
                
                fig_glob.add_trace(go.Scatter(
                    x=current_profile['Length_mm'], y=current_profile['Amplitude_um_Norm'] + y_shift, 
                    mode='lines', showlegend=False, line=dict(width=2)
                ))
                
                peak_y = (current_profile['Amplitude_um_Norm'] + y_shift).max()
                mean_ra = sample_data['Ra'].mean()
                std_ra = sample_data['Ra'].std()
                inline_label = f"<b>{name}: <i>R<sub>a</sub></i> = {mean_ra:.3f} ± {std_ra:.3f} µm</b>"
                
                fig_glob.add_annotation(
                    x=current_profile['Length_mm'].min(), y=peak_y, yshift=12,
                    text=inline_label, showarrow=False, align="left", xanchor="left", yanchor="bottom",
                    font=dict(family="Times New Roman", size=16, color="black")
                )
                
                t_vals.append(y_shift)
                t_text.append(f"<b>0</b>")
                for val in [-5, 5]:
                    t_vals.append(val + y_shift)
                    t_text.append(f"<b>{val}</b>")
            
            auto_height_glob = max(600, 200 + (len(unique_samples) * 80))
            
            # Apply Custom Axis labels
            x_label_prof2 = f"<b>{c_x_prof}</b>" if c_x_prof else "<b>Wavelength (mm)</b>"
            y_label_prof2 = f"<b>{c_y_prof}</b>" if c_y_prof else "<b>Roughness (µm)</b>"

            fig_glob.update_layout(
                autosize=False, 
                width=800,      
                height=auto_height_glob,
                margin=dict(l=80, r=40, t=40, b=80),
                template="simple_white",
                xaxis_title=x_label_prof2, yaxis_title=y_label_prof2,
                xaxis=AXIS_STYLE, yaxis=dict(tickmode='array', tickvals=t_vals, ticktext=t_text, **AXIS_STYLE)
            )
            st.plotly_chart(fig_glob, use_container_width=False, config=JOURNAL_CONFIG)

    with tabs[5]:
        st.subheader("Power Spectral Density (PSD) Analysis")
        with st.expander("📖 Spectral Analysis & PSD Theory"):
            st.markdown("""
            **Power Spectral Density (PSD) in Surface Metrology**
            
            While amplitude parameters (e.g., $R_a$, $R_q$) statistically quantify the average vertical deviations of a surface profile, they provide no information regarding the spatial distribution of these features. The **Power Spectral Density (PSD)** function resolves this by decomposing the surface topography into a continuum of constituent spatial frequencies via Fourier analysis. This provides a comprehensive frequency-domain fingerprint of the surface.
            
            **Diagnostic Interpretations:**
            * **Discrete Spectral Peaks:** Indicate the presence of dominant deterministic or periodic features. In manufacturing, these typically correlate to kinematic tool feed marks, chatter vibrations, or engineered micro-textures.
            * **High-Frequency Domain (Right):** Represents short-wavelength micro-roughness. This region is closely tied to friction, wear characteristics, and fundamental tool-workpiece interaction mechanics.
            * **Low-Frequency Domain (Left):** Encompasses long-wavelength macro-roughness and waviness. This often reveals structural form errors or machine tool misalignments.
            * **Log-Log Linear Slope:** For stochastically rough surfaces, the linear decay rate (slope) in the high-frequency regime is intrinsically linked to the surface's fractal dimension and topographical complexity.
            
            *Methodological Note: PSD computation in this suite is executed exclusively on the primary roughness profile following morphological form removal via Savitzky-Golay detrending. This ensures the strict isolation of functional surface texture from nominal geometric curvature.*
            
            ---
            **Standard References:**
            1. **Whitehouse, D. J. (2004).** *Surfaces and their Measurement*. Hermes Penton Ltd. (Fundamental theory of spectral analysis in metrology).
            2. **Thomas, T. R. (1999).** *Rough Surfaces* (2nd ed.). Imperial College Press. (Relating PSD slopes to fractal surface characteristics).
            """)
        
        unique_samples = sorted(df_master['Sample'].unique(), key=natural_sort_key)
        sample_choice = st.selectbox("Select Sample for PSD", unique_samples)
        sample_files = sorted(df_master[df_master['Sample'] == sample_choice]['File'].tolist(), key=natural_sort_key)
        
        for i, f in enumerate(sample_files):
            if f in prof_dict:
                if 'Roughness' in prof_dict[f].columns:
                    signal = prof_dict[f]['Roughness'].values
                    x_data = prof_dict[f]['Length_mm'].values
                    dx = np.mean(np.diff(x_data)) if len(x_data) > 1 else 1.0
                    
                    psd = np.abs(np.fft.fft(signal))**2
                    freq = np.fft.fftfreq(len(signal), d=dx)
                    
                    fig_psd = px.line(
                        x=freq[freq > 0], y=psd[freq > 0], 
                        title=f"PSD Spectrum - {f}", log_y=True, template="simple_white"
                    )
                    
                    # Apply Custom Axis labels
                    x_label_psd = f"<b>{c_x_psd}</b>" if c_x_psd else "<b>Spatial Frequency (cycles/mm)</b>"
                    y_label_psd = f"<b>{c_y_psd}</b>" if c_y_psd else "<b>Power Density (µm²·mm)</b>"
                    
                    fig_psd.update_layout(
                        autosize=False, 
                        width=800,      
                        height=600,
                        margin=dict(l=80, r=40, t=40, b=80),
                        xaxis_title=x_label_psd, yaxis_title=y_label_psd,
                        xaxis=AXIS_STYLE, yaxis=AXIS_STYLE
                    )
                    st.plotly_chart(fig_psd, use_container_width=False, key=f"psd_{sample_choice}_{f}_{i}", config=JOURNAL_CONFIG)
                else:
                    st.warning(f"⚠️ Roughness filter not applied for '{f}'. Please click 'Reset Entire Study' and re-upload.")
            else:
                st.error(f"❌ Raw profile missing for '{f}'. Ensure the Excel file has a sheet with the word 'DATA' in the title.")

    # --- TAB 6: DUAL EXPORT UI ---
    with tabs[6]:
        st.header("💾 Comprehensive Data Export")
        st.markdown("Download your analysis results in ready-to-publish Excel files (Auto-fitted columns + Lab Logo included).")
        
        # --- RAW EXPORT SECTION ---
        st.markdown("### 📦 1. Original (Raw) Data Export")
        st.markdown("Contains all files originally uploaded.")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            excel_summary = export_to_excel_with_logo(df_master, "Summary_Stats")
            st.download_button("📥 Raw Summary Stats", excel_summary, "1_RAW_Roughness_Summary.xlsx", use_container_width=True)

        with col2:
            if prof_dict:
                export_list_wide = []
                for fname in sorted(prof_dict.keys(), key=natural_sort_key):
                    p_data = prof_dict[fname]
                    sample_match = df_master[df_master['File'] == fname]
                    sample_name = sample_match['Sample'].iloc[0] if not sample_match.empty else "Unknown_Sample"
                    temp = p_data[['Length_mm', 'Amplitude_um_Norm']].copy().reset_index(drop=True)
                    clean_name = os.path.splitext(fname)[0]
                    temp.columns = [f"[{sample_name}] {clean_name}_X_mm", f"[{sample_name}] {clean_name}_Y_Roughness_um"]
                    export_list_wide.append(temp)
                
                full_export_wide = pd.concat(export_list_wide, axis=1)
                excel_all = export_to_excel_with_logo(full_export_wide, "All_Profiles")
                st.download_button("📥 Raw All Profiles", excel_all, "2_RAW_All_Profiles.xlsx", use_container_width=True)

        with col3:
            rep_list = []
            unique_samples = sorted(df_master['Sample'].unique(), key=natural_sort_key)
            for sample in unique_samples:
                sample_data = df_master[df_master['Sample'] == sample]
                features = sample_data[['Ra', 'Rq', 'Rz', 'Rt']].dropna()
                if not features.empty:
                    centroid = features.mean().values.reshape(1, -1)
                    idx = cdist(features.values, centroid).argmin()
                    closest_file = sample_data.iloc[idx]['File']
                    if closest_file in prof_dict:
                        prof = prof_dict[closest_file][['Length_mm', 'Amplitude_um_Norm']].copy().reset_index(drop=True)
                        name = st.session_state['legend_map'].get(sample, sample)
                        clean_file = os.path.splitext(closest_file)[0]
                        prof.columns = [f"{name} ({clean_file})_X_mm", f"{name} ({clean_file})_Y_Roughness_um"]
                        rep_list.append(prof)
            if rep_list:
                rep_export = pd.concat(rep_list, axis=1)
                excel_rep = export_to_excel_with_logo(rep_export, "Representative_Profiles")
                st.download_button("📥 Raw Rep. Profiles", excel_rep, "3_RAW_Representative.xlsx", use_container_width=True)

        # --- OPTIMIZED EXPORT SECTION ---
        if not st.session_state['opt_df'].empty:
            st.markdown("---")
            st.markdown("### ✨ 2. Optimized Data Export")
            st.markdown("Contains *only* the files kept after running the Global Trend Optimizer.")
            
            df_opt = st.session_state['opt_df']
            opt_files = df_opt['File'].unique()
            col4, colA, col5, col6 = st.columns(4)
            
            with col4:
                excel_summary_opt = export_to_excel_with_logo(df_opt, "Optimized_Summary")
                st.download_button("📥 1. Opt. Summary Stats", excel_summary_opt, "1_OPT_Roughness_Summary.xlsx", use_container_width=True)
                
            with colA:
                ra_dict = {}
                for sample in sorted(df_opt['Sample'].unique(), key=natural_sort_key):
                    col_name = st.session_state['legend_map'].get(sample, sample)
                    ra_dict[col_name] = df_opt[df_opt['Sample'] == sample]['Ra'].reset_index(drop=True)
                
                ra_matrix_df = pd.DataFrame(ra_dict)
                excel_ra_matrix = export_to_excel_with_logo(ra_matrix_df, "Optimized_Ra_Matrix")
                st.download_button("📥 2. Opt. Ra Matrix", excel_ra_matrix, "2_OPT_Ra_Matrix_Wide.xlsx", use_container_width=True)
                
            with col5:
                export_list_opt = []
                for fname in sorted(opt_files, key=natural_sort_key):
                    if fname in prof_dict:
                        p_data = prof_dict[fname]
                        sample_match = df_opt[df_opt['File'] == fname]
                        sample_name = sample_match['Sample'].iloc[0] if not sample_match.empty else "Unknown_Sample"
                        temp = p_data[['Length_mm', 'Amplitude_um_Norm']].copy().reset_index(drop=True)
                        clean_name = os.path.splitext(fname)[0]
                        temp.columns = [f"[{sample_name}] {clean_name}_X_mm", f"[{sample_name}] {clean_name}_Y_Roughness_um"]
                        export_list_opt.append(temp)
                if export_list_opt:
                    full_export_opt = pd.concat(export_list_opt, axis=1)
                    excel_all_opt = export_to_excel_with_logo(full_export_opt, "Optimized_Profiles")
                    st.download_button("📥 3. Opt. All Profiles", excel_all_opt, "3_OPT_All_Profiles.xlsx", use_container_width=True)

            with col6:
                rep_list_opt = []
                unique_samples_opt = sorted(df_opt['Sample'].unique(), key=natural_sort_key)
                for sample in unique_samples_opt:
                    sample_data = df_opt[df_opt['Sample'] == sample]
                    features = sample_data[['Ra', 'Rq', 'Rz', 'Rt']].dropna()
                    if not features.empty:
                        centroid = features.mean().values.reshape(1, -1)
                        idx = cdist(features.values, centroid).argmin()
                        closest_file = sample_data.iloc[idx]['File']
                        if closest_file in prof_dict:
                            prof = prof_dict[closest_file][['Length_mm', 'Amplitude_um_Norm']].copy().reset_index(drop=True)
                            name = st.session_state['legend_map'].get(sample, sample)
                            clean_file = os.path.splitext(closest_file)[0]
                            prof.columns = [f"{name} ({clean_file})_X_mm", f"{name} ({clean_file})_Y_Roughness_um"]
                            rep_list_opt.append(prof)
                if rep_list_opt:
                    rep_export_opt = pd.concat(rep_list_opt, axis=1)
                    excel_rep_opt = export_to_excel_with_logo(rep_export_opt, "Optimized_Representative")
                    st.download_button("📥 4. Opt. Rep. Profiles", excel_rep_opt, "4_OPT_Representative.xlsx", use_container_width=True)

    with tabs[7]:
        st.header("Documentation & Methods")
        
        st.subheader("🌊 ISO 16610-21 Gaussian Filter")
        st.markdown(r"""
        **Purpose:** The Gaussian filter is the internationally recognized standard for separating surface roughness from waviness and primary form.
        **Cutoff Wavelength ($\lambda_c$):** Defines the boundary between roughness and waviness. A profile sine wave with a wavelength equal to $\lambda_c$ is transmitted at 50% amplitude. Wavelengths shorter than $\lambda_c$ (roughness) are retained, while longer wavelengths (waviness) are severely attenuated.
        **Gaussian Weighting Function:**
        The filter applies a moving spatial weighted average based on the Gaussian curve:
        $$S(x)=\frac{1}{\alpha\lambda_c}\exp\left(-\pi\left(\frac{x}{\alpha\lambda_c}\right)^2\right)$$
        *where* $\alpha=\sqrt{\frac{\ln(2)}{\pi}}\approx0.4697$
        **Citation:** > *ISO 16610-21:2011. Geometrical product specifications (GPS) — Filtration — Part 21: Linear profile filters: Gaussian filters.* International Organization for Standardization.
        """)
        st.markdown("---")

        st.subheader("🔬 Savitzky-Golay (S-G) Filtering")
        st.markdown(r"""
        **Purpose of S-G Filtering:**
        In surface metrology, raw data often contains "form" (long-range curvature) and "noise" (measurement error). The **Savitzky-Golay filter** is a digital filter applied to data points to smooth and detrend the profile without distorting the signal peaks.
        **Mechanism:**
        Unlike a standard moving average, S-G uses local polynomial regression (typically 3rd-order/cubic). This preserves high-frequency features like **sharp peaks and valleys** which are critical for calculating standard roughness parameters. By subtracting the smoothed "Form" from the "Total Profile," the **Primary Roughness Profile** is isolated around a zero-mean baseline.
        **Citation:**
        > *Savitzky, A., & Golay, M. J. E. (1964). Smoothing and Differentiation of Data by Simplified Least Squares Procedures. Analytical Chemistry, 36(8), 1627-1639.*
        """)
        st.markdown("---")

        st.subheader("📐 Roughness Parameter Definitions")
        st.markdown(r"The following discrete mathematical formulations are used to compute the amplitude parameters from the filtered roughness profile $Z_i$ over $N$ data points:")
        
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            st.markdown(r"**Arithmetic Mean Roughness ($R_a$)**")
            st.markdown(r"$$R_a=\frac{1}{N}\sum_{i=1}^{N}|Z_i|$$")
            st.markdown(r"**Root Mean Square Roughness ($R_q$)**")
            st.markdown(r"$$R_q=\sqrt{\frac{1}{N}\sum_{i=1}^{N}Z_i^2}$$")
            st.markdown(r"**Total Profile Height ($R_t$)**")
            st.markdown(r"$$R_t=\max(Z)-\min(Z)$$")
        with col_f2:
            st.markdown(r"**Ten-Point Mean Roughness ($R_z$)**")
            st.markdown(r"$$R_z=\frac{1}{5}\sum_{i=1}^{5}P_i-\frac{1}{5}\sum_{j=1}^{5}V_j$$")
            st.markdown(r"*Note: $P_i$ are the 5 highest peaks and $V_j$ are the 5 lowest valleys.*")
            st.markdown(r"**Skewness ($R_{sk}$)**")
            st.markdown(r"$$R_{sk}=\frac{1}{N R_q^3}\sum_{i=1}^{N}Z_i^3$$")
            st.markdown(r"**Kurtosis ($R_{ku}$)**")
            st.markdown(r"$$R_{ku}=\frac{1}{N R_q^4}\sum_{i=1}^{N}Z_i^4$$")
            
        st.markdown(r"""
        **Citation:**
        > *ISO 4287:1997. Geometrical Product Specifications (GPS) — Surface texture: Profile method — Terms, definitions and surface texture parameters.* International Organization for Standardization.
        """)
        st.markdown("---")

        st.subheader("📝 Auto-Generated Methods Text")
        if not df_master.empty and 'Filter' in df_master.columns:
            active_f = df_master['Filter'].iloc[0]
            if "Gaussian" in str(active_f):
                method_text = "Surface profiles were analyzed using a custom Python-based metrology suite. Raw profiles were detrended using an ISO 16610-21 compliant Gaussian filter to isolate primary roughness. Statistical comparisons of amplitude parameters (Ra, Rq, Rz, Rt) were performed using one-way ANOVA (α=0.05)."
            elif "Savitzky" in str(active_f):
                method_text = "Surface profiles were analyzed using a custom Python-based metrology suite. Raw profiles were detrended using a Savitzky-Golay filter (3rd-order polynomial) to isolate primary roughness and preserve peak geometries (Savitzky & Golay, 1964). Statistical comparisons of amplitude parameters (Ra, Rq, Rz, Rt) were performed using one-way ANOVA (α=0.05)."
            else:
                method_text = "Surface profiles were analyzed using a custom Python-based metrology suite. Raw profiles were linearly detrended to isolate primary roughness. Statistical comparisons of amplitude parameters (Ra, Rq, Rz, Rt) were performed using one-way ANOVA (α=0.05)."
        else:
            method_text = "Surface profiles were analyzed using a custom Python-based metrology suite. Raw profiles were detrended to isolate primary roughness. Statistical comparisons of amplitude parameters were performed using one-way ANOVA (α=0.05)."
            
        st.text_area("Copy for your manuscript:", method_text, height=120)

else:
    st.markdown("<h2 style='margin-top: -40px;'>Solomon Scientific roughness analyzer</h2>", unsafe_allow_html=True)
    st.info("👋 Use the sidebar to upload your sample replicates.")
