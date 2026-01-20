import streamlit as st
import pandas as pd
import numpy as np
import glob
import os
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from prophet import Prophet
from sklearn.metrics import mean_squared_error
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde

# ==========================================
# 1. APP CONFIGURATION & THEME ENGINE
# ==========================================
st.set_page_config(
    page_title="Aadhaar Command Center", 
    layout="wide", 
    page_icon="🇮🇳",
    initial_sidebar_state="expanded"
)

# Enforce Dark Mode CSS
st.markdown("""
<style>
.stApp { background-color: #0E1117; color: #FAFAFA; }
[data-testid="stSidebar"] { background-color: #262730; color: #FAFAFA; }
.stMetricLabel { color: #FAFAFA !important; }
div[data-testid="stMetricValue"] { color: #00ffbf !important; }
h1, h2, h3, h4, h5, h6 { color: #FAFAFA !important; }
.stDataFrame { border: 1px solid #444; }
.insight-box { background-color: #1E1E1E; padding: 15px; border-radius: 5px; border-left: 5px solid #00ffbf; margin-bottom: 10px; font-size: 15px; }
.remedy-box { background-color: #2D2D2D; padding: 15px; border-radius: 5px; border-left: 5px solid #ff4b4b; font-size: 15px; }
</style>
""", unsafe_allow_html=True)

# Helper for Insights/Remedies
def show_insight(insight, remedy):
    st.markdown(f"""
    <div style='margin-top: 10px; margin-bottom: 20px;'>
        <div class='insight-box'><strong> Analysis Strategy:</strong><br>{insight}</div>
        <div class='remedy-box'><strong> Actionable Remedy:</strong><br>{remedy}</div>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------
# PATTERN RECOGNITION ENGINE
# ---------------------------------------------------------
def analyze_distribution(data, metric_name):
    """
    Analyzes the distribution of a numeric series to find peaks, skewness, and modality.
    Returns a natural language interpretation.
    """
    if len(data) < 5: return "Insufficient data for pattern analysis.", ""
    
    clean_data = data.dropna()
    if clean_data.empty: return "No data available.", ""

    # 1. Calculate Statistics
    skew = clean_data.skew()
    
    # 2. Find Peaks using Kernel Density Estimate (Smooths the histogram)
    try:
        density = gaussian_kde(clean_data)
        xs = np.linspace(clean_data.min(), clean_data.max(), 200)
        ys = density(xs)
        peaks, _ = find_peaks(ys)
        peak_values = xs[peaks]
    except:
        peak_values = []

    # 3. Interpret Pattern
    interpretation = ""
    remedy_hint = ""
    
    # Modality
    if len(peak_values) == 1:
        dist_type = "Unimodal (Single Peak)"
        interpretation += f"The {metric_name} distribution is **{dist_type}**, centered around {int(peak_values[0]):,} units. "
        if abs(skew) > 1:
            skew_dir = "Right (Positive)" if skew > 0 else "Left (Negative)"
            interpretation += f"It is significantly **skewed to the {skew_dir}**, indicating a 'Power Law' where a few entities dominate volume. "
            remedy_hint = "Allocate resources proportionally to the 'Tail' vs the 'Head'."
        else:
            interpretation += "It follows a standard **Normal Distribution** (Bell Curve). "
            remedy_hint = "Standardize processes across all regions as deviation is low."
            
    elif len(peak_values) == 2:
        dist_type = "Bimodal (Two Peaks)"
        interpretation += f"The {metric_name} distribution is **{dist_type}**, with distinct peaks at {int(peak_values[0]):,} and {int(peak_values[1]):,}. "
        interpretation += "This strongly suggests a **'Digital Divide'** or **'Two-Speed'** operational reality. "
        remedy_hint = "Do not apply a 'One-Size-Fits-All' policy. Create separate strategies for 'High-Volume' and 'Low-Volume' clusters."
        
    elif len(peak_values) > 2:
        dist_type = "Multimodal (Complex)"
        interpretation += f"The distribution is **{dist_type}**, showing multiple clusters of activity. "
        remedy_hint = "Investigate sub-clusters for specific local drivers (e.g., Urban vs Rural vs Tribal)."
    else:
        dist_type = "Uniform/Flat"
        interpretation += "The data is relatively flat with no distinct peaks. "

    return interpretation, remedy_hint

# ==========================================
# 2. DATA LOADER (REAL DATA ONLY)
# ==========================================
@st.cache_data
def load_data(pattern):
    folder_path = 'processed_data'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path) # Safety create
        
    files = glob.glob(os.path.join(folder_path, f"*{pattern}*.csv"))
    if not files: return pd.DataFrame()
    
    # Robust loading
    df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    df.columns = df.columns.str.strip()
    
    # Date parsing
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
        df = df.dropna(subset=['date'])
    return df

# Load Datasets
with st.spinner('🇮🇳 Accessing Secure UIDAI Database...'):
    df_bio = load_data("biometric")
    df_demo = load_data("demographic")
    df_enrol = load_data("enrolment")
    if df_enrol.empty: df_enrol = load_data("enrollment")

# Check for empty data
data_exists = not (df_bio.empty and df_demo.empty and df_enrol.empty)

# ==========================================
# 3. SIDEBAR NAVIGATION
# ==========================================
st.sidebar.title("🇮🇳 Aadhaar Command Center")

# Force Plotly Dark Theme
plot_theme = "plotly_dark"

# Navigation
page = st.sidebar.radio("Mission Control", 
    [" Operational Overview", 
     " Smart Insights", 
     " Multivariate & Statistical Lab", 
     " Anomaly Detection", 
     " Volume Forecasting",
     " Advanced Plotter",
     " Processed Data Menu"])

st.sidebar.markdown("---")
if data_exists:
    st.sidebar.success("🟢 System Online")
    st.sidebar.caption(f"Biometric Records: {len(df_bio):,}")
    st.sidebar.caption(f"Demographic Records: {len(df_demo):,}")
    st.sidebar.caption(f"Enrollment Records: {len(df_enrol):,}")
else:
    st.sidebar.error("🔴 Data Offline")

# ==========================================
# PAGE 1: OPERATIONAL OVERVIEW
# ==========================================
if page == " Operational Overview":
    st.title(" Operational Overview")
    st.markdown("Real-time insights derived from Enrolment, Biometric, and Demographic data streams.")
    
    # KPI Row
    k1, k2, k3 = st.columns(3)
    k1.metric("Total Biometric Updates", f"{len(df_bio):,}")
    k2.metric("Total Demographic Updates", f"{len(df_demo):,}")
    k3.metric("New Enrollments", f"{len(df_enrol):,}")
    
    st.markdown("---")
    
    # ROW 1: ENROLLMENT INSIGHTS
    st.subheader("1. Enrollment: Fresh Growth vs. Anomalies")
    col_e1, col_e2 = st.columns(2)
    
    if not df_enrol.empty:
        state_enrol = df_enrol.groupby('state')[['age_0_5', 'age_18_greater']].sum().reset_index()
        state_enrol['Total'] = state_enrol['age_0_5'] + state_enrol['age_18_greater']
        
        # Chart 1: Child Enrollment % (Updated to Percentage)
        with col_e1:
            # Calculate Percentage
            state_enrol['child_ratio'] = (state_enrol['age_0_5'] / state_enrol['Total']) * 100
            
            # Sort by Percentage
            top_fresh = state_enrol.sort_values('child_ratio', ascending=False).head(10)
            
            fig_fresh = px.bar(top_fresh, x='state', y='child_ratio', 
                               title="Child Enrollment Share (%)",
                               labels={'child_ratio': '% Children in New Enrollments'},
                               color='child_ratio', color_continuous_scale='Teal', template=plot_theme)
            st.plotly_chart(fig_fresh, width="stretch")
            show_insight(
                "We check the share of children (age 0-5) in total fresh enrollments.<br>Formula:<br>Child Share % = age_0_5 ÷ Total Enrollments × 100<br>High % = healthy birth registration linkage.",
                "If Child Share is low (<20%), it means the state is still doing too many adult enrollments (catch-up mode).<br>Target: Maintain Child Share > 60%."
            )
        
        # Chart 2: High Adult Enrollment Risk
        with col_e2:
            state_enrol['adult_ratio'] = (state_enrol['age_18_greater'] / state_enrol['Total']) * 100
            top_adult = state_enrol[state_enrol['Total'] > 1000].sort_values('adult_ratio', ascending=False).head(10)
            
            fig_adult = px.bar(top_adult, x='state', y='adult_ratio',
                               title="High Adult Enrollment Risk",
                               labels={'adult_ratio': '% Adults in New Enrollments'},
                               color='adult_ratio', color_continuous_scale='Redor', template=plot_theme)
            st.plotly_chart(fig_adult, width="stretch")
            show_insight(
                "We calculate:<br>Adult Share % = age_18_greater ÷ Total Enrollments × 100<br>Normally this should be very low, because most adults already have Aadhaar.",
                "If Adult Share > 25%, mark as risk zone.<br>Do:<br>Extra document check (POI/POA)<br>Operator audit<br>CCTV review of centers."
            )

    # ROW 2: BIOMETRIC INSIGHTS
    st.subheader("2. Biometric Health: Compliance Gaps")
    col_b1, col_b2 = st.columns(2)
    
    if not df_bio.empty:
        state_bio = df_bio.groupby('state')[['bio_age_5_17', 'bio_age_17_']].sum().reset_index()
        state_bio['Total'] = state_bio.sum(axis=1, numeric_only=True)
        
        # Chart 3: States Missing Child Updates
        with col_b1:
            state_bio['child_share'] = (state_bio['bio_age_5_17'] / state_bio['Total']) * 100
            laggards = state_bio[state_bio['Total'] > 5000].sort_values('child_share', ascending=True).head(10)
            
            fig_lag = px.bar(laggards, x='state', y='child_share',
                             title="States Missing Child Updates",
                             labels={'child_share': '% Child Updates (Target > 40%)'},
                             color='child_share', color_continuous_scale='RdYlGn', template=plot_theme)
            st.plotly_chart(fig_lag, width="stretch")
            show_insight(
                "Measure:<br>Child Update % = bio_age_5_17 ÷ Total Biometric × 100<br>Target set by UIDAI ≈ 40%+.",
                "Link school admission with update receipt.<br>Run school camps every quarter.<br>KPI:<br>Child updates per month per center ≥ 120."
            )
        
        # Chart 4: Volume Stack
        with col_b2:
            top_vol = state_bio.sort_values('Total', ascending=False).head(10)
            fig_stack = px.bar(top_vol, x='state', y=['bio_age_5_17', 'bio_age_17_'],
                               title="Total Biometric Workload",
                               barmode='stack', template=plot_theme,
                               color_discrete_map={'bio_age_5_17': '#00b894', 'bio_age_17_': '#636efa'})
            st.plotly_chart(fig_stack, width="stretch")
            show_insight(
                "This stack visualizes the raw operational load on Permanent Enrollment Centers (PECs).",
                "Operational bottleneck detected in top 3 states. Increase the number of authorised private operators in urban centers."
            )

# ==========================================
# PAGE 2: SMART INSIGHTS
# ==========================================
elif page == " Smart Insights":
    st.title(" Deep Dive & Experimental Findings")
    
    t1, t2 = st.tabs(["Automated Operational Insights", "Advanced Behavioral Analytics"])
    
    # --- TAB 1: AUTOMATED ANOMALIES ---
    with t1:
        st.write("### Automated District-Level Analysis")
        
        # INSIGHT 1: Child Enrollment Gap
        st.markdown("#### 1. Child Enrollment Gap Index")
        c1, c2 = st.columns([1, 1.5]) 
        
        with c1:
            if not df_enrol.empty:
                dist = df_enrol.groupby(['state', 'district'])[['age_5_17', 'age_18_greater']].sum().reset_index()
                dist['total'] = dist['age_5_17'] + dist['age_18_greater']
                dist = dist[dist['total'] > 500]
                dist['child_pct'] = (dist['age_5_17'] / dist['total']) * 100
                
                # Z-Score
                state_means = dist.groupby('state')['child_pct'].mean().rename('avg')
                dist = dist.merge(state_means, on='state')
                std_devs = dist.groupby('state')['child_pct'].transform('std').replace(0,1)
                dist['z_score'] = (dist['child_pct'] - dist['avg']) / std_devs
                
                laggards = dist[dist['z_score'] < -1.5].sort_values('z_score').head(10)
                
                # Dataframe
                st.dataframe(
                    laggards[['state', 'district', 'child_pct', 'avg']]
                    .style.format({"child_pct": "{:.1f}%", "avg": "{:.1f}%"}),
                    width="stretch"
                )
        
        with c2:
            # Visual for Laggards
            fig_lag = px.bar(laggards, x='district', y='child_pct', color='z_score',
                             title="Districts with Critical Child Enrollment Gaps",
                             labels={'child_pct': 'Child Enrollment %', 'district': 'District'},
                             color_continuous_scale='Reds_r', template=plot_theme)
            st.plotly_chart(fig_lag, width="stretch")
            show_insight(
                "These districts show a statistically significant deficit in child enrollment compared to their own state average.",
                "Direct District Magistrates (DMs) to form 'Block-Level Monitoring Committees'. Compare district school rosters with Aadhaar databases."
            )

        st.markdown("---")

        # INSIGHT 2: Enrollments Without Follow-up Updates (Ghost Districts)
        st.markdown("#### Enrollments Without Follow-up Updates")
        c3, c4 = st.columns([1, 1.5])
        
        with c3:
            if not df_enrol.empty and not df_bio.empty:
                # FIX: Explicit column selection to prevent Pincode summation
                enrol_cols = df_enrol.select_dtypes(include=np.number).columns.drop(['pincode'], errors='ignore')
                bio_cols = df_bio.select_dtypes(include=np.number).columns.drop(['pincode'], errors='ignore')
                
                e_vol = df_enrol.groupby(['state', 'district'])[enrol_cols].sum().sum(axis=1).rename('E_Vol')
                b_vol = df_bio.groupby(['state', 'district'])[bio_cols].sum().sum(axis=1).rename('B_Vol')
                
                merged = pd.concat([e_vol, b_vol], axis=1).dropna()
                merged['Ratio'] = merged['B_Vol'] / merged['E_Vol']
                
                ghosts = merged[merged['E_Vol'] > 1000].sort_values('Ratio').head(10)
                
                # Dataframe
                st.dataframe(ghosts.style.format({"Ratio": "{:.4f}", "E_Vol": "{:.0f}", "B_Vol": "{:.0f}"}), width="stretch")

        with c4:
            # Visual for Ghosts
            ghosts['District'] = ghosts.index.get_level_values(1)
            fig_ghost = px.scatter(ghosts, x='E_Vol', y='B_Vol', size='E_Vol', color='Ratio',
                                   title="Enrollments Without Follow-up Updates",
                                   labels={'E_Vol': 'New Enrollments', 'B_Vol': 'Biometric Updates'},
                                   hover_name='District', template=plot_theme)
            st.plotly_chart(fig_ghost, width="stretch")
            show_insight(
                "We compare:<br>E_Vol = Total Enrollments<br>B_Vol = Total Biometric Updates<br>Ratio = B_Vol ÷ E_Vol<br>Healthy district → Ratio around 0.6 – 1.2.",
                "If Ratio < 0.2:<br>Convert camps to permanent centers<br>Start incentive for updates<br>Send field team to verify center existence."
            )

    # --- TAB 2: ADVANCED BEHAVIORAL ANALYTICS ---
    with t2:
        st.write("### Strategic & Behavioral Insights")
        
        # 1. State Growth vs Update Status (Maturity Matrix)
        st.subheader("1. State Growth vs Update Status")
        if data_exists:
            # FIX: Explicit column selection
            e_cols = df_enrol.select_dtypes(include=np.number).columns.drop(['pincode'], errors='ignore')
            b_cols = df_bio.select_dtypes(include=np.number).columns.drop(['pincode'], errors='ignore')
            d_cols = df_demo.select_dtypes(include=np.number).columns.drop(['pincode'], errors='ignore')

            s_acq = df_enrol.groupby('state')[e_cols].sum().sum(axis=1).rename('Acquisition')
            s_maint = (df_bio.groupby('state')[b_cols].sum().sum(axis=1) + 
                       df_demo.groupby('state')[d_cols].sum().sum(axis=1)).rename('Maintenance')
            
            matrix = pd.concat([s_acq, s_maint], axis=1).dropna()
            # Normalize
            matrix['Growth_Share'] = (matrix['Acquisition'] / matrix['Acquisition'].sum()) * 100
            matrix['Maturity_Share'] = (matrix['Maintenance'] / matrix['Maintenance'].sum()) * 100
            matrix['State'] = matrix.index
            
            # Median Lines for Quadrants
            med_x = matrix['Growth_Share'].median()
            med_y = matrix['Maturity_Share'].median()
            
            fig_mat = px.scatter(matrix, x='Growth_Share', y='Maturity_Share', 
                                 text='State', hover_name='State', size='Maturity_Share',
                                 title="State Growth vs Update Status", template=plot_theme,
                                 color='Maturity_Share', color_continuous_scale='Plasma')
            
            # Add Quadrant Lines
            fig_mat.add_vline(x=med_x, line_dash="dash", line_color="red", annotation_text="Median Growth")
            fig_mat.add_hline(y=med_y, line_dash="dash", line_color="red", annotation_text="Median Maturity")
            fig_mat.update_traces(textposition='top center')
            
            st.plotly_chart(fig_mat, width="stretch")
            show_insight(
                "Two measures:<br>Growth Share = State Enrollment ÷ All India Enrollment × 100<br>Maturity Share = (Bio + Demo Updates) ÷ All India Updates × 100<br>This shows whether state is:<br>New & growing<br>Mature & stable.",
                "Mature states → invest in update kiosks not new kits.<br>New states → focus on mobile enrollment vans."
            )

        st.markdown("---")
        
        # 2. Monthly Rush Pattern (Seasonality)
        st.subheader("2. Monthly Rush Pattern")
        if not df_bio.empty and 'date' in df_bio.columns:
            df_bio['Month'] = df_bio['date'].dt.month_name()
            df_bio['Month_Num'] = df_bio['date'].dt.month
            
            monthly = df_bio.groupby(['Month_Num', 'Month'])[['bio_age_5_17', 'bio_age_17_']].mean().reset_index()
            monthly = monthly.sort_values('Month_Num')
            
            fig_seas = px.line(monthly, x='Month', y=['bio_age_5_17', 'bio_age_17_'], markers=True,
                               title="Monthly Rush Pattern", template=plot_theme,
                               labels={'value': 'Avg Daily Updates', 'variable': 'Age Group'})
            st.plotly_chart(fig_seas, width="stretch")
            show_insight(
                "We take average daily updates per month.<br>Detect peaks during school admission months.",
                "From May–July:<br>+30% operators<br>extra machines<br>extended timing 8am–8pm."
            )

        st.markdown("---")

        # 3. Address Change Activity (Churn Index)
        st.subheader("3. Address Change Activity")
        if data_exists:
            # FIX: Explicit column selection
            b_cols = df_bio.select_dtypes(include=np.number).columns.drop(['pincode'], errors='ignore')
            d_cols = df_demo.select_dtypes(include=np.number).columns.drop(['pincode'], errors='ignore')

            d_vol = df_demo.groupby('state')[d_cols].sum().sum(axis=1)
            b_vol = df_bio.groupby('state')[b_cols].sum().sum(axis=1)
            churn = pd.DataFrame({'Demo': d_vol, 'Bio': b_vol}).dropna()
            churn['Churn_Index'] = churn['Demo'] / churn['Bio']
            churn = churn[churn['Bio'] > 1000] # Filter noise
            
            c_high, c_low = st.columns(2)
            
            with c_high:
                top_churn = churn.sort_values('Churn_Index', ascending=False).head(8)
                fig_ch = px.bar(top_churn, x=top_churn.index, y='Churn_Index', 
                                title="Address Change Activity", 
                                color='Churn_Index', color_continuous_scale='Reds', template=plot_theme)
                st.plotly_chart(fig_ch, width="stretch")
                show_insight(
                    "Churn Index = Demo Updates ÷ Biometric Updates<br>1.0 → many address/phone changes<br><0.5 → very stable population.",
                    "High churn:<br>Promote online SSUP<br>more update kiosks<br>Low churn:<br>focus on quality biometric machines."
                )
                
            with c_low:
                low_churn = churn.sort_values('Churn_Index', ascending=True).head(8)
                fig_cl = px.bar(low_churn, x=low_churn.index, y='Churn_Index', 
                                title="Stable Populations (Low Change)", 
                                color='Churn_Index', color_continuous_scale='Blues', template=plot_theme)
                st.plotly_chart(fig_cl, width="stretch")
                st.write("") 

# ==========================================
# PAGE 3: MULTIVARIATE & STATISTICAL LAB
# ==========================================
elif page == " Multivariate & Statistical Lab":
    st.title(" Multivariate & Statistical Lab")
    st.info("Advanced multidimensional analysis of Operational Streams.")
    
    if data_exists:
        # Prepare Data for all analyses
        s_enrol = df_enrol.groupby('state')[['age_0_5', 'age_5_17', 'age_18_greater']].sum()
        s_bio = df_bio.groupby('state')[['bio_age_5_17', 'bio_age_17_']].sum()
        s_demo = df_demo.groupby('state')[['demo_age_5_17', 'demo_age_17_']].sum()
        
        # Merge basic volumes
        df_tri = pd.concat([s_enrol.sum(axis=1), s_bio.sum(axis=1), s_demo.sum(axis=1)], axis=1, keys=['Enrol', 'Bio', 'Demo']).dropna()
        df_tri['State'] = df_tri.index
        
        # --- SECTION 1: MULTIVARIATE ANALYSIS ---
        st.header("1. Multivariate Analysis(5-Dimensions)")
        # Create Multivariate Dataset
        df_multi = df_tri.copy()
        df_multi['Child_Index'] = (s_enrol['age_0_5'] / s_enrol.sum(axis=1)) * 100
        df_multi['Adult_Index'] = (s_enrol['age_18_greater'] / s_enrol.sum(axis=1)) * 100
        df_multi = df_multi.sort_values('Enrol', ascending=False).head(15).reset_index(drop=True)
        df_multi['State_ID'] = df_multi.index
        
        fig_par = px.parallel_coordinates(df_multi, 
                                          dimensions=['Enrol', 'Bio', 'Demo', 'Child_Index', 'Adult_Index'],
                                          color="State_ID", 
                                          title="Multidimensional State Trajectory (Top 15 States)",
                                          template=plot_theme)
        st.plotly_chart(fig_par, width="stretch")
        with st.expander("Show State ID Legend (Who is who?)"):
            st.caption("Since Parallel Coordinate plots only accept numbers, use this table to identify the State IDs.")
            
            # Create a clean lookup table
            legend_df = df_multi[['State_ID', 'State']].sort_values('State_ID')
            
            # Display as a horizontal metric row or a table
            st.dataframe(
                legend_df.set_index('State_ID').T, # Transpose for horizontal view
                width='stretch'
            )
        show_insight(
            "Enrollment, biometric, and demographic updates are reviewed together to understand the overall balance of services in each state.<br><br>Child share is calculated as:<br>age_0_5 ÷ total enrollments × 100<br><br>Adult share is calculated as:<br>age_18+ ÷ total enrollments × 100.<br><br>Crossing patterns indicate that services are not evenly delivered.",
            "Update counters are to be added where enrollments are high but updates are low.<br>School and Anganwadi drives are to be conducted where adult share is high.<br>A target of Child share above 35% and Bio at least 60% of Enrol is to be maintained."
        )

        st.markdown("---")

        # --- SECTION 2: TRIVARIATE ANALYSIS ---
        st.header("2. Trivariate Analysis")
        df_tri['Log_Enrol'] = np.log1p(df_tri['Enrol'])
        df_tri['Log_Bio'] = np.log1p(df_tri['Bio'])
        df_tri['Log_Demo'] = np.log1p(df_tri['Demo'])
        
        fig_3d = px.scatter_3d(df_tri, x='Log_Enrol', y='Log_Bio', z='Log_Demo',
                               color='Log_Demo', size='Log_Bio', text='State',
                               hover_name='State', template=plot_theme,
                               title="3D Operational Synergy (Log Scale)",
                               color_continuous_scale='Viridis')
        fig_3d.update_layout(height=700, scene=dict(
            xaxis_title='Enrollment (Log)', yaxis_title='Biometric (Log)', zaxis_title='Demographic (Log)'
        ))
        st.plotly_chart(fig_3d, width="stretch")
        show_insight(
            "Enrollment, biometric, and demographic services are compared together using log scale so that large and small states are treated equally.<br>States leaning toward one axis are considered to have a missing service component.",
            "Additional biometric machines and operator training are to be provided where Bio is weak.<br>Online address update options are to be promoted where Demo is low.<br>All three services are to be offered in a single visit at every center."
        )
        
        st.markdown("---")
        
        # --- SECTION 3: BIVARIATE ANALYSIS ---
        st.header("3. Bivariate Analysis")
        b1, b2 = st.columns(2)
        with b1:
            fig_b1 = px.scatter(df_tri, x='Enrol', y='Bio', trendline='ols', 
                                title="Correlation: Enrolment vs Biometric", template=plot_theme)
            st.plotly_chart(fig_b1, width="stretch")
            show_insight(
                "The relationship between enrollments and biometric updates is examined using a trend line.<br>Points falling below the line are treated as biometric backlog.",
                "More devices are to be placed in backlog locations.<br>Waiting time for updates is to be reduced.<br>Bio volume is expected to remain at least 70% of Enrol."
            )
            
        with b2:
            fig_b2 = px.scatter(df_tri, x='Demo', y='Bio', trendline='ols', 
                                title="Correlation: Demographic vs Biometric", template=plot_theme)
            st.plotly_chart(fig_b2, width="stretch")
            show_insight(
                "The link between demographic and biometric updates is assessed to check whether both are done together.<br>Weak linkage suggests that citizens are required to visit twice.",
                "A prompt for same-day biometric update is to be enabled in software.<br>Operators are to be trained for combined update packets."
            )
            
        st.markdown("---")

        # --- SECTION 4: UNIVARIATE ANALYSIS ---
        st.header("4. Univariate Analysis & Pattern Recognition")
        st.caption("Distribution analysis of operational volumes across all states.")
        
        u1, u2 = st.columns(2)
        with u1:
            fig_u1 = px.histogram(df_tri, x='Enrol', nbins=15, 
                                  title="Distribution: Enrollment Volumes", template=plot_theme)
            st.plotly_chart(fig_u1, width="stretch")
            
            # Dynamic Interpretation
            insight_txt, remedy_txt = analyze_distribution(df_tri['Enrol'], "Enrollment")
            show_insight(insight_txt, remedy_txt)
            
        with u2:
            fig_u2 = px.histogram(df_tri, x='Bio', nbins=15, 
                                  title="Distribution: Biometric Update Volumes", template=plot_theme)
            st.plotly_chart(fig_u2, width="stretch")
            
            # Dynamic Interpretation
            insight_txt, remedy_txt = analyze_distribution(df_tri['Bio'], "Biometric")
            show_insight(insight_txt, remedy_txt)

# ==========================================
# PAGE 4: ANOMALY DETECTION
# ==========================================
elif page == " Anomaly Detection":
    st.title(" High-Risk Anomaly Investigation")
    
    target_type = st.selectbox("Select Anomaly Target", ["Biometric Updates", "Enrollment", "Demographic Updates"])
    
    selected_df = pd.DataFrame()
    explanation = ""
    
    if target_type == "Biometric Updates" and not df_bio.empty:
        pin = df_bio.groupby('pincode')[['bio_age_5_17', 'bio_age_17_']].sum().reset_index()
        pin['total'] = pin['bio_age_5_17'] + pin['bio_age_17_']
        pin['ratio'] = np.where(pin['total']>0, pin['bio_age_5_17']/pin['total'], 0)
        selected_df = pin
        reason = "Compliance Gap (Zero Child Updates)"
        explanation = f"""
        **📊 Understanding the Metrics:**
        - **Ratio Calculation:** `Child Updates (5-17) / Total Updates`.
        - **Graph Logic:** The Y-axis shows this ratio (0.0 to 1.0), while the X-axis shows total volume.
        - **Reason for Suspicion:** {reason}. A low ratio at high volume implies the center is refusing mandatory child updates (low effort) in favor of easier adult updates.
        """
        
    elif target_type == "Enrollment" and not df_enrol.empty:
        pin = df_enrol.groupby('pincode')[['age_0_5', 'age_18_greater']].sum().reset_index()
        pin['total'] = pin['age_0_5'] + pin['age_18_greater']
        pin['ratio'] = np.where(pin['total']>0, pin['age_18_greater']/pin['total'], 0)
        selected_df = pin
        reason = "Adult Catch-up / Potential Fraud"
        explanation = f"""
        **📊 Understanding the Metrics:**
        - **Ratio Calculation:** `Adult Enrollments (18+) / Total Enrollments`.
        - **Graph Logic:** The Y-axis shows this ratio (0.0 to 1.0), while the X-axis shows total volume.
        - **Reason for Suspicion:** {reason}. A high ratio is anomalous because >99% of adults are already enrolled. Spikes indicate potential 'Ghost' creation.
        """

    elif target_type == "Demographic Updates" and not df_demo.empty:
        pin = df_demo.groupby('pincode')[['demo_age_5_17', 'demo_age_17_']].sum().reset_index()
        pin['total'] = pin['demo_age_5_17'] + pin['demo_age_17_']
        pin['ratio'] = np.where(pin['total']>0, pin['demo_age_17_']/pin['total'], 0)
        selected_df = pin
        reason = "Mass Address Update Mill"
        explanation = f"""
        ** Understanding the Metrics:**
        - **Ratio Calculation:** `Adult Updates (18+) / Total Updates`.
        - **Graph Logic:** The Y-axis shows this ratio (0.0 to 1.0), while the X-axis shows total volume.
        - **Reason for Suspicion:** {reason}. Unusually high concentrations of adult address changes often signal fraudulent 'Address Update Mills'.
        """

    if not selected_df.empty:
        # Show Explanation First
        st.info(explanation)
        
        iso = IsolationForest(contamination=0.05, random_state=42)
        X = selected_df[['total', 'ratio']].fillna(0)
        selected_df['anomaly'] = iso.fit_predict(X)
        
        fig = px.scatter(selected_df, x='total', y='ratio', 
                         color=selected_df['anomaly'].astype(str),
                         color_discrete_map={'-1':'red', '1':'blue'}, 
                         log_x=True,
                         title=f"{target_type} Anomalies (Red = Outliers)",
                         labels={'ratio': 'Key Risk Ratio', 'total': 'Volume'},
                         hover_data=['pincode'], template=plot_theme)
        st.plotly_chart(fig, width="stretch")
        
        show_insight(
            "For each pincode:<br>Risk Ratio (Biometric) = Child Updates ÷ Total Updates<br>Risk Ratio (Enrollment) = Adult Enroll ÷ Total Enroll<br>Isolation Forest finds outliers using: [total , ratio].",
            "Red points = immediate action<br>suspend operator<br>physical verification<br>packet re-check."
        )
        
        st.error(f" High Risk Investigation Required: {reason}")
        anomalies = selected_df[selected_df['anomaly']==-1].sort_values('total', ascending=False).head(10)
        anomalies['Reason_For_Suspicion'] = reason
        st.dataframe(anomalies[['pincode', 'total', 'ratio', 'Reason_For_Suspicion']], width="stretch")

# ==========================================
# PAGE 5: FORECASTING (PROPHET WITH INTERPRETATION)
# ==========================================
elif page == " Volume Forecasting":
    st.title(" Volume Forecasting (Prophet)")
    st.info("Using Facebook Prophet to forecast operational loads with real-world seasonality.")
    
    ds_choice = st.selectbox("Select Dataset", ["Biometric", "Demographic", "Enrollment"])
    
    if st.button(" Generate Forecast"):
        if ds_choice == "Biometric": df_t = df_bio
        elif ds_choice == "Demographic": df_t = df_demo
        else: df_t = df_enrol
        
        numeric_cols = [
            c for c in df_t.select_dtypes(include=np.number).columns
            if c.lower() not in ['pincode', 'operator_id', 'station_id']
        ]

        daily = df_t.groupby('date')[numeric_cols].sum().sum(axis=1).reset_index(name='y')
        daily = daily.rename(columns={'date': 'ds'})
        
        if len(daily) > 30:
            with st.spinner("🤖 Prophet Model Training in Progress..."):
                m = Prophet()
                m.fit(daily)
                future = m.make_future_dataframe(periods=30)
                forecast = m.predict(future)
                
                # Custom Plotly Chart
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=daily['ds'], y=daily['y'], name='Actual Volume', 
                                         mode='lines', line=dict(color='#00CC96')))
                fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='Prophet Forecast', 
                                         line=dict(color='#EF553B', dash='dash')))
                fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], showlegend=False, 
                                         line=dict(width=0), mode='lines'))
                fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], name='Uncertainty Interval',
                                         line=dict(width=0), mode='lines', fill='tonexty', fillcolor='rgba(239, 85, 59, 0.2)'))
                
                fig.update_layout(title=f"30-Day Operational Forecast: {ds_choice}", template=plot_theme)
                st.plotly_chart(fig, width="stretch")
                
                # Dynamic Forecast Interpretation
                last_actual = daily['y'].iloc[-1]
                avg_forecast = forecast['yhat'].iloc[-30:].mean()
                trend_direction = "Surging 📈" if avg_forecast > last_actual * 1.1 else ("Declining 📉" if avg_forecast < last_actual * 0.9 else "Stable ➡️")
                
                show_insight(
                    "Prophet predicts next 30 days using:<br>trend<br>weekly pattern<br>yearly pattern.<br>Compare:<br>Avg Forecast ÷ Last Actual.",
                    "If >1.1 → prepare for rush<br>If <0.9 → use time for training<br>Else → keep normal staffing."
                )
        else:
            st.error("Insufficient data points for Prophet model (Need > 30 days).")

# ==========================================
# PAGE 6: ADVANCED TESTING (RESTRICTED CHART TYPES)
# ==========================================
elif page == " Advanced Plotter":
    st.title(" Advanced Sandbox")
    st.caption("Custom Chart Builder using Live Data")
    
    ds = st.selectbox("Data Source", ["Biometric", "Demographic", "Enrollment"])
    if ds == "Biometric": df_s = df_bio
    elif ds == "Demographic": df_s = df_demo
    else: df_s = df_enrol
    
    # 1. Filter out 'pincode' from valid columns
    valid_x = [c for c in df_s.columns if 'pincode' not in c.lower()]
    valid_y = [c for c in df_s.select_dtypes(include=np.number).columns if 'pincode' not in c.lower()]
    
    c1, c2, c3 = st.columns(3)
    x_ax = c1.selectbox("X Axis", valid_x)
    y_ax = c2.selectbox("Y Axis", valid_y)
    
    # 2. DYNAMIC CHART TYPE LOGIC
    is_numeric_x = pd.api.types.is_numeric_dtype(df_s[x_ax])
    is_date_x = pd.api.types.is_datetime64_any_dtype(df_s[x_ax]) or 'date' in x_ax.lower()
    
    if is_date_x:
        chart_options = ["Line", "Bar", "Scatter"]
    elif is_numeric_x:
        chart_options = ["Scatter", "Histogram"]
    else:
        chart_options = ["Bar", "Scatter", "Histogram"]
        
    ctype = c3.selectbox("Chart Type", chart_options)
    
    if st.button("Plot"):
        if x_ax == y_ax:
            st.error("⚠️ Invalid Plot Configuration: X and Y axes cannot be the same parameter.")
        else:
            plot_df = df_s.head(5000) 
            if ctype == "Bar": fig = px.bar(plot_df, x=x_ax, y=y_ax, template=plot_theme)
            elif ctype == "Line": fig = px.line(plot_df, x=x_ax, y=y_ax, template=plot_theme)
            elif ctype == "Scatter": fig = px.scatter(plot_df, x=x_ax, y=y_ax, template=plot_theme)
            elif ctype == "Histogram": fig = px.histogram(plot_df, x=x_ax, template=plot_theme)
            st.plotly_chart(fig, width="stretch")

# ==========================================
# PAGE 7: DATA MENU
# ==========================================
elif page == " Processed Data Menu":
    st.title(" Processed Data Viewer")
    st.info("View raw processed data. Use slider to navigate.")
    
    view_opt = st.selectbox("Dataset", ["Biometric", "Enrollment", "Demographic"])
    df_v = df_bio if view_opt=="Biometric" else (df_enrol if view_opt=="Enrollment" else df_demo)
    
    page_size = 100
    total_pages = max(1, len(df_v)//page_size)
    p_num = st.slider("Page", 1, total_pages, 1)
    
    start = (p_num-1)*page_size
    
    display_df = df_v.iloc[start:start+page_size].copy()
    
    if 'date' in display_df.columns:
        display_df['date'] = display_df['date'].dt.strftime('%Y-%m-%d')
        
    st.dataframe(display_df, width="stretch")