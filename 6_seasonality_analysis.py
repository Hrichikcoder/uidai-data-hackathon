import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import glob

def load_data(pattern):
    path = os.path.join('processed_data', pattern)
    files = glob.glob(path)
    if not files: return pd.DataFrame()
    df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    
    if 'pincode' in df.columns:
        df = df.drop(columns=['pincode'])
    return df

print("--- [8] LOADING DATA ---")
df_enrol = load_data('*enrolment*.csv')
df_bio = load_data('*biometric*.csv')

df_enrol['date'] = pd.to_datetime(df_enrol['date'], format='%d-%m-%Y', errors='coerce')
df_bio['date'] = pd.to_datetime(df_bio['date'], format='%d-%m-%Y', errors='coerce')

df_enrol = df_enrol.dropna(subset=['date'])
df_bio = df_bio.dropna(subset=['date'])

def get_monthly_trend(df, metric_cols):
    df['Month'] = df['date'].dt.month_name()
    df['Month_Num'] = df['date'].dt.month
    
    monthly = df.groupby(['Month_Num', 'Month'])[metric_cols].mean().reset_index()
    monthly = monthly.sort_values('Month_Num')
    return monthly

bio_cols = ['bio_age_5_17', 'bio_age_17_']
bio_monthly = get_monthly_trend(df_bio.copy(), bio_cols)

enrol_cols = ['age_0_5', 'age_18_greater']
enrol_monthly = get_monthly_trend(df_enrol.copy(), enrol_cols)

fig = make_subplots(
    rows=2, cols=1, 
    subplot_titles=("<b>1. Biometric Rush Pattern (Updates)</b>", 
                    "<b>2. Enrolment Rush Pattern (New Aadhaars)</b>"),
    vertical_spacing=0.15
)

fig.add_trace(
    go.Scatter(x=bio_monthly['Month'], y=bio_monthly['bio_age_5_17'], 
               mode='lines+markers', name='Teenagers (5-17)',
               line=dict(color='#00CC96', width=3), marker=dict(size=8)),
    row=1, col=1
)
fig.add_trace(
    go.Scatter(x=bio_monthly['Month'], y=bio_monthly['bio_age_17_'], 
               mode='lines+markers', name='Adults (17+)',
               line=dict(color='#636EFA', width=3), marker=dict(size=8)),
    row=1, col=1
)

fig.add_trace(
    go.Scatter(x=enrol_monthly['Month'], y=enrol_monthly['age_0_5'], 
               mode='lines+markers', name='Children (0-5)',
               line=dict(color='#FFA15A', width=3), marker=dict(size=8)),
    row=2, col=1
)
fig.add_trace(
    go.Scatter(x=enrol_monthly['Month'], y=enrol_monthly['age_18_greater'], 
               mode='lines+markers', name='Adults (18+)',
               line=dict(color='#EF553B', width=3), marker=dict(size=8)),
    row=2, col=1
)

fig.update_layout(
    template='plotly_dark',
    height=800,
    title_text="<b>Monthly Rush Pattern Analysis (Seasonality)</b>",
    showlegend=True,
    xaxis_title="Month",
    xaxis2_title="Month",
    yaxis_title="Avg Volume",
    yaxis2_title="Avg Volume"
)

fig.show()

print("\n=== SEASONALITY STATISTICS ===")

def analyze_seasonality(df, name, vol_col):
    if df.empty: return
    
    peak_row = df.loc[df[vol_col].idxmax()]
    low_row = df.loc[df[vol_col].idxmin()]
    avg_vol = df[vol_col].mean()
    
    seasonality_factor = peak_row[vol_col] / avg_vol if avg_vol > 0 else 0
    
    print(f"\n--- {name} Analysis ---")
    print(f"Peak Month:      {peak_row['Month']} (Vol: {peak_row[vol_col]:.0f})")
    print(f"Lowest Month:    {low_row['Month']} (Vol: {low_row[vol_col]:.0f})")
    print(f"Average Volume:  {avg_vol:.0f}")
    print(f"Volatility:      Peak is {seasonality_factor:.1f}x higher than average")

    if seasonality_factor > 1.5:
        print(">> INSIGHT: Highly Seasonal. Staffing must be flexible.")
    else:
        print(">> INSIGHT: Stable Demand. Fixed staffing is safe.")

analyze_seasonality(bio_monthly, "Biometric (Children 5-17)", 'bio_age_5_17')

analyze_seasonality(enrol_monthly, "Enrolment (Babies 0-5)", 'age_0_5')