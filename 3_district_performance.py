import pandas as pd
import plotly.express as px
import glob
import os
import numpy as np

def load_data():
    path = os.path.join('processed_data', '*enrolment*.csv')
    files = glob.glob(path)
    if not files: raise FileNotFoundError(f"No files found in {path}")
    return pd.concat((pd.read_csv(f) for f in files), ignore_index=True)

print("--- [3] LOADING DATA ---")
df = load_data()

# --- PRE-PROCESSING ---
# Ensure we strictly sum the volume columns (Avoiding 'Year'/'Pincode' additions)
# The explicit addition below is safe and correct.
df['Total_Enrolment'] = df['age_0_5'] + df['age_5_17'] + df['age_18_greater']

# Aggregation: Sum by District
# Note: This sums all records found. If data is daily, this gives the total for the period.
district_stats = df.groupby(['state', 'district'])['Total_Enrolment'].sum().reset_index()

# LOGIC: Use Median to avoid Capital City Skew
state_stats = district_stats.groupby('state')['Total_Enrolment'].median().reset_index()
state_stats.rename(columns={'Total_Enrolment': 'State_Median'}, inplace=True)

merged = pd.merge(district_stats, state_stats, on='state')

# Deviation from Median
merged['Deviation'] = merged['Total_Enrolment'] - merged['State_Median']

# --- LOCALIZED LABELING STRATEGY ---
merged['Label'] = ''
for state in merged['state'].unique():
    state_data = merged[merged['state'] == state]
    # Label only the absolute best performer per state (Highest positive deviation)
    idx_max = state_data['Deviation'].idxmax()
    merged.at[idx_max, 'Label'] = state_data.at[idx_max, 'district']

print("\nGenerating visual...")

# --- PLOTTING ---
# Keeping the graph structure exactly as requested
fig = px.scatter(
    merged, 
    x='State_Median', 
    y='Total_Enrolment', 
    color='Deviation',
    text='Label', 
    hover_data=['state', 'district'],
    title='District Performance: Total vs State Median',
    template='plotly_dark',
    color_continuous_scale='RdYlGn', 
    labels={'State_Median': 'State Median Enrolment', 'Total_Enrolment': 'District Total Enrolment'}
)

# --- VISUAL ENHANCEMENTS ---

# 1. Diagonal Line (The Median Line)
fig.add_shape(type="line", line=dict(dash='dash', color='white', width=1), 
              x0=0, y0=0, x1=63000, y1=63000)

# 2. Zone Annotations
fig.add_annotation(x=2000, y=58000, text="LEADING ZONE<br>(Above State Avg)", 
                   showarrow=False, font=dict(color="#00FF00", size=12), align="left")

fig.add_annotation(x=10000, y=2000, text="LAGGING ZONE<br>(Below State Avg)", 
                   showarrow=False, font=dict(color="#FF0000", size=12), align="right")

# 3. Apply Custom Axis Ranges (Hardcoded as per request)
fig.update_layout(
    xaxis=dict(range=[0, 12000]), 
    yaxis=dict(range=[0, 63000])
)

# 4. Text Positioning
fig.update_traces(textposition='top center', textfont_size=9)

fig.show()

# --- [5] STATISTICAL ANALYSIS (CORRECTED) ---
print("\n=== DISTRICT PERFORMANCE STATS ===")

# 1. Data Integrity Check
print(f"Total Districts Analyzed: {len(merged)}")
print(f"Global Median Enrolment:  {merged['Total_Enrolment'].median():,.0f}")
print(f"Max District Enrolment:   {merged['Total_Enrolment'].max():,.0f}")

# 2. Top Performers (Highest Volume)
print("\n--- Top 5 Districts (Highest Enrolment) ---")
top_districts = merged.nlargest(5, 'Total_Enrolment')[['state', 'district', 'Total_Enrolment', 'State_Median']]
print(top_districts.to_string(index=False))

# 3. Top Deviators (Outperforming their State Median the most)
print("\n--- Top 5 Outperformers (Highest Deviation above State Median) ---")
top_deviators = merged.nlargest(5, 'Deviation')[['state', 'district', 'Total_Enrolment', 'Deviation']]
print(top_deviators.to_string(index=False))

# 4. Bottom Performers (Lowest Volume)
print("\n--- Bottom 5 Districts (Lowest Enrolment) ---")
bottom_districts = merged.nsmallest(5, 'Total_Enrolment')[['state', 'district', 'Total_Enrolment', 'State_Median']]
print(bottom_districts.to_string(index=False))