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

print("--- [1] LOADING DATA ---")
df = load_data()

# 1. Aggregate Data by State
state_group = df.groupby('state')[['age_0_5', 'age_5_17', 'age_18_greater']].sum().reset_index()

# 2. Calculate TOTAL Enrolment to Normalize
state_group['Total_Enrolment'] = state_group['age_0_5'] + state_group['age_5_17'] + state_group['age_18_greater']

# 3. Calculate Normalized Percentages (Handling division by zero safely)
state_group['Child_Share_Pct'] = (state_group['age_0_5'] / state_group['Total_Enrolment']) * 100
state_group['Adult_Share_Pct'] = (state_group['age_18_greater'] / state_group['Total_Enrolment']) * 100

# 4. Determine Status based on Shares
# If Child % > Adult %, Awareness is Good.
state_group['Status'] = state_group.apply(
    lambda x: 'High Awareness (Child > Adult)' if x['Child_Share_Pct'] > x['Adult_Share_Pct'] 
    else 'Poor Awareness (Adult > Child)', axis=1
)

# --- STATISTICAL OUTPUT (Normalized) ---
print("\n=== STATISTICAL ANALYSIS: ENROLMENT SHARE (%) ===")
stats = state_group['Child_Share_Pct'].dropna()

print(f"Mean Child Share:    {stats.mean():.2f}%")
print(f"Median Child Share:  {stats.median():.2f}%")
print(f"Std Deviation:       {stats.std():.2f}")
print(f"Min Share:           {stats.min():.2f}%")
print(f"Max Share:           {stats.max():.2f}%")

print("\n--- TOP 5 STATES (Best Awareness - Highest % of Children) ---")
# Sorting by Percentage gives the true "Best Performing" states regardless of size
print(state_group[['state', 'Child_Share_Pct', 'Total_Enrolment']].sort_values('Child_Share_Pct', ascending=False).head(5).to_string(index=False))

print("\n--- BOTTOM 5 STATES (Poor Awareness - Lowest % of Children) ---")
print(state_group[['state', 'Child_Share_Pct', 'Total_Enrolment']].sort_values('Child_Share_Pct', ascending=True).head(5).to_string(index=False))
print("\nGenerating visual...")

# --- PLOTTING (Normalized Stacked Bar) ---
# We transform data to 'long' format for a stacked bar chart of percentages
df_melted = state_group.melt(
    id_vars=['state'], 
    value_vars=['Child_Share_Pct', 'Adult_Share_Pct'],
    var_name='Age_Group', 
    value_name='Percentage'
)

fig = px.bar(
    df_melted, 
    x='state', 
    y='Percentage',
    color='Age_Group',
    title='Enrolment Awareness (Normalized): Child vs Adult Share %',
    labels={'Percentage': 'Share of Total Enrolment (%)'},
    color_discrete_map={'Child_Share_Pct': '#00CC96', 'Adult_Share_Pct': '#EF553B'},
    template='plotly_dark'
)

# Add a horizontal line at 50% to show the "tipping point"
fig.add_shape(type="line", line=dict(dash='dash', color='white'), 
              x0=-0.5, y0=50, x1=len(state_group['state'])-0.5, y1=50)

# Sort the visual by Child Share so the trend is visible
fig.update_layout(xaxis={'categoryorder':'total descending'})
fig.show()