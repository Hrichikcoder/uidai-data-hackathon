import pandas as pd
import plotly.express as px
import glob
import os
import numpy as np

def load_data():
    path = os.path.join('processed_data', '*biometric*.csv')
    files = glob.glob(path)
    if not files: raise FileNotFoundError(f"No files found in {path}")
    return pd.concat((pd.read_csv(f) for f in files), ignore_index=True)

print("--- [2] LOADING DATA ---")
df = load_data()

# 1. Aggregate
bio_updates = df.groupby('state')[['bio_age_5_17', 'bio_age_17_']].sum().reset_index()

# 2. Normalize: Calculate 'Mandatory Compliance Rate'
# Formula: (Child Updates / Total Biometric Updates) * 100
# This makes a small state comparable to a large state.
bio_updates['Total_Updates'] = bio_updates['bio_age_5_17'] + bio_updates['bio_age_17_']
bio_updates['Mandatory_Compliance_Pct'] = (bio_updates['bio_age_5_17'] / bio_updates['Total_Updates']) * 100

# --- STATISTICAL ANALYSIS (Normalized) ---
print("\n=== STATISTICAL ANALYSIS: MANDATORY UPDATE COMPLIANCE (%) ===")
stats = bio_updates['Mandatory_Compliance_Pct'].dropna()

print(f"Mean Compliance:       {stats.mean():.2f}%")
print(f"Median Compliance:     {stats.median():.2f}%")
print(f"Std Deviation:         {stats.std():.2f}")
print(f"Min Compliance:        {stats.min():.2f}%")
print(f"Max Compliance:        {stats.max():.2f}%")

print("\n--- TOP 5 STATES (Highest Focus on Mandatory Updates) ---")
print(bio_updates[['state', 'Mandatory_Compliance_Pct', 'Total_Updates']].sort_values('Mandatory_Compliance_Pct', ascending=False).head(5).to_string(index=False))

print("\n--- BOTTOM 5 STATES (Lowest Focus on Mandatory Updates) ---")
print(bio_updates[['state', 'Mandatory_Compliance_Pct', 'Total_Updates']].sort_values('Mandatory_Compliance_Pct', ascending=True).head(5).to_string(index=False))
print("\nGenerating visual...")

# --- PLOTTING (Normalized) ---
fig = px.bar(
    bio_updates.sort_values('Mandatory_Compliance_Pct', ascending=False), 
    x='state', 
    y='Mandatory_Compliance_Pct',
    title='Mandatory Biometric Compliance (Normalized): Share of Age 5-17 Updates',
    labels={'Mandatory_Compliance_Pct': 'Mandatory Updates Share (%)'},
    color='Mandatory_Compliance_Pct', 
    color_continuous_scale='Viridis',
    template='plotly_dark'
)
fig.add_hline(y=stats.mean(), line_dash="dash", line_color="white", annotation_text="National Average")
fig.show()