import pandas as pd
import plotly.express as px
import glob
import os
import numpy as np

def load_data(pattern):
    path = os.path.join('processed_data', pattern)
    files = glob.glob(path)
    return pd.concat((pd.read_csv(f) for f in files), ignore_index=True) if files else pd.DataFrame()

print("--- [7] LOADING DATA ---")
df_bio = load_data('*biometric*.csv')
df_demo = load_data('*demographic*.csv')

df_bio['Total_Bio'] = df_bio['bio_age_5_17'] + df_bio['bio_age_17_']
bio_sum = df_bio.groupby('state')['Total_Bio'].sum().reset_index()

df_demo['Total_Demo'] = df_demo['demo_age_5_17'] + df_demo['demo_age_17_']
demo_sum = df_demo.groupby('state')['Total_Demo'].sum().reset_index()

churn = pd.merge(bio_sum, demo_sum, on='state')

churn['Stability_Ratio'] = churn['Total_Bio'] / churn['Total_Demo']

print("\nGenerating visual...")

fig = px.bar(
    churn.sort_values('Stability_Ratio'), 
    x='state', 
    y='Stability_Ratio',
    title='<b>Demographic Churn Index</b><br>(High Ratio = Stable | Low Ratio = High Migration)',
    template='plotly_dark',
    labels={'Stability_Ratio': 'Stability Ratio (Bio / Demo)'}
)

fig.add_hline(y=1, line_dash="dash", line_color="white", opacity=0.5, 
              annotation_text="High Churn Threshold (Ratio < 1)", 
              annotation_position="bottom right")

fig.show()

print("\n=== MIGRATION & CHURN STATISTICS ===")

avg_ratio = churn['Stability_Ratio'].mean()
median_ratio = churn['Stability_Ratio'].median()

print(f"Total States:        {len(churn)}")
print(f"National Avg Ratio:  {avg_ratio:.2f}")
print(f"Median Ratio:        {median_ratio:.2f}")

high_churn = churn[churn['Stability_Ratio'] < 1.0].sort_values('Stability_Ratio')

print(f"\n--- [RISK] High Churn / Migration States (Ratio < 1.0) ---")
if not high_churn.empty:
    print(high_churn[['state', 'Stability_Ratio']].to_string(index=False))
else:
    print("None. All states have Ratio > 1.0 (Stable).")

high_stability = churn.nlargest(5, 'Stability_Ratio')

print(f"\n--- [STABLE] Top 5 Most Stable States (Bio Heavy) ---")
print(high_stability[['state', 'Stability_Ratio']].to_string(index=False))

std_dev = churn['Stability_Ratio'].std()
outlier_threshold = avg_ratio + (2 * std_dev)
outliers = churn[churn['Stability_Ratio'] > outlier_threshold]

if not outliers.empty:
    print(f"\n--- [ANOMALY] Statistical Outliers (Ratio > {outlier_threshold:.2f}) ---")
    print(outliers[['state', 'Stability_Ratio']].to_string(index=False))