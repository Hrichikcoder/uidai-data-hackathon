import pandas as pd
import plotly.express as px
import glob
import os
import numpy as np

def load_data(pattern):
    path = os.path.join('processed_data', pattern)
    files = glob.glob(path)
    return pd.concat((pd.read_csv(f) for f in files), ignore_index=True) if files else pd.DataFrame()

print("--- [4] LOADING DATA ---")
df_bio = load_data('*biometric*.csv')
df_enrol = load_data('*enrolment*.csv')

# --- PRE-PROCESSING ---

# DEFINING EXCLUSIONS: 
# Fix for "10k+" error: Exclude temporal (Year/Month) and structural (Codes) numbers 
# so they don't get added to the volume counts.
cols_to_drop = [
    'pincode', 'state_code', 'district_code', 'sub_district_code', 
    'year', 'month', 'day', 'date', 'age'
]

# Helper to filter columns case-insensitively
def get_clean_numeric_cols(df, drop_list):
    numerics = df.select_dtypes(include=np.number).columns
    # Drop columns if their lowercase name is in the drop list
    return [c for c in numerics if c.lower() not in drop_list]

# 1. Safety: Select only relevant metric columns
enrol_cols = get_clean_numeric_cols(df_enrol, cols_to_drop)
bio_cols = get_clean_numeric_cols(df_bio, cols_to_drop)

# 2. Aggregation: Sum by State
state_enrol = df_enrol.groupby('state')[enrol_cols].sum().sum(axis=1).rename('Total_Enrol')
state_bio = df_bio.groupby('state')[bio_cols].sum().sum(axis=1).rename('Total_Bio')

merged = pd.concat([state_enrol, state_bio], axis=1).dropna()
merged['State'] = merged.index

# 3. NORMALIZATION
total_india_enrol = merged['Total_Enrol'].sum()
total_india_bio = merged['Total_Bio'].sum()

merged['Growth_Share'] = (merged['Total_Enrol'] / total_india_enrol) * 100
merged['Maintenance_Share'] = (merged['Total_Bio'] / total_india_bio) * 100

# Handling zeros for Log Scale
merged = merged[merged['Growth_Share'] > 0]
merged = merged[merged['Maintenance_Share'] > 0]

# 4. QUADRANT LOGIC (Using Medians)
med_growth = merged['Growth_Share'].median()
med_maint = merged['Maintenance_Share'].median()

def get_quadrant(row):
    high_growth = row['Growth_Share'] >= med_growth
    high_maint = row['Maintenance_Share'] >= med_maint
    
    if high_growth and high_maint:
        return '1. Powerhouse (High Growth, High Updates)'
    elif not high_growth and high_maint:
        return '2. Mature (Low Growth, High Updates)'
    elif high_growth and not high_maint:
        return '4. Lagging (High Growth, Low Updates)'
    else:
        return '3. Inactive (Low Growth, Low Updates)'

merged['Quadrant'] = merged.apply(get_quadrant, axis=1)

print("\nGenerating visual...")

# --- PLOTTING ---
fig = px.scatter(
    merged, 
    x='Growth_Share', 
    y='Maintenance_Share', 
    color='Quadrant',
    size='Total_Enrol', 
    text='State', 
    hover_name='State',
    log_x=True,    
    log_y=True,    
    title="<b>State Maintenance Strategy Matrix (Log Scale)</b>",
    labels={
        'Growth_Share': 'Growth Share % (Log Scale)', 
        'Maintenance_Share': 'Maintenance Share % (Log Scale)'
    },
    template='plotly_dark',
    color_discrete_map={
        '1. Powerhouse (High Growth, High Updates)': '#00CC96', 
        '2. Mature (Low Growth, High Updates)': '#636EFA',      
        '3. Lagging (High Growth, Low Updates)': '#EF553B',     
        '4. Inactive (Low Growth, Low Updates)': '#AB63FA'      
    }
)

# --- QUADRANT LINES & ANNOTATIONS ---
fig.add_vline(x=med_growth, line_dash="dash", line_color="white", opacity=0.5)
fig.add_hline(y=med_maint, line_dash="dash", line_color="white", opacity=0.5)

max_x = merged['Growth_Share'].max()
max_y = merged['Maintenance_Share'].max()

fig.add_annotation(x=np.log10(max_x), y=np.log10(max_y), text="<b>POWERHOUSE</b>", showarrow=False, 
                   font=dict(color="#00CC96", size=14), xref="x", yref="y", xanchor="right", opacity=0.3)

fig.add_annotation(x=np.log10(max_x), y=np.log10(merged['Maintenance_Share'].min()), text="<b>RISK ZONE</b>", showarrow=False, 
                   font=dict(color="#EF553B", size=14), xref="x", yref="y", xanchor="right", yanchor="bottom", opacity=0.5)

fig.update_traces(
    textposition='top center', 
    textfont_size=10
)

fig.update_layout(
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(t=80, l=50, r=50, b=50)
)

fig.show()

# --- [5] STATISTICAL ANALYSIS PRINT ---
print("\n--- [5] STATISTICAL ANALYSIS ---")

# 1. Correlation
correlation = merged[['Growth_Share', 'Maintenance_Share']].corr(method='pearson').iloc[0, 1]
print(f"Correlation (Growth vs Maintenance): {correlation:.4f}")
if correlation > 0.8:
    print(">> Insight: Strong linear link. High growth = High maintenance.")
elif correlation > 0.5:
    print(">> Insight: Moderate link. Growth drives maintenance, but exceptions exist.")
else:
    print(">> Insight: Weak link. Maintenance behavior is independent of growth.")

# 2. Quadrant Summary
print("\n--- Quadrant Distribution ---")
quad_summary = merged.groupby('Quadrant').agg(
    Count=('State', 'count'),
    Avg_Growth=('Growth_Share', 'mean'),
    Avg_Maint=('Maintenance_Share', 'mean')
).sort_values('Count', ascending=False)
print(quad_summary)

# 3. Maintenance Ratio Anomalies
merged['Maint_Growth_Ratio'] = merged['Maintenance_Share'] / merged['Growth_Share']

print("\n--- Top 5 'Maintenance-Heavy' States (Ratio > 1) ---")
print(merged.nlargest(5, 'Maint_Growth_Ratio')[['State', 'Maint_Growth_Ratio']].to_string(index=False))

print("\n--- Top 5 'Growth-Risk' States (Ratio < 1) ---")
print(merged.nsmallest(5, 'Maint_Growth_Ratio')[['State', 'Maint_Growth_Ratio']].to_string(index=False))