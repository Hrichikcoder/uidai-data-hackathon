import pandas as pd
import plotly.express as px
import glob
import os
import numpy as np

def load_data(pattern):
    path = os.path.join('processed_data', pattern)
    files = glob.glob(path)
    return pd.concat((pd.read_csv(f) for f in files), ignore_index=True) if files else pd.DataFrame()

print("--- [5] LOADING DATA ---")
df_enrol = load_data('*enrolment*.csv')
df_bio = load_data('*biometric*.csv')
df_demo = load_data('*demographic*.csv')

# --- PRE-PROCESSING ---
# 1. Enrolment Metrics
df_enrol['Total_Enrol'] = df_enrol['age_0_5'] + df_enrol['age_5_17'] + df_enrol['age_18_greater']
enrol_group = df_enrol.groupby('state')[['Total_Enrol', 'age_0_5']].sum().reset_index()

# NORMALIZE X: Child Enrolment Share (Quality of Growth)
enrol_group['Growth_Quality_Pct'] = (enrol_group['age_0_5'] / enrol_group['Total_Enrol']) * 100

# 2. Update Metrics
df_bio['Total_Bio'] = df_bio['bio_age_5_17'] + df_bio['bio_age_17_']
bio_sum = df_bio.groupby('state')['Total_Bio'].sum().reset_index()

df_demo['Total_Demo'] = df_demo['demo_age_5_17'] + df_demo['demo_age_17_']
demo_sum = df_demo.groupby('state')['Total_Demo'].sum().reset_index()

updates = pd.merge(bio_sum, demo_sum, on='state')
updates['Total_Updates'] = updates['Total_Bio'] + updates['Total_Demo']

# 3. Merge for Matrix
matrix = pd.merge(enrol_group, updates, on='state')

# NORMALIZE Y: Maintenance Intensity (Updates per Enrolment)
matrix['Maintenance_Intensity'] = matrix['Total_Updates'] / matrix['Total_Enrol']

# --- QUADRANT LOGIC ---
# Calculate Medians to divide the graph
med_quality = matrix['Growth_Quality_Pct'].median()
med_intensity = matrix['Maintenance_Intensity'].median()

def get_quadrant(row):
    high_quality = row['Growth_Quality_Pct'] >= med_quality
    high_intensity = row['Maintenance_Intensity'] >= med_intensity
    
    if high_quality and high_intensity:
        return '1. Sustainable (High Child Growth, High Updates)' # Ideal
    elif not high_quality and high_intensity:
        return '2. Legacy (Low Child Growth, High Updates)'      # Mature/Adult heavy
    elif high_quality and not high_intensity:
        return '3. Emerging (High Child Growth, Low Updates)'    # New Gen/Early stage
    else:
        return '4. Catch-Up (Low Child Growth, Low Updates)'     # Risk/Lagging

matrix['Zone'] = matrix.apply(get_quadrant, axis=1)

print("\nGenerating visual...")

# --- PLOTTING ---
fig = px.scatter(
    matrix, 
    x='Growth_Quality_Pct', 
    y='Maintenance_Intensity',
    text='state',
    size='Total_Enrol',  # Bubble size = Volume
    color='Zone',        # Color by Quadrant
    title='<b>Aadhaar Maturity Matrix</b><br>',
    labels={
        'Growth_Quality_Pct': 'Growth Quality (Child Enrolment %)', 
        'Maintenance_Intensity': 'Maintenance Intensity (Updates/Enrol)'
    },
    template='plotly_dark',
    color_discrete_map={
        '1. Sustainable (High Child Growth, High Updates)': '#00CC96', # Green
        '2. Legacy (Low Child Growth, High Updates)': '#636EFA',      # Blue
        '3. Emerging (High Child Growth, Low Updates)': '#FFA15A',    # Orange
        '4. Catch-Up (Low Child Growth, Low Updates)': '#EF553B'      # Red
    }
)

# --- ANNOTATIONS & GUIDES ---
# Add Median Lines
fig.add_vline(x=med_quality, line_dash="dash", line_color="white", opacity=0.3)
fig.add_hline(y=med_intensity, line_dash="dash", line_color="white", opacity=0.3)

# Add Corner Labels for easier reading
max_x, max_y = matrix['Growth_Quality_Pct'].max(), matrix['Maintenance_Intensity'].max()
min_x = matrix['Growth_Quality_Pct'].min()

# Label: Sustainable (Top Right)
fig.add_annotation(x=max_x, y=max_y, text="<b>SUSTAINABLE</b><br>(Organic Growth + High Upkeep)", 
                   showarrow=False, font=dict(color="#00CC96", size=12), xanchor="right", yanchor="top")

# Label: Legacy (Top Left)
fig.add_annotation(x=min_x, y=max_y, text="<b>LEGACY</b><br>(Adult Focus + High Upkeep)", 
                   showarrow=False, font=dict(color="#636EFA", size=12), xanchor="left", yanchor="top")

# Label: Catch-Up (Bottom Left)
fig.add_annotation(x=min_x, y=0, text="<b>CATCH-UP</b><br>(Adult Focus + Low Upkeep)", 
                   showarrow=False, font=dict(color="#EF553B", size=12), xanchor="left", yanchor="bottom")

# Styling
fig.update_traces(textposition='top center', textfont_size=10)
fig.update_layout(
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(t=80, l=50, r=50, b=50)
)

fig.show()
print("\n=== AADHAAR MATURITY STATISTICS ===")

print(f"Total States Analyzed: {len(matrix)}")
print(f"Median Growth Quality (Child %): {matrix['Growth_Quality_Pct'].median():.2f}%")
print(f"Median Maintenance Intensity:    {matrix['Maintenance_Intensity'].median():.4f}")

print("\n--- Zone Distribution ---")
print(matrix['Zone'].value_counts().to_string())

print("\n--- Top 5 Sustainable States (High Vol) ---")
sustainable_states = matrix[matrix['Zone'].str.contains('Sustainable')]
if not sustainable_states.empty:
    print(sustainable_states.nlargest(5, 'Total_Enrol')[['state', 'Growth_Quality_Pct', 'Maintenance_Intensity']].to_string(index=False))
else:
    print("No states found in Sustainable zone.")

print("\n--- Top 5 Highest Maintenance Intensity (Updates/Enrol) ---")
print(matrix.nlargest(5, 'Maintenance_Intensity')[['state', 'Maintenance_Intensity', 'Zone']].to_string(index=False))

print("\n--- Top 5 Highest Child Growth Share (Age 0-5 %) ---")
print(matrix.nlargest(5, 'Growth_Quality_Pct')[['state', 'Growth_Quality_Pct', 'Zone']].to_string(index=False))