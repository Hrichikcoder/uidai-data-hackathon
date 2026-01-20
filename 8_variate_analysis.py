import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import glob
import os

def load_data(pattern):
    path = os.path.join('processed_data', pattern)
    files = glob.glob(path)
    
    if not files:
        return pd.DataFrame()
    
    df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    return df

df_enrol = load_data('*enrolment*.csv')
df_bio = load_data('*biometric*.csv')
df_demo = load_data('*demographic*.csv')

datasets = [
    ("Enrollment", df_enrol),
    ("Demographic", df_demo),
    ("Biometric", df_bio)
]

for name, df in datasets:
    if df.empty:
        continue
    
    top_states = df['state'].value_counts().head(10).reset_index()
    top_states.columns = ['state', 'count']
    
    fig = px.bar(top_states, x='state', y='count',
                 title=f"Top 10 Frequency: state ({name})",
                 labels={'count': 'Count', 'state': 'state'},
                 template='plotly_dark')
    
    fig.show()

if not df_enrol.empty and not df_bio.empty and not df_demo.empty:
    enrol_cols = ['age_0_5', 'age_5_17', 'age_18_greater']
    bio_cols = ['bio_age_5_17', 'bio_age_17_']
    demo_cols = ['demo_age_5_17', 'demo_age_17_']
    
    s_enrol = df_enrol.groupby('state')[enrol_cols].sum().sum(axis=1)
    s_bio = df_bio.groupby('state')[bio_cols].sum().sum(axis=1)
    s_demo = df_demo.groupby('state')[demo_cols].sum().sum(axis=1)
    
    merged = pd.concat([
        s_enrol.rename('Enrol'),
        s_bio.rename('Bio'),
        s_demo.rename('Demo')
    ], axis=1).dropna()
    
    merged['State'] = merged.index
    
    s_enrol_norm = (merged['Enrol'] - merged['Enrol'].min()) / (merged['Enrol'].max() - merged['Enrol'].min())
    s_bio_norm = (merged['Bio'] - merged['Bio'].min()) / (merged['Bio'].max() - merged['Bio'].min())
    
    bivariate_df = pd.DataFrame({
        'Enrolment Score': s_enrol_norm,
        'Biometric Score': s_bio_norm,
        'State': merged['State']
    })
    
    fig_bi = px.scatter(bivariate_df, x='Enrolment Score', y='Biometric Score',
                        trendline='ols',
                        text='State',
                        title="Bivariate: Enrolment vs Biometric (Normalized 0-1)",
                        labels={'Enrolment Score': 'Enrolment Score (0=Low, 1=High)', 
                                'Biometric Score': 'Biometric Score (0=Low, 1=High)'},
                        template='plotly_dark')
    
    fig_bi.update_traces(textposition='top center')
    fig_bi.show()

    merged['Log_Enrol'] = np.log1p(merged['Enrol'])
    merged['Log_Bio'] = np.log1p(merged['Bio'])
    merged['Log_Demo'] = np.log1p(merged['Demo'])

    fig_tri = px.scatter_3d(merged, x='Log_Enrol', y='Log_Bio', z='Log_Demo',
                            color='Demo', size='Bio', 
                            hover_name='State',
                            title="Trivariate Operational Synergy (Log Scale)",
                            template='plotly_dark')
    fig_tri.show()

    merged['Child_Share_Pct'] = (df_enrol.groupby('state')['age_0_5'].sum() / s_enrol) * 100
    merged['Adult_Share_Pct'] = (df_enrol.groupby('state')['age_18_greater'].sum() / s_enrol) * 100
    
    df_multi = merged.sort_values('Enrol', ascending=False).head(15).reset_index(drop=True)
    df_multi['State_ID'] = df_multi.index 

    fig_multi = go.Figure(data=
        go.Parcoords(
            line = dict(
                color = df_multi['State_ID'],
                colorscale = 'Turbo',
                showscale = True,
                cmin = 0,
                cmax = len(df_multi) - 1,
                colorbar = dict(
                    tickvals = df_multi['State_ID'],
                    ticktext = df_multi['State'],
                    title = 'State'
                )
            ),
            dimensions = list([
                dict(label = 'Enrolment', values = df_multi['Enrol']),
                dict(label = 'Biometric', values = df_multi['Bio']),
                dict(label = 'Demographic', values = df_multi['Demo']),
                dict(label = 'Child Share %', values = df_multi['Child_Share_Pct']),
                dict(label = 'Adult Share %', values = df_multi['Adult_Share_Pct']),
                dict(
                    range = [0, len(df_multi)-1],
                    label = 'State Name', 
                    values = df_multi['State_ID'],
                    tickvals = df_multi['State_ID'],
                    ticktext = df_multi['State']
                )
            ])
        )
    )
    fig_multi.update_layout(
        title="Multidimensional State Trajectory (Top 15 States)",
        template='plotly_dark'
    )
    fig_multi.show()