import matplotlib.pyplot as plt
import pandas as pd
import altair as alt
import plotly.graph_objects as go

df = pd.read_csv('/Users/emilymoy/Desktop/DS4200/final proj 4200/alzheimers_prediction_dataset.csv')

df_under78 = df[df['Age'] < 78]
df_under78['Age Group'] = '<78'

df_above78 = df[df['Age'] >= 78]
df_above78['Age Group'] = '78+'

df_all = df.copy()
df_all['Age Group'] = 'All Ages'

df_combined = pd.concat([df_all, df_under78, df_above78])

subset = df_combined[['Alzheimer’s Diagnosis', 'Genetic Risk Factor (APOE-ε4 allele)', 'Age Group']]
counts = subset.value_counts().reset_index(name='counts')
counts.columns = ['Alzheimer’s Diagnosis', 'Genetic Risk Factor', 'Age Group', 'counts']

age_filter = alt.param(
    name='age_group',
    bind=alt.binding_radio(options=['All Ages', '<78', '78+'], name='Age Group: '),
    value='All Ages'
)

chart = alt.Chart(counts).mark_bar().encode(
    x=alt.X('Alzheimer’s Diagnosis'),
    y=alt.Y('counts', stack='normalize'),
    color=alt.Color('Genetic Risk Factor',
                    scale=alt.Scale(domain=['Yes', 'No'], range=['#d35400', '#f7dc6f'])),
    tooltip=['Alzheimer’s Diagnosis', 'Genetic Risk Factor', 'counts']
).add_params(
    age_filter
).transform_filter(
    alt.datum['Age Group'] == age_filter
).properties(
    width=400,
    height=300,
    title='Genetic Risk Factor by Alzheimer’s Diagnosis'
).interactive()

chart.save('genetic_risk_ages_chart.html')