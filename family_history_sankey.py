import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import altair as alt
import plotly.graph_objects as go

df = pd.read_csv('alzheimers_prediction_dataset.csv')

df_over78 = df[df["Age"] >= 78]

subset_over78 = df_over78[["Alzheimer’s Diagnosis", "Family History of Alzheimer’s"]].copy()

counts_values = subset_over78.value_counts().reset_index(name="counts")

labels = ['Alzheimer’s: Yes', 'Alzheimer’s: No', 'Family History: Yes', 'Family History: No']
label_to_index = {label: i for i, label in enumerate(labels)}

fig = go.Figure(data=[go.Sankey(
    node=dict(
        pad=15,
        thickness=20,
        line=dict(color='black', width=0.5),
        label=labels
    ),
    link=dict(
        source=counts_values['Alzheimer’s Diagnosis'].map(lambda x: label_to_index[f'Alzheimer’s: {x}']),
        target=counts_values['Family History of Alzheimer’s'].map(lambda x: label_to_index[f'Family History: {x}']),
        value=counts_values['counts']
    )
)])

fig.update_layout(title_text='Family History vs Alzheimer’s Diagnosis')
fig.with_html('family_history_sankey.html')
