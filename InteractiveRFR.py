import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import base64
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

df_raw = pd.read_csv('alzheimers_prediction_dataset.csv')
df_raw = df_raw[df_raw['Country'] == 'USA']
df_raw = df_raw[df_raw['Age'] >= 78]  # Filter for Age >= 75

x_feat_list = [
    'Age',                            # Numerical
    'Gender',                         # Categorical (e.g., Male/Female)
    'Education Level',                # Numerical
    'BMI',                            # Numerical
    'Physical Activity Level',        # Categorical (e.g., Low/Medium/High)
    'Smoking Status',                 # Categorical (e.g., Yes/No/Former)
    'Alcohol Consumption',            # Categorical (e.g., Low/Medium/High)
    'Diabetes',                       # Categorical (Yes/No)
    'Family History of Alzheimer’s',  # Categorical (Yes/No)
    'Depression Level',               # Categorical (e.g., Low/Medium/High)
    'Sleep Quality',                  # Categorical (e.g., Poor/Fair/Good)
    'Dietary Habits',                 # Categorical (e.g., Poor/Fair/Good)
    'Air Pollution Exposure',         # Categorical (e.g., Low/Medium/High)
    'Employment Status',              # Categorical (e.g., Employed/Unemployed/Retired)
    'Marital Status',                 # Categorical (e.g., Single/Married/Divorced)
    'Genetic Risk Factor (APOE-ε4 allele)',  # Categorical (Yes/No)
    'Social Engagement Level',        # Categorical (e.g., Low/Medium/High)
    'Income Level',                   # Categorical (e.g., Low/Medium/High)
]

x_df = df_raw.loc[:, x_feat_list]

mapping_dict = {
    'Gender': {'Male': 0, 'Female': 1},
    'Physical Activity Level': {'Low': 0, 'Medium': 1, 'High': 2},
    'Smoking Status': {'Never': 0, 'Former': 0, 'Current': 1},
    'Alcohol Consumption': {'Never': 0, 'Occasionally': 1, 'Regularly': 2},
    'Diabetes': {'No': 0, 'Yes': 1},
    'Family History of Alzheimer’s': {'No': 0, 'Yes': 1},
    'Depression Level': {'Low': 0, 'Medium': 1, 'High': 2},
    'Sleep Quality': {'Poor': 0, 'Average': 1, 'Good': 2},
    'Dietary Habits': {'Unhealthy': 0, 'Average': 1, 'Healthy': 2},
    'Air Pollution Exposure': {'Low': 0, 'Medium': 1, 'High': 2},
    'Employment Status': {'Unemployed': 0, 'Employed': 1, 'Retired': 2},
    'Marital Status': {'Single': 0, 'Married': 1, 'Widowed': 2},
    'Genetic Risk Factor (APOE-ε4 allele)': {'No': 0, 'Yes': 1},
    'Social Engagement Level': {'Low': 0, 'Medium': 1, 'High': 2},
    'Income Level': {'Low': 0, 'Medium': 1, 'High': 2},
    'Stress Levels': {'Low': 0, 'Medium': 1, 'High': 2},
}

for col, mapping in mapping_dict.items():
    if col in x_df.columns:
        x_df[col] = x_df[col].map(mapping)

x = x_df.values

target_df = df_raw[['Alzheimer’s Diagnosis']]
target_mapping = {'No': 0, 'Yes': 1}
target_df['Alzheimer’s Diagnosis'] = target_df['Alzheimer’s Diagnosis'].map(target_mapping).fillna(-1)

y = df_raw['Alzheimer’s Diagnosis']
if y.dtype == 'object':
    le = LabelEncoder()
    y = le.fit_transform(y)

rfr = RandomForestRegressor()
rfr.fit(x, y)


Dict_Of_Charts = {
    "Nature vs Nature": {
        "Nature": ['Age', 'Gender', 'Genetic Risk Factor (APOE-ε4 allele)', 'Family History of Alzheimer’s', 'Diabetes'],
        "Nurture": ['Physical Activity Level', 'Smoking Status', 'Alcohol Consumption', 'Sleep Quality', 'Dietary Habits', 'Employment Status', 'Marital Status', 'Social Engagement Level', 'Income Level', 'Depression Level', 'Air Pollution Exposure', 'BMI', 'Education Level']
    },
    "Impact of Life Choices": {
        "Yes": ['Physical Activity Level', 'Smoking Status', 'Alcohol Consumption', 'Dietary Habits', 'Social Engagement Level', 'BMI', 'Education Level'],
        "Debatable": ['Sleep Quality', 'Employment Status', 'Income Level'],
        "No": ['Age', 'Gender', 'Genetic Risk Factor (APOE-ε4 allele)', 'Family History of Alzheimer’s', 'Diabetes', 'Air Pollution Exposure', 'Depression Level', 'Marital Status']
    },
    "Importance of Taking Care of Your Body": {
        "Body Factors": ['Physical Activity Level', 'Smoking Status', 'Alcohol Consumption', 'Depression Level', 'Sleep Quality', 'Dietary Habits', 'BMI'],
        "Other": ['Gender', 'Education Level', 'Age', 'Diabetes', 'Family History of Alzheimer’s', 'Air Pollution Exposure', 'Employment Status', 'Marital Status', 'Genetic Risk Factor (APOE-ε4 allele)', 'Social Engagement Level', 'Income Level']
    }
}
# [Previous imports and code up to the Plotly section remain unchanged]

# Import Plotly for interactive visualization
import plotly.graph_objects as go

# Map feature names to their importances
feat_import_dict = dict(zip(x_feat_list, rfr.feature_importances_))

# Compute grouped importances based on Dict_Of_Charts, scaled by 100
grouped_importances = {}
for main_category, subcategories in Dict_Of_Charts.items():
    grouped_importances[main_category] = {}
    for subcategory, features in subcategories.items():
        # Sum importances and scale by 100
        sum_importance = 100 * sum(feat_import_dict.get(feature, 0) for feature in features)
        grouped_importances[main_category][subcategory] = sum_importance

# Define unique subcategories and assign colors
unique_subcategories = list(set(
    subcategory for subcategories in Dict_Of_Charts.values() for subcategory in subcategories
))
# Use Plotly's qualitative color palette (adjust as needed)
colors = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
]  # Extend if more than 10 subcategories
subcategory_colors = {subcat: colors[i % len(colors)] for i, subcat in enumerate(unique_subcategories)}

# Create the Plotly figure
fig = go.Figure()

# Initial data (first main category)
initial_main_category = list(Dict_Of_Charts.keys())[0]
initial_subcategories = list(grouped_importances[initial_main_category].keys())
initial_importances = list(grouped_importances[initial_main_category].values())
initial_hover = [
    f"Summed Importance: {imp:.2f}%<br>Features: {', '.join(Dict_Of_Charts[initial_main_category][subcat])}"
    for subcat, imp in zip(initial_subcategories, initial_importances)
]
initial_colors = [subcategory_colors[subcat] for subcat in initial_subcategories]

# Add a single trace with initial data
fig.add_trace(
    go.Bar(
        x=initial_subcategories,
        y=initial_importances,
        hovertext=initial_hover,
        hoverinfo="text",
        marker=dict(color=initial_colors),  # Set initial colors
        showlegend=False
    )
)

# Create dropdown buttons for each main category
buttons = []
for main_category in Dict_Of_Charts:
    subcategories = list(grouped_importances[main_category].keys())
    importances = list(grouped_importances[main_category].values())
    hover_texts = [
        f"Summed Importance: {imp:.2f}%<br>Features: {', '.join(Dict_Of_Charts[main_category][subcat])}"
        for subcat, imp in zip(subcategories, importances)
    ]
    colors = [subcategory_colors[subcat] for subcat in subcategories]
    button = dict(
        label=main_category,
        method="update",
        args=[
            {
                "x": [subcategories],
                "y": [importances],
                "hovertext": [hover_texts],
                "marker.color": [colors]  # Update colors dynamically
            },
            {
                "title": f"Grouped RFR Factor Importance for: {main_category}",
                "xaxis": {"title": ""}
            }
        ]
    )
    buttons.append(button)

# Update the layout with the dropdown menu
fig.update_layout(
    updatemenus=[
        dict(
            buttons=buttons,
            direction="down",
            showactive=True,
            x=0.1,
            xanchor="right",
            y=1.1,
            yanchor="top"
        )
    ],
    title=f"Totaled RFR Percentage Importances for: {list(Dict_Of_Charts.keys())[0]}",
    xaxis_title="",
    yaxis_title="Total Importance (%)",
    showlegend=False  # No legend needed with single trace
)

# Save the interactive chart to an HTML file
with open('interactive_chart.html', 'w') as f:
    f.write(fig.to_html(full_html=True, include_plotlyjs='cdn'))
