import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import base64
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

df_raw = pd.read_csv('alzheimers_prediction_dataset.csv')
#df_raw = df_raw[df_raw['Country'] == 'USA']
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

# Modified plot_feat_import function with annotations
def plot_feat_import(feat_list, feat_import, sort=True, limit=None):
    if sort:
        # Sort in descending order so most important features are first
        idx = np.argsort(feat_import)[::-1]
        feat_list = [feat_list[_idx] for _idx in idx]
        feat_import = feat_import[idx]

    if limit is not None:
        # Take the top 'limit' features (most important due to descending sort)
        feat_list = feat_list[:limit]
        feat_import = feat_import[:limit]

    plt.figure(figsize=(8, 6))
    plt.barh(feat_list, feat_import)
    plt.gca().invert_yaxis()  # Invert y-axis to place most important at top
    plt.xlabel('Feature Importance (Mean MSE Reduction)')
    plt.title('RandomForestRegressor Alzheimers Features Compared')

    # Add annotations for each bar
    max_import = max(feat_import)
    offset = 0.05 * max_import  # Small offset proportional to max importance
    for i, (feature, importance) in enumerate(zip(feat_list, feat_import)):
        plt.text(importance + offset, i, f"{importance:.4f}", va='center', fontsize=8)

    plt.tight_layout()
    plt.savefig('feature_importance.png')
    return plt