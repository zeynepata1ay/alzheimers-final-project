import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import base64
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# Load data
df_raw = pd.read_csv('alzheimers_prediction_dataset.csv')

# Optional: reduce size for faster testing
# df_raw = df_raw[df_raw['Age'] >= 78].sample(n=3000, random_state=42)
df_raw = df_raw[df_raw['Age'] >= 78]  # Focus on high-risk group

x_feat_list = [
    'Age',
    'Gender',
    'Education Level',
    'BMI',
    'Physical Activity Level',
    'Smoking Status',
    'Alcohol Consumption',
    'Diabetes',
    'Family History of Alzheimer’s',
    'Depression Level',
    'Sleep Quality',
    'Dietary Habits',
    'Air Pollution Exposure',
    'Employment Status',
    'Marital Status',
    'Genetic Risk Factor (APOE-ε4 allele)',
    'Social Engagement Level',
    'Income Level',
]

x_df = df_raw.loc[:, x_feat_list]

# Encode categorical features
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

# Encode the target variable
target_mapping = {'No': 0, 'Yes': 1}
df_raw['Alzheimer’s Diagnosis'] = df_raw['Alzheimer’s Diagnosis'].map(target_mapping).fillna(-1)
y = df_raw['Alzheimer’s Diagnosis']

if y.dtype == 'object':
    le = LabelEncoder()
    y = le.fit_transform(y)

# Train RandomForest (optimized for performance)
rfr = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
rfr.fit(x, y)

# Plot function
def plot_feat_import(feat_list, feat_import, sort=True, limit=None):
    if sort:
        idx = np.argsort(feat_import)[::-1]
        feat_list = [feat_list[_idx] for _idx in idx]
        feat_import = feat_import[idx]

    if limit is not None:
        feat_list = feat_list[:limit]
        feat_import = feat_import[:limit]

    plt.figure(figsize=(8, 6))
    plt.barh(feat_list, feat_import)
    plt.gca().invert_yaxis()
    plt.xlabel('Feature Importance (Mean MSE Reduction)')
    plt.title("RandomForestRegressor Alzheimer's Predictors")

    max_import = max(feat_import)
    offset = 0.05 * max_import
    for i, (feature, importance) in enumerate(zip(feat_list, feat_import)):
        plt.text(importance + offset, i, f"{importance:.4f}", va='center', fontsize=8)

    plt.tight_layout()
    plt.savefig('feature_importance.png')
    return plt

plt_obj = plot_feat_import(x_feat_list, rfr.feature_importances_)

with open('feature_importance.png', 'rb') as image_file:
    encoded_image = base64.b64encode(image_file.read()).decode()
