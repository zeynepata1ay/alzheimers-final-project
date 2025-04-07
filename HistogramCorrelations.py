import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Correlation values provided
correlation_values = [
    -0.008328281051729740, 0.0017340694944942100, 0.0030336088256354200, 
    -0.005524099282097010, -0.007282501866280190, 0.0018583647140248900, 
    0.005382523706367580, 0.009737586361950930, 0.0026313867823815800, 
    -0.010124941837901700, 0.1571097540100460, 0.0000732217316612, 
    -0.00688007160246397, -0.008988978185730870, -0.005762616502633710, 
    0.002960642368298270, 0.004515050627882150, 0.1968698914641000, 
    -0.003956458542691580, 0.0004527511129614070, 0.0021469330174309800, 
    0.0006500627052818190
]

# Columns (features) corresponding to the correlation values
columns = [
    "Age", "Gender", "Education Level", "BMI", "Physical Activity Level", 
    "Smoking Status", "Alcohol Consumption", "Diabetes", "Hypertension", 
    "Cholesterol Level", "Family History of Alzheimer’s", "Depression Level", 
    "Sleep Quality", "Dietary Habits", "Air Pollution Exposure", 
    "Employment Status", "Marital Status", "Genetic Risk Factor (APOE-ε4 allele)", 
    "Social Engagement Level", "Income Level", "Stress Levels", 
    "Urban vs Rural Living"
]

# Corrected title
title = "Alzheimer’s Correlation Values, Post 78 Years of Age"

# Create the bar plot
plt.figure(figsize=(12, 6))
sns.barplot(x=columns, y=correlation_values, color='skyblue')
plt.title(title)
plt.xlabel('Features')
plt.ylabel('Correlation with Alzheimer’s Diagnosis')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for readability
plt.grid(True, axis='y', alpha=0.3)  # Add horizontal grid lines
plt.tight_layout()  # Adjust layout to prevent label cutoff
plt.savefig('correlation_plot.png', dpi=300)