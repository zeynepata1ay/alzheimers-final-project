import pandas as pd
import altair as alt

df = pd.read_csv("alzheimers_prediction_dataset.csv")
df_78plus = df[df["Age"] >= 78]

subset = df_78plus[["Alzheimer’s Diagnosis", "Family History of Alzheimer’s"]].copy()
counts = subset.value_counts().reset_index(name="Count")
counts.columns = ["Alzheimer’s Diagnosis", "Family History", "Count"]

chart = alt.Chart(counts).mark_bar().encode(
    x=alt.X("Alzheimer’s Diagnosis", title="Alzheimer’s Diagnosis"),
    y=alt.Y("Count", stack="normalize", title="Percentage"),
    color=alt.Color("Family History", scale=alt.Scale(domain=["Yes", "No"], range=["#c39bd3", "#7f8c8d"])),
    tooltip=["Alzheimer’s Diagnosis", "Family History", "Count"]
).properties(
    width=400,
    height=300,
    title="Family History by Alzheimer’s Diagnosis (78+ Years Old)"
).interactive()

chart.save("family_history_chart.html")