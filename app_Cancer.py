import pandas as pd
import numpy as np
import gradio as gr
import shap
import matplotlib.pyplot as plt
import plotly.express as px

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler

# Load and preprocess data
df = pd.read_csv('/Users/adityaatul/Downloads/data.csv')
X = df.drop(['id', 'Unnamed: 32', 'diagnosis'], axis=1)
y = df['diagnosis']

# Scale features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Train model
model = LogisticRegression(solver='liblinear')
model.fit(X_scaled, y)

# SHAP Explainer
explainer = shap.Explainer(model, X_scaled)

# Feature list
input_labels = X.columns.tolist()

# Gradio predict function
def predict_cancer(*inputs):
    input_array = np.array(inputs).reshape(1, -1)
    input_scaled = scaler.transform(input_array)

    pred = model.predict(input_scaled)[0]
    proba = model.predict_proba(input_scaled)[0]

    # SHAP Plot
    shap_values = explainer(input_scaled)
    plt.figure(figsize=(8, 4))
    shap.plots.bar(shap_values[0], show=False)
    shap_path = "shap_plot.png"
    plt.savefig(shap_path, bbox_inches='tight')
    plt.close()

    # Probability Bar
    prob_df = pd.DataFrame({
        'Diagnosis': model.classes_,
        'Probability': proba
    })
    prob_fig = px.bar(prob_df, x='Diagnosis', y='Probability', color='Diagnosis',
                      color_discrete_sequence=['green', 'red'],
                      title='Prediction Probability', range_y=[0, 1])

    return f"**🔬 Prediction: {pred}**", shap_path, prob_fig

# Use gr.Number instead of Slider
inputs = [
    gr.Number(label=label, value=val) for label, val in zip(input_labels, X.iloc[0])
]

# Gradio interface
gr.Interface(
    fn=predict_cancer,
    inputs=inputs,
    outputs=[
        gr.Markdown(),
        gr.Image(type="filepath", label="SHAP Feature Explanation"),
        gr.Plot(label="Prediction Confidence")
    ],
    title="🔬 Breast Cancer Detection using Logistic Regression",
    description=(
        "🧬 Based on input features like radius, perimeter, texture, and smoothness, "
        "this model determines whether the tumor is **Malignant** or **Benign**.\n"
        "The SHAP graph shows how each feature contributed to the prediction.\n\n"
        "---\n"
        "**Created by Aditya Atul** | Scientific ML Interface | Powered by SHAP, Scikit-learn, Gradio"
    ),
    examples=[X.iloc[i].tolist() for i in [0, 100, 200]]
).launch(server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", 7860))
        )
