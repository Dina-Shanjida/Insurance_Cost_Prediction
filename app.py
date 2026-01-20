import gradio as gr
import pandas as pd
import pickle 
import numpy as np


with open("insurance_rf_pipeline.pkl", "rb") as file:
    model = pickle.load(file)


def predict_insurance(age , sex , bmi , children , smoker , region):

    input_df = pd.DataFrame([[
        age , sex , bmi , children , smoker , region
    ]],columns= ['age', 'sex', 'bmi', 'children', 'smoker', 'region']

    
    )

    prediction = model.predict(input_df)[0]

    return f"Predicted insurance amount: {prediction}"


inputs = [
    gr.Number(label= "Age"),
    gr.Radio(["M" , "F"], label="Gender"),
    gr.Number(label="BMI"),
    gr.Number(label="No of children"),
    gr.Radio(["yes", 'no'], label= "Smoker"),
    gr.Dropdown(["southwest", "southeast", "northwest", "northeast"], label= "Region")
]

app = gr.Interface(
    fn = predict_insurance,
    inputs= inputs,
    outputs= "text",
    title= "Insurance Amount Prediction"
)

app.launch(share= True)

