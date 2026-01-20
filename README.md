# Insurance Cost Prediction

Predicting insurance costs using machine learning.

This project uses structured data to build a model that predicts the **insurance charges** of individuals based on factors like age, BMI, smoking status, region, etc.

Demo available on **Hugging Face Spaces**.

👉 **Hugging Face:** https://huggingface.co/spaces/dinasanjida/Insurance-Cost-Prediction  
👉 **Dataset:** https://www.kaggle.com/datasets/mirichoi0218/insurance

---

## Overview

The goal of this project is to build a machine learning pipeline that:

1. Loads the insurance dataset  
2. Preprocesses the data  
3. Trains a regression model  
4. Evaluates performance  
5. Saves the trained model  
6. Interfaces with a user application (web app / Space)

---

## Dataset Description

**Source:** Kaggle – Insurance dataset  
URL: https://www.kaggle.com/datasets/mirichoi0218/insurance

**Features:**

| Feature  | Description                                |
| -------- | ------------------------------------------|
| age      | Age of the policyholder                    |
| sex      | Gender                                    |
| bmi      | Body Mass Index                           |
| children | Number of children covered by insurance  |
| smoker   | Whether the person smokes (‘yes’/‘no’)   |
| region   | Residential area                          |
| charges  | Insurance cost (target variable)          |

---

## Project Structure

```

Insurance_Cost_Prediction/
│
├── data/                     # Dataset files
├── notebooks/                # Jupyter notebooks for exploration
├── src/                      # Source code scripts
│   ├── preprocessing.py      # Data preprocessing functions
│   ├── train.py              # Model training script
│   ├── evaluate.py           # Model evaluation script
│   └── predict.py            # Model inference script
├── model/                    # Saved trained model
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation

```
---

## Pipeline / Workflow

1. **Data Loading**  
   Load and explore the dataset.

2. **Preprocessing**  
   Encode categorical variables and scale features as needed.

3. **Model Training**  
   Train a regression model on the training data.

4. **Evaluation**  
   Evaluate model performance using metrics like RMSE, MAE, and R².

5. **Model Saving**  
   Save the trained model for later use.

6. **Deployment / Interface**  
   Integrate the model with a web interface (hosted on Hugging Face Spaces) for interactive predictions.

---

## Quick Start

```bash
# Clone the repository
git clone https://github.com/Dina-Shanjida/Insurance_Cost_Prediction.git
cd Insurance_Cost_Prediction

# Install dependencies
pip install -r requirements.txt

# Train the model
python src/train.py

# Evaluate the model
python src/evaluate.py

# Run the app (if available)
python app.py
