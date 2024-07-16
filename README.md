# Predicting Miscarriage in Women using Big Data & IoT

## Introduction

Miscarriage Prediction Model is a Python-based application designed to provide expecting parents and healthcare professionals with an early assessment of the risk of miscarriage. By machine learning algorithms, the model analyzes various input parameters related to maternal health, medical history, and lifestyle factors to generate probabilistic predictions.

## Dependencies

- Python (>=3.6)
- scikit-learn
- pandas
- numpy

## About Project

<p> The machine learning algorithm used in the provided code is XGBoost (Extreme Gradient Boosting). XGBoost is a popular and powerful gradient boosting algorithm that is commonly used for classification and regression tasks. It works by building a series of decision trees sequentially, where each subsequent tree corrects the errors made by the previous ones.

In the provided code, the XGBClassifier class from the XGBoost library is used to train a classification model to predict whether a miscarriage occurred or not based on the input features such as age, BMI, number of previous miscarriages, activity level, location, body temperature, heart rate, stress level, and blood pressure level.

<i>Here's a brief overview of the key steps involved in the code: </i> <br>
<b>Loading the Data: </b> The data is loaded from a CSV file using Pandas. <br>
<b>Data Preprocessing: </b> The data is split into features (X) and the target variable (y). Categorical variables are encoded using label encoding.<br>
<b>Model Training: </b> The XGBoost classifier is trained on the training data.<br>
<b>Streamlit App: </b> The Streamlit application is created to take user input for various health parameters and predict whether a miscarriage is likely to occur based on the trained model.<br>
<b>Prediction and Output: </b>The model predicts the probability of miscarriage based on the user input and displays the result along with suggestions for improving health. <br>Additionally, the feature importance is visualized to show which features have the most significant impact on the prediction. </p>

 Clone this repository to your local machine:
   ```bash
   git clone https://github.com/git-sumana/miscarriage-prediction.git
