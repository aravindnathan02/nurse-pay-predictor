# Nurse Pay Rate Prediction System

[**Streamlit Link**](https://nurse-pay-predictor.streamlit.app/)

## Overview

A machine learning-powered Streamlit application that predicts hourly pay rates for healthcare professionals using synthetic data and multiple predictive models.

## Features

- Pay rate predictions for various nursing and healthcare roles
- Market analysis of pay rates across different locations
- Historical trend visualization
- Machine learning models:
  - Random Forest (not loaded due to size limitations)
  - XGBoost (best performing)
  - LSTM Time Series Prediction



## Prerequisites

- Python 3.9+
- Libraries in requirements.txt

## Installation

> git clone https://github.com/yourusername/nurse-pay-prediction.git \
> cd nurse-pay-prediction \
> pip install -r requirements.txt

## Running the Application

1. Train models and generate data:
> python pre_deployment.py

2. Launch Streamlit app:
> streamlit run streamlit_app.py

## Model Training

The application uses:
- Synthetic data generation
- Feature preprocessing
- Multiple regression models
- Time series analysis

## Deployment

Deployed on Streamlit Community Cloud

## Technologies

- Python
- Streamlit
- Scikit-learn
- XGBoost
- TensorFlow
- Pandas
- Matplotlib

## Contributors
Aravind Vaithianathan (aravindnathan02)
