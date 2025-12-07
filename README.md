# Flood Probability Prediction System
A comprehensive machine learning web application built with Streamlit that predicts flood probability percentages by analyzing multiple environmental factors. Features interactive visualizations, batch predictions, and real-time risk assessment with model comparison dashboard.
# Key Features

Advanced ML Pipeline: Tested 8+ regression models including Random Forest, XGBoost, Gradient Boosting, SVR, and Neural Networks
High Accuracy: Best model achieves R² > 0.95 with low RMSE and MAE
Interactive Streamlit Dashboard: Professional UI with custom CSS styling
Dual Prediction Modes:

Single prediction with manual input
Batch prediction via CSV file upload


Real-time Risk Assessment: Automatic categorization into 4 risk levels (Low, Moderate, High, Critical)
Visual Analytics:

Interactive Plotly gauge meters
Risk distribution pie charts
Model comparison bar graphs


Model Comparison Dashboard: Compare R², RMSE, MAE, MAPE across all models
Export Functionality: Download batch prediction results as CSV
Actionable Recommendations: Specific guidance for each risk level

# Tech Stack
Frontend & UI:

Streamlit (Web framework)
Plotly (Interactive charts & gauges)
Custom CSS styling

# Machine Learning:

Python 3.14
Scikit-learn (Multiple regression models)
Pandas & NumPy (Data processing)
Pickle (Model serialization)

# Models Implemented:

Random Forest Regressor
Gradient Boosting Regressor
XGBoost Regressor
LightGBM Regressor
CatBoost Regressor
AdaBoost Regressor
Decision Tree Regressor
Extra Tree Regressor

# Model Performance

Best Model R² Score: > 0.95
Evaluation Metrics: R², Adjusted R², RMSE, MAE, MAPE, Max Error
Prediction Output: Continuous probability percentage (0-100%)
Processing Speed: Real-time predictions in < 2 seconds
