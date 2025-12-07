import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px


st.set_page_config(
    page_title="Flood Probability Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown(""" 
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        text-align: center;
        font-size: 2rem;
        font-weight: bold;
        margin: 2rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #1f77b4;
    }
    </style>
""", unsafe_allow_html=True)

# Load model and scaler
@st.cache_resource
def load_models():
    with open('flood_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('feature_names.pkl', 'rb') as f:
        features = pickle.load(f)
    with open('model_results.pkl', 'rb') as f:
        results = pickle.load(f)
    return model, scaler, features, results

try:
    model, scaler, feature_names, model_results = load_models()
    model_results = model_results.drop_duplicates(subset=['Model'], keep='first').reset_index(drop=True)
    model_loaded = True
    best_model_name = model_results.iloc[0]['Model']
    best_r2 = model_results.iloc[0]['R2']
    best_rmse = model_results.iloc[0]['RMSE']
except Exception as e:
    model_loaded = False
    st.error(f" Error loading model files: {str(e)}")
    st.info("Please ensure flood_model.pkl, scaler.pkl, and feature_names.pkl are in the same directory.")

# Header
st.markdown('<div class="main-header"> Flood Probability Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Advanced ML-based Flood Risk Assessment System</div>', unsafe_allow_html=True)

if model_loaded:
    st.success(f" Model Loaded: **{best_model_name}** | R² Score: **{best_r2:.4f}** | RMSE: **{best_rmse:.4f}**")
    
    # Sidebar
    st.sidebar.header(" Model Information")
    st.sidebar.markdown(f""" Best Model: {best_model_name}
                        
    Performance Metrics:
    - R² Score: {best_r2:.4f}
    - RMSE: {best_rmse:.4f}
    - MAE: {model_results.iloc[0]['MAE']:.4f}
    - MAPE: {model_results.iloc[0]['MAPE (%)']:.2f}%
    """)

    st.sidebar.markdown("---")
    st.sidebar.header(" All Models Comparison")
    
    # Display only unique models in sidebar (fixed duplication issue)
    comparison_df = model_results[['Model', 'R2', 'RMSE', 'MAE']].head(8).copy()
    st.sidebar.dataframe(
        comparison_df,
        hide_index=True,
        use_container_width=True
    )
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs([" Single Prediction", " Batch Prediction", " Model Analysis"])
    
    # Tab 1: Single Prediction
    with tab1:
        st.header("Enter Feature Values")
        
        # Create input fields dynamically based on feature names
        col_per_row = 3
        num_features = len(feature_names)
        
        input_data = {}
        
        for i in range(0, num_features, col_per_row):
            cols = st.columns(col_per_row)
            for j in range(col_per_row):
                if i + j < num_features:
                    feature = feature_names[i + j]
                    with cols[j]:
                        input_data[feature] = st.number_input(
                            feature,
                            value=0.0,
                            step=0.1,
                            format="%.2f",
                            key=f"input_{feature}"
                        )
        
        st.markdown("---")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            predict_button = st.button(" Predict Flood Probability", use_container_width=True, type="primary")
        
        if predict_button:
            # Prepare input
            input_df = pd.DataFrame([input_data])
            
            # Scale input
            input_scaled = scaler.transform(input_df)
            
            # Make prediction
            prediction = model.predict(input_scaled)[0]
            
            # Display prediction
            st.markdown(f'<div class="prediction-box"> Flood Probability: {prediction:.2f}%</div>', unsafe_allow_html=True)
            
            # Risk level
            if prediction < 20:
                risk_level = " Low Risk"
                risk_color = "green"
                recommendation = "The area has a low flood risk. Standard precautions are sufficient."
            elif prediction < 50:
                risk_level = " Moderate Risk"
                risk_color = "orange"
                recommendation = "Moderate flood risk detected. Stay alert and prepare emergency supplies."
            elif prediction < 75:
                risk_level = " High Risk"
                risk_color = "darkorange"
                recommendation = "High flood risk! Prepare evacuation plans and monitor weather updates."
            else:
                risk_level = " Critical Risk"
                risk_color = "red"
                recommendation = "Critical flood risk! Immediate action required. Follow evacuation orders."
            
            # Display risk assessment
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Risk Level")
                st.markdown(f"<h2 style='color: {risk_color};'>{risk_level}</h2>", unsafe_allow_html=True)
            
            with col2:
                st.markdown("### Recommendation")
                st.info(recommendation)
            
            # Gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=prediction,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Flood Probability", 'font': {'size': 24}},
                delta={'reference': 50, 'increasing': {'color': "red"}},
                gauge={
                    'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': "darkblue"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 20], 'color': 'lightgreen'},
                        {'range': [20, 50], 'color': 'lightyellow'},
                        {'range': [50, 75], 'color': 'orange'},
                        {'range': [75, 100], 'color': 'lightcoral'}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 75
                    }
                }
            ))
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    # Tab 2: Batch Prediction
    with tab2:
        st.header("Upload CSV for Batch Prediction")
        st.info(" Upload a CSV file with the same features as the training data.")
        
        uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
        
        if uploaded_file is not None:
            try:
                batch_df = pd.read_csv(uploaded_file)
                st.success(f" File uploaded successfully! {len(batch_df)} rows detected.")
                
                st.subheader("Preview Data")
                st.dataframe(batch_df.head(10), use_container_width=True)
                
                if st.button(" Run Batch Prediction", type="primary"):
                    # Check if all required features are present
                    missing_features = set(feature_names) - set(batch_df.columns)
                    if missing_features:
                        st.error(f" Missing features: {missing_features}")
                    else:
                        # Make predictions
                        X_batch = batch_df[feature_names]
                        X_batch_scaled = scaler.transform(X_batch)
                        predictions = model.predict(X_batch_scaled)
                        
                        # Add predictions to dataframe
                        result_df = batch_df.copy()
                        result_df['Predicted_FloodProbability'] = predictions
                        
                        # Categorize risk
                        def categorize_risk(prob):
                            if prob < 20:
                                return "Low"
                            elif prob < 50:
                                return "Moderate"
                            elif prob < 75:
                                return "High"
                            else:
                                return "Critical"
                        
                        result_df['Risk_Level'] = result_df['Predicted_FloodProbability'].apply(categorize_risk)
                        
                        st.subheader("Prediction Results")
                        st.dataframe(result_df, use_container_width=True)
                        
                        # Summary statistics
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Average Probability", f"{predictions.mean():.2f}%")
                        col2.metric("Max Probability", f"{predictions.max():.2f}%")
                        col3.metric("Min Probability", f"{predictions.min():.2f}%")
                        col4.metric("Std Deviation", f"{predictions.std():.2f}%")
                        
                        # Risk distribution
                        risk_counts = result_df['Risk_Level'].value_counts()
                        fig = px.pie(
                            values=risk_counts.values,
                            names=risk_counts.index,
                            title="Risk Level Distribution",
                            color=risk_counts.index,
                            color_discrete_map={
                                'Low': 'green',
                                'Moderate': 'yellow',
                                'High': 'orange',
                                'Critical': 'red'
                            }
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Download results
                        csv = result_df.to_csv(index=False)
                        st.download_button(
                            label=" Download Predictions",
                            data=csv,
                            file_name="flood_predictions.csv",
                            mime="text/csv",
                            type="primary"
                        )
            
            except Exception as e:
                st.error(f" Error processing file: {str(e)}")
    
    # Tab 3: Model Analysis
    with tab3:
        st.header("Model Performance Analysis")
        
        # Model comparison 
        st.subheader("All Models Comparison")
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=model_results['Model'],
            y=model_results['R2'],
            name='R² Score',
            marker_color='skyblue',
            text=model_results['R2'].round(4),
            textposition='outside'
        ))
        fig.update_layout(
            title="Model R² Score Comparison",
            xaxis_title="Model",
            yaxis_title="R² Score",
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Metrics table 
        st.subheader("Detailed Metrics")
        st.dataframe(
            model_results[['Model', 'R2', 'Adjusted R²', 'RMSE', 'MAE', 'MAPE (%)']],
            hide_index=True,
            use_container_width=True
        )
        
        # Best model info
        st.subheader(" Best Model Details")
        best_col1, best_col2 = st.columns(2)
        
        with best_col1:
            st.metric("Model Name", best_model_name)
            st.metric("R² Score", f"{best_r2:.4f}")
            st.metric("RMSE", f"{best_rmse:.4f}")
        
        with best_col2:
            st.metric("MAE", f"{model_results.iloc[0]['MAE']:.4f}")
            st.metric("MAPE", f"{model_results.iloc[0]['MAPE (%)']:.2f}%")
            st.metric("Max Error", f"{model_results.iloc[0]['Max Error']:.4f}")

else:
    st.warning(" Please ensure all model files are available to use the app.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p> Flood Probability Predictor | Powered by Machine Learning</p>
    <p>Built with Streamlit & Scikit-learn</p>
</div>
""", unsafe_allow_html=True)