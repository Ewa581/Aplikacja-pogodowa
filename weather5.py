import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from models import AnomalyDetector, AnomalyEnsemble
from utils import load_data, generate_sample_data

# App configuration - FIXED: Changed to standard weather emoji
st.set_page_config(
    page_title="Weather Anomalies Detection",
    page_icon="⛅",  # Changed from special character to standard emoji
    layout="wide"
)

# Sidebar - Data Input
st.sidebar.header("Data Input")
uploaded_file = st.sidebar.file_uploader(
    "Upload Weather Data (Excel)",
    type=['xlsx', 'xls'],
    help="Should contain columns: date, temperature_C, rain_mm, wind_speed_kmh"
)

# Load or generate data
try:
    df = load_data(uploaded_file)
    df['date'] = pd.to_datetime(df['date'])  # Ensure datetime format
except Exception as e:
    st.error(f"Data loading error: {str(e)}")
    st.info("Using sample data instead")
    df = generate_sample_data()

# Sidebar - Date Range Selection
st.sidebar.header("Date Range")
date_range = st.sidebar.date_input(
    "Select analysis period",
    value=[df['date'].min(), df['date'].max()],
    min_value=df['date'].min(),
    max_value=df['date'].max()
)

# Filter data by date range
if len(date_range) == 2:
    mask = (df['date'] >= pd.to_datetime(date_range[0])) & \
           (df['date'] <= pd.to_datetime(date_range[1]))
    df_filtered = df.loc[mask].copy()
else:
    df_filtered = df.copy()
    st.warning("Please select a valid date range")

# Sidebar - Model Selection
st.sidebar.header("Detection Method")
use_ensemble = st.sidebar.checkbox("Use Ensemble Detection", False)

if not use_ensemble:
    # Single model selection
    model_choice = st.sidebar.selectbox(
        "Algorithm",
        options=[
            "Z-Score (Statistical)",
            "Isolation Forest",
            "One-Class SVM",
            "Local Outlier Factor (LOF)",
            "DBSCAN"
        ],
        index=0
    )

    # Model Parameters
    params = {}
    if model_choice == "Z-Score (Statistical)":
        params['threshold'] = st.sidebar.slider(
            "Z-Score Threshold", 1.0, 5.0, 2.0, 0.1)
    elif model_choice == "Isolation Forest":
        params['contamination'] = st.sidebar.slider(
            "Expected Anomaly Fraction", 0.01, 0.5, 0.05, 0.01)
    elif model_choice == "One-Class SVM":
        params['nu'] = st.sidebar.slider(
            "Nu (Outlier Fraction)", 0.01, 0.5, 0.05, 0.01)
    elif model_choice == "Local Outlier Factor (LOF)":
        params['n_neighbors'] = st.sidebar.slider(
            "Number of Neighbors", 5, 50, 20)
    elif model_choice == "DBSCAN":
        params['eps'] = st.sidebar.slider(
            "EPS", 0.1, 2.0, 0.5, 0.1)
        params['min_samples'] = st.sidebar.slider(
            "Min Samples", 2, 20, 5)
else:
    # Ensemble configuration
    st.sidebar.header("Ensemble Configuration")
    selected_models = st.sidebar.multiselect(
        "Select models to combine",
        options=["Z-Score (Statistical)", "Isolation Forest", 
                "One-Class SVM", "LOF", "DBSCAN"],
        default=["Isolation Forest", "LOF", "One-Class SVM"]
    )
    
    voting_method = st.sidebar.radio(
        "Voting Strategy",
        options=["soft (any model)", "hard (majority)"],
        index=0
    )
    
    # Set parameters for each model
    ensemble_params = {}
    if "Z-Score (Statistical)" in selected_models:
        st.sidebar.subheader("Z-Score Parameters")
        ensemble_params["Z-Score"] = {
            'threshold': st.sidebar.slider(
                "Z-Score Threshold", 1.0, 5.0, 2.0, 0.1,
                key='zscore_thresh'
            )
        }
    
    if "Isolation Forest" in selected_models:
        st.sidebar.subheader("Isolation Forest Parameters")
        ensemble_params["Isolation Forest"] = {
            'contamination': st.sidebar.slider(
                "Contamination", 0.01, 0.5, 0.05, 0.01,
                key='if_contam'
            )
        }
    
    if "One-Class SVM" in selected_models:
        st.sidebar.subheader("One-Class SVM Parameters")
        ensemble_params["One-Class SVM"] = {
            'nu': st.sidebar.slider(
                "Nu", 0.01, 0.5, 0.05, 0.01,
                key='ocsvm_nu'
            )
        }
    
    if "Local Outlier Factor (LOF)" in selected_models:
        st.sidebar.subheader("LOF Parameters")
        ensemble_params["LOF"] = {
            'n_neighbors': st.sidebar.slider(
                "Neighbors", 5, 50, 20,
                key='lof_neighbors'
            )
        }
    
    if "DBSCAN" in selected_models:
        st.sidebar.subheader("DBSCAN Parameters")
        ensemble_params["DBSCAN"] = {
            'eps': st.sidebar.slider(
                "EPS", 0.1, 2.0, 0.5, 0.1,
                key='dbscan_eps'
            ),
            'min_samples': st.sidebar.slider(
                "Min Samples", 2, 20, 5,
                key='dbscan_samples'
            )
        }

# Feature Selection
st.sidebar.header("Features")
features = st.sidebar.multiselect(
    "Select features for detection",
    options=['temperature_C', 'rain_mm', 'wind_speed_kmh'],
    default=['temperature_C', 'rain_mm']
)

# Main App Area - FIXED: Changed to standard weather emoji
st.title("⛅ Weather Anomalies Detection")
st.markdown(f"**Current Mode:** {'Ensemble' if use_ensemble else 'Single Model'}")

# Run Detection
if st.button("Detect Anomalies"):
    if not features:
        st.error("Please select at least one feature")
    else:
        with st.spinner("Running anomaly detection..."):
            if use_ensemble and len(selected_models) > 1:
                # Convert model names to expected format
                model_mapping = {
                    "Z-Score (Statistical)": "Z-Score",
                    "Local Outlier Factor (LOF)": "LOF",
                    "One-Class SVM": "One-Class SVM",
                    "Isolation Forest": "Isolation Forest",
                    "DBSCAN": "DBSCAN"
                }
                models_config = {
                    model_mapping[model]: params 
                    for model, params in ensemble_params.items()
                }
                
                # Run ensemble detection
                anomalies, model_votes = AnomalyEnsemble.ensemble_detect(
                    df_filtered,
                    features,
                    models_config,
                    voting='soft' if voting_method.startswith('soft') else 'hard'
                )
                df_filtered['anomaly'] = anomalies
                
                # Calculate agreement score
                df_filtered['agreement'] = model_votes.mean(axis=1)
                
                st.success(f"Ensemble detection complete! Found {sum(anomalies)} anomalies")
                
                # Show model agreement
                st.subheader("Model Agreement")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Voting Results**")
                    st.dataframe(
                        model_votes.head(10).style.applymap(
                            lambda x: 'background-color: red' if x == 1 else ''
                        )
                    )
                
                with col2:
                    st.markdown("**Correlation Matrix**")
                    fig_corr = px.imshow(
                        model_votes.corr(),
                        text_auto=True,
                        aspect="auto",
                        color_continuous_scale='Blues'
                    )
                    st.plotly_chart(fig_corr, use_container_width=True)
                
            else:
                # Single model detection
                if model_choice == "Z-Score (Statistical)":
                    anomalies = AnomalyDetector.detect_with_zscore(
                        df_filtered, features, params['threshold'])
                elif model_choice == "Isolation Forest":
                    anomalies = AnomalyDetector.detect_with_isolation_forest(
                        df_filtered, features, params['contamination'])
                elif model_choice == "One-Class SVM":
                    anomalies = AnomalyDetector.detect_with_ocsvm(
                        df_filtered, features, params['nu'])
                elif model_choice == "Local Outlier Factor (LOF)":
                    anomalies = AnomalyDetector.detect_with_lof(
                        df_filtered, features, params['n_neighbors'])
                elif model_choice == "DBSCAN":
                    anomalies = AnomalyDetector.detect_with_dbscan(
                        df_filtered, features, params['eps'], params['min_samples'])
                
                df_filtered['anomaly'] = anomalies
                st.success(f"Detection complete! Found {sum(anomalies)} anomalies")

# Display Results
if 'anomaly' in df_filtered.columns:
    # Metrics
    st.subheader("Detection Results")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Days", len(df_filtered))
    col2.metric("Anomaly Days", sum(df_filtered['anomaly']))
    col3.metric("Anomaly Rate", 
               f"{sum(df_filtered['anomaly'])/len(df_filtered)*100:.1f}%")
    
    # Time Series Plot
    st.subheader("Time Series with Anomalies")
    selected_metric = st.selectbox(
        "Select metric to visualize",
        options=features
    )
    
    fig = px.scatter(
        df_filtered,
        x='date',
        y=selected_metric,
        color='anomaly',
        color_discrete_map={True: "red", False: "blue"},
        title=f"{selected_metric.replace('_', ' ').title()} with Anomalies",
        hover_data=df_filtered.columns
    )
    fig.update_traces(
        marker=dict(size=8),
        selector=dict(mode='markers')
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Ensemble-specific visualizations
    if use_ensemble and 'agreement' in df_filtered.columns:
        st.subheader("Ensemble Confidence Analysis")
        
        tab1, tab2 = st.tabs(["Confidence Over Time", "Confidence Distribution"])
        
        with tab1:
            fig_time = px.scatter(
                df_filtered,
                x='date',
                y='agreement',
                color=selected_metric,
                size=np.where(df_filtered['anomaly'], 10, 3),
                hover_name='date',
                title="Model Agreement Over Time"
            )
            st.plotly_chart(fig_time, use_container_width=True)
        
        with tab2:
            fig_dist = px.histogram(
                df_filtered,
                x='agreement',
                color='anomaly',
                nbins=20,
                barmode='overlay',
                title="Confidence Score Distribution"
            )
            st.plotly_chart(fig_dist, use_container_width=True)
    
    # Anomaly Details
    st.subheader("Anomaly Details")
    st.dataframe(
        df_filtered[df_filtered['anomaly']].sort_values('date'),
        height=300,
        use_container_width=True
    )

# Data Preview
with st.expander("Show Raw Data"):
    st.dataframe(df_filtered, use_container_width=True)

# Download Button
if 'anomaly' in df_filtered.columns:
    csv = df_filtered.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Results",
        data=csv,
        file_name="weather_anomalies.csv",
        mime="text/csv"
    )