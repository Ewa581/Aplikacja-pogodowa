import os
from flask import Flask, render_template, request, jsonify
import pandas as pd
import psycopg2
from datetime import datetime
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, RepeatVector, TimeDistributed
from tensorflow.keras.callbacks import EarlyStopping
import json

app = Flask(__name__)

# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'database': 'postgres',
    'user': 'postgres',
    'password': 'postgres'
}

def get_db_connection():
    return psycopg2.connect(**DB_CONFIG)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_csv():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and file.filename.endswith('.csv'):
        try:
            df = pd.read_csv(file)
            
            # Validate CSV structure
            if 'time' not in df.columns or 'averageAirTemp' not in df.columns:
                return jsonify({'error': 'CSV must contain "time" and "averageAirTemp" columns'}), 400
            
            # Connect to database
            conn = get_db_connection()
            cur = conn.cursor()
            
            # Insert data
            for _, row in df.iterrows():
                cur.execute(
                    "INSERT INTO temperature_data (time, average_air_temp) VALUES (%s, %s)",
                    (row['time'], row['averageAirTemp'])
                )
            
            conn.commit()
            cur.close()
            conn.close()
            
            return jsonify({'message': 'Data uploaded successfully'}), 200
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    else:
        return jsonify({'error': 'Invalid file format. Please upload a CSV file'}), 400

@app.route('/data', methods=['GET'])
def get_data():
    start_date = request.args.get('start')
    end_date = request.args.get('end')
    
    try:
        conn = get_db_connection()
        query = "SELECT time, average_air_temp FROM temperature_data"
        params = []
        
        if start_date and end_date:
            query += " WHERE time BETWEEN %s AND %s"
            params.extend([start_date, end_date])
        elif start_date:
            query += " WHERE time >= %s"
            params.append(start_date)
        elif end_date:
            query += " WHERE time <= %s"
            params.append(end_date)
            
        query += " ORDER BY time"
        
        df = pd.read_sql(query, conn, params=params if params else None)
        conn.close()
        
        return jsonify({
            'time': df['time'].astype(str).tolist(),
            'temperature': df['average_air_temp'].tolist()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def prepare_data_for_models(df):
    # Convert time to numerical features (hour, day of week, etc.)
    df['time'] = pd.to_datetime(df['time'])
    df['hour'] = df['time'].dt.hour
    df['day_of_week'] = df['time'].dt.dayofweek
    df['day_of_year'] = df['time'].dt.dayofyear
    
    # Scale features
    scaler = StandardScaler()
    features = df[['average_air_temp', 'hour', 'day_of_week', 'day_of_year']]
    scaled_features = scaler.fit_transform(features)
    
    return scaled_features, df['average_air_temp'].values

def train_lstm_autoencoder(X, timesteps=2024):
    # Reshape data for LSTM [samples, timesteps, features]
    X = X.reshape((-1, timesteps, X.shape[1]))
    
    # Define model
    model = Sequential([
        LSTM(64, activation='relu', input_shape=(timesteps, X.shape[2]), return_sequences=True),
        LSTM(32, activation='relu', return_sequences=False),
        RepeatVector(timesteps),
        LSTM(32, activation='relu', return_sequences=True),
        LSTM(64, activation='relu', return_sequences=True),
        TimeDistributed(Dense(X.shape[2]))
    ])
    
    model.compile(optimizer='adam', loss='mse')
    
    # Early stopping
    early_stop = EarlyStopping(monitor='val_loss', patience=5, mode='min')
    
    # Train model
    model.fit(X, X, epochs=50, batch_size=32, validation_split=0.1, callbacks=[early_stop], verbose=0)
    
    return model

def detect_anomalies_with_lstm(model, X, timesteps=24):
    X_reshaped = X.reshape((-1, timesteps, X.shape[1]))
    reconstructions = model.predict(X_reshaped)
    mse = np.mean(np.power(X_reshaped - reconstructions, 2), axis=(1, 2))
    threshold = np.percentile(mse, 95)  # 95th percentile as threshold
    anomalies = mse > threshold
    return anomalies, mse

@app.route('/detect_anomalies', methods=['POST'])
def detect_anomalies():
    try:
        data = request.json
        start_date = data.get('start')
        end_date = data.get('end')
        selected_models = data.get('models', [])
        
        # Get data from database
        conn = get_db_connection()
        query = "SELECT time, average_air_temp FROM temperature_data WHERE time BETWEEN %s AND %s ORDER BY time"
        df = pd.read_sql(query, conn, params=[start_date, end_date])
        conn.close()
        
        if len(df) < 10:
            return jsonify({'error': 'Not enough data points for analysis'}), 400
        
        # Prepare data
        X, temperatures = prepare_data_for_models(df)
        
        results = {}
        all_predictions = []
        
        # Isolation Forest
        if 'isolation_forest' in selected_models:
            clf = IsolationForest(contamination=0.05)
            preds = clf.fit_predict(X)
            results['isolation_forest'] = (preds == -1).tolist()
            all_predictions.append(preds == -1)
        
        # One-Class SVM
        if 'one_class_svm' in selected_models:
            svm = OneClassSVM(nu=0.05)
            preds = svm.fit_predict(X)
            results['one_class_svm'] = (preds == -1).tolist()
            all_predictions.append(preds == -1)
        
        # DBSCAN
        if 'dbscan' in selected_models:
            dbscan = DBSCAN(eps=0.5, min_samples=5)
            preds = dbscan.fit_predict(X)
            results['dbscan'] = (preds == -1).tolist()
            all_predictions.append(preds == -1)
        
        # Gaussian Mixture
        if 'gaussian_mixture' in selected_models:
            gmm = GaussianMixture(n_components=3)
            gmm.fit(X)
            scores = gmm.score_samples(X)
            threshold = np.percentile(scores, 5)
            results['gaussian_mixture'] = (scores < threshold).tolist()
            all_predictions.append(scores < threshold)
        
        # LSTM Autoencoder
        if 'lstm_autoencoder' in selected_models:
            lstm_model = train_lstm_autoencoder(X)
            anomalies, scores = detect_anomalies_with_lstm(lstm_model, X)
            results['lstm_autoencoder'] = anomalies.tolist()
            all_predictions.append(anomalies)
        
        # Combined model predictions
        if len(all_predictions) > 0:
            combined = np.mean(all_predictions, axis=0)
            results['combined'] = (combined > 0.5).tolist()
        
        # Prepare response
        response = {
            'time': df['time'].astype(str).tolist(),
            'temperature': temperatures.tolist(),
            'predictions': results
        }
        
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)