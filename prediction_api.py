"""
Weather Prediction API
Flask server to serve weather predictions
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import torch
import joblib
import json
from datetime import datetime, timedelta
import os

import config

app = Flask(__name__)
CORS(app)

class PredictionService:
    def __init__(self, model_dir=None):
        self.model_dir = model_dir or config.MODEL_DIR
        self.model = None
        self.scaler_X = None
        self.scaler_y = None
        self.metadata = None
        
        self.model_aq = None
        self.scaler_aq_X = None
        self.scaler_aq_y = None
        self.metadata_aq = None
        
        self.model_type = 'lstm'  # or 'xgboost'
        
        self.load_latest_model()
    
    def load_latest_model(self):
        """Load the most recent trained weather and air quality models"""
        # Load weather model
        try:
            # Find latest LSTM model
            lstm_files = [f for f in os.listdir(self.model_dir) if f.startswith('lstm_weather_')]
            if lstm_files:
                latest_lstm = sorted(lstm_files)[-1]
                timestamp = latest_lstm.split('_')[-1].replace('.pth', '')
                
                model_path = f'{self.model_dir}/{latest_lstm}'
                scaler_X_path = f'{self.model_dir}/scaler_X_{timestamp}.pkl'
                scaler_y_path = f'{self.model_dir}/scaler_y_{timestamp}.pkl'
                metadata_path = f'{self.model_dir}/lstm_metadata_{timestamp}.json'
                
                # Load metadata
                with open(metadata_path, 'r') as f:
                    self.metadata = json.load(f)
                
                # Load scalers
                self.scaler_X = joblib.load(scaler_X_path)
                self.scaler_y = joblib.load(scaler_y_path)
                
                # Load LSTM model
                from train_lstm_model import WeatherLSTM
                device = torch.device('cpu')
                self.model = WeatherLSTM(
                    input_size=self.metadata['input_size'],
                    hidden_size=128,
                    num_layers=2,
                    output_size=self.metadata['output_size']
                ).to(device)
                
                self.model.load_state_dict(torch.load(model_path, map_location=device))
                self.model.eval()
                
                print(f"✓ Loaded Weather LSTM model from {timestamp}")
        except Exception as e:
            print(f"Error loading Weather model: {e}")

        # Load air quality model
        try:
            lstm_aq_files = [f for f in os.listdir(self.model_dir) if f.startswith('lstm_air_quality_')]
            if lstm_aq_files:
                latest_lstm_aq = sorted(lstm_aq_files)[-1]
                timestamp_aq = latest_lstm_aq.split('_')[-1].replace('.pth', '')
                
                model_aq_path = f'{self.model_dir}/{latest_lstm_aq}'
                scaler_aq_X_path = f'{self.model_dir}/scaler_aq_X_{timestamp_aq}.pkl'
                scaler_aq_y_path = f'{self.model_dir}/scaler_aq_y_{timestamp_aq}.pkl'
                metadata_aq_path = f'{self.model_dir}/lstm_aq_metadata_{timestamp_aq}.json'
                
                # Load metadata
                with open(metadata_aq_path, 'r') as f:
                    self.metadata_aq = json.load(f)
                
                # Load scalers
                self.scaler_aq_X = joblib.load(scaler_aq_X_path)
                self.scaler_aq_y = joblib.load(scaler_aq_y_path)
                
                # Load LSTM model
                from train_aq_lstm_model import AirQualityLSTM
                device = torch.device('cpu')
                self.model_aq = AirQualityLSTM(
                    input_size=self.metadata_aq['input_size'],
                    hidden_size=128,
                    num_layers=2,
                    output_size=self.metadata_aq['output_size']
                ).to(device)
                
                self.model_aq.load_state_dict(torch.load(model_aq_path, map_location=device))
                self.model_aq.eval()
                
                print(f"✓ Loaded Air Quality LSTM model from {timestamp_aq}")
        except Exception as e:
            print(f"Error loading Air Quality model: {e}")
    
    def _ensure_datetime(self, df):
        if 'datetime' not in df.columns:
            if 'timestamp' in df.columns:
                df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
            else:
                raise ValueError('Missing datetime or timestamp column')
        df['datetime'] = pd.to_datetime(df['datetime'])
        return df

    def _add_time_features(self, df):
        df = self._ensure_datetime(df)
        df['hour'] = df['datetime'].dt.hour
        df['month'] = df['datetime'].dt.month
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        return df

    def _prepare_dataframe(self, data):
        if isinstance(data, dict):
            records = []
            for key, value in data.items():
                if not isinstance(value, dict):
                    continue
                row = dict(value)
                if str(key).isdigit():
                    row['timestamp'] = int(key)
                row['datetime'] = row.get('datetime') or row.get('timestamp')
                records.append(row)
            df = pd.DataFrame(records)
        else:
            df = pd.DataFrame(data)

        df = self._ensure_datetime(df)
        df = df.sort_values('datetime').reset_index(drop=True)
        return df

    def predict(self, recent_data):
        """
        Make prediction for next 48 hours
        
        Args:
            recent_data: DataFrame with last 24 hours of sensor readings
        
        Returns:
            Dictionary with predictions for each parameter (weather and air quality)
        """
        if self.model is None and self.model_aq is None:
            raise ValueError("No model loaded")
        
        # Standardize columns
        df = recent_data.copy()
        df.columns = df.columns.str.lower()
        df = df.rename(columns={
            'voc_index': 'voc',
            'nox_index': 'nox',
            'pm2_5': 'pm2_5',
            'pm1_0': 'pm1_0',
            'pm10': 'pm10',
            'co2': 'co2',
            'co': 'co',
            'light_intensity': 'light_intensity',
            'wind_speed': 'wind_speed',
            'wind_direction': 'wind_direction',
            'humidity': 'humidity',
            'temperature': 'temperature',
            'pressure': 'pressure',
            'rainfall': 'rainfall'
        })
        
        df = self._add_time_features(df)
        result = {}
        
        # Weather Predictions
        if self.model is not None:
            try:
                feature_cols = self.metadata['feature_columns']
                # Make sure all features exist
                for col in feature_cols:
                    if col not in df.columns:
                        df[col] = 0.0
                        
                X_seq = df[feature_cols].values
                
                # Ensure correct sequence length
                expected_length = self.metadata['lookback_hours'] * 60
                if len(X_seq) < expected_length:
                    # Pad with last value if needed
                    padding = np.repeat(X_seq[-1:], expected_length - len(X_seq), axis=0)
                    X_seq = np.vstack([padding, X_seq])
                elif len(X_seq) > expected_length:
                    X_seq = X_seq[-expected_length:]
                
                # Normalize
                X_scaled = self.scaler_X.transform(X_seq)
                X_scaled = X_scaled.reshape(1, expected_length, len(feature_cols))
                
                # Predict
                with torch.no_grad():
                    X_tensor = torch.FloatTensor(X_scaled)
                    predictions = self.model(X_tensor).numpy()
                
                # Inverse transform
                predictions = self.scaler_y.inverse_transform(predictions)
                
                # Format weather results
                for i, param in enumerate(self.metadata['target_params']):
                    result[param] = float(predictions[0, i])
            except Exception as e:
                print(f"Weather prediction error: {e}")
        
        # Air Quality Predictions
        if self.model_aq is not None:
            try:
                feature_cols_aq = self.metadata_aq['feature_columns']
                # Make sure all features exist
                for col in feature_cols_aq:
                    if col not in df.columns:
                        df[col] = 0.0
                        
                X_seq_aq = df[feature_cols_aq].values
                
                # Ensure correct sequence length
                expected_length_aq = self.metadata_aq['lookback_hours'] * 60
                if len(X_seq_aq) < expected_length_aq:
                    # Pad with last value if needed
                    padding = np.repeat(X_seq_aq[-1:], expected_length_aq - len(X_seq_aq), axis=0)
                    X_seq_aq = np.vstack([padding, X_seq_aq])
                elif len(X_seq_aq) > expected_length_aq:
                    X_seq_aq = X_seq_aq[-expected_length_aq:]
                
                # Normalize
                X_scaled_aq = self.scaler_aq_X.transform(X_seq_aq)
                X_scaled_aq = X_scaled_aq.reshape(1, expected_length_aq, len(feature_cols_aq))
                
                # Predict
                with torch.no_grad():
                    X_tensor_aq = torch.FloatTensor(X_scaled_aq)
                    predictions_aq = self.model_aq(X_tensor_aq).numpy()
                
                # Inverse transform
                predictions_aq = self.scaler_aq_y.inverse_transform(predictions_aq)
                
                # Format air quality results
                for i, param in enumerate(self.metadata_aq['target_params']):
                    result[param] = float(predictions_aq[0, i])
            except Exception as e:
                print(f"Air quality prediction error: {e}")
        
        result['forecast_time'] = datetime.now().isoformat()
        result['forecast_horizon_hours'] = self.metadata['forecast_horizon'] if self.metadata else (self.metadata_aq['forecast_horizon'] if self.metadata_aq else 48)
        
        return result

# Initialize service
prediction_service = PredictionService()

@app.route('/', methods=['GET'])
def home():
    """API home page"""
    return jsonify({
        'service': 'Weather & Air Quality Prediction API',
        'version': '1.0',
        'weather_model_loaded': prediction_service.model is not None,
        'air_quality_model_loaded': prediction_service.model_aq is not None,
        'endpoints': {
            '/predict': 'POST - Make weather & air quality prediction',
            '/health': 'GET - Check API health',
            '/model-info': 'GET - Get models information'
        }
    })

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'weather_model_loaded': prediction_service.model is not None,
        'air_quality_model_loaded': prediction_service.model_aq is not None,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/model-info', methods=['GET'])
def model_info():
    """Get models information"""
    if prediction_service.metadata is None and prediction_service.metadata_aq is None:
        return jsonify({'error': 'No model loaded'}), 500
    
    return jsonify({
        'weather_metadata': prediction_service.metadata,
        'air_quality_metadata': prediction_service.metadata_aq
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Make weather prediction
    
    Expected JSON payload:
    {
        "data": [
            {
                "datetime": "2024-01-01T00:00:00",
                "temperature": 25.0,
                "humidity": 65.0,
                ...
            },
            ...
        ]
    }
    """
    try:
        data = request.get_json() or {}

        if 'data' not in data:
            return jsonify({'error': 'Missing "data" field'}), 400

        df = pd.DataFrame(data['data'])
        predictions = prediction_service.predict(df)

        return jsonify({
            'success': True,
            'predictions': predictions
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/predict-from-firebase', methods=['POST'])
def predict_from_firebase():
    """
    Fetch latest data from Firebase and make prediction.

    The service uses secure configuration from the environment or .env file.
    Optional request payload fields:
    {
        "firebase_url": "https://your-project.firebaseio.com"
    }
    """
    try:
        import firebase_admin
        from firebase_admin import db
        from firebase_admin import credentials

        data = request.get_json() or {}
        firebase_url = data.get('firebase_url') or config.FIREBASE_DATABASE_URL
        if not firebase_url:
            return jsonify({'error': 'Missing "firebase_url" field and no FIREBASE_DATABASE_URL configured'}), 400

        if not firebase_admin._apps:
            if config.FIREBASE_CREDENTIALS and os.path.exists(config.FIREBASE_CREDENTIALS):
                cred = credentials.Certificate(config.FIREBASE_CREDENTIALS)
                firebase_admin.initialize_app(cred, {
                    'databaseURL': firebase_url
                })
            else:
                firebase_admin.initialize_app(options={
                    'databaseURL': firebase_url
                })

        ref = db.reference('/weather_data')
        cutoff_time = int((datetime.now() - timedelta(hours=24)).timestamp())
        query = ref.order_by_key().start_at(str(cutoff_time))
        recent_data = query.get()

        if not recent_data:
            return jsonify({'error': 'No recent data found in Firebase'}), 404

        records = []
        for key, reading in recent_data.items():
            if not isinstance(reading, dict):
                continue
            row = dict(reading)
            if str(key).isdigit():
                row['timestamp'] = int(key)
                row['datetime'] = datetime.fromtimestamp(int(key))
            if 'weather' in reading and isinstance(reading['weather'], dict):
                row.update(reading['weather'])
            if 'air_quality' in reading and isinstance(reading['air_quality'], dict):
                row.update(reading['air_quality'])
            records.append(row)

        if not records:
            return jsonify({'error': 'No valid weather records found in Firebase'}), 404

        df = pd.DataFrame(records)
        predictions = prediction_service.predict(df)
        return jsonify({
            'success': True,
            'predictions': predictions
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    print("\n=== Weather Prediction API ===")
    print(f"Model loaded: {prediction_service.model is not None}")
    print("\nStarting server on http://localhost:5000")
    print("Press Ctrl+C to stop\n")
    
    app.run(host='0.0.0.0', port=5000, debug=False)
