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

app = Flask(__name__)
CORS(app)

class PredictionService:
    def __init__(self, model_dir='models'):
        self.model_dir = model_dir
        self.model = None
        self.scaler_X = None
        self.scaler_y = None
        self.metadata = None
        self.model_type = 'lstm'  # or 'xgboost'
        
        self.load_latest_model()
    
    def load_latest_model(self):
        """Load the most recent trained model"""
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
                
                self.model_type = 'lstm'
                print(f"✓ Loaded LSTM model from {timestamp}")
                return True
        
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
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
            Dictionary with predictions for each parameter
        """
        if self.model is None:
            raise ValueError("No model loaded")
        
        try:
            # Prepare features
            feature_cols = self.metadata['feature_columns']
            X_seq = recent_data[feature_cols].values
            
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
            
            # Format results
            result = {}
            for i, param in enumerate(self.metadata['target_params']):
                result[param] = float(predictions[0, i])
            
            result['forecast_time'] = datetime.now().isoformat()
            result['forecast_horizon_hours'] = self.metadata['forecast_horizon']
            
            return result
        
        except Exception as e:
            raise ValueError(f"Prediction error: {str(e)}")

# Initialize service
prediction_service = PredictionService()

@app.route('/', methods=['GET'])
def home():
    """API home page"""
    return jsonify({
        'service': 'Weather Prediction API',
        'version': '1.0',
        'model_type': prediction_service.model_type,
        'endpoints': {
            '/predict': 'POST - Make weather prediction',
            '/health': 'GET - Check API health',
            '/model-info': 'GET - Get model information'
        }
    })

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': prediction_service.model is not None,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/model-info', methods=['GET'])
def model_info():
    """Get model information"""
    if prediction_service.metadata is None:
        return jsonify({'error': 'No model loaded'}), 500
    
    return jsonify({
        'model_type': prediction_service.model_type,
        'metadata': prediction_service.metadata
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
    Fetch latest data from Firebase and make prediction
    
    Expected JSON payload:
    {
        "firebase_url": "https://your-project.firebaseio.com",
        "auth_token": "your_token"
    }
    """
    try:
        import firebase_admin
        from firebase_admin import db

        data = request.get_json() or {}
        firebase_url = data.get('firebase_url')
        if not firebase_url:
            return jsonify({'error': 'Missing "firebase_url" field'}), 400

        if not firebase_admin._apps:
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
