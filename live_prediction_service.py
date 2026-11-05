"""
Live Prediction Service
Automatically fetches latest data from Firebase and makes real-time predictions
"""

import firebase_admin
from firebase_admin import credentials, db
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import torch
import joblib
import json
import os
from flask import Flask, jsonify
from flask_cors import CORS
import threading
import time

import config

app = Flask(__name__)
CORS(app)

class LivePredictionService:
    def __init__(self, firebase_url, credentials_path):
        self.firebase_url = firebase_url
        self.model = None
        self.scaler_X = None
        self.scaler_y = None
        self.metadata = None
        
        # Cache for predictions
        self.current_prediction = None
        self.last_prediction_time = None
        self.prediction_cache_minutes = 15  # Update predictions every 15 minutes
        
        # Initialize Firebase
        self.init_firebase(credentials_path)
        
        # Load model
        self.load_model()
        
        # Start background prediction updater
        self.start_background_updater()
    
    def init_firebase(self, credentials_path):
        """Initialize Firebase connection"""
        if not firebase_admin._apps:
            cred = credentials.Certificate(credentials_path)
            firebase_admin.initialize_app(cred, {
                'databaseURL': self.firebase_url
            })
        print("✓ Firebase initialized")
    
    def load_model(self):
        """Load the latest trained model"""
        models_dir = 'models'
        
        # Find latest LSTM model
        lstm_files = [f for f in os.listdir(models_dir) if f.startswith('lstm_weather_')]
        if not lstm_files:
            raise ValueError("No trained models found!")
        
        latest_lstm = sorted(lstm_files)[-1]
        timestamp = latest_lstm.split('_')[-1].replace('.pth', '')
        
        # Load metadata
        metadata_path = f'{models_dir}/lstm_metadata_{timestamp}.json'
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        # Load scalers
        self.scaler_X = joblib.load(f'{models_dir}/scaler_X_{timestamp}.pkl')
        self.scaler_y = joblib.load(f'{models_dir}/scaler_y_{timestamp}.pkl')
        
        # Load model
        from train_lstm_model import WeatherLSTM
        self.model = WeatherLSTM(
            input_size=self.metadata['input_size'],
            hidden_size=128,
            num_layers=2,
            output_size=self.metadata['output_size']
        )
        
        model_path = f'{models_dir}/{latest_lstm}'
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.model.eval()
        
        print(f"✓ Loaded model from {timestamp}")
    
    def fetch_recent_data(self, hours=24):
        """Fetch recent data from Firebase"""
        ref = db.reference('/weather_data')
        
        # Calculate cutoff timestamp
        cutoff = int((datetime.now() - timedelta(hours=hours)).timestamp())
        
        # Query data
        query = ref.order_by_key().start_at(str(cutoff))
        data = query.get()
        
        if not data:
            return None
        
        # Convert to DataFrame
        records = []
        for timestamp, reading in data.items():
            record = {
                'timestamp': int(timestamp),
                'datetime': datetime.fromtimestamp(int(timestamp))
            }
            
            if 'weather' in reading:
                record.update(reading['weather'])
            
            if 'air_quality' in reading:
                record.update(reading['air_quality'])
            
            records.append(record)
        
        df = pd.DataFrame(records)
        df = df.sort_values('timestamp')
        
        return df
    
    def add_time_features(self, df):
        """Add temporal features"""
        df['hour'] = df['datetime'].dt.hour
        df['day_of_week'] = df['datetime'].dt.dayofweek
        df['month'] = df['datetime'].dt.month
        
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        return df
    
    def make_prediction(self):
        """Fetch latest data and make prediction"""
        try:
            print(f"\n[{datetime.now()}] Updating prediction...")
            
            # Fetch recent data
            df = self.fetch_recent_data(hours=24)
            if df is None or len(df) < 100:
                print("Not enough recent data for prediction")
                return None
            
            # Add time features and ensure the model feature set is available
            df = self.add_time_features(df)
            feature_cols = self.metadata['feature_columns']
            df = df.reindex(columns=feature_cols)
            df = df.ffill().bfill().fillna(0)
            
            missing_cols = [col for col in feature_cols if col not in df.columns]
            if missing_cols:
                print(f"Missing model feature columns: {missing_cols}")
                return None
            
            expected_length = self.metadata['lookback_hours'] * 60
            X_seq = df.tail(expected_length).values
            
            if len(X_seq) == 0:
                print("No data available to build prediction sequence")
                return None
            
            if len(X_seq) < expected_length:
                padding = np.repeat(X_seq[0:1], expected_length - len(X_seq), axis=0)
                X_seq = np.vstack([padding, X_seq])
            
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
            result = {
                'forecast_time': datetime.now().isoformat(),
                'forecast_horizon_hours': self.metadata['forecast_horizon'],
                'predictions': {},
                'current_conditions': {},
                'data_points_used': len(df)
            }
            
            # Predictions
            for i, param in enumerate(self.metadata['target_params']):
                result['predictions'][param] = float(predictions[0, i])
            
            # Current conditions (last reading)
            last_row = df.iloc[-1]
            for param in self.metadata['target_params']:
                if param in last_row:
                    result['current_conditions'][param] = float(last_row[param])
            
            print(f"✓ Prediction updated")
            return result
            
        except Exception as e:
            print(f"✗ Prediction error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def update_prediction_cache(self):
        """Update cached prediction"""
        prediction = self.make_prediction()
        if prediction:
            self.current_prediction = prediction
            self.last_prediction_time = datetime.now()
    
    def background_updater(self):
        """Background thread to update predictions periodically"""
        consecutive_errors = 0
        max_consecutive_errors = 10
        while True:
            try:
                self.update_prediction_cache()
                consecutive_errors = 0  # reset on success
                time.sleep(self.prediction_cache_minutes * 60)
            except Exception as e:
                consecutive_errors += 1
                backoff = min(300, 60 * consecutive_errors)  # max 5 min backoff
                print(f"Background updater error ({consecutive_errors}/{max_consecutive_errors}): {e}")
                if consecutive_errors >= max_consecutive_errors:
                    print("Max consecutive errors reached. Background updater pausing for 30 minutes.")
                    time.sleep(1800)
                    consecutive_errors = 0
                else:
                    time.sleep(backoff)
    
    def start_background_updater(self):
        """Start background prediction updater"""
        thread = threading.Thread(target=self.background_updater, daemon=True)
        thread.start()
        print(f"✓ Background updater started (updates every {self.prediction_cache_minutes} min)")
    
    def get_prediction(self):
        """Get current cached prediction"""
        if self.current_prediction is None:
            # Generate first prediction synchronously
            self.update_prediction_cache()
        
        return self.current_prediction

# Global service instance
service = None

def init_service():
    """Initialize the prediction service"""
    global service

    firebase_url = config.FIREBASE_DATABASE_URL
    credentials_path = config.FIREBASE_CREDENTIALS

    if not firebase_url or not credentials_path:
        raise ValueError(
            'Missing Firebase configuration. Use .env or environment variables to set FIREBASE_DATABASE_URL and FIREBASE_CREDENTIALS.'
        )

    if not os.path.exists(credentials_path):
        raise ValueError(f"Firebase credentials not found at {credentials_path}")

    service = LivePredictionService(firebase_url, credentials_path)
    print("✓ Live prediction service initialized")

@app.route('/')
def home():
    """Home page"""
    return jsonify({
        'service': 'Live Weather Prediction Service',
        'description': 'Automatically updates predictions from real-time Firebase data',
        'endpoints': {
            '/predict': 'GET - Get latest weather prediction',
            '/health': 'GET - Service health check',
            '/update': 'POST - Force prediction update'
        }
    })

@app.route('/health')
def health():
    """Health check"""
    status = {
        'status': 'healthy',
        'service_running': service is not None,
        'model_loaded': service.model is not None if service else False,
        'timestamp': datetime.now().isoformat()
    }
    
    if service and service.last_prediction_time:
        status['last_prediction'] = service.last_prediction_time.isoformat()
        status['prediction_age_minutes'] = (datetime.now() - service.last_prediction_time).seconds / 60
    
    return jsonify(status)

@app.route('/predict')
def predict():
    """Get latest prediction"""
    if not service:
        return jsonify({'error': 'Service not initialized'}), 500
    
    prediction = service.get_prediction()
    
    if prediction is None:
        return jsonify({'error': 'No prediction available yet'}), 503
    
    return jsonify({
        'success': True,
        'prediction': prediction,
        'cached': True,
        'cache_age_minutes': (datetime.now() - service.last_prediction_time).seconds / 60
    })

@app.route('/update', methods=['POST'])
def force_update():
    """Force prediction update"""
    if not service:
        return jsonify({'error': 'Service not initialized'}), 500
    
    service.update_prediction_cache()
    
    return jsonify({
        'success': True,
        'message': 'Prediction updated',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/current')
def current_conditions():
    """Get current sensor readings"""
    if not service:
        return jsonify({'error': 'Service not initialized'}), 500
    
    try:
        df = service.fetch_recent_data(hours=1)
        if df is None or len(df) == 0:
            return jsonify({'error': 'No recent data'}), 404
        
        latest = df.iloc[-1]
        
        return jsonify({
            'timestamp': latest['timestamp'],
            'datetime': latest['datetime'].isoformat(),
            'conditions': {
                'temperature': float(latest.get('temperature', 0)),
                'humidity': float(latest.get('humidity', 0)),
                'pressure': float(latest.get('pressure', 0)),
                'wind_speed': float(latest.get('wind_speed', 0)),
                'rainfall': float(latest.get('rainfall', 0)),
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🌤️  Live Weather Prediction Service")
    print("="*60)
    
    # Initialize service
    init_service()
    
    print("\nStarting Flask server on http://localhost:5000")
    print("Predictions auto-update every 15 minutes")
    print("Press Ctrl+C to stop\n")
    
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
