"""
XGBoost Baseline Air Quality Prediction Model
Predicts air quality parameters for next 48 hours using XGBoost
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import joblib
import json
from datetime import datetime
import os

class AirQualityXGBoostPredictor:
    def __init__(self, lookback_hours=24, forecast_horizon=48):
        """
        lookback_hours: How many hours of past data to use
        forecast_horizon: How many hours to predict into future
        """
        self.lookback_hours = lookback_hours
        self.forecast_horizon = forecast_horizon
        self.models = {}
        self.scalers = {}
        self.feature_columns = []
        
        # Parameters to predict
        self.target_params = [
            'pm2_5', 'pm10', 'voc', 
            'nox', 'co2', 'co'
        ]
    
    def load_data(self, csv_path):
        """Load and preprocess data"""
        print(f"Loading data from {csv_path}...")
        df = pd.read_csv(csv_path)
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
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values('datetime').reset_index(drop=True)
        
        # Ensure 1-minute sampling
        df = df.set_index('datetime').resample('1min').mean(numeric_only=True).interpolate()
        df = df.reset_index()
        
        print(f"✓ Loaded {len(df)} records")
        return df
    
    def create_sequences(self, df):
        """Create time-series sequences for training"""
        print("\nCreating sequences for Air Quality...")
        
        # Features to use
        feature_cols = [
            'temperature', 'humidity', 'pressure', 'rainfall',
            'light_intensity', 'wind_speed', 'wind_direction',
            'voc', 'nox', 'pm2_5', 'pm10', 'co2', 'co',
            'hour_sin', 'hour_cos', 'month_sin', 'month_cos'
        ]
        
        # Filter available columns
        feature_cols = [col for col in feature_cols if col in df.columns]
        self.feature_columns = feature_cols
        
        X_list = []
        y_dict = {param: [] for param in self.target_params}
        
        # Convert hours to sample indices (1 sample per minute)
        lookback_samples = self.lookback_hours * 60
        forecast_samples = self.forecast_horizon * 60
        
        # Use step of 60 to build dataset efficiently
        for i in range(lookback_samples, len(df) - forecast_samples, 60):
            # Get past data (features)
            past_data = df.iloc[i - lookback_samples:i][feature_cols].values
            
            # Flatten past data and add statistical features
            features = []
            for col_idx, col in enumerate(feature_cols):
                col_data = past_data[:, col_idx]
                features.extend([
                    col_data[-1],  # Current value
                    np.mean(col_data),  # Mean
                    np.std(col_data),  # Std dev
                    col_data[-1] - col_data[0],  # Trend
                    np.max(col_data),  # Max
                    np.min(col_data)   # Min
                ])
            
            X_list.append(features)
            
            # Get future targets (average over forecast period)
            future_data = df.iloc[i:i + forecast_samples]
            for param in self.target_params:
                if param in df.columns:
                    y_dict[param].append(future_data[param].mean())
        
        X = np.array(X_list)
        print(f"✓ Created {len(X)} sequences with {X.shape[1]} features")
        
        return X, y_dict
    
    def train(self, csv_path):
        """Train XGBoost models"""
        print("\n=== Training XGBoost Air Quality Models ===")
        
        df = self.load_data(csv_path)
        X, y_dict = self.create_sequences(df)
        
        # Split data (temporal split for time-series)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        
        # Train separate model for each target parameter
        for param in self.target_params:
            if param not in y_dict or len(y_dict[param]) == 0:
                print(f"Skipping {param} (no data)")
                continue
            
            print(f"\n--- Training {param} model ---")
            
            y = np.array(y_dict[param])
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            self.scalers[param] = scaler
            
            # Train XGBoost
            model = xgb.XGBRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            )
            
            model.fit(
                X_train_scaled, y_train,
                eval_set=[(X_test_scaled, y_test)],
                early_stopping_rounds=20,
                verbose=False
            )
            
            # Evaluate
            y_pred = model.predict(X_test_scaled)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            print(f"  RMSE: {rmse:.4f}")
            print(f"  MAE:  {mae:.4f}")
            print(f"  R²:   {r2:.4f}")
            
            self.models[param] = model
        
        print("\n✓ Training complete!")
        return self.models
    
    def save_models(self, output_dir='models'):
        """Save trained models"""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        for param, model in self.models.items():
            model_path = f'{output_dir}/xgboost_aq_{param}_{timestamp}.json'
            model.save_model(model_path)
            
            scaler_path = f'{output_dir}/scaler_aq_{param}_{timestamp}.pkl'
            joblib.dump(self.scalers[param], scaler_path)
        
        # Save metadata
        metadata = {
            'lookback_hours': self.lookback_hours,
            'forecast_horizon': self.forecast_horizon,
            'feature_columns': self.feature_columns,
            'target_params': self.target_params,
            'timestamp': timestamp
        }
        
        with open(f'{output_dir}/metadata_aq_{timestamp}.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n✓ Models saved to {output_dir}/")

def main():
    csv_path = 'data/weather_data_latest.csv'
    if not os.path.exists(csv_path):
        if os.path.exists('moratuwa_sensors_20k_dataset.csv'):
            csv_path = 'moratuwa_sensors_20k_dataset.csv'
        else:
            raise FileNotFoundError(f"Neither data/weather_data_latest.csv nor moratuwa_sensors_20k_dataset.csv found.")
    
    predictor = AirQualityXGBoostPredictor(
        lookback_hours=24,
        forecast_horizon=48
    )
    
    predictor.train(csv_path)
    predictor.save_models()

if __name__ == '__main__':
    main()
