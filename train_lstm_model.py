"""
LSTM Weather Prediction Model
Uses LSTM neural network for multi-step weather forecasting
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import json
from datetime import datetime
import matplotlib.pyplot as plt
import os

WIND_DIRECTION_TO_DEGREES = {
    'N': 0, 'NNE': 22.5, 'NE': 45, 'ENE': 67.5,
    'E': 90, 'ESE': 112.5, 'SE': 135, 'SSE': 157.5,
    'S': 180, 'SSW': 202.5, 'SW': 225, 'WSW': 247.5,
    'W': 270, 'WNW': 292.5, 'NW': 315, 'NNW': 337.5,
}

CSV_COLUMN_MAP = {
    'Timestamp': 'datetime',
    'Temperature': 'temperature',
    'Humidity': 'humidity',
    'Pressure': 'pressure',
    'Rainfall': 'rainfall',
    'Light_Intensity': 'light_intensity',
    'Wind_Speed': 'wind_speed',
    'Wind_Direction': 'wind_direction',
    'VOC_Index': 'voc',
    'NOx_Index': 'nox',
    'PM2_5': 'pm2_5',
    'PM1_0': 'pm1_0',
    'PM10': 'pm10',
    'AQI': 'aqi',
    'aqi': 'aqi',
    'CO2': 'co2',
    'CO': 'co',
}

class WeatherDataset(Dataset):
    """PyTorch Dataset for weather time-series"""
    def __init__(self, sequences, targets):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

class WeatherLSTM(nn.Module):
    """LSTM model for weather prediction"""
    def __init__(self, input_size, hidden_size=128, num_layers=2, output_size=5, dropout=0.2):
        super(WeatherLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention mechanism
        self.attention = nn.Linear(hidden_size, 1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Apply attention
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attention_weights * lstm_out, dim=1)
        
        # Fully connected layers
        out = self.relu(self.fc1(context))
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out

class WeatherLSTMPredictor:
    def __init__(self, lookback_hours=24, forecast_horizon=48):
        self.lookback_hours = lookback_hours
        self.forecast_horizon = forecast_horizon
        self.model = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.feature_columns = []
        self.target_params = [
            'temperature', 'humidity', 'pressure', 
            'wind_speed', 'rainfall'
        ]
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
    
    def load_data(self, csv_path):
        """Load and preprocess data"""
        print(f"Loading data from {csv_path}...")
        df = pd.read_csv(csv_path)
        df = df.rename(columns={k: v for k, v in CSV_COLUMN_MAP.items() if k in df.columns})

        if 'datetime' not in df.columns:
            raise ValueError('CSV must include a datetime or Timestamp column')

        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values('datetime').reset_index(drop=True)

        if 'wind_direction' in df.columns and df['wind_direction'].dtype == object:
            df['wind_direction'] = df['wind_direction'].map(WIND_DIRECTION_TO_DEGREES).fillna(180)

        numeric_cols = [
            'temperature', 'humidity', 'pressure', 'rainfall', 'light_intensity',
            'wind_speed', 'wind_direction', 'voc', 'nox', 'pm2_5', 'pm10', 'co2', 'co'
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        df['hour'] = df['datetime'].dt.hour
        df['month'] = df['datetime'].dt.month
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

        # Resample to 1-hour intervals
        df = df.set_index('datetime').resample('1h').mean(numeric_only=True).interpolate()
        df = df.reset_index()
        
        print(f"✓ Loaded {len(df)} records")
        return df
    
    def create_sequences(self, df):
        """Create LSTM sequences"""
        print("\nCreating LSTM sequences...")
        
        # Feature columns
        feature_cols = [
            'temperature', 'humidity', 'pressure', 'rainfall',
            'light_intensity', 'wind_speed', 'wind_direction',
            'voc', 'pm2_5', 'co2',
            'hour_sin', 'hour_cos', 'month_sin', 'month_cos'
        ]
        
        feature_cols = [col for col in feature_cols if col in df.columns]
        self.feature_columns = feature_cols
        
        # Target columns
        target_cols = [col for col in self.target_params if col in df.columns]
        
        X_list = []
        y_list = []
        
        lookback_samples = self.lookback_hours
        forecast_samples = self.forecast_horizon
        
        for i in range(lookback_samples, len(df) - forecast_samples, 1):  # Step by hour
            # Input sequence (past data)
            X_seq = df.iloc[i - lookback_samples:i][feature_cols].values
            X_list.append(X_seq)
            
            # Target (future average)
            y_future = df.iloc[i:i + forecast_samples][target_cols].mean().values
            y_list.append(y_future)
        
        X = np.array(X_list)
        y = np.array(y_list)
        
        print(f"✓ Created {len(X)} sequences")
        print(f"  Input shape: {X.shape} (samples, timesteps, features)")
        print(f"  Output shape: {y.shape} (samples, targets)")
        
        return X, y
    
    def train(self, csv_path, epochs=100, batch_size=32, learning_rate=0.001):
        """Train LSTM model"""
        print("\n=== Training LSTM Model ===")
        
        # Load and prepare data
        df = self.load_data(csv_path)
        X, y = self.create_sequences(df)
        
        # Split data (80% train, 20% test)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Normalize features
        n_samples, n_timesteps, n_features = X_train.shape
        X_train_reshaped = X_train.reshape(-1, n_features)
        X_test_reshaped = X_test.reshape(-1, n_features)
        
        X_train_scaled = self.scaler_X.fit_transform(X_train_reshaped)
        X_test_scaled = self.scaler_X.transform(X_test_reshaped)
        
        X_train_scaled = X_train_scaled.reshape(n_samples, n_timesteps, n_features)
        X_test_scaled = X_test_scaled.reshape(len(X_test), n_timesteps, n_features)
        
        # Normalize targets
        y_train_scaled = self.scaler_y.fit_transform(y_train)
        y_test_scaled = self.scaler_y.transform(y_test)
        
        # Create datasets
        train_dataset = WeatherDataset(X_train_scaled, y_train_scaled)
        test_dataset = WeatherDataset(X_test_scaled, y_test_scaled)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Initialize model
        self.model = WeatherLSTM(
            input_size=n_features,
            hidden_size=128,
            num_layers=2,
            output_size=len(self.target_params),
            dropout=0.2
        ).to(self.device)
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        
        # Training loop
        train_losses = []
        test_losses = []
        best_test_loss = float('inf')
        
        print("\nTraining...")
        for epoch in range(epochs):
            # Train
            self.model.train()
            train_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            train_losses.append(train_loss)
            
            # Evaluate
            self.model.eval()
            test_loss = 0
            with torch.no_grad():
                for X_batch, y_batch in test_loader:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    outputs = self.model(X_batch)
                    loss = criterion(outputs, y_batch)
                    test_loss += loss.item()
            
            test_loss /= len(test_loader)
            test_losses.append(test_loss)
            
            scheduler.step(test_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}")
            
            # Save best model
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                best_model_state = self.model.state_dict()
        
        # Load best model
        self.model.load_state_dict(best_model_state)
        
        # Final evaluation
        print("\n=== Final Evaluation ===")
        self.evaluate(test_loader)
        
        return train_losses, test_losses
    
    def evaluate(self, test_loader):
        """Evaluate model performance"""
        self.model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(self.device)
                outputs = self.model(X_batch)
                all_preds.append(outputs.cpu().numpy())
                all_targets.append(y_batch.numpy())
        
        predictions = np.vstack(all_preds)
        targets = np.vstack(all_targets)
        
        # Inverse transform
        predictions = self.scaler_y.inverse_transform(predictions)
        targets = self.scaler_y.inverse_transform(targets)
        
        # Calculate metrics for each parameter
        for i, param in enumerate(self.target_params):
            rmse = np.sqrt(mean_squared_error(targets[:, i], predictions[:, i]))
            mae = mean_absolute_error(targets[:, i], predictions[:, i])
            r2 = r2_score(targets[:, i], predictions[:, i])
            
            print(f"\n{param}:")
            print(f"  RMSE: {rmse:.4f}")
            print(f"  MAE:  {mae:.4f}")
            print(f"  R²:   {r2:.4f}")
    
    def save_model(self, output_dir='models'):
        """Save trained model and scalers"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save model
        model_path = f'{output_dir}/lstm_weather_{timestamp}.pth'
        torch.save(self.model.state_dict(), model_path)
        
        # Save scalers
        joblib.dump(self.scaler_X, f'{output_dir}/scaler_X_{timestamp}.pkl')
        joblib.dump(self.scaler_y, f'{output_dir}/scaler_y_{timestamp}.pkl')
        
        # Save metadata
        metadata = {
            'lookback_hours': self.lookback_hours,
            'forecast_horizon': self.forecast_horizon,
            'feature_columns': self.feature_columns,
            'target_params': self.target_params,
            'input_size': len(self.feature_columns),
            'output_size': len(self.target_params),
            'timestamp': timestamp
        }
        
        with open(f'{output_dir}/lstm_metadata_{timestamp}.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n✓ Model saved to {output_dir}/")

def main():
    csv_path = os.getenv('TRAINING_CSV', 'data/moratuwa_open_meteo.csv')
    if not os.path.exists(csv_path):
        fallback = 'moratuwa_sensors_20k_dataset.csv'
        if os.path.exists(fallback):
            csv_path = fallback
        else:
            raise FileNotFoundError(
                f'Training dataset not found: {csv_path}. '
                'Run: python fetch_moratuwa_data.py --start 2026-04-01'
            )
    
    predictor = WeatherLSTMPredictor(
        lookback_hours=24,
        forecast_horizon=48
    )
    
    train_losses, test_losses = predictor.train(
        csv_path,
        epochs=int(os.getenv('TRAINING_EPOCHS', '100')),
        batch_size=32,
        learning_rate=0.001
    )
    
    predictor.save_model()

if __name__ == '__main__':
    main()
