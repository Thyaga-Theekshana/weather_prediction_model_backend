"""
Automated Model Retraining System
Automatically retrains models when new data is available and performance degrades
"""

import os
import time
import schedule
import joblib
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# Import training modules
from firebase_export import fetch_all_data, flatten_data, clean_data, add_time_features
from train_lstm_model import WeatherLSTMPredictor
from train_xgboost_model import WeatherXGBoostPredictor

class AutoRetrainingSystem:
    def __init__(self, 
                 retrain_interval_days=7,
                 min_new_data_points=10080,  # 1 week at 1min sampling
                 performance_threshold=0.15):  # 15% degradation triggers retrain
        
        self.retrain_interval_days = retrain_interval_days
        self.min_new_data_points = min_new_data_points
        self.performance_threshold = performance_threshold
        
        self.models_dir = 'models'
        self.data_dir = 'data'
        self.logs_dir = 'logs'
        
        # Create directories
        for dir_path in [self.models_dir, self.data_dir, self.logs_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        self.last_train_time = None
        self.last_data_count = 0
        self.performance_history = []
        
        self.load_state()
    
    def load_state(self):
        """Load previous training state"""
        state_file = f'{self.logs_dir}/training_state.json'
        if os.path.exists(state_file):
            with open(state_file, 'r') as f:
                state = json.load(f)
                self.last_train_time = datetime.fromisoformat(state.get('last_train_time', '2000-01-01'))
                self.last_data_count = state.get('last_data_count', 0)
                self.performance_history = state.get('performance_history', [])
            print(f"✓ Loaded training state from {state_file}")
        else:
            print("No previous training state found")
    
    def save_state(self):
        """Save current training state"""
        state = {
            'last_train_time': self.last_train_time.isoformat() if self.last_train_time else None,
            'last_data_count': self.last_data_count,
            'performance_history': self.performance_history
        }
        state_file = f'{self.logs_dir}/training_state.json'
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2)
        print(f"✓ Saved training state to {state_file}")
    
    def fetch_latest_data(self):
        """Fetch latest data from Firebase"""
        print("\n=== Fetching Latest Data from Firebase ===")
        
        try:
            # Fetch all data
            firebase_data = fetch_all_data()
            if not firebase_data:
                print("No data available in Firebase")
                return None
            
            # Convert to DataFrame
            df = flatten_data(firebase_data)
            df = clean_data(df)
            df = add_time_features(df)
            
            # Save to CSV
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            csv_path = f'{self.data_dir}/weather_data_{timestamp}.csv'
            df.to_csv(csv_path, index=False)
            
            print(f"✓ Fetched {len(df)} records")
            print(f"✓ Saved to {csv_path}")
            
            return csv_path, len(df)
        
        except Exception as e:
            print(f"✗ Error fetching data: {e}")
            return None, 0
    
    def should_retrain(self, current_data_count):
        """Determine if retraining is needed"""
        reasons = []
        
        # Check 1: Never trained before
        if self.last_train_time is None:
            reasons.append("Initial training")
            return True, reasons
        
        # Check 2: Scheduled interval reached
        days_since_training = (datetime.now() - self.last_train_time).days
        if days_since_training >= self.retrain_interval_days:
            reasons.append(f"Scheduled retrain ({days_since_training} days since last training)")
        
        # Check 3: Significant new data available
        new_data_points = current_data_count - self.last_data_count
        if new_data_points >= self.min_new_data_points:
            reasons.append(f"New data available ({new_data_points} new points)")
        
        # Check 4: Performance degradation (if we have history with real metrics)
        valid_history = [p for p in self.performance_history if p.get('rmse') is not None]
        if len(valid_history) >= 2:
            recent_performance = valid_history[-1]['rmse']
            best_performance = min([p['rmse'] for p in valid_history])
            if best_performance > 0:
                degradation = (recent_performance - best_performance) / best_performance
                if degradation > self.performance_threshold:
                    reasons.append(f"Performance degraded by {degradation*100:.1f}%")
        
        should_train = len(reasons) > 0
        return should_train, reasons
    
    def train_models(self, csv_path):
        """Train both LSTM and XGBoost models"""
        print("\n=== Training Models ===")
        
        results = {}
        
        try:
            # Train LSTM
            print("\n--- LSTM Model ---")
            lstm_predictor = WeatherLSTMPredictor(
                lookback_hours=24,
                forecast_horizon=48
            )
            
            train_losses, test_losses = lstm_predictor.train(
                csv_path,
                epochs=50,  # Reduced for faster automated training
                batch_size=32
            )
            
            lstm_predictor.save_model(self.models_dir)
            results['lstm'] = {
                'final_train_loss': train_losses[-1],
                'final_test_loss': test_losses[-1],
                'status': 'success'
            }
            
        except Exception as e:
            print(f"✗ LSTM training failed: {e}")
            results['lstm'] = {'status': 'failed', 'error': str(e)}
        
        try:
            # Train XGBoost
            print("\n--- XGBoost Model ---")
            xgb_predictor = WeatherXGBoostPredictor(
                lookback_hours=24,
                forecast_horizon=48
            )
            
            xgb_predictor.train(csv_path)
            xgb_predictor.save_models(self.models_dir)
            results['xgboost'] = {'status': 'success'}
            
        except Exception as e:
            print(f"✗ XGBoost training failed: {e}")
            results['xgboost'] = {'status': 'failed', 'error': str(e)}
        
        return results
    
    def evaluate_current_model(self):
        """Evaluate current LSTM model performance on held-out recent data.

        Loads the latest saved model and its training data, reconstructs the
        test split (last 20% of samples), runs predictions, and returns real
        RMSE, MAE, and R² averaged across all target parameters.

        Returns:
            dict with 'rmse', 'mae', 'r2' keys. Values are None if no model
            or data are available yet.
        """
        import glob
        import json as _json
        import torch
        import joblib
        import numpy as np
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

        models_dir = self.models_dir

        # ── 1. Find the latest LSTM model ──────────────────────────────────
        lstm_files = sorted(glob.glob(f'{models_dir}/lstm_weather_*.pth'))
        if not lstm_files:
            print("⚠ No trained LSTM model found for evaluation — skipping.")
            return {'rmse': None, 'mae': None, 'r2': None}

        latest_pth = lstm_files[-1]
        timestamp = os.path.basename(latest_pth)[len('lstm_weather_'):].replace('.pth', '')

        metadata_path = f'{models_dir}/lstm_metadata_{timestamp}.json'
        scaler_X_path = f'{models_dir}/scaler_X_{timestamp}.pkl'
        scaler_y_path = f'{models_dir}/scaler_y_{timestamp}.pkl'

        missing = [p for p in [metadata_path, scaler_X_path, scaler_y_path] if not os.path.exists(p)]
        if missing:
            print(f"⚠ Evaluation skipped — missing artefacts: {missing}")
            return {'rmse': None, 'mae': None, 'r2': None}

        with open(metadata_path, 'r') as fh:
            metadata = _json.load(fh)

        scaler_X = joblib.load(scaler_X_path)
        scaler_y = joblib.load(scaler_y_path)

        try:
            from train_lstm_model import WeatherLSTM, WeatherLSTMPredictor
        except ImportError as exc:
            print(f"⚠ Could not import LSTM classes: {exc}")
            return {'rmse': None, 'mae': None, 'r2': None}

        device = torch.device('cpu')
        model = WeatherLSTM(
            input_size=metadata['input_size'],
            hidden_size=128,
            num_layers=2,
            output_size=metadata['output_size'],
        ).to(device)
        model.load_state_dict(torch.load(latest_pth, map_location=device))
        model.eval()

        # ── 2. Find the latest training CSV ────────────────────────────────
        csv_candidates = sorted(glob.glob(f'{self.data_dir}/weather_data_*.csv'))
        # Also accept the bundled dataset as a last resort
        if not csv_candidates and os.path.exists('moratuwa_sensors_20k_dataset.csv'):
            csv_candidates = ['moratuwa_sensors_20k_dataset.csv']
        if not csv_candidates:
            print("⚠ No training CSV found for evaluation — skipping.")
            return {'rmse': None, 'mae': None, 'r2': None}

        csv_path = csv_candidates[-1]
        print(f"  Evaluating with data: {csv_path}")

        # ── 3. Re-create sequences using the predictor ─────────────────────
        try:
            predictor = WeatherLSTMPredictor(
                lookback_hours=metadata['lookback_hours'],
                forecast_horizon=metadata['forecast_horizon'],
            )
            df = predictor.load_data(csv_path)
            X, y = predictor.create_sequences(df)
        except Exception as exc:
            print(f"⚠ Could not build evaluation sequences: {exc}")
            return {'rmse': None, 'mae': None, 'r2': None}

        if len(X) < 10:
            print("⚠ Too few sequences for meaningful evaluation.")
            return {'rmse': None, 'mae': None, 'r2': None}

        # ── 4. Use the held-out 20% test split (same as training) ──────────
        split_idx = int(len(X) * 0.8)
        X_test = X[split_idx:]
        y_test = y[split_idx:]

        n_samples, n_timesteps, n_features = X_test.shape
        X_test_reshaped = X_test.reshape(-1, n_features)
        X_test_scaled = scaler_X.transform(X_test_reshaped).reshape(n_samples, n_timesteps, n_features)
        y_test_scaled = scaler_y.transform(y_test)

        # ── 5. Run inference ───────────────────────────────────────────────
        from torch.utils.data import DataLoader, TensorDataset
        dataset = TensorDataset(
            torch.FloatTensor(X_test_scaled),
            torch.FloatTensor(y_test_scaled),
        )
        loader = DataLoader(dataset, batch_size=64, shuffle=False)

        all_preds, all_targets = [], []
        with torch.no_grad():
            for X_batch, y_batch in loader:
                preds = model(X_batch.to(device)).cpu().numpy()
                all_preds.append(preds)
                all_targets.append(y_batch.numpy())

        predictions = np.vstack(all_preds)
        targets = np.vstack(all_targets)

        # Inverse-transform back to physical units
        predictions = scaler_y.inverse_transform(predictions)
        targets = scaler_y.inverse_transform(targets)

        # ── 6. Compute aggregate metrics across all target parameters ───────
        rmse_vals, mae_vals, r2_vals = [], [], []
        target_params = metadata.get('target_params', [])
        for i, param in enumerate(target_params):
            rmse_vals.append(float(np.sqrt(mean_squared_error(targets[:, i], predictions[:, i]))))
            mae_vals.append(float(mean_absolute_error(targets[:, i], predictions[:, i])))
            r2_vals.append(float(r2_score(targets[:, i], predictions[:, i])))
            print(f"    {param}: RMSE={rmse_vals[-1]:.4f}  MAE={mae_vals[-1]:.4f}  R²={r2_vals[-1]:.4f}")

        result = {
            'rmse': round(float(np.mean(rmse_vals)), 4),
            'mae': round(float(np.mean(mae_vals)), 4),
            'r2': round(float(np.mean(r2_vals)), 4),
            'per_param': {p: {'rmse': rmse_vals[i], 'mae': mae_vals[i], 'r2': r2_vals[i]}
                          for i, p in enumerate(target_params)},
            'n_test_samples': len(predictions),
            'evaluated_at': datetime.now().isoformat(),
        }
        print(f"  ✓ Evaluation complete — RMSE={result['rmse']}  MAE={result['mae']}  R²={result['r2']}")
        return result

    
    def log_training(self, results, data_count):
        """Log training results"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'data_count': data_count,
            'training_results': results,
            'performance': self.evaluate_current_model()
        }
        
        # Append to performance history (only when we have real metrics)
        perf = log_entry['performance']
        if perf.get('rmse') is not None:
            self.performance_history.append({
                'timestamp': log_entry['timestamp'],
                'rmse': perf['rmse']
            })
        
        # Keep only last 30 entries
        self.performance_history = self.performance_history[-30:]
        
        # Save to log file
        log_file = f'{self.logs_dir}/training_log.jsonl'
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
        
        print(f"\n✓ Training logged to {log_file}")
    
    def run_training_cycle(self):
        """Execute one training cycle"""
        print("\n" + "="*60)
        print(f"Automated Training Cycle - {datetime.now()}")
        print("="*60)
        
        # Fetch latest data
        result = self.fetch_latest_data()
        if result is None:
            print("✗ Failed to fetch data, skipping this cycle")
            return
        
        csv_path, data_count = result
        
        # Check if retraining is needed
        should_train, reasons = self.should_retrain(data_count)
        
        if not should_train:
            print("\n✓ No retraining needed")
            print(f"  Current data: {data_count} points")
            print(f"  Last training: {self.last_train_time}")
            return
        
        print(f"\n🔄 Retraining triggered:")
        for reason in reasons:
            print(f"  • {reason}")
        
        # Train models
        results = self.train_models(csv_path)
        
        # Update state
        self.last_train_time = datetime.now()
        self.last_data_count = data_count
        
        # Log results
        self.log_training(results, data_count)
        
        # Save state
        self.save_state()
        
        print("\n" + "="*60)
        print("✓ Training cycle complete!")
        print("="*60)
    
    def start_scheduler(self, check_interval_hours=6):
        """Start automated scheduling"""
        print("\n" + "="*60)
        print("🤖 Automated Retraining System Started")
        print("="*60)
        print(f"Check interval: Every {check_interval_hours} hours")
        print(f"Retrain interval: Every {self.retrain_interval_days} days")
        print(f"Min new data required: {self.min_new_data_points} points")
        print("="*60)
        
        # Schedule periodic checks
        schedule.every(check_interval_hours).hours.do(self.run_training_cycle)
        
        # Run once immediately
        self.run_training_cycle()
        
        # Keep running
        print("\nScheduler running... (Press Ctrl+C to stop)")
        try:
            while True:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
        except KeyboardInterrupt:
            print("\n\n✓ Scheduler stopped")
            self.save_state()

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Automated Model Retraining System')
    parser.add_argument('--retrain-days', type=int, default=7,
                       help='Retrain every N days (default: 7)')
    parser.add_argument('--check-hours', type=int, default=6,
                       help='Check for retraining every N hours (default: 6)')
    parser.add_argument('--min-data', type=int, default=10080,
                       help='Minimum new data points to trigger retrain (default: 10080 = 1 week)')
    parser.add_argument('--once', action='store_true',
                       help='Run once and exit (no scheduling)')
    
    args = parser.parse_args()
    
    # Create system
    system = AutoRetrainingSystem(
        retrain_interval_days=args.retrain_days,
        min_new_data_points=args.min_data
    )
    
    if args.once:
        # Run once and exit
        system.run_training_cycle()
    else:
        # Start continuous scheduling
        system.start_scheduler(check_interval_hours=args.check_hours)

if __name__ == '__main__':
    main()
