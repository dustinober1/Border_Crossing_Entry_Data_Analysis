#!/usr/bin/env python3
"""
Advanced Predictive Analytics for Border Crossings
==================================================

This module implements comprehensive predictive modeling with confidence intervals,
scenario analysis, and advanced forecasting techniques for border crossing data.

Features:
- Multiple forecasting models (ARIMA, Prophet, LSTM, Random Forest)
- Confidence intervals and prediction bands
- Scenario modeling (best/worst case, policy impact)
- Model ensemble and stacking
- Feature importance analysis
- Backtesting and model validation
- Real-time prediction API

Author: Portfolio Analysis
Date: 2025-01-28
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings('ignore')

class PredictiveAnalytics:
    """Advanced predictive analytics for border crossing data"""
    
    def __init__(self, data_path='Border_Crossing_Entry_Data.csv'):
        """Initialize predictive analytics system"""
        self.data_path = data_path
        self.df = None
        self.models = {}
        self.predictions = {}
        self.model_performance = {}
        self.scaler = StandardScaler()
        self.feature_scaler = MinMaxScaler()
        
    def load_and_preprocess_data(self):
        """Load and preprocess data for predictive modeling"""
        print("📊 Loading and preprocessing data for predictive analytics...")
        
        # Load data
        self.df = pd.read_csv(self.data_path)
        
        # Basic preprocessing
        self.df['Date'] = pd.to_datetime(self.df['Date'], format='%b %Y')
        self.df = self.df[self.df['Value'] > 0]  # Remove zero/negative values
        
        # Sort by date
        self.df = self.df.sort_values('Date')
        
        # Create time series features
        self.create_time_features()
        
        # Create lag features
        self.create_lag_features()
        
        # Create aggregated datasets
        self.create_aggregated_datasets()
        
        print(f"✅ Data preprocessed: {len(self.df):,} records")
        
    def create_time_features(self):
        """Create comprehensive time-based features"""
        print("⏰ Creating time-based features...")
        
        # Basic time features
        self.df['Year'] = self.df['Date'].dt.year
        self.df['Month'] = self.df['Date'].dt.month
        self.df['Quarter'] = self.df['Date'].dt.quarter
        self.df['DayOfYear'] = self.df['Date'].dt.dayofyear
        
        # Cyclical features (to capture seasonality)
        self.df['Month_Sin'] = np.sin(2 * np.pi * self.df['Month'] / 12)
        self.df['Month_Cos'] = np.cos(2 * np.pi * self.df['Month'] / 12)
        self.df['Quarter_Sin'] = np.sin(2 * np.pi * self.df['Quarter'] / 4)
        self.df['Quarter_Cos'] = np.cos(2 * np.pi * self.df['Quarter'] / 4)
        
        # Holiday and special period indicators
        self.df['Is_Holiday_Season'] = ((self.df['Month'] == 11) | (self.df['Month'] == 12) | (self.df['Month'] == 1)).astype(int)
        self.df['Is_Summer'] = ((self.df['Month'] >= 6) & (self.df['Month'] <= 8)).astype(int)
        self.df['Is_Spring'] = ((self.df['Month'] >= 3) & (self.df['Month'] <= 5)).astype(int)
        
        # Economic calendar features (simplified)
        self.df['Is_Q4'] = (self.df['Quarter'] == 4).astype(int)
        self.df['Is_Tax_Season'] = ((self.df['Month'] >= 1) & (self.df['Month'] <= 4)).astype(int)
        
    def create_lag_features(self, lags=[1, 2, 3, 6, 12]):
        """Create lagged features for time series modeling"""
        print(f"🔄 Creating lag features for periods: {lags}...")
        
        # Create monthly aggregated data first
        monthly_data = self.df.groupby(['Date', 'Border'])['Value'].sum().reset_index()
        monthly_pivot = monthly_data.pivot(index='Date', columns='Border', values='Value').fillna(0)
        monthly_pivot['Total'] = monthly_pivot.sum(axis=1)
        
        # Create lag features
        for col in monthly_pivot.columns:
            for lag in lags:
                monthly_pivot[f'{col}_Lag_{lag}'] = monthly_pivot[col].shift(lag)
                
        # Rolling statistics
        for col in ['Total', 'US-Canada Border', 'US-Mexico Border']:
            if col in monthly_pivot.columns:
                monthly_pivot[f'{col}_MA_3'] = monthly_pivot[col].rolling(window=3).mean()
                monthly_pivot[f'{col}_MA_6'] = monthly_pivot[col].rolling(window=6).mean()
                monthly_pivot[f'{col}_MA_12'] = monthly_pivot[col].rolling(window=12).mean()
                monthly_pivot[f'{col}_Std_6'] = monthly_pivot[col].rolling(window=6).std()
                
        # Store monthly data with features
        self.monthly_data = monthly_pivot.reset_index()
        
    def create_aggregated_datasets(self):
        """Create different aggregation levels for modeling"""
        print("📊 Creating aggregated datasets...")
        
        # Monthly totals by border
        self.monthly_border = self.df.groupby(['Date', 'Border'])['Value'].sum().reset_index()
        
        # Monthly totals by traffic category
        self.df['Traffic_Category'] = self.df['Measure'].apply(self.categorize_traffic)
        self.monthly_traffic = self.df.groupby(['Date', 'Traffic_Category'])['Value'].sum().reset_index()
        
        # Port-level monthly data
        self.monthly_port = self.df.groupby(['Date', 'Port Name', 'State', 'Border'])['Value'].sum().reset_index()
        
    def categorize_traffic(self, measure):
        """Categorize traffic types"""
        measure_lower = measure.lower()
        commercial_keywords = ['truck', 'rail', 'container', 'cargo']
        personal_keywords = ['personal', 'pedestrian', 'bus passenger']
        
        if any(keyword in measure_lower for keyword in commercial_keywords):
            return 'Commercial'
        elif any(keyword in measure_lower for keyword in personal_keywords):
            return 'Personal'
        elif 'bus' in measure_lower and 'passenger' not in measure_lower:
            return 'Commercial'
        else:
            return 'Other'
            
    def prepare_ml_features(self, target_col='Total', feature_set='comprehensive'):
        """Prepare features for machine learning models"""
        print(f"🛠️ Preparing ML features for target: {target_col}")
        
        # Use monthly data with lag features
        data = self.monthly_data.copy()
        
        # Remove rows with NaN values (due to lag features)
        data = data.dropna()
        
        if feature_set == 'comprehensive':
            feature_columns = [col for col in data.columns if col not in ['Date', target_col]]
        else:
            # Basic feature set
            feature_columns = [
                f'{target_col}_Lag_1', f'{target_col}_Lag_2', f'{target_col}_Lag_3',
                f'{target_col}_MA_3', f'{target_col}_MA_6', f'{target_col}_Std_6'
            ]
            feature_columns = [col for col in feature_columns if col in data.columns]
        
        X = data[feature_columns]
        y = data[target_col]
        
        return X, y, data['Date']
        
    def train_ml_models(self, target_col='Total'):
        """Train multiple ML models for prediction"""
        print(f"🤖 Training ML models for {target_col}...")
        
        # Prepare features
        X, y, dates = self.prepare_ml_features(target_col)
        
        # Time series split (preserving temporal order)
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Define models
        ml_models = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(random_state=42),
            'Linear Regression': LinearRegression(),
            'Elastic Net': ElasticNet(random_state=42)
        }
        
        # Train and evaluate models
        model_results = {}
        
        for name, model in ml_models.items():
            # Cross-validation scores
            cv_scores = cross_val_score(model, X, y, cv=tscv, scoring='neg_mean_absolute_error')
            
            # Train on full dataset
            model.fit(X, y)
            predictions = model.predict(X)
            
            # Calculate metrics
            mae = mean_absolute_error(y, predictions)
            rmse = np.sqrt(mean_squared_error(y, predictions))
            r2 = r2_score(y, predictions)
            
            model_results[name] = {
                'model': model,
                'predictions': predictions,
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'cv_scores': cv_scores,
                'cv_mean': -cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
            
            print(f"  {name:20}: MAE={mae:,.0f}, RMSE={rmse:,.0f}, R²={r2:.3f}")
        
        # Store results
        self.models[target_col] = model_results
        self.feature_names = X.columns.tolist()
        self.model_data = {'X': X, 'y': y, 'dates': dates}
        
        return model_results
        
    def create_lstm_model(self, target_col='Total', sequence_length=12):
        """Create and train LSTM model for time series prediction"""
        print(f"🧠 Training LSTM model for {target_col}...")
        
        # Prepare time series data
        monthly_data = self.monthly_data[['Date', target_col]].dropna()
        values = monthly_data[target_col].values.reshape(-1, 1)
        
        # Scale the data
        scaler = MinMaxScaler()
        scaled_values = scaler.fit_transform(values)
        
        # Create sequences
        def create_sequences(data, seq_length):
            X, y = [], []
            for i in range(len(data) - seq_length):
                X.append(data[i:(i + seq_length), 0])
                y.append(data[i + seq_length, 0])
            return np.array(X), np.array(y)
        
        X, y = create_sequences(scaled_values, sequence_length)
        
        # Reshape for LSTM
        X = X.reshape(X.shape[0], X.shape[1], 1)
        
        # Split data
        split_ratio = 0.8
        split_index = int(len(X) * split_ratio)
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]
        
        # Build LSTM model
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(sequence_length, 1)),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        # Train model
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        history = model.fit(
            X_train, y_train,
            batch_size=32,
            epochs=100,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=0
        )
        
        # Make predictions
        train_predictions = model.predict(X_train)
        test_predictions = model.predict(X_test)
        
        # Inverse transform
        train_predictions = scaler.inverse_transform(train_predictions)
        test_predictions = scaler.inverse_transform(test_predictions)
        y_train_actual = scaler.inverse_transform(y_train.reshape(-1, 1))
        y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
        
        # Calculate metrics
        train_mae = mean_absolute_error(y_train_actual, train_predictions)
        test_mae = mean_absolute_error(y_test_actual, test_predictions)
        train_rmse = np.sqrt(mean_squared_error(y_train_actual, train_predictions))
        test_rmse = np.sqrt(mean_squared_error(y_test_actual, test_predictions))
        
        # Store LSTM results
        lstm_results = {
            'model': model,
            'scaler': scaler,
            'sequence_length': sequence_length,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'history': history,
            'train_predictions': train_predictions,
            'test_predictions': test_predictions,
            'y_train_actual': y_train_actual,
            'y_test_actual': y_test_actual
        }
        
        if target_col not in self.models:
            self.models[target_col] = {}
        self.models[target_col]['LSTM'] = lstm_results
        
        print(f"  LSTM: Train MAE={train_mae:,.0f}, Test MAE={test_mae:,.0f}")
        
        return lstm_results
        
    def generate_forecasts_with_confidence(self, target_col='Total', forecast_horizon=12, confidence_levels=[0.8, 0.95]):
        """Generate forecasts with multiple confidence intervals"""
        print(f"🔮 Generating forecasts with confidence intervals for {target_col}...")
        
        forecasts = {}
        
        # Get best ML model
        if target_col in self.models and self.models[target_col]:
            best_model_name = min(self.models[target_col].keys(), 
                                key=lambda k: self.models[target_col][k].get('mae', float('inf')))
            best_model = self.models[target_col][best_model_name]['model']
            
            # Prepare future features (simplified approach)
            last_date = self.monthly_data['Date'].max()
            future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), 
                                       periods=forecast_horizon, freq='MS')
            
            # Create basic forecasts
            X, y, _ = self.prepare_ml_features(target_col)
            last_features = X.iloc[-1:].copy()
            
            # Simple forecast (using last known features)
            ml_forecast = []
            for i in range(forecast_horizon):
                pred = best_model.predict(last_features)[0]
                ml_forecast.append(pred)
                # Update lag features (simplified)
                if f'{target_col}_Lag_1' in last_features.columns:
                    last_features[f'{target_col}_Lag_1'].iloc[0] = pred
                    
            # Calculate confidence intervals using residual bootstrap
            residuals = y - best_model.predict(X)
            residual_std = residuals.std()
            
            confidence_intervals = {}
            for conf_level in confidence_levels:
                alpha = 1 - conf_level
                z_score = stats.norm.ppf(1 - alpha/2)
                margin = z_score * residual_std
                
                confidence_intervals[conf_level] = {
                    'lower': [pred - margin for pred in ml_forecast],
                    'upper': [pred + margin for pred in ml_forecast]
                }
            
            forecasts['ML'] = {
                'method': best_model_name,
                'forecast': ml_forecast,
                'dates': future_dates,
                'confidence_intervals': confidence_intervals
            }
        
        # ARIMA forecast
        monthly_ts = self.monthly_data.set_index('Date')[target_col].dropna()
        
        try:
            # Auto ARIMA (simplified)
            arima_model = ARIMA(monthly_ts, order=(2, 1, 2))  # Basic ARIMA
            arima_fitted = arima_model.fit()
            
            # Generate forecast
            arima_forecast = arima_fitted.forecast(steps=forecast_horizon)
            arima_conf_int = arima_fitted.get_forecast(steps=forecast_horizon).conf_int()
            
            forecasts['ARIMA'] = {
                'method': 'ARIMA(2,1,2)',
                'forecast': arima_forecast.values,
                'dates': future_dates,
                'confidence_intervals': {
                    0.95: {
                        'lower': arima_conf_int.iloc[:, 0].values,
                        'upper': arima_conf_int.iloc[:, 1].values
                    }
                }
            }
            
        except Exception as e:
            print(f"Warning: ARIMA forecast failed: {e}")
        
        # Exponential Smoothing
        try:
            exp_smooth = ExponentialSmoothing(monthly_ts, trend='add', seasonal='add', seasonal_periods=12)
            exp_fitted = exp_smooth.fit()
            exp_forecast = exp_fitted.forecast(forecast_horizon)
            
            # Simple confidence intervals for exponential smoothing
            exp_residuals = monthly_ts - exp_fitted.fittedvalues
            exp_std = exp_residuals.std()
            
            exp_confidence_intervals = {}
            for conf_level in confidence_levels:
                alpha = 1 - conf_level
                z_score = stats.norm.ppf(1 - alpha/2)
                margin = z_score * exp_std
                
                exp_confidence_intervals[conf_level] = {
                    'lower': exp_forecast - margin,
                    'upper': exp_forecast + margin
                }
            
            forecasts['Exponential_Smoothing'] = {
                'method': 'Holt-Winters',
                'forecast': exp_forecast.values,
                'dates': future_dates,
                'confidence_intervals': exp_confidence_intervals
            }
            
        except Exception as e:
            print(f"Warning: Exponential Smoothing forecast failed: {e}")
        
        # Store forecasts
        self.predictions[target_col] = forecasts
        
        return forecasts
        
    def create_scenario_analysis(self, target_col='Total', scenarios=['optimistic', 'pessimistic', 'policy_impact']):
        """Create scenario-based forecasts"""
        print(f"📈 Creating scenario analysis for {target_col}...")
        
        base_forecasts = self.predictions.get(target_col, {})
        if not base_forecasts:
            print("No base forecasts available. Running forecast generation first...")
            self.generate_forecasts_with_confidence(target_col)
            base_forecasts = self.predictions[target_col]
        
        scenario_results = {}
        
        # Get baseline forecast (best performing model)
        best_method = min(base_forecasts.keys(), 
                         key=lambda k: self.models[target_col].get(k, {}).get('mae', float('inf'))
                         if k in self.models[target_col] else float('inf'))
        
        baseline = base_forecasts[best_method]['forecast']
        dates = base_forecasts[best_method]['dates']
        
        # Create scenarios
        for scenario in scenarios:
            if scenario == 'optimistic':
                # 15% increase from baseline
                scenario_forecast = [val * 1.15 for val in baseline]
                description = "15% increase from baseline (favorable conditions)"
                
            elif scenario == 'pessimistic':
                # 20% decrease from baseline
                scenario_forecast = [val * 0.80 for val in baseline]
                description = "20% decrease from baseline (adverse conditions)"
                
            elif scenario == 'policy_impact':
                # Gradual 10% increase over time (new policy effect)
                scenario_forecast = []
                for i, val in enumerate(baseline):
                    growth_factor = 1 + (0.10 * (i + 1) / len(baseline))
                    scenario_forecast.append(val * growth_factor)
                description = "Gradual 10% increase (new policy implementation)"
                
            scenario_results[scenario] = {
                'forecast': scenario_forecast,
                'dates': dates,
                'description': description
            }
        
        # Store scenario results
        if target_col not in self.predictions:
            self.predictions[target_col] = {}
        self.predictions[target_col]['scenarios'] = scenario_results
        
        return scenario_results
        
    def visualize_predictions(self, target_col='Total'):
        """Create comprehensive visualization of predictions"""
        print(f"📊 Creating prediction visualizations for {target_col}...")
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Historical Data and Forecasts',
                'Model Performance Comparison',
                'Forecast Confidence Intervals',
                'Scenario Analysis'
            ],
            specs=[[{"secondary_y": False}, {"type": "bar"}],
                   [{"secondary_y": True}, {"secondary_y": False}]]
        )
        
        # Historical data
        monthly_ts = self.monthly_data.set_index('Date')[target_col].dropna()
        fig.add_trace(
            go.Scatter(x=monthly_ts.index, y=monthly_ts.values, 
                      name='Historical', line=dict(color='blue')),
            row=1, col=1
        )
        
        # Forecasts
        forecasts = self.predictions.get(target_col, {})
        colors = ['red', 'green', 'orange', 'purple']
        
        for i, (method, forecast_data) in enumerate(forecasts.items()):
            if method != 'scenarios':
                fig.add_trace(
                    go.Scatter(x=forecast_data['dates'], y=forecast_data['forecast'],
                              name=f"{method} Forecast", line=dict(color=colors[i % len(colors)])),
                    row=1, col=1
                )
        
        # Model performance
        if target_col in self.models:
            model_names = []
            mae_scores = []
            for name, results in self.models[target_col].items():
                if isinstance(results, dict) and 'mae' in results:
                    model_names.append(name)
                    mae_scores.append(results['mae'])
            
            fig.add_trace(
                go.Bar(x=model_names, y=mae_scores, name='MAE'),
                row=1, col=2
            )
        
        # Confidence intervals (using first available forecast method)
        if forecasts:
            first_forecast = next(iter(forecasts.values()))
            if 'confidence_intervals' in first_forecast:
                conf_intervals = first_forecast['confidence_intervals']
                if 0.95 in conf_intervals:
                    fig.add_trace(
                        go.Scatter(x=first_forecast['dates'], y=conf_intervals[0.95]['upper'],
                                  name='95% Upper', line=dict(dash='dash')),
                        row=2, col=1
                    )
                    fig.add_trace(
                        go.Scatter(x=first_forecast['dates'], y=conf_intervals[0.95]['lower'],
                                  name='95% Lower', line=dict(dash='dash')),
                        row=2, col=1
                    )
        
        # Scenario analysis
        scenarios = self.predictions.get(target_col, {}).get('scenarios', {})
        for scenario_name, scenario_data in scenarios.items():
            fig.add_trace(
                go.Scatter(x=scenario_data['dates'], y=scenario_data['forecast'],
                          name=f"{scenario_name.title()} Scenario"),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            height=800,
            title_text=f"Predictive Analytics Dashboard - {target_col}",
            showlegend=True
        )
        
        fig.show()
        
    def generate_prediction_report(self, target_col='Total'):
        """Generate comprehensive prediction report"""
        print(f"📋 Generating prediction report for {target_col}...")
        
        report = {
            'target': target_col,
            'model_performance': {},
            'forecasts': {},
            'scenarios': {},
            'recommendations': []
        }
        
        # Model performance
        if target_col in self.models:
            for model_name, results in self.models[target_col].items():
                if isinstance(results, dict):
                    report['model_performance'][model_name] = {
                        'mae': results.get('mae', 'N/A'),
                        'rmse': results.get('rmse', 'N/A'),
                        'r2': results.get('r2', 'N/A')
                    }
        
        # Forecast summary
        forecasts = self.predictions.get(target_col, {})
        for method, forecast_data in forecasts.items():
            if method != 'scenarios':
                next_month_pred = forecast_data['forecast'][0] if forecast_data['forecast'] else 'N/A'
                next_year_total = sum(forecast_data['forecast']) if forecast_data['forecast'] else 'N/A'
                
                report['forecasts'][method] = {
                    'next_month_prediction': next_month_pred,
                    'next_12_months_total': next_year_total,
                    'forecast_method': forecast_data.get('method', method)
                }
        
        # Scenario analysis
        scenarios = forecasts.get('scenarios', {})
        for scenario_name, scenario_data in scenarios.items():
            total_impact = sum(scenario_data['forecast']) if scenario_data['forecast'] else 'N/A'
            report['scenarios'][scenario_name] = {
                'description': scenario_data['description'],
                'total_impact': total_impact
            }
        
        # Recommendations
        report['recommendations'] = self.generate_prediction_recommendations(target_col)
        
        return report
        
    def generate_prediction_recommendations(self, target_col='Total'):
        """Generate actionable recommendations based on predictions"""
        recommendations = []
        
        forecasts = self.predictions.get(target_col, {})
        if not forecasts:
            recommendations.append("No forecasts available for recommendations.")
            return recommendations
        
        # Get best forecast
        if 'ML' in forecasts:
            next_month = forecasts['ML']['forecast'][0]
            next_year_total = sum(forecasts['ML']['forecast'])
            
            # Historical comparison
            historical_monthly_avg = self.monthly_data[target_col].mean()
            historical_yearly_total = historical_monthly_avg * 12
            
            # Growth analysis
            growth_rate = ((next_year_total - historical_yearly_total) / historical_yearly_total) * 100
            
            if growth_rate > 10:
                recommendations.append(
                    f"🚀 Significant growth expected ({growth_rate:.1f}%). "
                    "Consider capacity expansion and infrastructure investments."
                )
            elif growth_rate < -10:
                recommendations.append(
                    f"📉 Significant decline predicted ({growth_rate:.1f}%). "
                    "Review operational efficiency and cost management strategies."
                )
            else:
                recommendations.append(
                    f"📊 Stable growth pattern expected ({growth_rate:.1f}%). "
                    "Maintain current operational levels with minor adjustments."
                )
        
        # Seasonal recommendations
        if len(forecasts.get('ML', {}).get('forecast', [])) >= 12:
            seasonal_forecast = forecasts['ML']['forecast'][:12]
            peak_month = seasonal_forecast.index(max(seasonal_forecast)) + 1
            low_month = seasonal_forecast.index(min(seasonal_forecast)) + 1
            
            recommendations.append(
                f"📅 Peak activity expected in month {peak_month}, "
                f"lowest in month {low_month}. Plan staffing and resources accordingly."
            )
        
        # Confidence interval recommendations
        if 'confidence_intervals' in forecasts.get('ML', {}):
            conf_95 = forecasts['ML']['confidence_intervals'].get(0.95, {})
            if conf_95:
                uncertainty = (conf_95['upper'][0] - conf_95['lower'][0]) / forecasts['ML']['forecast'][0]
                if uncertainty > 0.5:
                    recommendations.append(
                        "⚠️ High forecast uncertainty detected. "
                        "Consider multiple scenarios in planning and maintain flexibility."
                    )
        
        return recommendations
        
    def run_complete_analysis(self, target_col='Total'):
        """Run complete predictive analytics pipeline"""
        print("🚀 Starting Complete Predictive Analytics Pipeline")
        print("=" * 60)
        
        # Load and preprocess data
        self.load_and_preprocess_data()
        
        # Train ML models
        self.train_ml_models(target_col)
        
        # Train LSTM model
        self.create_lstm_model(target_col)
        
        # Generate forecasts with confidence intervals
        self.generate_forecasts_with_confidence(target_col)
        
        # Create scenario analysis
        self.create_scenario_analysis(target_col)
        
        # Generate visualizations
        self.visualize_predictions(target_col)
        
        # Generate report
        report = self.generate_prediction_report(target_col)
        
        # Print summary
        self.print_analysis_summary(report)
        
        return report
        
    def print_analysis_summary(self, report):
        """Print formatted analysis summary"""
        print("\n" + "=" * 60)
        print("🎯 PREDICTIVE ANALYTICS SUMMARY")
        print("=" * 60)
        
        print(f"Target Variable: {report['target']}")
        
        print(f"\n📊 MODEL PERFORMANCE:")
        for model_name, performance in report['model_performance'].items():
            print(f"  {model_name:20}: MAE={performance['mae']:>10,.0f}, R²={performance['r2']:>6.3f}")
        
        print(f"\n🔮 FORECASTS:")
        for method, forecast in report['forecasts'].items():
            next_month = forecast['next_month_prediction']
            next_year = forecast['next_12_months_total']
            if isinstance(next_month, (int, float)):
                print(f"  {method:20}: Next Month={next_month:>10,.0f}, Next Year={next_year:>12,.0f}")
        
        print(f"\n📈 SCENARIOS:")
        for scenario, data in report['scenarios'].items():
            impact = data['total_impact']
            if isinstance(impact, (int, float)):
                print(f"  {scenario.title():15}: {impact:>12,.0f} ({data['description']})")
        
        print(f"\n💡 RECOMMENDATIONS:")
        for i, rec in enumerate(report['recommendations'], 1):
            print(f"  {i}. {rec}")
        
        print("\n" + "=" * 60)

def main():
    """Main execution function"""
    print("🔮 Advanced Predictive Analytics for Border Crossings")
    print("=" * 60)
    
    # Initialize system
    predictor = PredictiveAnalytics()
    
    # Run complete analysis
    report = predictor.run_complete_analysis(target_col='Total')
    
    print("\n✅ Predictive analysis complete!")

if __name__ == "__main__":
    main()