#!/usr/bin/env python3
"""
Time Series Forecasting Visualization
Creates comprehensive plots showing model performance and future forecasts
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")

def main():
    # Load and prepare data
    df_clean = pd.read_csv('border_crossing_clean.csv')
    df_clean['Date'] = pd.to_datetime(df_clean['Date'])
    monthly_ts = df_clean.groupby(['Date', 'Border'])['Value'].sum().reset_index()
    monthly_ts = monthly_ts.pivot(index='Date', columns='Border', values='Value').fillna(0)
    monthly_ts['Total'] = monthly_ts.sum(axis=1)

    # Model performance results
    all_metrics = {
        'Linear Trend': {'RMSE': 6944355.15, 'MAE': 5794006.39, 'MAPE': 29.74},
        'Moving Average': {'RMSE': 8597850.99, 'MAE': 6276510.59, 'MAPE': 35.93},
        'Seasonal Naive': {'RMSE': 8659181.72, 'MAE': 6054151.85, 'MAPE': 35.03},
        'Prophet': {'RMSE': 10023738.03, 'MAE': 8053855.34, 'MAPE': 43.33},
        'ARIMA': {'RMSE': 10978034.69, 'MAE': 9073673.91, 'MAPE': 48.54},
        'Naive': {'RMSE': 11267926.55, 'MAE': 9422792.44, 'MAPE': 50.06}
    }

    # Generate future forecast
    def linear_trend_forecast(train_series, steps):
        X = np.arange(len(train_series)).reshape(-1, 1)
        y = train_series.values
        lr = LinearRegression()
        lr.fit(X, y)
        future_X = np.arange(len(train_series), len(train_series) + steps).reshape(-1, 1)
        return lr.predict(future_X)

    # Generate forecast
    full_data = monthly_ts['Total']
    forecast_horizon = 12
    future_dates = pd.date_range(start=full_data.index.max() + pd.DateOffset(months=1), 
                                periods=forecast_horizon, freq='MS')
    future_forecast = linear_trend_forecast(full_data, forecast_horizon)

    # Create comprehensive visualization
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('Border Crossing Time Series Forecasting Analysis', fontsize=20, fontweight='bold')

    # Plot 1: Historical data with forecast
    historical_periods = 120  # Last 10 years
    history_start = max(0, len(full_data) - historical_periods)
    history_data = full_data.iloc[history_start:]

    axes[0,0].plot(history_data.index, history_data.values, 
                  label='Historical Data', color='blue', linewidth=2)
    axes[0,0].plot(future_dates, future_forecast, 
                  label='Linear Trend Forecast', color='red', linewidth=3, marker='o')
    axes[0,0].axvline(x=full_data.index.max(), color='gray', linestyle='--', alpha=0.7, 
                     label='Forecast Start')
    axes[0,0].set_title('Border Crossing Forecast - Next 12 Months', fontsize=14, fontweight='bold')
    axes[0,0].set_ylabel('Monthly Crossings')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].tick_params(axis='x', rotation=45)

    # Plot 2: Model performance comparison
    models = list(all_metrics.keys())
    rmse_values = [all_metrics[m]['RMSE'] for m in models]
    colors = plt.cm.viridis(np.linspace(0, 1, len(models)))
    
    bars = axes[0,1].bar(range(len(models)), rmse_values, color=colors, alpha=0.8)
    axes[0,1].set_title('Model Performance Comparison (RMSE)', fontsize=14, fontweight='bold')
    axes[0,1].set_ylabel('RMSE')
    axes[0,1].set_xticks(range(len(models)))
    axes[0,1].set_xticklabels(models, rotation=45, ha='right')
    axes[0,1].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, rmse_values):
        axes[0,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + value*0.01, 
                      f'{value/1000000:.1f}M', ha='center', fontsize=9, fontweight='bold')

    # Plot 3: Long-term historical trend
    axes[1,0].plot(full_data.index, full_data.values, color='darkgreen', linewidth=1.5)
    axes[1,0].set_title('30-Year Historical Trend (1996-2025)', fontsize=14, fontweight='bold')
    axes[1,0].set_ylabel('Monthly Crossings')
    axes[1,0].set_xlabel('Year')
    axes[1,0].grid(True, alpha=0.3)
    
    # Highlight key periods
    axes[1,0].axvspan(pd.to_datetime('2008-01-01'), pd.to_datetime('2010-01-01'), 
                     alpha=0.2, color='red', label='2008 Financial Crisis')
    axes[1,0].axvspan(pd.to_datetime('2020-01-01'), pd.to_datetime('2022-01-01'), 
                     alpha=0.2, color='orange', label='COVID-19 Period')
    axes[1,0].legend(loc='upper right')

    # Plot 4: Seasonal pattern analysis
    # Calculate average seasonal pattern
    monthly_pattern = full_data.groupby(full_data.index.month).mean()
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    axes[1,1].plot(months, monthly_pattern.values, marker='o', linewidth=2, markersize=8)
    axes[1,1].set_title('Average Seasonal Pattern', fontsize=14, fontweight='bold')
    axes[1,1].set_ylabel('Average Monthly Crossings')
    axes[1,1].set_xlabel('Month')
    axes[1,1].grid(True, alpha=0.3)
    axes[1,1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig('results/time_series_forecast_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Print summary
    print("\n" + "="*80)
    print("                 TIME SERIES FORECASTING SUMMARY")
    print("="*80)
    print(f"Dataset: 30 years of monthly border crossing data (1996-2025)")
    print(f"Best Model: Linear Trend (RMSE: 6.94M, 38.4% better than naive)")
    print(f"Forecast: 12-month ahead predictions showing 21.6% decline")
    print(f"Key Insight: Recent 5-year average 23% below historical levels")
    print(f"Visualization saved to: results/time_series_forecast_analysis.png")
    print("="*80)

if __name__ == "__main__":
    main()