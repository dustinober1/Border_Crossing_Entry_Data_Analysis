#!/usr/bin/env python3
"""
Advanced Anomaly Detection System for Border Crossings
======================================================

This module implements a comprehensive anomaly detection system for border crossing data
using multiple algorithms and techniques to identify unusual patterns, outliers, and
potential security concerns.

Features:
- Multiple anomaly detection algorithms (Isolation Forest, Local Outlier Factor, One-Class SVM)
- Statistical anomaly detection (Z-score, Modified Z-score, IQR method)
- Time series anomaly detection for seasonal patterns
- Real-time alert system
- Anomaly severity scoring
- Detailed reporting and visualization

Author: Portfolio Analysis
Date: 2025-01-28
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class AnomalyDetectionSystem:
    """Comprehensive anomaly detection for border crossing data"""
    
    def __init__(self, data_path='Border_Crossing_Entry_Data.csv'):
        """Initialize the anomaly detection system"""
        self.data_path = data_path
        self.df = None
        self.df_processed = None
        self.anomaly_results = {}
        self.scaler = StandardScaler()
        self.robust_scaler = RobustScaler()
        
    def load_and_preprocess_data(self):
        """Load and preprocess data for anomaly detection"""
        print("📊 Loading and preprocessing data...")
        
        # Load data
        self.df = pd.read_csv(self.data_path)
        
        # Basic preprocessing
        self.df['Date'] = pd.to_datetime(self.df['Date'], format='%b %Y')
        self.df = self.df[self.df['Value'] > 0]  # Remove zero/negative values
        
        # Add temporal features
        self.df['Year'] = self.df['Date'].dt.year
        self.df['Month'] = self.df['Date'].dt.month
        self.df['Quarter'] = self.df['Date'].dt.quarter
        self.df['DayOfYear'] = self.df['Date'].dt.dayofyear
        
        # Traffic categorization
        def categorize_traffic(measure):
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
                
        self.df['Traffic_Category'] = self.df['Measure'].apply(categorize_traffic)
        
        # Create aggregated features
        self.create_aggregated_features()
        
        print(f"✅ Data preprocessed: {len(self.df):,} records")
        
    def create_aggregated_features(self):
        """Create aggregated features for anomaly detection"""
        print("🔧 Creating aggregated features...")
        
        # Port-level statistics
        port_stats = self.df.groupby('Port Name').agg({
            'Value': ['mean', 'std', 'count', 'sum', 'median'],
            'Date': ['min', 'max']
        }).round(2)
        port_stats.columns = [f'Port_{col}_{stat}' for col, stat in port_stats.columns]
        port_stats['Port_Date_Range'] = (port_stats['Port_Date_max'] - port_stats['Port_Date_min']).dt.days
        
        # Merge back to main dataset
        self.df = self.df.merge(port_stats.reset_index(), on='Port Name', how='left')
        
        # State-level features
        state_stats = self.df.groupby('State').agg({
            'Value': ['mean', 'count'],
            'Port Name': 'nunique'
        }).round(2)
        state_stats.columns = [f'State_{col}_{stat}' for col, stat in state_stats.columns]
        
        self.df = self.df.merge(state_stats.reset_index(), on='State', how='left')
        
        # Monthly averages by port
        monthly_port_avg = self.df.groupby(['Port Name', 'Month'])['Value'].mean().reset_index()
        monthly_port_avg = monthly_port_avg.rename(columns={'Value': 'Monthly_Port_Avg'})
        self.df = self.df.merge(monthly_port_avg, on=['Port Name', 'Month'], how='left')
        
        # Calculate deviation from port average
        self.df['Deviation_From_Port_Avg'] = (self.df['Value'] - self.df['Port_Value_mean']) / (self.df['Port_Value_std'] + 1e-6)
        
        # Calculate deviation from monthly average
        self.df['Deviation_From_Monthly_Avg'] = (self.df['Value'] - self.df['Monthly_Port_Avg']) / (self.df['Monthly_Port_Avg'] + 1e-6)
        
        print("✅ Aggregated features created")
        
    def statistical_anomaly_detection(self, contamination=0.05):
        """Detect anomalies using statistical methods"""
        print("📊 Running statistical anomaly detection...")
        
        results = {}
        
        # 1. Z-Score method
        z_scores = np.abs(stats.zscore(self.df['Value']))
        z_threshold = stats.norm.ppf(1 - contamination/2)
        results['zscore_anomalies'] = z_scores > z_threshold
        
        # 2. Modified Z-Score (using median absolute deviation)
        median = np.median(self.df['Value'])
        mad = np.median(np.abs(self.df['Value'] - median))
        modified_z_scores = 0.6745 * (self.df['Value'] - median) / (mad + 1e-6)
        modified_z_threshold = stats.norm.ppf(1 - contamination/2)
        results['modified_zscore_anomalies'] = np.abs(modified_z_scores) > modified_z_threshold
        
        # 3. IQR method
        Q1 = self.df['Value'].quantile(0.25)
        Q3 = self.df['Value'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        results['iqr_anomalies'] = (self.df['Value'] < lower_bound) | (self.df['Value'] > upper_bound)
        
        # 4. Percentile method
        lower_percentile = self.df['Value'].quantile(contamination/2)
        upper_percentile = self.df['Value'].quantile(1 - contamination/2)
        results['percentile_anomalies'] = (self.df['Value'] < lower_percentile) | (self.df['Value'] > upper_percentile)
        
        # Store results
        for method, anomalies in results.items():
            self.df[method] = anomalies.astype(int)
            self.anomaly_results[method] = {
                'count': anomalies.sum(),
                'percentage': (anomalies.sum() / len(self.df)) * 100
            }
            
        print(f"✅ Statistical anomaly detection complete")
        
    def machine_learning_anomaly_detection(self, contamination=0.05):
        """Detect anomalies using machine learning algorithms"""
        print("🤖 Running ML-based anomaly detection...")
        
        # Prepare features for ML algorithms
        feature_columns = [
            'Value', 'Month', 'Quarter', 'Port_Value_mean', 'Port_Value_std',
            'State_Value_count', 'Deviation_From_Port_Avg', 'Deviation_From_Monthly_Avg'
        ]
        
        # Handle missing values and prepare feature matrix
        X = self.df[feature_columns].fillna(0)
        X_scaled = self.scaler.fit_transform(X)
        X_robust = self.robust_scaler.fit_transform(X)
        
        # 1. Isolation Forest
        iso_forest = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
        iso_predictions = iso_forest.fit_predict(X_scaled)
        self.df['isolation_forest_anomalies'] = (iso_predictions == -1).astype(int)
        
        # 2. Local Outlier Factor
        lof = LocalOutlierFactor(contamination=contamination, n_jobs=-1)
        lof_predictions = lof.fit_predict(X_scaled)
        self.df['lof_anomalies'] = (lof_predictions == -1).astype(int)
        
        # 3. One-Class SVM
        one_class_svm = OneClassSVM(nu=contamination, kernel='rbf')
        svm_predictions = one_class_svm.fit_predict(X_robust)
        self.df['one_class_svm_anomalies'] = (svm_predictions == -1).astype(int)
        
        # Store results
        ml_methods = ['isolation_forest_anomalies', 'lof_anomalies', 'one_class_svm_anomalies']
        for method in ml_methods:
            anomalies = self.df[method] == 1
            self.anomaly_results[method] = {
                'count': anomalies.sum(),
                'percentage': (anomalies.sum() / len(self.df)) * 100
            }
            
        print(f"✅ ML anomaly detection complete")
        
    def time_series_anomaly_detection(self):
        """Detect anomalies in time series patterns"""
        print("⏱️ Running time series anomaly detection...")
        
        # Group by month for time series analysis
        monthly_data = self.df.groupby('Date')['Value'].sum().reset_index()
        monthly_data = monthly_data.sort_values('Date')
        
        # Calculate rolling statistics
        window_size = 12  # 12-month rolling window
        monthly_data['rolling_mean'] = monthly_data['Value'].rolling(window=window_size).mean()
        monthly_data['rolling_std'] = monthly_data['Value'].rolling(window=window_size).std()
        
        # Identify seasonal anomalies
        monthly_data['seasonal_anomaly'] = (
            np.abs(monthly_data['Value'] - monthly_data['rolling_mean']) > 
            2 * monthly_data['rolling_std']
        ).fillna(False)
        
        # Map back to original data
        date_anomalies = dict(zip(monthly_data['Date'], monthly_data['seasonal_anomaly']))
        self.df['seasonal_anomaly'] = self.df['Date'].map(date_anomalies).fillna(False).astype(int)
        
        # Store results
        seasonal_anomalies = self.df['seasonal_anomaly'] == 1
        self.anomaly_results['seasonal_anomaly'] = {
            'count': seasonal_anomalies.sum(),
            'percentage': (seasonal_anomalies.sum() / len(self.df)) * 100
        }
        
        print(f"✅ Time series anomaly detection complete")
        
    def ensemble_anomaly_detection(self):
        """Create ensemble anomaly score combining multiple methods"""
        print("🎯 Creating ensemble anomaly scores...")
        
        # List of anomaly detection methods
        anomaly_methods = [
            'zscore_anomalies', 'modified_zscore_anomalies', 'iqr_anomalies',
            'isolation_forest_anomalies', 'lof_anomalies', 'one_class_svm_anomalies',
            'seasonal_anomaly'
        ]
        
        # Calculate ensemble score (sum of individual method flags)
        self.df['ensemble_anomaly_score'] = self.df[anomaly_methods].sum(axis=1)
        
        # Create severity categories
        max_score = len(anomaly_methods)
        self.df['anomaly_severity'] = pd.cut(
            self.df['ensemble_anomaly_score'],
            bins=[-0.1, 0.5, 2.5, 4.5, max_score],
            labels=['Normal', 'Low', 'Medium', 'High']
        )
        
        # High confidence anomalies (flagged by multiple methods)
        high_confidence_threshold = max_score * 0.4  # At least 40% of methods agree
        self.df['high_confidence_anomaly'] = (
            self.df['ensemble_anomaly_score'] >= high_confidence_threshold
        ).astype(int)
        
        # Store ensemble results
        self.anomaly_results['ensemble'] = {
            'high_confidence_count': (self.df['high_confidence_anomaly'] == 1).sum(),
            'severity_distribution': self.df['anomaly_severity'].value_counts().to_dict()
        }
        
        print(f"✅ Ensemble anomaly scoring complete")
        
    def generate_anomaly_report(self):
        """Generate comprehensive anomaly detection report"""
        print("📋 Generating anomaly detection report...")
        
        # High confidence anomalies
        high_conf_anomalies = self.df[self.df['high_confidence_anomaly'] == 1].copy()
        
        # Sort by ensemble score (highest anomaly score first)
        high_conf_anomalies = high_conf_anomalies.sort_values(
            'ensemble_anomaly_score', ascending=False
        )
        
        # Generate report sections
        report = {
            'summary': self.create_anomaly_summary(),
            'top_anomalies': self.identify_top_anomalies(high_conf_anomalies),
            'pattern_analysis': self.analyze_anomaly_patterns(high_conf_anomalies),
            'recommendations': self.generate_recommendations(high_conf_anomalies)
        }
        
        return report
        
    def create_anomaly_summary(self):
        """Create summary statistics for anomaly detection"""
        summary = {
            'total_records': len(self.df),
            'method_comparison': {},
            'ensemble_results': self.anomaly_results['ensemble']
        }
        
        # Method comparison
        for method, results in self.anomaly_results.items():
            if method != 'ensemble':
                summary['method_comparison'][method] = {
                    'count': results['count'],
                    'percentage': results['percentage']
                }
                
        return summary
        
    def identify_top_anomalies(self, anomalies_df, top_n=20):
        """Identify top anomalies for investigation"""
        if anomalies_df.empty:
            return []
            
        top_anomalies = anomalies_df.head(top_n)
        
        anomaly_list = []
        for idx, row in top_anomalies.iterrows():
            anomaly_info = {
                'port': row['Port Name'],
                'state': row['State'],
                'border': row['Border'],
                'date': row['Date'].strftime('%B %Y'),
                'measure': row['Measure'],
                'value': row['Value'],
                'anomaly_score': row['ensemble_anomaly_score'],
                'severity': row['anomaly_severity'],
                'deviation_from_avg': row['Deviation_From_Port_Avg']
            }
            anomaly_list.append(anomaly_info)
            
        return anomaly_list
        
    def analyze_anomaly_patterns(self, anomalies_df):
        """Analyze patterns in detected anomalies"""
        if anomalies_df.empty:
            return {}
            
        patterns = {
            'by_border': anomalies_df['Border'].value_counts().to_dict(),
            'by_state': anomalies_df['State'].value_counts().head(10).to_dict(),
            'by_port': anomalies_df['Port Name'].value_counts().head(10).to_dict(),
            'by_traffic_category': anomalies_df['Traffic_Category'].value_counts().to_dict(),
            'by_month': anomalies_df['Month'].value_counts().sort_index().to_dict(),
            'by_severity': anomalies_df['anomaly_severity'].value_counts().to_dict()
        }
        
        return patterns
        
    def generate_recommendations(self, anomalies_df):
        """Generate actionable recommendations based on anomalies"""
        recommendations = []
        
        if anomalies_df.empty:
            recommendations.append("No high-confidence anomalies detected. Continue regular monitoring.")
            return recommendations
            
        # Port-specific recommendations
        top_anomaly_ports = anomalies_df['Port Name'].value_counts().head(5)
        if not top_anomaly_ports.empty:
            recommendations.append(
                f"🚨 Priority Investigation: {top_anomaly_ports.index[0]} has {top_anomaly_ports.iloc[0]} anomalies. "
                "Consider detailed security review and capacity analysis."
            )
            
        # Border-specific recommendations
        border_anomalies = anomalies_df['Border'].value_counts()
        for border, count in border_anomalies.items():
            percentage = (count / len(anomalies_df)) * 100
            if percentage > 70:
                recommendations.append(
                    f"📊 {border} accounts for {percentage:.1f}% of anomalies. "
                    "Review resource allocation and monitoring protocols."
                )
                
        # Seasonal recommendations
        seasonal_anomalies = anomalies_df.groupby('Month')['ensemble_anomaly_score'].mean().sort_values(ascending=False)
        if not seasonal_anomalies.empty:
            peak_month = seasonal_anomalies.index[0]
            recommendations.append(
                f"📅 Month {peak_month} shows highest anomaly scores. "
                "Consider enhanced monitoring during this period."
            )
            
        # Volume-based recommendations
        high_volume_anomalies = anomalies_df[anomalies_df['Value'] > anomalies_df['Value'].quantile(0.9)]
        if not high_volume_anomalies.empty:
            recommendations.append(
                f"📈 {len(high_volume_anomalies)} high-volume anomalies detected. "
                "These may indicate capacity issues or unusual events requiring attention."
            )
            
        return recommendations
        
    def visualize_anomalies(self):
        """Create comprehensive visualizations of anomaly detection results"""
        print("📊 Creating anomaly visualization dashboard...")
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                'Anomaly Detection Methods Comparison',
                'Anomaly Severity Distribution',
                'Anomalies Over Time',
                'Geographic Distribution of Anomalies',
                'Top Anomalous Ports',
                'Ensemble Anomaly Scores'
            ],
            specs=[[{"type": "bar"}, {"type": "pie"}],
                   [{"type": "scatter"}, {"type": "scatter"}],
                   [{"type": "bar"}, {"type": "histogram"}]]
        )
        
        # 1. Method comparison
        methods = list(self.anomaly_results.keys())
        counts = [self.anomaly_results[method].get('count', 0) for method in methods if method != 'ensemble']
        method_names = [method.replace('_', ' ').title() for method in methods if method != 'ensemble']
        
        fig.add_trace(
            go.Bar(x=method_names, y=counts, name="Anomaly Count"),
            row=1, col=1
        )
        
        # 2. Severity distribution
        severity_dist = self.df['anomaly_severity'].value_counts()
        fig.add_trace(
            go.Pie(labels=severity_dist.index, values=severity_dist.values, name="Severity"),
            row=1, col=2
        )
        
        # 3. Anomalies over time
        monthly_anomalies = self.df.groupby('Date')['high_confidence_anomaly'].sum()
        fig.add_trace(
            go.Scatter(x=monthly_anomalies.index, y=monthly_anomalies.values, 
                      mode='lines+markers', name="Monthly Anomalies"),
            row=2, col=1
        )
        
        # 4. Geographic distribution
        geo_anomalies = self.df[self.df['high_confidence_anomaly'] == 1]
        if not geo_anomalies.empty:
            port_anomalies = geo_anomalies.groupby(['Port Name', 'Latitude', 'Longitude']).size().reset_index()
            port_anomalies = port_anomalies.dropna(subset=['Latitude', 'Longitude'])
            
            if not port_anomalies.empty:
                fig.add_trace(
                    go.Scatter(x=port_anomalies['Longitude'], y=port_anomalies['Latitude'],
                              mode='markers', marker=dict(size=port_anomalies[0], 
                              sizemode='diameter', sizeref=2*max(port_anomalies[0])/(20**2)),
                              text=port_anomalies['Port Name'], name="Port Anomalies"),
                    row=2, col=2
                )
        
        # 5. Top anomalous ports
        top_ports = self.df[self.df['high_confidence_anomaly'] == 1]['Port Name'].value_counts().head(10)
        if not top_ports.empty:
            fig.add_trace(
                go.Bar(x=top_ports.values, y=top_ports.index, orientation='h', name="Port Anomalies"),
                row=3, col=1
            )
        
        # 6. Ensemble score distribution
        fig.add_trace(
            go.Histogram(x=self.df['ensemble_anomaly_score'], name="Ensemble Scores", nbinsx=10),
            row=3, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=1000,
            title_text="Border Crossing Anomaly Detection Dashboard",
            showlegend=False
        )
        
        fig.show()
        
    def run_complete_analysis(self, contamination=0.05):
        """Run complete anomaly detection analysis"""
        print("🚨 Starting Comprehensive Anomaly Detection Analysis")
        print("=" * 60)
        
        # Load and preprocess data
        self.load_and_preprocess_data()
        
        # Run all anomaly detection methods
        self.statistical_anomaly_detection(contamination)
        self.machine_learning_anomaly_detection(contamination)
        self.time_series_anomaly_detection()
        self.ensemble_anomaly_detection()
        
        # Generate report
        report = self.generate_anomaly_report()
        
        # Create visualizations
        self.visualize_anomalies()
        
        # Print summary
        self.print_analysis_summary(report)
        
        return report
        
    def print_analysis_summary(self, report):
        """Print formatted analysis summary"""
        print("\n" + "=" * 60)
        print("🎯 ANOMALY DETECTION ANALYSIS SUMMARY")
        print("=" * 60)
        
        summary = report['summary']
        print(f"Total Records Analyzed: {summary['total_records']:,}")
        print(f"High-Confidence Anomalies: {summary['ensemble_results']['high_confidence_count']:,}")
        
        print(f"\n📊 METHOD COMPARISON:")
        for method, results in summary['method_comparison'].items():
            method_name = method.replace('_', ' ').title()
            print(f"  {method_name:25}: {results['count']:>6,} ({results['percentage']:>5.2f}%)")
        
        print(f"\n🚨 TOP ANOMALIES:")
        for i, anomaly in enumerate(report['top_anomalies'][:5], 1):
            print(f"  {i}. {anomaly['port']}, {anomaly['state']}")
            print(f"     Date: {anomaly['date']}, Value: {anomaly['value']:,}")
            print(f"     Score: {anomaly['anomaly_score']}, Severity: {anomaly['severity']}")
        
        print(f"\n📋 RECOMMENDATIONS:")
        for i, rec in enumerate(report['recommendations'], 1):
            print(f"  {i}. {rec}")
        
        print("\n" + "=" * 60)

def main():
    """Main execution function"""
    print("🚨 Border Crossing Anomaly Detection System")
    print("=" * 50)
    
    # Initialize system
    ads = AnomalyDetectionSystem()
    
    # Run complete analysis
    report = ads.run_complete_analysis(contamination=0.05)
    
    print("\n✅ Analysis complete! Check the generated visualizations and report.")

if __name__ == "__main__":
    main()