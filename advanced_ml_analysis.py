#!/usr/bin/env python3
"""
Advanced Machine Learning Analysis for Border Crossing Data
============================================================

This module implements advanced ML techniques for pattern detection:
1. Clustering analysis to identify crossing patterns
2. Classification models for transportation mode prediction
3. Feature engineering for temporal patterns
4. Dimensionality reduction for data exploration
5. Ensemble methods for robust predictions

Author: Portfolio Analysis
Date: 2025-01-28
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, silhouette_score
from sklearn.feature_selection import SelectKBest, f_classif
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

class BorderCrossingMLAnalyzer:
    """Advanced ML analysis for border crossing patterns"""
    
    def __init__(self, data_path='Border_Crossing_Entry_Data.csv'):
        """Initialize the analyzer with data loading and preprocessing"""
        self.df = pd.read_csv(data_path)
        self.df_processed = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def preprocess_data(self):
        """Comprehensive data preprocessing for ML analysis"""
        print("🔧 Preprocessing data for ML analysis...")
        
        # Create working copy
        self.df_processed = self.df.copy()
        
        # Convert date
        self.df_processed['Date'] = pd.to_datetime(self.df_processed['Date'], format='%b %Y')
        
        # Extract temporal features
        self.df_processed['Year'] = self.df_processed['Date'].dt.year
        self.df_processed['Month'] = self.df_processed['Date'].dt.month
        self.df_processed['Quarter'] = self.df_processed['Date'].dt.quarter
        self.df_processed['Season'] = self.df_processed['Month'].map({
            12: 'Winter', 1: 'Winter', 2: 'Winter',
            3: 'Spring', 4: 'Spring', 5: 'Spring',
            6: 'Summer', 7: 'Summer', 8: 'Summer',
            9: 'Fall', 10: 'Fall', 11: 'Fall'
        })
        
        # Remove zero/negative values
        self.df_processed = self.df_processed[self.df_processed['Value'] > 0]
        
        # Create traffic categories
        commercial_keywords = ['truck', 'rail', 'container', 'cargo']
        personal_keywords = ['personal', 'pedestrian', 'bus passenger']
        
        def categorize_traffic(measure):
            measure_lower = measure.lower()
            if any(keyword in measure_lower for keyword in commercial_keywords):
                return 'Commercial'
            elif any(keyword in measure_lower for keyword in personal_keywords):
                return 'Personal'
            elif 'bus' in measure_lower and 'passenger' not in measure_lower:
                return 'Commercial'
            else:
                return 'Other'
        
        self.df_processed['Traffic_Category'] = self.df_processed['Measure'].apply(categorize_traffic)
        
        # Create log-transformed value for better distribution
        self.df_processed['Log_Value'] = np.log1p(self.df_processed['Value'])
        
        # Encode categorical variables
        categorical_cols = ['Border', 'State', 'Measure', 'Traffic_Category', 'Season']
        for col in categorical_cols:
            le = LabelEncoder()
            self.df_processed[f'{col}_Encoded'] = le.fit_transform(self.df_processed[col])
            self.label_encoders[col] = le
        
        print(f"✅ Preprocessing complete. Dataset shape: {self.df_processed.shape}")
        
    def feature_engineering(self):
        """Create advanced features for ML analysis"""
        print("🛠️ Engineering advanced features...")
        
        # Port-level aggregations
        port_stats = self.df_processed.groupby('Port Name').agg({
            'Value': ['mean', 'std', 'count', 'sum'],
            'Log_Value': ['mean', 'std']
        }).round(2)
        port_stats.columns = [f'Port_{stat}_{col}' for col, stat in port_stats.columns]
        
        # Merge back to main dataset
        self.df_processed = self.df_processed.merge(
            port_stats.reset_index(), on='Port Name', how='left'
        )
        
        # State-level features
        state_stats = self.df_processed.groupby('State').agg({
            'Value': ['mean', 'count'],
            'Port Name': 'nunique'
        }).round(2)
        state_stats.columns = [f'State_{stat}_{col}' for col, stat in state_stats.columns]
        
        self.df_processed = self.df_processed.merge(
            state_stats.reset_index(), on='State', how='left'
        )
        
        # Temporal features
        monthly_stats = self.df_processed.groupby('Month')['Value'].mean()
        self.df_processed['Month_Avg'] = self.df_processed['Month'].map(monthly_stats)
        
        # Seasonal indicators
        self.df_processed['Is_Peak_Season'] = (
            self.df_processed['Month'].isin([3, 4, 5])
        ).astype(int)
        
        # Volume categories
        self.df_processed['Volume_Category'] = pd.cut(
            self.df_processed['Value'],
            bins=[0, 100, 1000, 10000, 100000, float('inf')],
            labels=['Very_Low', 'Low', 'Medium', 'High', 'Very_High']
        )
        
        print("✅ Feature engineering complete")
        
    def clustering_analysis(self, n_clusters=5):
        """Perform clustering analysis to identify patterns"""
        print(f"🎯 Performing clustering analysis with {n_clusters} clusters...")
        
        # Select features for clustering
        cluster_features = [
            'Log_Value', 'Month', 'Quarter', 'Border_Encoded',
            'Traffic_Category_Encoded', 'Port_mean_Value', 'State_count_Value'
        ]
        
        # Prepare data
        X_cluster = self.df_processed[cluster_features].copy()
        X_cluster = X_cluster.fillna(X_cluster.mean())
        X_cluster_scaled = self.scaler.fit_transform(X_cluster)
        
        # K-Means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.df_processed['Cluster_KMeans'] = kmeans.fit_predict(X_cluster_scaled)
        
        # DBSCAN clustering
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        self.df_processed['Cluster_DBSCAN'] = dbscan.fit_predict(X_cluster_scaled)
        
        # Calculate silhouette scores
        kmeans_silhouette = silhouette_score(X_cluster_scaled, self.df_processed['Cluster_KMeans'])
        
        # Only calculate DBSCAN silhouette if we have more than 1 cluster
        n_dbscan_clusters = len(set(self.df_processed['Cluster_DBSCAN'])) - (1 if -1 in self.df_processed['Cluster_DBSCAN'] else 0)
        if n_dbscan_clusters > 1:
            dbscan_silhouette = silhouette_score(X_cluster_scaled, self.df_processed['Cluster_DBSCAN'])
        else:
            dbscan_silhouette = -1
        
        print(f"K-Means Silhouette Score: {kmeans_silhouette:.3f}")
        print(f"DBSCAN Silhouette Score: {dbscan_silhouette:.3f}")
        print(f"DBSCAN found {n_dbscan_clusters} clusters")
        
        # Visualize clusters
        self.visualize_clusters()
        
        return kmeans_silhouette, dbscan_silhouette
        
    def visualize_clusters(self):
        """Create visualizations for clustering results"""
        print("📊 Creating cluster visualizations...")
        
        # PCA for dimensionality reduction
        cluster_features = [
            'Log_Value', 'Month', 'Quarter', 'Border_Encoded',
            'Traffic_Category_Encoded', 'Port_mean_Value', 'State_count_Value'
        ]
        
        X_viz = self.df_processed[cluster_features].fillna(self.df_processed[cluster_features].mean())
        X_viz_scaled = self.scaler.fit_transform(X_viz)
        
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_viz_scaled)
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Border Crossing Pattern Analysis - Clustering Results', fontsize=16, fontweight='bold')
        
        # K-Means clusters in PCA space
        scatter = axes[0,0].scatter(X_pca[:, 0], X_pca[:, 1], 
                                  c=self.df_processed['Cluster_KMeans'], 
                                  cmap='viridis', alpha=0.6)
        axes[0,0].set_title('K-Means Clustering (PCA Space)')
        axes[0,0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        axes[0,0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        plt.colorbar(scatter, ax=axes[0,0])
        
        # DBSCAN clusters
        scatter2 = axes[0,1].scatter(X_pca[:, 0], X_pca[:, 1], 
                                   c=self.df_processed['Cluster_DBSCAN'], 
                                   cmap='plasma', alpha=0.6)
        axes[0,1].set_title('DBSCAN Clustering (PCA Space)')
        axes[0,1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        axes[0,1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        plt.colorbar(scatter2, ax=axes[0,1])
        
        # Cluster characteristics - K-Means
        cluster_stats = self.df_processed.groupby('Cluster_KMeans').agg({
            'Value': 'mean',
            'Border': lambda x: x.mode().iloc[0],
            'Traffic_Category': lambda x: x.mode().iloc[0]
        })
        
        axes[1,0].bar(range(len(cluster_stats)), cluster_stats['Value'])
        axes[1,0].set_title('Average Volume by K-Means Cluster')
        axes[1,0].set_xlabel('Cluster')
        axes[1,0].set_ylabel('Average Crossings')
        
        # Volume distribution by cluster
        self.df_processed.boxplot(column='Log_Value', by='Cluster_KMeans', ax=axes[1,1])
        axes[1,1].set_title('Log Volume Distribution by Cluster')
        axes[1,1].set_xlabel('Cluster')
        axes[1,1].set_ylabel('Log(Value + 1)')
        
        plt.tight_layout()
        plt.show()
        
        # Print cluster characteristics
        print("\n🔍 K-Means Cluster Characteristics:")
        for cluster_id in sorted(self.df_processed['Cluster_KMeans'].unique()):
            cluster_data = self.df_processed[self.df_processed['Cluster_KMeans'] == cluster_id]
            print(f"\nCluster {cluster_id} ({len(cluster_data)} records):")
            print(f"  Average Volume: {cluster_data['Value'].mean():,.0f}")
            print(f"  Dominant Border: {cluster_data['Border'].mode().iloc[0]}")
            print(f"  Dominant Traffic: {cluster_data['Traffic_Category'].mode().iloc[0]}")
            print(f"  Top State: {cluster_data['State'].mode().iloc[0]}")
            
    def classification_analysis(self):
        """Build classification models for pattern prediction"""
        print("🤖 Building classification models...")
        
        # Prepare features and target
        feature_cols = [
            'Month', 'Quarter', 'Border_Encoded', 'State_Encoded',
            'Port_mean_Value', 'State_count_Value', 'Is_Peak_Season',
            'Log_Value'
        ]
        
        X = self.df_processed[feature_cols].fillna(self.df_processed[feature_cols].mean())
        y = self.df_processed['Traffic_Category_Encoded']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Random Forest Classifier
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        
        # Predictions and evaluation
        y_pred = rf_model.predict(X_test)
        
        # Cross-validation score
        cv_scores = cross_val_score(rf_model, X, y, cv=5)
        
        print(f"Random Forest CV Score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': rf_model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        # Visualize feature importance
        plt.figure(figsize=(10, 6))
        sns.barplot(data=feature_importance, x='Importance', y='Feature')
        plt.title('Feature Importance for Traffic Category Classification')
        plt.tight_layout()
        plt.show()
        
        print("\n📈 Feature Importance Rankings:")
        for idx, row in feature_importance.iterrows():
            print(f"  {row['Feature']}: {row['Importance']:.3f}")
            
        # Classification report
        y_test_labels = [self.label_encoders['Traffic_Category'].classes_[i] for i in y_test]
        y_pred_labels = [self.label_encoders['Traffic_Category'].classes_[i] for i in y_pred]
        
        print("\n📊 Classification Report:")
        print(classification_report(y_test_labels, y_pred_labels))
        
        return rf_model, feature_importance
        
    def anomaly_detection(self):
        """Implement anomaly detection for unusual crossing patterns"""
        print("🚨 Performing anomaly detection...")
        
        # Prepare features for anomaly detection
        anomaly_features = [
            'Log_Value', 'Month', 'Border_Encoded', 'Traffic_Category_Encoded',
            'Port_mean_Value', 'State_count_Value'
        ]
        
        X_anomaly = self.df_processed[anomaly_features].fillna(
            self.df_processed[anomaly_features].mean()
        )
        
        # Isolation Forest
        iso_forest = IsolationForest(contamination=0.05, random_state=42)
        anomaly_labels = iso_forest.fit_predict(X_anomaly)
        
        self.df_processed['Is_Anomaly'] = (anomaly_labels == -1).astype(int)
        
        # Analyze anomalies
        anomalies = self.df_processed[self.df_processed['Is_Anomaly'] == 1]
        normal = self.df_processed[self.df_processed['Is_Anomaly'] == 0]
        
        print(f"Found {len(anomalies)} anomalies ({len(anomalies)/len(self.df_processed)*100:.2f}%)")
        
        # Visualize anomalies
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Anomaly Detection Results', fontsize=16, fontweight='bold')
        
        # Anomalies in value space
        axes[0,0].scatter(normal['Date'], normal['Value'], alpha=0.5, label='Normal', s=20)
        axes[0,0].scatter(anomalies['Date'], anomalies['Value'], 
                         color='red', label='Anomaly', s=50, alpha=0.8)
        axes[0,0].set_title('Anomalies Over Time')
        axes[0,0].set_ylabel('Crossing Volume')
        axes[0,0].legend()
        axes[0,0].set_yscale('log')
        
        # Anomalies by border
        anomaly_by_border = anomalies.groupby('Border').size()
        normal_by_border = normal.groupby('Border').size()
        
        x = np.arange(len(anomaly_by_border))
        width = 0.35
        
        axes[0,1].bar(x - width/2, normal_by_border.values, width, label='Normal', alpha=0.7)
        axes[0,1].bar(x + width/2, anomaly_by_border.values, width, label='Anomaly', alpha=0.7)
        axes[0,1].set_title('Anomalies by Border')
        axes[0,1].set_xticks(x)
        axes[0,1].set_xticklabels(anomaly_by_border.index, rotation=45)
        axes[0,1].legend()
        
        # Top anomalous ports
        top_anomaly_ports = anomalies.groupby(['Port Name', 'State']).size().nlargest(10)
        axes[1,0].barh(range(len(top_anomaly_ports)), top_anomaly_ports.values)
        axes[1,0].set_title('Top 10 Ports with Anomalies')
        axes[1,0].set_yticks(range(len(top_anomaly_ports)))
        axes[1,0].set_yticklabels([f"{port}, {state}" for (port, state) in top_anomaly_ports.index])
        axes[1,0].invert_yaxis()
        
        # Anomaly patterns
        anomaly_patterns = anomalies.groupby(['Traffic_Category', 'Season']).size().unstack(fill_value=0)
        sns.heatmap(anomaly_patterns, annot=True, cmap='Reds', ax=axes[1,1])
        axes[1,1].set_title('Anomaly Patterns by Traffic Category and Season')
        
        plt.tight_layout()
        plt.show()
        
        # Print anomaly insights
        print("\n🔍 Anomaly Analysis:")
        print(f"Average anomaly volume: {anomalies['Value'].mean():,.0f}")
        print(f"Average normal volume: {normal['Value'].mean():,.0f}")
        print(f"Anomaly volume ratio: {anomalies['Value'].mean() / normal['Value'].mean():.2f}x")
        
        print("\nTop anomalous records:")
        top_anomalies = anomalies.nlargest(5, 'Value')[['Port Name', 'State', 'Border', 'Date', 'Measure', 'Value']]
        print(top_anomalies.to_string(index=False))
        
        return anomalies, iso_forest
        
    def run_complete_analysis(self):
        """Run the complete ML analysis pipeline"""
        print("🚀 Starting Advanced ML Analysis Pipeline")
        print("=" * 60)
        
        # Step 1: Preprocessing
        self.preprocess_data()
        
        # Step 2: Feature Engineering
        self.feature_engineering()
        
        # Step 3: Clustering Analysis
        kmeans_score, dbscan_score = self.clustering_analysis()
        
        # Step 4: Classification Analysis
        rf_model, feature_importance = self.classification_analysis()
        
        # Step 5: Anomaly Detection
        anomalies, anomaly_model = self.anomaly_detection()
        
        print("\n" + "=" * 60)
        print("✅ Advanced ML Analysis Complete!")
        print("=" * 60)
        
        # Return results summary
        results = {
            'kmeans_silhouette': kmeans_score,
            'dbscan_silhouette': dbscan_score,
            'rf_model': rf_model,
            'feature_importance': feature_importance,
            'anomalies': anomalies,
            'anomaly_model': anomaly_model,
            'processed_data': self.df_processed
        }
        
        return results

def main():
    """Main execution function"""
    print("🎯 Advanced Machine Learning Analysis for Border Crossing Data")
    print("=" * 70)
    
    # Initialize analyzer
    analyzer = BorderCrossingMLAnalyzer()
    
    # Run complete analysis
    results = analyzer.run_complete_analysis()
    
    print(f"\n📊 Analysis Results Summary:")
    print(f"- K-Means Silhouette Score: {results['kmeans_silhouette']:.3f}")
    print(f"- DBSCAN Silhouette Score: {results['dbscan_silhouette']:.3f}")
    print(f"- Anomalies Detected: {len(results['anomalies'])}")
    print(f"- Feature Importance Top 3:")
    for idx, row in results['feature_importance'].head(3).iterrows():
        print(f"  {idx+1}. {row['Feature']}: {row['Importance']:.3f}")

if __name__ == "__main__":
    main()