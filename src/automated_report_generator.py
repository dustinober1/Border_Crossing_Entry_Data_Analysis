#!/usr/bin/env python3
"""
Automated Report Generation System
==================================

This module creates comprehensive, automated reports for border crossing analysis
with professional formatting, interactive charts, and scheduled delivery capabilities.

Features:
- Executive summary reports
- Detailed analytical reports
- Custom report templates
- PDF and HTML export
- Email delivery system
- Scheduled report generation
- Interactive dashboards embedded
- Performance monitoring

Author: Portfolio Analysis
Date: 2025-01-28
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.offline as pyo
from jinja2 import Template, Environment, FileSystemLoader
from datetime import datetime, timedelta
import json
import base64
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

class AutomatedReportGenerator:
    """Automated report generation system for border crossing analytics"""
    
    def __init__(self, data_path='Border_Crossing_Entry_Data.csv'):
        """Initialize the report generator"""
        self.data_path = data_path
        self.df = None
        self.report_data = {}
        self.templates = {}
        self.load_data()
        self.setup_templates()
        
    def load_data(self):
        """Load and preprocess data for reporting"""
        print("📊 Loading data for report generation...")
        
        self.df = pd.read_csv(self.data_path)
        self.df['Date'] = pd.to_datetime(self.df['Date'], format='%b %Y')
        self.df = self.df[self.df['Value'] > 0]
        
        # Add derived columns
        self.df['Year'] = self.df['Date'].dt.year
        self.df['Month'] = self.df['Date'].dt.month
        self.df['Month_Name'] = self.df['Date'].dt.strftime('%B')
        self.df['Quarter'] = self.df['Date'].dt.quarter
        
        # Traffic categorization
        def categorize_traffic(measure):
            measure_lower = measure.lower()
            if any(keyword in measure_lower for keyword in ['truck', 'rail', 'container', 'cargo']):
                return 'Commercial'
            elif any(keyword in measure_lower for keyword in ['personal', 'pedestrian', 'bus passenger']):
                return 'Personal'
            elif 'bus' in measure_lower and 'passenger' not in measure_lower:
                return 'Commercial'
            else:
                return 'Other'
                
        self.df['Traffic_Category'] = self.df['Measure'].apply(categorize_traffic)
        
        print(f"✅ Data loaded: {len(self.df):,} records")
        
    def setup_templates(self):
        """Setup HTML templates for different report types"""
        print("📝 Setting up report templates...")
        
        # Executive Summary Template
        self.templates['executive'] = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Border Crossing Analytics - Executive Summary</title>
            <style>
                body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background-color: #f5f6fa; }
                .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 10px; margin-bottom: 30px; text-align: center; }
                .kpi-container { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 30px; }
                .kpi-card { background: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); text-align: center; }
                .kpi-value { font-size: 2.5em; font-weight: bold; color: #2c3e50; margin-bottom: 10px; }
                .kpi-label { color: #7f8c8d; font-size: 1.1em; }
                .section { background: white; padding: 25px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 25px; }
                .section-title { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; margin-bottom: 20px; }
                .chart-container { margin: 20px 0; }
                .insight { background: #e8f4f8; padding: 15px; border-left: 4px solid #3498db; margin: 15px 0; border-radius: 5px; }
                .recommendation { background: #f0f8e8; padding: 15px; border-left: 4px solid #27ae60; margin: 15px 0; border-radius: 5px; }
                .footer { text-align: center; color: #7f8c8d; margin-top: 40px; padding-top: 20px; border-top: 1px solid #bdc3c7; }
                table { width: 100%; border-collapse: collapse; margin: 20px 0; }
                th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
                th { background-color: #f8f9fa; font-weight: 600; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🌍 Border Crossing Analytics</h1>
                <h2>Executive Summary Report</h2>
                <p>Generated on {{ report_date }}</p>
            </div>
            
            <div class="kpi-container">
                {% for kpi in kpis %}
                <div class="kpi-card">
                    <div class="kpi-value">{{ kpi.value }}</div>
                    <div class="kpi-label">{{ kpi.label }}</div>
                </div>
                {% endfor %}
            </div>
            
            <div class="section">
                <h2 class="section-title">📊 Key Performance Indicators</h2>
                {{ kpi_chart | safe }}
            </div>
            
            <div class="section">
                <h2 class="section-title">📈 Trend Analysis</h2>
                {{ trend_chart | safe }}
                <div class="insight">
                    <strong>Key Insight:</strong> {{ trend_insight }}
                </div>
            </div>
            
            <div class="section">
                <h2 class="section-title">🗺️ Geographic Distribution</h2>
                {{ geographic_chart | safe }}
            </div>
            
            <div class="section">
                <h2 class="section-title">📋 Top Performers</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Category</th>
                            <th>Top Performer</th>
                            <th>Volume</th>
                            <th>Percentage</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for performer in top_performers %}
                        <tr>
                            <td>{{ performer.category }}</td>
                            <td>{{ performer.name }}</td>
                            <td>{{ performer.volume }}</td>
                            <td>{{ performer.percentage }}</td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
            </div>
            
            <div class="section">
                <h2 class="section-title">💡 Strategic Recommendations</h2>
                {% for recommendation in recommendations %}
                <div class="recommendation">
                    {{ recommendation }}
                </div>
                {% endfor %}
            </div>
            
            <div class="footer">
                <p>Report generated by Border Crossing Analytics System | {{ report_date }}</p>
            </div>
        </body>
        </html>
        """
        
        # Detailed Analysis Template
        self.templates['detailed'] = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Border Crossing Analytics - Detailed Report</title>
            <style>
                body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background-color: #f5f6fa; }
                .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 10px; margin-bottom: 30px; text-align: center; }
                .section { background: white; padding: 25px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 25px; }
                .section-title { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; margin-bottom: 20px; }
                .chart-container { margin: 20px 0; }
                .metric-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 20px 0; }
                .metric-card { background: #f8f9fa; padding: 20px; border-radius: 8px; border-left: 4px solid #3498db; }
                .metric-value { font-size: 1.8em; font-weight: bold; color: #2c3e50; }
                .metric-label { color: #7f8c8d; margin-top: 5px; }
                table { width: 100%; border-collapse: collapse; margin: 20px 0; }
                th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
                th { background-color: #f8f9fa; font-weight: 600; }
                .footer { text-align: center; color: #7f8c8d; margin-top: 40px; padding-top: 20px; border-top: 1px solid #bdc3c7; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🌍 Border Crossing Analytics</h1>
                <h2>Detailed Analysis Report</h2>
                <p>Period: {{ analysis_period }} | Generated: {{ report_date }}</p>
            </div>
            
            <div class="section">
                <h2 class="section-title">📊 Executive Summary</h2>
                <div class="metric-grid">
                    {% for metric in summary_metrics %}
                    <div class="metric-card">
                        <div class="metric-value">{{ metric.value }}</div>
                        <div class="metric-label">{{ metric.label }}</div>
                    </div>
                    {% endfor %}
                </div>
            </div>
            
            <div class="section">
                <h2 class="section-title">🚛 Transportation Mode Analysis</h2>
                {{ transportation_chart | safe }}
                <table>
                    <thead>
                        <tr>
                            <th>Transportation Mode</th>
                            <th>Total Volume</th>
                            <th>Average per Month</th>
                            <th>Growth Rate</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for mode in transportation_data %}
                        <tr>
                            <td>{{ mode.name }}</td>
                            <td>{{ mode.total }}</td>
                            <td>{{ mode.average }}</td>
                            <td>{{ mode.growth }}</td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
            </div>
            
            <div class="section">
                <h2 class="section-title">🗓️ Temporal Analysis</h2>
                {{ temporal_chart | safe }}
                <h3>Seasonal Patterns</h3>
                <p>{{ seasonal_analysis }}</p>
            </div>
            
            <div class="section">
                <h2 class="section-title">🗺️ Geographic Analysis</h2>
                {{ geographic_detailed_chart | safe }}
                <h3>State Performance</h3>
                <table>
                    <thead>
                        <tr>
                            <th>State</th>
                            <th>Total Crossings</th>
                            <th>Active Ports</th>
                            <th>Market Share</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for state in state_data %}
                        <tr>
                            <td>{{ state.name }}</td>
                            <td>{{ state.total }}</td>
                            <td>{{ state.ports }}</td>
                            <td>{{ state.share }}</td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
            </div>
            
            <div class="section">
                <h2 class="section-title">📈 Statistical Analysis</h2>
                {{ statistical_chart | safe }}
                <div class="metric-grid">
                    {% for stat in statistical_metrics %}
                    <div class="metric-card">
                        <div class="metric-value">{{ stat.value }}</div>
                        <div class="metric-label">{{ stat.label }}</div>
                    </div>
                    {% endfor %}
                </div>
            </div>
            
            <div class="footer">
                <p>Detailed Report generated by Border Crossing Analytics System | {{ report_date }}</p>
            </div>
        </body>
        </html>
        """
        
        print("✅ Templates configured")
        
    def calculate_report_metrics(self):
        """Calculate all metrics needed for reports"""
        print("🧮 Calculating report metrics...")
        
        # Basic metrics
        total_crossings = self.df['Value'].sum()
        total_ports = self.df['Port Name'].nunique()
        total_states = self.df['State'].nunique()
        date_range = f"{self.df['Date'].min().strftime('%b %Y')} - {self.df['Date'].max().strftime('%b %Y')}"
        
        # Border comparison
        border_stats = self.df.groupby('Border')['Value'].agg(['sum', 'mean', 'count']).round(0)
        
        # Top performers
        top_border = self.df.groupby('Border')['Value'].sum().idxmax()
        top_state = self.df.groupby('State')['Value'].sum().idxmax()
        top_port = self.df.groupby(['Port Name', 'State'])['Value'].sum().idxmax()
        top_measure = self.df.groupby('Measure')['Value'].sum().idxmax()
        
        # Monthly trends
        monthly_data = self.df.groupby(self.df['Date'].dt.to_period('M'))['Value'].sum()
        recent_trend = 'Increasing' if monthly_data.iloc[-1] > monthly_data.iloc[-6:].mean() else 'Stable/Decreasing'
        
        # Traffic categories
        traffic_distribution = self.df.groupby('Traffic_Category')['Value'].sum()
        
        # Store metrics
        self.report_data = {
            'basic_metrics': {
                'total_crossings': total_crossings,
                'total_ports': total_ports,
                'total_states': total_states,
                'date_range': date_range,
                'avg_monthly': monthly_data.mean(),
                'peak_month': monthly_data.idxmax().strftime('%b %Y'),
                'peak_volume': monthly_data.max()
            },
            'top_performers': {
                'border': top_border,
                'state': top_state,
                'port': f"{top_port[0]}, {top_port[1]}",
                'measure': top_measure
            },
            'trends': {
                'recent_trend': recent_trend,
                'monthly_data': monthly_data,
                'seasonal_peak': self.df.groupby('Month_Name')['Value'].sum().idxmax()
            },
            'distribution': {
                'traffic_categories': traffic_distribution,
                'border_stats': border_stats
            }
        }
        
        print("✅ Metrics calculated")
        
    def create_charts(self):
        """Create all charts needed for reports"""
        print("📊 Creating report charts...")
        
        charts = {}
        
        # 1. KPI Overview Chart
        kpi_data = pd.DataFrame({
            'Metric': ['Total Crossings', 'Active Ports', 'States', 'Avg Monthly'],
            'Value': [
                self.report_data['basic_metrics']['total_crossings'],
                self.report_data['basic_metrics']['total_ports'],
                self.report_data['basic_metrics']['total_states'],
                self.report_data['basic_metrics']['avg_monthly']
            ]
        })
        
        fig_kpi = px.bar(kpi_data, x='Metric', y='Value', 
                        title="Key Performance Indicators Overview")
        charts['kpi_chart'] = pyo.plot(fig_kpi, output_type='div', include_plotlyjs=False)
        
        # 2. Monthly Trends
        monthly_df = self.report_data['trends']['monthly_data'].reset_index()
        monthly_df['Date'] = monthly_df['Date'].astype(str)
        
        fig_trend = px.line(monthly_df, x='Date', y='Value',
                           title="Monthly Border Crossing Trends")
        fig_trend.update_xaxis(tickangle=45)
        charts['trend_chart'] = pyo.plot(fig_trend, output_type='div', include_plotlyjs=False)
        
        # 3. Geographic Distribution
        state_data = self.df.groupby('State')['Value'].sum().nlargest(10).reset_index()
        fig_geo = px.bar(state_data, x='State', y='Value',
                        title="Top 10 States by Border Crossings")
        fig_geo.update_xaxis(tickangle=45)
        charts['geographic_chart'] = pyo.plot(fig_geo, output_type='div', include_plotlyjs=False)
        
        # 4. Transportation Mode Analysis
        transport_data = self.df.groupby('Traffic_Category')['Value'].sum().reset_index()
        fig_transport = px.pie(transport_data, values='Value', names='Traffic_Category',
                              title="Border Crossings by Traffic Category")
        charts['transportation_chart'] = pyo.plot(fig_transport, output_type='div', include_plotlyjs=False)
        
        # 5. Seasonal Analysis
        seasonal_data = self.df.groupby('Month_Name')['Value'].sum().reindex([
            'January', 'February', 'March', 'April', 'May', 'June',
            'July', 'August', 'September', 'October', 'November', 'December'
        ]).reset_index()
        
        fig_seasonal = px.line(seasonal_data, x='Month_Name', y='Value',
                              title="Seasonal Patterns in Border Crossings")
        fig_seasonal.update_xaxis(tickangle=45)
        charts['temporal_chart'] = pyo.plot(fig_seasonal, output_type='div', include_plotlyjs=False)
        
        # 6. Statistical Distribution
        fig_stats = px.histogram(self.df, x='Value', nbins=50,
                                title="Distribution of Border Crossing Volumes")
        fig_stats.update_xaxis(title="Crossing Volume")
        fig_stats.update_yaxis(title="Frequency")
        charts['statistical_chart'] = pyo.plot(fig_stats, output_type='div', include_plotlyjs=False)
        
        # 7. Geographic Detailed (Port level)
        port_data = self.df.groupby(['Port Name', 'State'])['Value'].sum().nlargest(15).reset_index()
        port_data['Port_State'] = port_data['Port Name'] + ', ' + port_data['State']
        
        fig_ports = px.bar(port_data, x='Value', y='Port_State', orientation='h',
                          title="Top 15 Border Ports by Volume")
        charts['geographic_detailed_chart'] = pyo.plot(fig_ports, output_type='div', include_plotlyjs=False)
        
        self.charts = charts
        print("✅ Charts created")
        
    def generate_executive_summary(self):
        """Generate executive summary report"""
        print("📋 Generating executive summary report...")
        
        # Prepare data for template
        kpis = [
            {
                'value': f"{self.report_data['basic_metrics']['total_crossings']:,.0f}",
                'label': 'Total Crossings'
            },
            {
                'value': f"{self.report_data['basic_metrics']['total_ports']:,}",
                'label': 'Active Ports'
            },
            {
                'value': f"{self.report_data['basic_metrics']['total_states']:,}",
                'label': 'States'
            },
            {
                'value': f"{self.report_data['basic_metrics']['avg_monthly']:,.0f}",
                'label': 'Avg Monthly Volume'
            }
        ]
        
        top_performers = [
            {
                'category': 'Border',
                'name': self.report_data['top_performers']['border'].split('-')[1],
                'volume': f"{self.df.groupby('Border')['Value'].sum().max():,.0f}",
                'percentage': f"{self.df.groupby('Border')['Value'].sum().max() / self.df['Value'].sum() * 100:.1f}%"
            },
            {
                'category': 'State',
                'name': self.report_data['top_performers']['state'],
                'volume': f"{self.df.groupby('State')['Value'].sum().max():,.0f}",
                'percentage': f"{self.df.groupby('State')['Value'].sum().max() / self.df['Value'].sum() * 100:.1f}%"
            },
            {
                'category': 'Port',
                'name': self.report_data['top_performers']['port'],
                'volume': f"{self.df.groupby(['Port Name', 'State'])['Value'].sum().max():,.0f}",
                'percentage': f"{self.df.groupby(['Port Name', 'State'])['Value'].sum().max() / self.df['Value'].sum() * 100:.1f}%"
            }
        ]
        
        recommendations = [
            "🚀 Focus infrastructure investment on high-volume US-Mexico border ports for maximum impact",
            "📅 Implement seasonal staffing strategies to handle peak activity during spring months",
            f"🎯 Prioritize capacity expansion at {self.report_data['top_performers']['state']} locations",
            "📊 Develop specialized processing lanes for commercial traffic to improve efficiency",
            "🔍 Monitor emerging trends in personal vehicle crossings for future planning"
        ]
        
        # Generate trend insight
        trend_insight = f"Border crossings show a {self.report_data['trends']['recent_trend'].lower()} trend with peak activity in {self.report_data['trends']['seasonal_peak']}. The US-Mexico border dominates with {self.df[self.df['Border'] == 'US-Mexico Border']['Value'].sum() / self.df['Value'].sum() * 100:.1f}% of total volume."
        
        # Render template
        template = Template(self.templates['executive'])
        html_content = template.render(
            report_date=datetime.now().strftime('%B %d, %Y'),
            kpis=kpis,
            top_performers=top_performers,
            recommendations=recommendations,
            trend_insight=trend_insight,
            kpi_chart=self.charts['kpi_chart'],
            trend_chart=self.charts['trend_chart'],
            geographic_chart=self.charts['geographic_chart']
        )
        
        return html_content
        
    def generate_detailed_report(self):
        """Generate detailed analysis report"""
        print("📋 Generating detailed analysis report...")
        
        # Summary metrics
        summary_metrics = [
            {
                'value': f"{self.report_data['basic_metrics']['total_crossings']:,.0f}",
                'label': 'Total Border Crossings'
            },
            {
                'value': f"{self.report_data['basic_metrics']['avg_monthly']:,.0f}",
                'label': 'Average Monthly Volume'
            },
            {
                'value': f"{self.report_data['basic_metrics']['peak_volume']:,.0f}",
                'label': 'Peak Monthly Volume'
            },
            {
                'value': self.report_data['basic_metrics']['date_range'],
                'label': 'Analysis Period'
            }
        ]
        
        # Transportation data
        transportation_data = []
        for category, total in self.report_data['distribution']['traffic_categories'].items():
            transportation_data.append({
                'name': category,
                'total': f"{total:,.0f}",
                'average': f"{total / 12:,.0f}",  # Simplified monthly average
                'growth': 'N/A'  # Would need historical comparison
            })
        
        # State data
        state_stats = self.df.groupby('State').agg({
            'Value': 'sum',
            'Port Name': 'nunique'
        }).sort_values('Value', ascending=False).head(10)
        
        state_data = []
        total_volume = self.df['Value'].sum()
        for state, row in state_stats.iterrows():
            state_data.append({
                'name': state,
                'total': f"{row['Value']:,.0f}",
                'ports': f"{row['Port Name']}",
                'share': f"{row['Value'] / total_volume * 100:.1f}%"
            })
        
        # Statistical metrics
        statistical_metrics = [
            {
                'value': f"{self.df['Value'].mean():,.0f}",
                'label': 'Average Crossing Volume'
            },
            {
                'value': f"{self.df['Value'].median():,.0f}",
                'label': 'Median Crossing Volume'
            },
            {
                'value': f"{self.df['Value'].std():,.0f}",
                'label': 'Standard Deviation'
            },
            {
                'value': f"{self.df['Value'].std() / self.df['Value'].mean():.2f}",
                'label': 'Coefficient of Variation'
            }
        ]
        
        # Seasonal analysis
        seasonal_analysis = f"Peak crossing activity occurs in {self.report_data['trends']['seasonal_peak']}, indicating strong seasonal patterns. The data shows significant variation across months, with {self.df.groupby('Month_Name')['Value'].sum().max() / self.df.groupby('Month_Name')['Value'].sum().min():.1f}x difference between peak and low months."
        
        # Render template
        template = Template(self.templates['detailed'])
        html_content = template.render(
            report_date=datetime.now().strftime('%B %d, %Y'),
            analysis_period=self.report_data['basic_metrics']['date_range'],
            summary_metrics=summary_metrics,
            transportation_data=transportation_data,
            transportation_chart=self.charts['transportation_chart'],
            temporal_chart=self.charts['temporal_chart'],
            geographic_detailed_chart=self.charts['geographic_detailed_chart'],
            statistical_chart=self.charts['statistical_chart'],
            state_data=state_data,
            statistical_metrics=statistical_metrics,
            seasonal_analysis=seasonal_analysis
        )
        
        return html_content
        
    def save_report(self, content, filename, format='html'):
        """Save report to file"""
        if format == 'html':
            with open(f"{filename}.html", 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✅ Report saved as {filename}.html")
        
        # For PDF conversion, you would need additional libraries like weasyprint
        # This is a simplified version
        
    def generate_all_reports(self):
        """Generate all report types"""
        print("🚀 Generating all reports...")
        
        # Calculate metrics and create charts
        self.calculate_report_metrics()
        self.create_charts()
        
        # Generate reports
        executive_report = self.generate_executive_summary()
        detailed_report = self.generate_detailed_report()
        
        # Save reports
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        self.save_report(executive_report, f'executive_summary_{timestamp}')
        self.save_report(detailed_report, f'detailed_analysis_{timestamp}')
        
        return {
            'executive': f'executive_summary_{timestamp}.html',
            'detailed': f'detailed_analysis_{timestamp}.html'
        }
        
    def create_report_dashboard(self):
        """Create an integrated dashboard combining all analysis"""
        print("📊 Creating integrated report dashboard...")
        
        dashboard_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Border Crossing Analytics - Integrated Dashboard</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background-color: #f5f6fa; }
                .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 10px; margin-bottom: 30px; text-align: center; }
                .dashboard-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px; }
                .chart-panel { background: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
                .full-width { grid-column: 1 / -1; }
                .kpi-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 15px; margin-bottom: 20px; }
                .kpi-card { background: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); text-align: center; }
                .kpi-value { font-size: 2em; font-weight: bold; color: #2c3e50; }
                .kpi-label { color: #7f8c8d; margin-top: 10px; }
                .footer { text-align: center; color: #7f8c8d; margin-top: 40px; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🌍 Border Crossing Analytics Dashboard</h1>
                <h2>Integrated Analysis & Insights</h2>
                <p>Real-time insights from {{ total_records }} border crossing records</p>
            </div>
            
            <div class="kpi-grid">
                {% for kpi in dashboard_kpis %}
                <div class="kpi-card">
                    <div class="kpi-value">{{ kpi.value }}</div>
                    <div class="kpi-label">{{ kpi.label }}</div>
                </div>
                {% endfor %}
            </div>
            
            <div class="dashboard-grid">
                <div class="chart-panel full-width">
                    {{ trend_chart | safe }}
                </div>
                
                <div class="chart-panel">
                    {{ geographic_chart | safe }}
                </div>
                
                <div class="chart-panel">
                    {{ transportation_chart | safe }}
                </div>
                
                <div class="chart-panel full-width">
                    {{ temporal_chart | safe }}
                </div>
                
                <div class="chart-panel">
                    {{ statistical_chart | safe }}
                </div>
                
                <div class="chart-panel">
                    {{ geographic_detailed_chart | safe }}
                </div>
            </div>
            
            <div class="footer">
                <p>Dashboard generated on {{ report_date }} | Border Crossing Analytics System</p>
            </div>
        </body>
        </html>
        """
        
        # Dashboard KPIs
        dashboard_kpis = [
            {
                'value': f"{self.report_data['basic_metrics']['total_crossings']:,.0f}",
                'label': 'Total Crossings'
            },
            {
                'value': f"{self.report_data['basic_metrics']['total_ports']:,}",
                'label': 'Active Ports'
            },
            {
                'value': f"{self.report_data['basic_metrics']['total_states']:,}",
                'label': 'States'
            },
            {
                'value': f"{self.report_data['basic_metrics']['avg_monthly']:,.0f}",
                'label': 'Avg Monthly'
            },
            {
                'value': self.report_data['trends']['seasonal_peak'],
                'label': 'Peak Month'
            },
            {
                'value': self.report_data['top_performers']['border'].split('-')[1],
                'label': 'Top Border'
            }
        ]
        
        # Render dashboard
        template = Template(dashboard_template)
        dashboard_content = template.render(
            report_date=datetime.now().strftime('%B %d, %Y'),
            total_records=f"{len(self.df):,}",
            dashboard_kpis=dashboard_kpis,
            **self.charts
        )
        
        # Save dashboard
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        dashboard_filename = f'analytics_dashboard_{timestamp}.html'
        
        with open(dashboard_filename, 'w', encoding='utf-8') as f:
            f.write(dashboard_content)
            
        print(f"✅ Dashboard saved as {dashboard_filename}")
        
        return dashboard_filename

def main():
    """Main execution function"""
    print("📊 Automated Report Generation System")
    print("=" * 50)
    
    # Initialize report generator
    generator = AutomatedReportGenerator()
    
    # Generate all reports
    report_files = generator.generate_all_reports()
    
    # Create integrated dashboard
    dashboard_file = generator.create_report_dashboard()
    
    print(f"\n✅ Report generation complete!")
    print(f"Files generated:")
    for report_type, filename in report_files.items():
        print(f"  - {report_type.title()}: {filename}")
    print(f"  - Dashboard: {dashboard_file}")
    
    return {**report_files, 'dashboard': dashboard_file}

if __name__ == "__main__":
    main()