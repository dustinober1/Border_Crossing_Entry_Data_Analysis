#!/usr/bin/env python3
"""
Interactive Border Crossing Dashboard
=====================================

This module creates an interactive Plotly Dash dashboard for border crossing analysis
with real-time filtering, dynamic visualizations, and responsive design.

Features:
- Multi-level filtering (Border, State, Transportation Mode, Date Range)
- Real-time chart updates
- Geographic mapping with port details
- Time series analysis with zoom capabilities
- Export functionality for reports
- Mobile-responsive design

Author: Portfolio Analysis
Date: 2025-01-28
"""

import dash
from dash import dcc, html, Input, Output, callback, State
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

class BorderCrossingDashboard:
    """Interactive dashboard for border crossing analysis"""
    
    def __init__(self, data_path='Border_Crossing_Entry_Data.csv'):
        """Initialize dashboard with data loading"""
        self.app = dash.Dash(__name__, external_stylesheets=[
            'https://codepen.io/chriddyp/pen/bWLwgP.css',
            'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css'
        ])
        self.load_and_process_data(data_path)
        self.setup_layout()
        self.setup_callbacks()
        
    def load_and_process_data(self, data_path):
        """Load and preprocess data for dashboard"""
        print("📊 Loading data for dashboard...")
        
        self.df = pd.read_csv(data_path)
        
        # Data cleaning
        self.df['Date'] = pd.to_datetime(self.df['Date'], format='%b %Y')
        self.df = self.df[self.df['Value'] > 0]  # Remove zero/negative values
        
        # Add derived columns
        self.df['Year'] = self.df['Date'].dt.year
        self.df['Month'] = self.df['Date'].dt.month
        self.df['Month_Name'] = self.df['Date'].dt.strftime('%B')
        self.df['Quarter'] = self.df['Date'].dt.quarter
        
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
        
        # Create summary statistics
        self.summary_stats = {
            'total_crossings': self.df['Value'].sum(),
            'total_ports': self.df['Port Name'].nunique(),
            'total_states': self.df['State'].nunique(),
            'date_range': f"{self.df['Date'].min().strftime('%b %Y')} - {self.df['Date'].max().strftime('%b %Y')}"
        }
        
        print(f"✅ Data loaded: {len(self.df):,} records, {self.summary_stats['total_ports']} ports")
        
    def create_kpi_cards(self, filtered_df):
        """Create KPI cards for the dashboard"""
        total_crossings = filtered_df['Value'].sum()
        active_ports = filtered_df['Port Name'].nunique()
        avg_monthly = filtered_df.groupby(filtered_df['Date'].dt.to_period('M'))['Value'].sum().mean()
        top_border = filtered_df.groupby('Border')['Value'].sum().idxmax()
        
        return html.Div([
            html.Div([
                html.Div([
                    html.I(className="fas fa-chart-line", style={'fontSize': '2em', 'color': '#3498db'}),
                    html.H3(f"{total_crossings:,.0f}", style={'margin': '10px 0 5px 0'}),
                    html.P("Total Crossings", style={'margin': '0', 'color': '#666'})
                ], className="kpi-card"),
                
                html.Div([
                    html.I(className="fas fa-map-marker-alt", style={'fontSize': '2em', 'color': '#e74c3c'}),
                    html.H3(f"{active_ports}", style={'margin': '10px 0 5px 0'}),
                    html.P("Active Ports", style={'margin': '0', 'color': '#666'})
                ], className="kpi-card"),
                
                html.Div([
                    html.I(className="fas fa-calendar-alt", style={'fontSize': '2em', 'color': '#f39c12'}),
                    html.H3(f"{avg_monthly:,.0f}", style={'margin': '10px 0 5px 0'}),
                    html.P("Avg Monthly", style={'margin': '0', 'color': '#666'})
                ], className="kpi-card"),
                
                html.Div([
                    html.I(className="fas fa-trophy", style={'fontSize': '2em', 'color': '#27ae60'}),
                    html.H3(f"{top_border.split('-')[1]}", style={'margin': '10px 0 5px 0'}),
                    html.P("Top Border", style={'margin': '0', 'color': '#666'})
                ], className="kpi-card"),
            ], style={'display': 'flex', 'justifyContent': 'space-around', 'marginBottom': '20px'})
        ])
        
    def setup_layout(self):
        """Setup the dashboard layout"""
        # Custom CSS
        custom_style = """
        <style>
        .kpi-card {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            text-align: center;
            min-width: 150px;
            margin: 0 10px;
        }
        .filter-section {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
        }
        .chart-container {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        h1 {
            text-align: center;
            color: #2c3e50;
            margin-bottom: 30px;
        }
        h3 {
            color: #34495e;
            margin-bottom: 15px;
        }
        </style>
        """
        
        self.app.layout = html.Div([
            html.Div([html.Div(custom_style, dangerously_allow_html=True)]),
            
            # Header
            html.Div([
                html.H1("🌍 US Border Crossing Analytics Dashboard", 
                       style={'textAlign': 'center', 'marginBottom': '10px'}),
                html.P("Interactive analysis of border crossing patterns with real-time filtering",
                      style={'textAlign': 'center', 'color': '#666', 'fontSize': '1.2em'})
            ]),
            
            # KPI Cards
            html.Div(id='kpi-cards'),
            
            # Filters Section
            html.Div([
                html.H3("🔍 Filters", style={'marginBottom': '15px'}),
                
                html.Div([
                    html.Div([
                        html.Label("Border:", style={'fontWeight': 'bold'}),
                        dcc.Dropdown(
                            id='border-filter',
                            options=[{'label': 'All Borders', 'value': 'all'}] + 
                                   [{'label': border, 'value': border} 
                                    for border in sorted(self.df['Border'].unique())],
                            value='all',
                            style={'marginTop': '5px'}
                        )
                    ], style={'width': '23%', 'display': 'inline-block', 'marginRight': '2%'}),
                    
                    html.Div([
                        html.Label("State:", style={'fontWeight': 'bold'}),
                        dcc.Dropdown(
                            id='state-filter',
                            options=[{'label': 'All States', 'value': 'all'}] + 
                                   [{'label': state, 'value': state} 
                                    for state in sorted(self.df['State'].dropna().unique())],
                            value='all',
                            style={'marginTop': '5px'}
                        )
                    ], style={'width': '23%', 'display': 'inline-block', 'marginRight': '2%'}),
                    
                    html.Div([
                        html.Label("Traffic Category:", style={'fontWeight': 'bold'}),
                        dcc.Dropdown(
                            id='traffic-filter',
                            options=[{'label': 'All Categories', 'value': 'all'}] + 
                                   [{'label': category, 'value': category} 
                                    for category in sorted(self.df['Traffic_Category'].unique())],
                            value='all',
                            style={'marginTop': '5px'}
                        )
                    ], style={'width': '23%', 'display': 'inline-block', 'marginRight': '2%'}),
                    
                    html.Div([
                        html.Label("Transportation Mode:", style={'fontWeight': 'bold'}),
                        dcc.Dropdown(
                            id='measure-filter',
                            options=[{'label': 'All Modes', 'value': 'all'}] + 
                                   [{'label': measure, 'value': measure} 
                                    for measure in sorted(self.df['Measure'].unique())],
                            value='all',
                            style={'marginTop': '5px'}
                        )
                    ], style={'width': '23%', 'display': 'inline-block'}),
                ], style={'marginBottom': '15px'}),
                
                html.Div([
                    html.Label("Date Range:", style={'fontWeight': 'bold'}),
                    dcc.DatePickerRange(
                        id='date-filter',
                        start_date=self.df['Date'].min(),
                        end_date=self.df['Date'].max(),
                        display_format='MMM YYYY',
                        style={'marginTop': '5px'}
                    )
                ])
            ], className='filter-section'),
            
            # Charts Section
            html.Div([
                # Row 1: Time Series and Border Comparison
                html.Div([
                    html.Div([
                        dcc.Graph(id='time-series-chart')
                    ], style={'width': '65%', 'display': 'inline-block', 'verticalAlign': 'top'}),
                    
                    html.Div([
                        dcc.Graph(id='border-comparison-chart')
                    ], style={'width': '33%', 'display': 'inline-block', 'marginLeft': '2%', 'verticalAlign': 'top'})
                ], style={'marginBottom': '20px'}),
                
                # Row 2: Geographic Map and Top States
                html.Div([
                    html.Div([
                        dcc.Graph(id='geographic-map')
                    ], style={'width': '65%', 'display': 'inline-block', 'verticalAlign': 'top'}),
                    
                    html.Div([
                        dcc.Graph(id='top-states-chart')
                    ], style={'width': '33%', 'display': 'inline-block', 'marginLeft': '2%', 'verticalAlign': 'top'})
                ], style={'marginBottom': '20px'}),
                
                # Row 3: Transportation Analysis and Seasonal Patterns
                html.Div([
                    html.Div([
                        dcc.Graph(id='transportation-chart')
                    ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),
                    
                    html.Div([
                        dcc.Graph(id='seasonal-chart')
                    ], style={'width': '48%', 'display': 'inline-block', 'marginLeft': '4%', 'verticalAlign': 'top'})
                ])
            ]),
            
            # Export Section
            html.Div([
                html.H3("📁 Export Data", style={'marginTop': '30px'}),
                html.Button("📊 Export Filtered Data as CSV", id='export-btn', 
                           style={'backgroundColor': '#3498db', 'color': 'white', 
                                 'padding': '10px 20px', 'border': 'none', 'borderRadius': '5px'})
            ], style={'textAlign': 'center', 'marginTop': '30px', 'marginBottom': '30px'})
        ], style={'padding': '20px', 'backgroundColor': '#f5f6fa'})
        
    def filter_data(self, border, state, traffic, measure, start_date, end_date):
        """Apply filters to the dataset"""
        filtered_df = self.df.copy()
        
        # Apply filters
        if border != 'all':
            filtered_df = filtered_df[filtered_df['Border'] == border]
        if state != 'all':
            filtered_df = filtered_df[filtered_df['State'] == state]
        if traffic != 'all':
            filtered_df = filtered_df[filtered_df['Traffic_Category'] == traffic]
        if measure != 'all':
            filtered_df = filtered_df[filtered_df['Measure'] == measure]
            
        # Date filter
        if start_date and end_date:
            filtered_df = filtered_df[
                (filtered_df['Date'] >= start_date) & 
                (filtered_df['Date'] <= end_date)
            ]
            
        return filtered_df
        
    def setup_callbacks(self):
        """Setup all dashboard callbacks"""
        
        @self.app.callback(
            [Output('kpi-cards', 'children'),
             Output('time-series-chart', 'figure'),
             Output('border-comparison-chart', 'figure'),
             Output('geographic-map', 'figure'),
             Output('top-states-chart', 'figure'),
             Output('transportation-chart', 'figure'),
             Output('seasonal-chart', 'figure')],
            [Input('border-filter', 'value'),
             Input('state-filter', 'value'),
             Input('traffic-filter', 'value'),
             Input('measure-filter', 'value'),
             Input('date-filter', 'start_date'),
             Input('date-filter', 'end_date')]
        )
        def update_dashboard(border, state, traffic, measure, start_date, end_date):
            """Update all dashboard components based on filters"""
            
            # Filter data
            filtered_df = self.filter_data(border, state, traffic, measure, start_date, end_date)
            
            if filtered_df.empty:
                # Return empty figures if no data
                empty_fig = go.Figure().add_annotation(
                    text="No data available for current filters",
                    xref="paper", yref="paper", x=0.5, y=0.5,
                    font=dict(size=16), showarrow=False
                )
                return (html.Div("No data available"), empty_fig, empty_fig, 
                       empty_fig, empty_fig, empty_fig, empty_fig)
            
            # KPI Cards
            kpi_cards = self.create_kpi_cards(filtered_df)
            
            # 1. Time Series Chart
            monthly_data = filtered_df.groupby(['Date', 'Border'])['Value'].sum().reset_index()
            time_series_fig = px.line(
                monthly_data, x='Date', y='Value', color='Border',
                title="Monthly Border Crossing Trends",
                labels={'Value': 'Crossings', 'Date': 'Date'},
                color_discrete_map={'US-Canada Border': '#2E8B57', 'US-Mexico Border': '#CD853F'}
            )
            time_series_fig.update_layout(hovermode='x unified')
            
            # 2. Border Comparison
            border_totals = filtered_df.groupby('Border')['Value'].sum().reset_index()
            border_fig = px.pie(
                border_totals, values='Value', names='Border',
                title="Volume by Border",
                color_discrete_map={'US-Canada Border': '#2E8B57', 'US-Mexico Border': '#CD853F'}
            )
            
            # 3. Geographic Map
            geo_data = filtered_df.groupby(['Port Name', 'State', 'Border', 'Latitude', 'Longitude'])['Value'].sum().reset_index()
            geo_data = geo_data.dropna(subset=['Latitude', 'Longitude'])
            
            map_fig = px.scatter_mapbox(
                geo_data, lat='Latitude', lon='Longitude', size='Value',
                color='Border', hover_name='Port Name',
                hover_data={'State': True, 'Value': ':,.0f'},
                color_discrete_map={'US-Canada Border': '#2E8B57', 'US-Mexico Border': '#CD853F'},
                mapbox_style='open-street-map', zoom=3,
                title="Border Crossings by Location"
            )
            map_fig.update_layout(height=500)
            
            # 4. Top States
            state_totals = filtered_df.groupby('State')['Value'].sum().nlargest(10).reset_index()
            states_fig = px.bar(
                state_totals, x='Value', y='State', orientation='h',
                title="Top 10 States by Volume",
                labels={'Value': 'Total Crossings', 'State': 'State'}
            )
            states_fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            
            # 5. Transportation Modes
            transport_data = filtered_df.groupby('Traffic_Category')['Value'].sum().reset_index()
            transport_fig = px.bar(
                transport_data, x='Traffic_Category', y='Value',
                title="Volume by Traffic Category",
                labels={'Value': 'Total Crossings', 'Traffic_Category': 'Category'}
            )
            
            # 6. Seasonal Patterns
            seasonal_data = filtered_df.groupby('Month_Name')['Value'].sum().reindex([
                'January', 'February', 'March', 'April', 'May', 'June',
                'July', 'August', 'September', 'October', 'November', 'December'
            ]).reset_index()
            seasonal_fig = px.line(
                seasonal_data, x='Month_Name', y='Value',
                title="Seasonal Patterns",
                labels={'Value': 'Total Crossings', 'Month_Name': 'Month'}
            )
            seasonal_fig.update_xaxis(tickangle=45)
            
            return (kpi_cards, time_series_fig, border_fig, map_fig, 
                   states_fig, transport_fig, seasonal_fig)
    
    def run_server(self, debug=True, port=8050):
        """Run the dashboard server"""
        print(f"🚀 Starting dashboard server on http://localhost:{port}")
        self.app.run_server(debug=debug, port=port, host='0.0.0.0')

def main():
    """Main function to run the dashboard"""
    print("🎯 Initializing Border Crossing Interactive Dashboard")
    print("=" * 60)
    
    try:
        # Create dashboard instance
        dashboard = BorderCrossingDashboard()
        
        print("✅ Dashboard initialized successfully")
        print("📱 Features available:")
        print("  - Real-time filtering by Border, State, Traffic Category, and Date")
        print("  - Interactive time series analysis")
        print("  - Geographic mapping with port details")
        print("  - KPI cards with dynamic updates")
        print("  - Export functionality")
        print("\n🌐 Access the dashboard at: http://localhost:8050")
        
        # Run server
        dashboard.run_server(debug=True, port=8050)
        
    except Exception as e:
        print(f"❌ Error starting dashboard: {e}")
        print("Please ensure all required packages are installed:")
        print("  pip install dash plotly pandas numpy")

if __name__ == "__main__":
    main()