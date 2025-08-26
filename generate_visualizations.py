#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Create results directory
import os
os.makedirs('results', exist_ok=True)
os.makedirs('results/visualizations', exist_ok=True)

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("Loading cleaned data...")
df_clean = pd.read_csv('border_crossing_clean.csv')
df_clean['Date'] = pd.to_datetime(df_clean['Date'])

# 1. BORDER COMPARISON ANALYSIS
print("Creating border comparison visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Border Crossing Analysis - Overview', fontsize=16, fontweight='bold')

# Total volume by border
border_totals = df_clean.groupby('Border')['Value'].sum().sort_values(ascending=False)
axes[0,0].bar(border_totals.index, border_totals.values, color=['#CD853F', '#2E8B57'])
axes[0,0].set_title('Total Border Crossing Volume', fontsize=14, fontweight='bold')
axes[0,0].set_ylabel('Total Crossings')
axes[0,0].tick_params(axis='x', rotation=45)
for i, v in enumerate(border_totals.values):
    axes[0,0].text(i, v + v*0.01, f'{v/1e9:.1f}B', ha='center', fontweight='bold')

# Average volume by border
border_avg = df_clean.groupby('Border')['Value'].mean().sort_values(ascending=False)
axes[0,1].bar(border_avg.index, border_avg.values, color=['#CD853F', '#2E8B57'])
axes[0,1].set_title('Average Border Crossing Volume', fontsize=14, fontweight='bold')
axes[0,1].set_ylabel('Average Crossings')
axes[0,1].tick_params(axis='x', rotation=45)
for i, v in enumerate(border_avg.values):
    axes[0,1].text(i, v + v*0.01, f'{v:,.0f}', ha='center', fontweight='bold')

# Number of ports by border
ports_by_border = df_clean.groupby('Border')['Port Name'].nunique().sort_values(ascending=False)
axes[1,0].bar(ports_by_border.index, ports_by_border.values, color=['#CD853F', '#2E8B57'])
axes[1,0].set_title('Number of Active Ports by Border', fontsize=14, fontweight='bold')
axes[1,0].set_ylabel('Number of Ports')
axes[1,0].tick_params(axis='x', rotation=45)
for i, v in enumerate(ports_by_border.values):
    axes[1,0].text(i, v + v*0.01, f'{v}', ha='center', fontweight='bold')

# Volume distribution (box plot)
df_clean.boxplot(column='Value', by='Border', ax=axes[1,1])
axes[1,1].set_title('Volume Distribution by Border', fontsize=14, fontweight='bold')
axes[1,1].set_ylabel('Crossing Volume (log scale)')
axes[1,1].set_yscale('log')
axes[1,1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('results/visualizations/01_border_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. TRANSPORTATION MODE ANALYSIS
print("Creating transportation mode analysis...")

transport_analysis = df_clean.groupby(['Border', 'Measure'])['Value'].sum().reset_index()
transport_pivot = transport_analysis.pivot(index='Measure', columns='Border', values='Value').fillna(0)
transport_pivot['Total'] = transport_pivot.sum(axis=1)
transport_pivot = transport_pivot.sort_values('Total', ascending=False)

fig, ax = plt.subplots(figsize=(16, 10))
top_measures = transport_pivot.head(12)
x = np.arange(len(top_measures))
width = 0.6

p1 = ax.bar(x, top_measures['US-Canada Border'], width, label='US-Canada Border', color='#2E8B57', alpha=0.8)
p2 = ax.bar(x, top_measures['US-Mexico Border'], width, bottom=top_measures['US-Canada Border'], 
           label='US-Mexico Border', color='#CD853F', alpha=0.8)

ax.set_title('Border Crossings by Transportation Mode', fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Transportation Mode', fontsize=12, fontweight='bold')
ax.set_ylabel('Total Crossings', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(top_measures.index, rotation=45, ha='right')
ax.legend(loc='upper right')

# Add value labels on bars
for i, (idx, row) in enumerate(top_measures.iterrows()):
    total = row['Total']
    if total > 1000000000:
        ax.text(i, total + total*0.02, f'{total/1000000000:.1f}B', ha='center', fontweight='bold')
    elif total > 1000000:
        ax.text(i, total + total*0.02, f'{total/1000000:.1f}M', ha='center', fontweight='bold')
    else:
        ax.text(i, total + total*0.02, f'{total/1000:.0f}K', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('results/visualizations/02_transportation_modes.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. TEMPORAL ANALYSIS
print("Creating temporal analysis...")

monthly_data = df_clean.groupby(['Month_Name', 'Border'])['Value'].sum().reset_index()
monthly_pivot = monthly_data.pivot(index='Month_Name', columns='Border', values='Value')

month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
               'July', 'August', 'September', 'October', 'November', 'December']
monthly_pivot = monthly_pivot.reindex([m for m in month_order if m in monthly_pivot.index])

fig, axes = plt.subplots(2, 1, figsize=(15, 12))

# Line plot for trends
axes[0].plot(monthly_pivot.index, monthly_pivot['US-Canada Border'], 
            marker='o', linewidth=3, label='US-Canada Border', color='#2E8B57')
axes[0].plot(monthly_pivot.index, monthly_pivot['US-Mexico Border'], 
            marker='s', linewidth=3, label='US-Mexico Border', color='#CD853F')
axes[0].set_title('Monthly Border Crossing Trends', fontsize=16, fontweight='bold')
axes[0].set_ylabel('Total Crossings', fontsize=12, fontweight='bold')
axes[0].legend(fontsize=12)
axes[0].grid(True, alpha=0.3)
axes[0].tick_params(axis='x', rotation=45)

# Stacked bar chart
monthly_pivot.plot(kind='bar', ax=axes[1], color=['#2E8B57', '#CD853F'], alpha=0.8)
axes[1].set_title('Monthly Border Crossings (Stacked)', fontsize=16, fontweight='bold')
axes[1].set_ylabel('Total Crossings', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Month', fontsize=12, fontweight='bold')
axes[1].legend(fontsize=12)
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('results/visualizations/03_temporal_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

# 4. GEOGRAPHIC ANALYSIS
print("Creating geographic analysis...")

state_totals = df_clean.groupby('State')['Value'].sum().sort_values(ascending=False)
port_totals = df_clean.groupby(['Port Name', 'State'])['Value'].sum().sort_values(ascending=False)

fig, axes = plt.subplots(2, 2, figsize=(18, 14))
fig.suptitle('Geographic Analysis - States and Ports', fontsize=16, fontweight='bold')

# Top 15 states by total volume
top_states = state_totals.head(14)  # All 14 states
colors = plt.cm.Set3(np.linspace(0, 1, len(top_states)))
bars = axes[0,0].bar(range(len(top_states)), top_states.values, color=colors)
axes[0,0].set_title('States by Total Border Crossings', fontsize=14, fontweight='bold')
axes[0,0].set_ylabel('Total Crossings')
axes[0,0].set_xticks(range(len(top_states)))
axes[0,0].set_xticklabels(top_states.index, rotation=45, ha='right')

# Add value labels
for i, v in enumerate(top_states.values):
    if v > 1000000000:
        axes[0,0].text(i, v + v*0.01, f'{v/1000000000:.1f}B', ha='center', fontweight='bold', fontsize=9)
    elif v > 1000000:
        axes[0,0].text(i, v + v*0.01, f'{v/1000000:.0f}M', ha='center', fontweight='bold', fontsize=9)
    else:
        axes[0,0].text(i, v + v*0.01, f'{v/1000:.0f}K', ha='center', fontweight='bold', fontsize=9)

# Border distribution by state
border_state = df_clean.groupby(['State', 'Border'])['Value'].sum().reset_index()
canada_states = border_state[border_state['Border'] == 'US-Canada Border']['State'].nunique()
mexico_states = border_state[border_state['Border'] == 'US-Mexico Border']['State'].nunique()

axes[0,1].pie([canada_states, mexico_states], labels=['US-Canada States', 'US-Mexico States'], 
              autopct='%1.0f', colors=['#2E8B57', '#CD853F'])
axes[0,1].set_title('Number of Border States', fontsize=14, fontweight='bold')

# Top ports by volume
top_ports = port_totals.head(15)
port_labels = [f"{port}\\n({state})" for port, state in top_ports.index]
axes[1,0].barh(range(len(top_ports)), top_ports.values, color='skyblue')
axes[1,0].set_title('Top 15 Ports by Total Volume', fontsize=14, fontweight='bold')
axes[1,0].set_xlabel('Total Crossings')
axes[1,0].set_yticks(range(len(top_ports)))
axes[1,0].set_yticklabels(port_labels, fontsize=8)
axes[1,0].invert_yaxis()

# Average crossings by state
state_avg = df_clean.groupby('State')['Value'].mean().sort_values(ascending=False).head(10)
axes[1,1].bar(range(len(state_avg)), state_avg.values, color='lightcoral')
axes[1,1].set_title('Top 10 States by Average Crossing Volume', fontsize=14, fontweight='bold')
axes[1,1].set_ylabel('Average Crossings')
axes[1,1].set_xticks(range(len(state_avg)))
axes[1,1].set_xticklabels(state_avg.index, rotation=45, ha='right')

plt.tight_layout()
plt.savefig('results/visualizations/04_geographic_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

# 5. TRAFFIC CATEGORY ANALYSIS
print("Creating traffic category analysis...")

# Categorize measures into commercial and personal traffic
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

df_clean['Traffic_Category'] = df_clean['Measure'].apply(categorize_traffic)

traffic_analysis = df_clean.groupby(['Border', 'Traffic_Category'])['Value'].sum().reset_index()
traffic_pivot = traffic_analysis.pivot(index='Traffic_Category', columns='Border', values='Value').fillna(0)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Traffic Category Analysis', fontsize=16, fontweight='bold')

# Traffic category distribution
traffic_total = df_clean.groupby('Traffic_Category')['Value'].sum().sort_values(ascending=False)
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
wedges, texts, autotexts = axes[0,0].pie(traffic_total.values, labels=traffic_total.index, autopct='%1.1f%%', 
                                        colors=colors, startangle=90)
axes[0,0].set_title('Traffic Distribution by Category', fontsize=14, fontweight='bold')

# Traffic by border
traffic_pivot.plot(kind='bar', ax=axes[0,1], color=['#2E8B57', '#CD853F'])
axes[0,1].set_title('Traffic Categories by Border', fontsize=14, fontweight='bold')
axes[0,1].set_ylabel('Total Crossings')
axes[0,1].legend(title='Border')
axes[0,1].tick_params(axis='x', rotation=45)

# Monthly trends by traffic category
monthly_traffic = df_clean.groupby(['Month_Name', 'Traffic_Category'])['Value'].sum().reset_index()
monthly_traffic_pivot = monthly_traffic.pivot(index='Month_Name', columns='Traffic_Category', values='Value').fillna(0)
monthly_traffic_pivot = monthly_traffic_pivot.reindex([m for m in month_order if m in monthly_traffic_pivot.index])

for category, color in zip(monthly_traffic_pivot.columns, colors):
    if category in monthly_traffic_pivot.columns:
        axes[1,0].plot(monthly_traffic_pivot.index, monthly_traffic_pivot[category], 
                      marker='o', label=category, linewidth=2, color=color)

axes[1,0].set_title('Monthly Trends by Traffic Category', fontsize=14, fontweight='bold')
axes[1,0].set_ylabel('Total Crossings')
axes[1,0].legend()
axes[1,0].tick_params(axis='x', rotation=45)
axes[1,0].grid(True, alpha=0.3)

# Top measures in each category
commercial_measures = df_clean[df_clean['Traffic_Category'] == 'Commercial'].groupby('Measure')['Value'].sum().head(5)
y_pos = np.arange(len(commercial_measures))
axes[1,1].barh(y_pos, commercial_measures.values, alpha=0.7, color='#FF6B6B')
axes[1,1].set_yticks(y_pos)
axes[1,1].set_yticklabels(commercial_measures.index, fontsize=10)
axes[1,1].set_title('Top 5 Commercial Traffic Measures', fontsize=14, fontweight='bold')
axes[1,1].set_xlabel('Total Crossings')

plt.tight_layout()
plt.savefig('results/visualizations/05_traffic_categories.png', dpi=300, bbox_inches='tight')
plt.close()

print("All visualizations saved to results/visualizations/")
print("Generated files:")
print("- 01_border_comparison.png")
print("- 02_transportation_modes.png") 
print("- 03_temporal_analysis.png")
print("- 04_geographic_analysis.png")
print("- 05_traffic_categories.png")