"""
Complete Guide: 20+ Chart Types for Time Series Data Analysis
Using Matplotlib, Pyplot, and Seaborn

Dataset: Bitcoin Trading Data (btc_1d_data_2018_to_2025.csv)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# DATA PREPARATION
# ============================================================================

# Load and prepare data
df = pd.read_csv('/content/drive/MyDrive/btc_1d_data_2018_to_2025.csv')
df['Open time'] = pd.to_datetime(df['Open time'])
df['Year'] = df['Open time'].dt.year
df['Month'] = df['Open time'].dt.month
df['Quarter'] = df['Open time'].dt.quarter
df['Day_of_Week'] = df['Open time'].dt.day_name()

# Calculate additional metrics
df['Price_Change'] = df['Close'] - df['Open']
df['Price_Change_Pct'] = ((df['Close'] - df['Open']) / df['Open']) * 100
df['Daily_Range'] = df['High'] - df['Low']
df['MA_7'] = df['Close'].rolling(window=7).mean()
df['MA_30'] = df['Close'].rolling(window=30).mean()

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============================================================================
# 1. LINE PLOT - Basic Time Series
# ============================================================================

plt.figure(figsize=(14, 6))
plt.plot(df['Open time'], df['Close'], linewidth=2, color='#2E86AB')
plt.title('1. Line Plot - Bitcoin Price Over Time', fontsize=16, fontweight='bold')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Close Price (USD)', fontsize=12)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('01_line_plot.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# 2. AREA PLOT - Filled Line Chart
# ============================================================================

plt.figure(figsize=(14, 6))
plt.fill_between(df['Open time'], df['Close'], alpha=0.5, color='#06D6A0')
plt.plot(df['Open time'], df['Close'], linewidth=2, color='#073B4C')
plt.title('2. Area Plot - Price with Fill', fontsize=16, fontweight='bold')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Close Price (USD)', fontsize=12)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('02_area_plot.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# 3. MULTI-LINE PLOT - Multiple Time Series
# ============================================================================

plt.figure(figsize=(14, 6))
plt.plot(df['Open time'], df['Close'], label='Close Price', linewidth=2, color='#E63946')
plt.plot(df['Open time'], df['MA_7'], label='7-Day MA', linewidth=2, linestyle='--', color='#457B9D')
plt.plot(df['Open time'], df['MA_30'], label='30-Day MA', linewidth=2, linestyle='--', color='#1D3557')
plt.title('3. Multi-Line Plot - Price with Moving Averages', fontsize=16, fontweight='bold')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Price (USD)', fontsize=12)
plt.legend(loc='best', fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('03_multiline_plot.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# 4. BAR PLOT - Categorical Time Data
# ============================================================================

yearly_avg = df.groupby('Year')['Close'].mean()
plt.figure(figsize=(12, 6))
bars = plt.bar(yearly_avg.index, yearly_avg.values, color=plt.cm.viridis(np.linspace(0, 1, len(yearly_avg))), 
               edgecolor='black', linewidth=1.5)
plt.title('4. Bar Plot - Average Annual Bitcoin Price', fontsize=16, fontweight='bold')
plt.xlabel('Year', fontsize=12)
plt.ylabel('Average Price (USD)', fontsize=12)
plt.xticks(yearly_avg.index)
for bar, val in zip(bars, yearly_avg.values):
    plt.text(bar.get_x() + bar.get_width()/2, val, f'${val:,.0f}', 
             ha='center', va='bottom', fontweight='bold')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('04_bar_plot.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# 5. HORIZONTAL BAR PLOT
# ============================================================================

monthly_volume = df.groupby('Month')['Volume'].sum().sort_values()
plt.figure(figsize=(10, 8))
colors = plt.cm.plasma(np.linspace(0, 1, 12))
plt.barh(range(12), monthly_volume.values, color=colors, edgecolor='black', linewidth=1.5)
plt.title('5. Horizontal Bar Plot - Total Volume by Month', fontsize=16, fontweight='bold')
plt.xlabel('Total Volume (BTC)', fontsize=12)
plt.ylabel('Month', fontsize=12)
plt.yticks(range(12), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('05_horizontal_bar.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# 6. STACKED BAR PLOT
# ============================================================================

quarterly_data = df.groupby(['Year', 'Quarter'])['Volume'].sum().unstack()
plt.figure(figsize=(14, 6))
quarterly_data.plot(kind='bar', stacked=True, colormap='tab10', 
                    edgecolor='black', linewidth=1, figsize=(14, 6))
plt.title('6. Stacked Bar Plot - Quarterly Volume by Year', fontsize=16, fontweight='bold')
plt.xlabel('Year', fontsize=12)
plt.ylabel('Volume (BTC)', fontsize=12)
plt.legend(title='Quarter', labels=['Q1', 'Q2', 'Q3', 'Q4'])
plt.xticks(rotation=0)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('06_stacked_bar.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# 7. GROUPED BAR PLOT
# ============================================================================

monthly_stats = df.groupby('Month')[['Open', 'Close']].mean()
x = np.arange(len(monthly_stats))
width = 0.35

plt.figure(figsize=(14, 6))
plt.bar(x - width/2, monthly_stats['Open'], width, label='Avg Open', 
        color='#FF6B6B', edgecolor='black')
plt.bar(x + width/2, monthly_stats['Close'], width, label='Avg Close', 
        color='#4ECDC4', edgecolor='black')
plt.title('7. Grouped Bar Plot - Open vs Close Prices by Month', fontsize=16, fontweight='bold')
plt.xlabel('Month', fontsize=12)
plt.ylabel('Price (USD)', fontsize=12)
plt.xticks(x, ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('07_grouped_bar.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# 8. PIE CHART - Distribution
# ============================================================================

quarterly_vol = df.groupby('Quarter')['Volume'].sum()
plt.figure(figsize=(10, 8))
colors_pie = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
explode = (0.05, 0.05, 0.05, 0.05)
plt.pie(quarterly_vol.values, labels=[f'Q{i}' for i in quarterly_vol.index], 
        autopct='%1.1f%%', startangle=90, colors=colors_pie, explode=explode,
        shadow=True, textprops={'fontsize': 12, 'fontweight': 'bold'})
plt.title('8. Pie Chart - Volume Distribution by Quarter', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('08_pie_chart.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# 9. DONUT CHART
# ============================================================================

yearly_trades = df.groupby('Year')['Number of trades'].sum()
plt.figure(figsize=(10, 8))
colors_donut = plt.cm.Set3(np.linspace(0, 1, len(yearly_trades)))
plt.pie(yearly_trades.values, labels=yearly_trades.index, autopct='%1.1f%%', 
        startangle=90, colors=colors_donut, pctdistance=0.85,
        textprops={'fontsize': 11, 'fontweight': 'bold'})
centre_circle = plt.Circle((0, 0), 0.70, fc='white')
plt.gca().add_artist(centre_circle)
plt.title('9. Donut Chart - Trading Activity by Year', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('09_donut_chart.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# 10. HISTOGRAM - Distribution Analysis
# ============================================================================

plt.figure(figsize=(12, 6))
plt.hist(df['Price_Change_Pct'], bins=50, color='#667eea', edgecolor='black', alpha=0.7)
plt.axvline(df['Price_Change_Pct'].mean(), color='red', linestyle='--', 
            linewidth=2, label=f'Mean: {df["Price_Change_Pct"].mean():.2f}%')
plt.axvline(df['Price_Change_Pct'].median(), color='green', linestyle='--', 
            linewidth=2, label=f'Median: {df["Price_Change_Pct"].median():.2f}%')
plt.title('10. Histogram - Daily Price Change Distribution', fontsize=16, fontweight='bold')
plt.xlabel('Price Change (%)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('10_histogram.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# 11. BOX PLOT - Statistical Distribution
# ============================================================================

monthly_price_data = [df[df['Month'] == m]['Close'].values for m in range(1, 13)]
plt.figure(figsize=(14, 6))
bp = plt.boxplot(monthly_price_data, patch_artist=True, 
                 labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
colors_box = plt.cm.rainbow(np.linspace(0, 1, 12))
for patch, color in zip(bp['boxes'], colors_box):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
plt.title('11. Box Plot - Price Distribution by Month', fontsize=16, fontweight='bold')
plt.xlabel('Month', fontsize=12)
plt.ylabel('Close Price (USD)', fontsize=12)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('11_box_plot.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# 12. VIOLIN PLOT - Distribution with Density
# ============================================================================

plt.figure(figsize=(14, 6))
yearly_price_data = [df[df['Year'] == year]['Close'].values for year in df['Year'].unique()]
parts = plt.violinplot(yearly_price_data, positions=range(len(df['Year'].unique())), 
                       showmeans=True, showmedians=True)
for i, pc in enumerate(parts['bodies']):
    pc.set_facecolor(plt.cm.viridis(i/len(parts['bodies'])))
    pc.set_alpha(0.7)
plt.title('12. Violin Plot - Price Distribution by Year', fontsize=16, fontweight='bold')
plt.xlabel('Year', fontsize=12)
plt.ylabel('Close Price (USD)', fontsize=12)
plt.xticks(range(len(df['Year'].unique())), df['Year'].unique())
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('12_violin_plot.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# 13. SCATTER PLOT - Correlation Analysis
# ============================================================================

plt.figure(figsize=(12, 8))
scatter = plt.scatter(df['Volume'], df['Close'], c=df['Open time'].astype(np.int64), 
                     cmap='viridis', alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
plt.colorbar(scatter, label='Time (older → newer)')
plt.title('13. Scatter Plot - Volume vs Price (colored by time)', fontsize=16, fontweight='bold')
plt.xlabel('Volume (BTC)', fontsize=12)
plt.ylabel('Close Price (USD)', fontsize=12)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('13_scatter_plot.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# 14. BUBBLE CHART - 3 Variables
# ============================================================================

sample_df = df.sample(n=500, random_state=42)  # Sample for clarity
plt.figure(figsize=(14, 8))
scatter = plt.scatter(sample_df['Volume'], sample_df['Close'], 
                     s=sample_df['Number of trades']/1e5, 
                     c=sample_df['Price_Change_Pct'], 
                     cmap='RdYlGn', alpha=0.6, edgecolors='black', linewidth=1)
plt.colorbar(scatter, label='Price Change (%)')
plt.title('14. Bubble Chart - Volume vs Price (size=trades, color=change%)', 
          fontsize=16, fontweight='bold')
plt.xlabel('Volume (BTC)', fontsize=12)
plt.ylabel('Close Price (USD)', fontsize=12)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('14_bubble_chart.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# 15. HEATMAP - 2D Pattern Analysis
# ============================================================================

monthly_avg_price = df.groupby(['Year', 'Month'])['Close'].mean().unstack()
plt.figure(figsize=(14, 8))
sns.heatmap(monthly_avg_price.T, annot=False, fmt='.0f', cmap='RdYlGn', 
            cbar_kws={'label': 'Avg Price (USD)'}, linewidths=0.5)
plt.title('15. Heatmap - Monthly Average Price by Year', fontsize=16, fontweight='bold')
plt.xlabel('Year', fontsize=12)
plt.ylabel('Month', fontsize=12)
plt.yticks(range(12), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], rotation=0)
plt.tight_layout()
plt.savefig('15_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# 16. CORRELATION HEATMAP
# ============================================================================

corr_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Number of trades']
correlation_matrix = df[corr_cols].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, square=True, linewidths=1, cbar_kws={'label': 'Correlation'})
plt.title('16. Correlation Heatmap - Trading Metrics', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('16_correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# 17. STACKED AREA CHART
# ============================================================================

quarterly_vol_by_year = df.groupby(['Year', 'Quarter'])['Volume'].sum().unstack()
plt.figure(figsize=(14, 6))
plt.stackplot(quarterly_vol_by_year.index, 
              quarterly_vol_by_year[1], quarterly_vol_by_year[2], 
              quarterly_vol_by_year[3], quarterly_vol_by_year[4],
              labels=['Q1', 'Q2', 'Q3', 'Q4'], 
              colors=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A'], alpha=0.8)
plt.title('17. Stacked Area Chart - Quarterly Volume Evolution', fontsize=16, fontweight='bold')
plt.xlabel('Year', fontsize=12)
plt.ylabel('Volume (BTC)', fontsize=12)
plt.legend(loc='upper left')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('17_stacked_area.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# 18. STEM PLOT - Discrete Time Series
# ============================================================================

recent_data = df.tail(60)  # Last 60 days
plt.figure(figsize=(14, 6))
markerline, stemlines, baseline = plt.stem(recent_data['Open time'], 
                                            recent_data['Price_Change_Pct'], 
                                            basefmt='r-')
plt.setp(markerline, 'markerfacecolor', '#06D6A0', 'markersize', 6)
plt.setp(stemlines, 'color', '#073B4C', 'linewidth', 1.5)
plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
plt.title('18. Stem Plot - Daily Price Changes (Last 60 Days)', fontsize=16, fontweight='bold')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Price Change (%)', fontsize=12)
plt.xticks(rotation=45)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('18_stem_plot.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# 19. ERROR BAR PLOT - Uncertainty Visualization
# ============================================================================

monthly_stats = df.groupby('Month').agg({
    'Close': ['mean', 'std']
}).reset_index()
monthly_stats.columns = ['Month', 'Mean', 'Std']

plt.figure(figsize=(14, 6))
plt.errorbar(monthly_stats['Month'], monthly_stats['Mean'], 
             yerr=monthly_stats['Std'], fmt='o-', linewidth=2, 
             markersize=10, capsize=8, capthick=2, 
             color='#E63946', ecolor='#457B9D')
plt.fill_between(monthly_stats['Month'], 
                 monthly_stats['Mean'] - monthly_stats['Std'], 
                 monthly_stats['Mean'] + monthly_stats['Std'], 
                 alpha=0.3, color='#457B9D')
plt.title('19. Error Bar Plot - Monthly Price with Standard Deviation', 
          fontsize=16, fontweight='bold')
plt.xlabel('Month', fontsize=12)
plt.ylabel('Price (USD)', fontsize=12)
plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('19_errorbar_plot.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# 20. CANDLESTICK PLOT (Custom Implementation)
# ============================================================================

sample_candles = df.tail(30)  # Last 30 days
fig, ax = plt.subplots(figsize=(14, 6))

for i, (idx, row) in enumerate(sample_candles.iterrows()):
    color = '#06D6A0' if row['Close'] >= row['Open'] else '#EF476F'
    
    # High-Low line
    ax.plot([i, i], [row['Low'], row['High']], color=color, linewidth=1.5)
    
    # Open-Close rectangle
    height = abs(row['Close'] - row['Open'])
    bottom = min(row['Open'], row['Close'])
    ax.add_patch(plt.Rectangle((i-0.3, bottom), 0.6, height, 
                                facecolor=color, edgecolor='black', linewidth=1))

ax.set_title('20. Candlestick Chart - Last 30 Days', fontsize=16, fontweight='bold')
ax.set_xlabel('Days', fontsize=12)
ax.set_ylabel('Price (USD)', fontsize=12)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('20_candlestick.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# 21. PAIR PLOT (Seaborn) - Multiple Variable Relationships
# ============================================================================

sample_pair = df.sample(n=500, random_state=42)[['Close', 'Volume', 
                                                   'Number of trades', 'Daily_Range']]
plt.figure(figsize=(12, 10))
pair_plot = sns.pairplot(sample_pair, diag_kind='kde', plot_kws={'alpha': 0.6, 's': 30}, 
                         diag_kws={'color': '#667eea'})
pair_plot.fig.suptitle('21. Pair Plot - Multivariate Relationships', 
                       fontsize=16, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('21_pairplot.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# 22. REGRESSION PLOT (Seaborn) - Trend Analysis
# ============================================================================

plt.figure(figsize=(12, 6))
sample_reg = df.sample(n=1000, random_state=42)
sns.regplot(x='Volume', y='Close', data=sample_reg, 
            scatter_kws={'alpha': 0.5, 's': 40, 'color': '#4ECDC4'}, 
            line_kws={'color': '#E63946', 'linewidth': 3})
plt.title('22. Regression Plot - Volume vs Price Trend', fontsize=16, fontweight='bold')
plt.xlabel('Volume (BTC)', fontsize=12)
plt.ylabel('Close Price (USD)', fontsize=12)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('22_regression_plot.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# 23. KDE PLOT (Kernel Density Estimation)
# ============================================================================

plt.figure(figsize=(12, 6))
sns.kdeplot(data=df, x='Price_Change_Pct', fill=True, color='#667eea', alpha=0.6)
plt.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Change')
plt.title('23. KDE Plot - Price Change Distribution Density', fontsize=16, fontweight='bold')
plt.xlabel('Price Change (%)', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('23_kde_plot.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# 24. JOINT PLOT (Seaborn) - Bivariate with Marginals
# ============================================================================

sample_joint = df.sample(n=1000, random_state=42)
joint = sns.jointplot(data=sample_joint, x='Volume', y='Close', 
                      kind='hex', color='#667eea', height=10)
joint.fig.suptitle('24. Joint Plot - Volume vs Price with Distributions', 
                   fontsize=16, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('24_joint_plot.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# 25. SWARM PLOT (Seaborn) - Categorical Distribution
# ============================================================================

sample_swarm = df.sample(n=500, random_state=42)
plt.figure(figsize=(14, 6))
sns.swarmplot(data=sample_swarm, x='Quarter', y='Daily_Range', 
              hue='Quarter', palette='Set2', size=5)
plt.title('25. Swarm Plot - Daily Range Distribution by Quarter', 
          fontsize=16, fontweight='bold')
plt.xlabel('Quarter', fontsize=12)
plt.ylabel('Daily Range (USD)', fontsize=12)
plt.legend(title='Quarter')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('25_swarm_plot.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ Successfully created 25 different chart types!")
print("\n📊 Chart Types Generated:")
print("1. Line Plot")
print("2. Area Plot")
print("3. Multi-Line Plot")
print("4. Bar Plot")
print("5. Horizontal Bar Plot")
print("6. Stacked Bar Plot")
print("7. Grouped Bar Plot")
print("8. Pie Chart")
print("9. Donut Chart")
print("10. Histogram")
print("11. Box Plot")
print("12. Violin Plot")
print("13. Scatter Plot")
print("14. Bubble Chart")
print("15. Heatmap")
print("16. Correlation Heatmap")
print("17. Stacked Area Chart")
print("18. Stem Plot")
print("19. Error Bar Plot")
print("20. Candlestick Chart")
print("21. Pair Plot (Seaborn)")
print("22. Regression Plot (Seaborn)")
print("23. KDE Plot")
print("24. Joint Plot (Seaborn)")
print("25. Swarm Plot (Seaborn)")
print("\n📁 All charts saved as high-resolution PNG files (300 DPI)")
