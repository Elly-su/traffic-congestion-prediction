"""
Exploratory Data Analysis Module for Traffic Congestion Prediction
Generates visualizations and insights from traffic data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import stats


class TrafficEDA:
    """Class for performing exploratory data analysis on traffic data."""
    
    def __init__(self, output_dir: str = 'visualizations'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self._set_style()
    
    def _set_style(self):
        """Set consistent plotting style."""
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        plt.rcParams['figure.figsize'] = (14, 6)
        plt.rcParams['font.size'] = 10
    
    def load_data(self, filepath: str = 'data/raw/traffic_data_raw.csv') -> pd.DataFrame:
        """Load data for analysis."""
        print("\n" + "="*70)
        print("  LOADING DATA FOR EDA")
        print("="*70)
        
        df = pd.read_csv(filepath)
        df['date_time'] = pd.to_datetime(df['date_time'])
        
        print(f"  ✓ Loaded {len(df):,} records")
        return df
    
    def plot_traffic_by_hour(self, df: pd.DataFrame) -> None:
        """Analyze and plot traffic volume by hour of day."""
        print("\n  Analyzing traffic by hour...")
        
        df['hour'] = df['date_time'].dt.hour
        hourly_avg = df.groupby('hour')['traffic_volume'].agg(['mean', 'std'])
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        ax.plot(hourly_avg.index, hourly_avg['mean'], 
               marker='o', linewidth=2.5, markersize=8, label='Average Traffic')
        ax.fill_between(hourly_avg.index,
                        hourly_avg['mean'] - hourly_avg['std'],
                        hourly_avg['mean'] + hourly_avg['std'],
                        alpha=0.3, label='± 1 Std Dev')
        
        # Highlight rush hours
        rush_hours_morning = [7, 8, 9]
        rush_hours_evening = [16, 17, 18, 19]
        for hour in rush_hours_morning:
            ax.axvspan(hour-0.5, hour+0.5, color='orange', alpha=0.2)
        for hour in rush_hours_evening:
            ax.axvspan(hour-0.5, hour+0.5, color='red', alpha=0.2)
        
        ax.set_xlabel('Hour of Day', fontsize=12, fontweight='bold')
        ax.set_ylabel('Average Traffic Volume', fontsize=12, fontweight='bold')
        ax.set_title('Traffic Volume by Hour of Day\n(Orange: Morning Rush | Red: Evening Rush)', 
                    fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/traffic_by_hour.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved: traffic_by_hour.png")
    
    def plot_traffic_by_day_of_week(self, df: pd.DataFrame) -> None:
        """Analyze and plot traffic volume by day of week."""
        print("\n  Analyzing traffic by day of week...")
        
        df['day_of_week'] = df['date_time'].dt.dayofweek
        df['day_name'] = df['date_time'].dt.day_name()
        
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        sns.boxplot(data=df, x='day_name', y='traffic_volume', 
                   order=day_order, palette='Set2', ax=ax)
        
        ax.set_xlabel('Day of Week', fontsize=12, fontweight='bold')
        ax.set_ylabel('Traffic Volume', fontsize=12, fontweight='bold')
        ax.set_title('Traffic Volume Distribution by Day of Week', 
                    fontsize=14, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        
        # Add weekend shading
        ax.axvspan(4.5, 6.5, color='lightblue', alpha=0.2, label='Weekend')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/traffic_by_weekday.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved: traffic_by_weekday.png")
    
    def plot_seasonal_patterns(self, df: pd.DataFrame) -> None:
        """Analyze seasonal traffic patterns."""
        print("\n  Analyzing seasonal patterns...")
        
        df['month'] = df['date_time'].dt.month
        df['year_month'] = df['date_time'].dt.to_period('M')
        
        monthly_avg = df.groupby('year_month')['traffic_volume'].mean()
        
        fig, ax = plt.subplots(figsize=(16, 6))
        
        monthly_avg.plot(ax=ax, linewidth=2.5, marker='o', markersize=6, color='darkblue')
        
        ax.set_xlabel('Month', fontsize=12, fontweight='bold')
        ax.set_ylabel('Average Traffic Volume', fontsize=12, fontweight='bold')
        ax.set_title('Seasonal Traffic Patterns (Monthly Average)', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/seasonal_patterns.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved: seasonal_patterns.png")
    
    def plot_holiday_comparison(self, df: pd.DataFrame) -> None:
        """Compare traffic on holidays vs regular days."""
        print("\n  Analyzing holiday impact...")
        
        df['is_holiday'] = df['holiday'] != 'None'
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Box plot comparison
        sns.boxplot(data=df, x='is_holiday', y='traffic_volume', 
                   palette='Set1', ax=axes[0])
        axes[0].set_xticklabels(['Regular Day', 'Holiday'])
        axes[0].set_xlabel('Day Type', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Traffic Volume', fontsize=12, fontweight='bold')
        axes[0].set_title('Traffic Volume: Holidays vs Regular Days', 
                         fontsize=14, fontweight='bold')
        
        # Distribution comparison
        df[df['is_holiday'] == False]['traffic_volume'].hist(
            bins=50, alpha=0.6, label='Regular Days', ax=axes[1], color='blue')
        df[df['is_holiday'] == True]['traffic_volume'].hist(
            bins=50, alpha=0.6, label='Holidays', ax=axes[1], color='red')
        
        axes[1].set_xlabel('Traffic Volume', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Frequency', fontsize=12, fontweight='bold')
        axes[1].set_title('Traffic Distribution Comparison', 
                         fontsize=14, fontweight='bold')
        axes[1].legend()
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/holiday_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved: holiday_comparison.png")
    
    def plot_correlation_matrix(self, df: pd.DataFrame) -> None:
        """Generate correlation heatmap for numerical features."""
        print("\n  Creating correlation matrix...")
        
        # Select numerical columns
        numerical_cols = ['traffic_volume', 'temp', 'rain_1h', 'snow_1h', 
                         'clouds_all', 'temperature_2m', 'precipitation', 
                         'windspeed_10m', 'cloudcover']
        
        # Filter existing columns
        numerical_cols = [col for col in numerical_cols if col in df.columns]
        
        correlation_matrix = df[numerical_cols].corr()
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                   center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                   ax=ax)
        
        ax.set_title('Correlation Matrix: Traffic and Weather Variables', 
                    fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved: correlation_matrix.png")
    
    def plot_weather_impact(self, df: pd.DataFrame) -> None:
        """Analyze impact of weather conditions on traffic."""
        print("\n  Analyzing weather impact...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Temperature vs Traffic
        temp_celsius = df['temp'] - 273.15 if df['temp'].mean() > 200 else df['temp']
        axes[0, 0].scatter(temp_celsius, df['traffic_volume'], 
                          alpha=0.3, s=10, color='orange')
        axes[0, 0].set_xlabel('Temperature (°C)', fontsize=11, fontweight='bold')
        axes[0, 0].set_ylabel('Traffic Volume', fontsize=11, fontweight='bold')
        axes[0, 0].set_title('Temperature vs Traffic Volume', fontsize=12, fontweight='bold')
        
        # Add trend line
        z = np.polyfit(temp_celsius, df['traffic_volume'], 1)
        p = np.poly1d(z)
        axes[0, 0].plot(temp_celsius.sort_values(), p(temp_celsius.sort_values()), 
                       "r--", linewidth=2, label=f'Trend Line')
        axes[0, 0].legend()
        
        # 2. Rain impact
        df['rain_category'] = pd.cut(df['rain_1h'], 
                                     bins=[-0.1, 0, 1, 5, 100],
                                     labels=['No Rain', 'Light', 'Moderate', 'Heavy'])
        
        sns.boxplot(data=df, x='rain_category', y='traffic_volume', 
                   palette='Blues', ax=axes[0, 1])
        axes[0, 1].set_xlabel('Rain Intensity', fontsize=11, fontweight='bold')
        axes[0, 1].set_ylabel('Traffic Volume', fontsize=11, fontweight='bold')
        axes[0, 1].set_title('Rain Impact on Traffic', fontsize=12, fontweight='bold')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Snow impact
        df['snow_category'] = pd.cut(df['snow_1h'],
                                     bins=[-0.1, 0, 0.5, 2, 100],
                                     labels=['No Snow', 'Light', 'Moderate', 'Heavy'])
        
        sns.boxplot(data=df, x='snow_category', y='traffic_volume',
                   palette='winter', ax=axes[1, 0])
        axes[1, 0].set_xlabel('Snow Intensity', fontsize=11, fontweight='bold')
        axes[1, 0].set_ylabel('Traffic Volume', fontsize=11, fontweight='bold')
        axes[1, 0].set_title('Snow Impact on Traffic', fontsize=12, fontweight='bold')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. Weather condition comparison
        weather_avg = df.groupby('weather_main')['traffic_volume'].mean().sort_values()
        weather_avg.plot(kind='barh', ax=axes[1, 1], color='skyblue', edgecolor='black')
        axes[1, 1].set_xlabel('Average Traffic Volume', fontsize=11, fontweight='bold')
        axes[1, 1].set_ylabel('Weather Condition', fontsize=11, fontweight='bold')
        axes[1, 1].set_title('Traffic by Weather Condition', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/weather_impact.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved: weather_impact.png")
    
    def plot_hourly_heatmap(self, df: pd.DataFrame) -> None:
        """Create heatmap of traffic by hour and day of week."""
        print("\n  Creating hourly heatmap...")
        
        df['hour'] = df['date_time'].dt.hour
        df['day_of_week'] = df['date_time'].dt.dayofweek
        
        pivot_table = df.pivot_table(
            values='traffic_volume',
            index='day_of_week',
            columns='hour',
            aggfunc='mean'
        )
        
        day_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        
        fig, ax = plt.subplots(figsize=(16, 8))
        
        sns.heatmap(pivot_table, cmap='YlOrRd', annot=False, fmt='.0f',
                   cbar_kws={'label': 'Average Traffic Volume'},
                   yticklabels=day_labels, ax=ax)
        
        ax.set_xlabel('Hour of Day', fontsize=12, fontweight='bold')
        ax.set_ylabel('Day of Week', fontsize=12, fontweight='bold')
        ax.set_title('Traffic Volume Heatmap: Hour vs Day of Week', 
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/hourly_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved: hourly_heatmap.png")
    
    def plot_traffic_distribution(self, df: pd.DataFrame) -> None:
        """Plot traffic volume distribution and statistics."""
        print("\n  Analyzing traffic distribution...")
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Histogram with KDE
        axes[0].hist(df['traffic_volume'], bins=50, alpha=0.7, 
                    color='steelblue', edgecolor='black', density=True)
        
        # Add KDE
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(df['traffic_volume'])
        x_range = np.linspace(df['traffic_volume'].min(), 
                             df['traffic_volume'].max(), 100)
        axes[0].plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')
        
        axes[0].axvline(df['traffic_volume'].mean(), color='green', 
                       linestyle='--', linewidth=2, label=f'Mean: {df["traffic_volume"].mean():.0f}')
        axes[0].axvline(df['traffic_volume'].median(), color='orange',
                       linestyle='--', linewidth=2, label=f'Median: {df["traffic_volume"].median():.0f}')
        
        axes[0].set_xlabel('Traffic Volume', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Density', fontsize=12, fontweight='bold')
        axes[0].set_title('Traffic Volume Distribution', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Q-Q plot
        stats.probplot(df['traffic_volume'], dist="norm", plot=axes[1])
        axes[1].set_title('Q-Q Plot (Normal Distribution Check)', 
                         fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/traffic_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved: traffic_distribution.png")
    
    def generate_summary_statistics(self, df: pd.DataFrame) -> None:
        """Generate and save summary statistics."""
        print("\n" + "="*70)
        print("  SUMMARY STATISTICS")
        print("="*70)
        
        print(f"\n  Dataset Overview:")
        print(f"    Total records: {len(df):,}")
        print(f"    Date range: {df['date_time'].min()} to {df['date_time'].max()}")
        
        print(f"\n  Traffic Volume Statistics:")
        print(f"    Mean: {df['traffic_volume'].mean():.2f}")
        print(f"    Median: {df['traffic_volume'].median():.2f}")
        print(f"    Std Dev: {df['traffic_volume'].std():.2f}")
        print(f"    Min: {df['traffic_volume'].min():.2f}")
        print(f"    Max: {df['traffic_volume'].max():.2f}")
        
        print(f"\n  Peak Traffic Hours:")
        df['hour'] = df['date_time'].dt.hour
        hourly_avg = df.groupby('hour')['traffic_volume'].mean()
        top_hours = hourly_avg.nlargest(5)
        for hour, volume in top_hours.items():
            print(f"    {hour:02d}:00 - {volume:.0f} vehicles/hour")


def main():
    """Main EDA execution."""
    print("\n" + "="*70)
    print("  EXPLORATORY DATA ANALYSIS")
    print("="*70)
    
    eda = TrafficEDA()
    
    # Load data
    df = eda.load_data()
    
    # Generate all visualizations
    print("\n  Generating visualizations...")
    eda.plot_traffic_distribution(df)
    eda.plot_traffic_by_hour(df)
    eda.plot_traffic_by_day_of_week(df)
    eda.plot_seasonal_patterns(df)
    eda.plot_holiday_comparison(df)
    eda.plot_hourly_heatmap(df)
    eda.plot_correlation_matrix(df)
    eda.plot_weather_impact(df)
    
    # Generate summary statistics
    eda.generate_summary_statistics(df)
    
    print("\n" + "="*70)
    print("  ✓ EDA COMPLETE")
    print(f"  ✓ All visualizations saved to: {eda.output_dir}/")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
