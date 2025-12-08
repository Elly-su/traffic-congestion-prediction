"""
Data Collection Module for Traffic Congestion Prediction
Downloads and integrates traffic and weather datasets from multiple sources.
"""

import pandas as pd
import numpy as np
import requests
import os
from datetime import datetime, timedelta
from typing import Tuple

# Set random seed for reproducibility
np.random.seed(42)


def download_traffic_data() -> pd.DataFrame:
    """
    Download the Metro Interstate Traffic Volume dataset.
    This function creates a comprehensive dataset based on the UCI Metro Interstate Traffic Volume dataset.
    
    For the actual implementation, students would download from:
    https://archive.ics.uci.edu/ml/datasets/Metro+Interstate+Traffic+Volume
    
    For this demonstration, we'll create a realistic synthetic dataset with similar characteristics.
    
    Returns:
        DataFrame containing traffic volume data with weather and temporal features
    """
    print("\n" + "="*70)
    print("  COLLECTING TRAFFIC DATA")
    print("="*70)
    
    # Generate 2 years of hourly data (17,520 records)
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2022, 1, 1)
    date_range = pd.date_range(start=start_date, end=end_date, freq='H')
    
    n_samples = len(date_range)
    print(f"  Generating {n_samples:,} hourly traffic observations...")
    
    # Create base traffic patterns
    hours = date_range.hour.values
    day_of_week = date_range.dayofweek.values
    
    # Base traffic volume calculation with realistic patterns
    base_volume = 3000  # Base traffic
    
    # Rush hour patterns (morning 7-9am, evening 4-7pm)
    rush_hour_boost = np.where(
        ((hours >= 7) & (hours <= 9)) | ((hours >= 16) & (hours <= 19)),
        2500, 0
    )
    
    # Weekday vs weekend patterns
    weekday_boost = np.where(day_of_week < 5, 1500, -800)
    
    # Hourly variation
    hourly_variation = 1000 * np.sin((hours - 6) * np.pi / 12)
    
    # Random noise
    noise = np.random.normal(0, 500, n_samples)
    
    # Calculate traffic volume
    traffic_volume = base_volume + rush_hour_boost + weekday_boost + hourly_variation + noise
    traffic_volume = np.maximum(traffic_volume, 500)  # Minimum traffic
    
    # Create weather data
    # Temperature (Kelvin) - seasonal variation
    month = date_range.month.values
    base_temp = 280 + 15 * np.sin((month - 3) * np.pi / 6)  # Seasonal variation
    temp = base_temp + np.random.normal(0, 5, n_samples)
    
    # Rain (mm/hour) - some hours have rain
    rain_prob = 0.15
    rain = np.where(np.random.random(n_samples) < rain_prob,
                   np.random.exponential(2, n_samples), 0)
    
    # Snow (mm/hour) - only in winter months
    snow_prob = np.where((month >= 11) | (month <= 3), 0.08, 0)
    snow = np.where(np.random.random(n_samples) < snow_prob,
                   np.random.exponential(1.5, n_samples), 0)
    
    # Cloud coverage (0-100%)
    clouds = np.random.beta(2, 2, n_samples) * 100
    
    # Weather impact on traffic
    weather_impact = -rain * 150 - snow * 300  # Bad weather reduces traffic
    traffic_volume = np.maximum(traffic_volume + weather_impact, 200)
    
    # Weather descriptions
    weather_conditions = []
    for i in range(n_samples):
        if snow[i] > 0:
            weather_conditions.append('Snow')
        elif rain[i] > 0:
            weather_conditions.append('Rain')
        elif clouds[i] > 70:
            weather_conditions.append('Clouds')
        elif clouds[i] > 30:
            weather_conditions.append('Partly Cloudy')
        else:
            weather_conditions.append('Clear')
    
    # Holiday data (US holidays)
    holidays = [
        '2020-01-01', '2020-07-04', '2020-11-26', '2020-12-25',
        '2021-01-01', '2021-07-04', '2021-11-25', '2021-12-25',
    ]
    holiday_dates = pd.to_datetime(holidays)
    is_holiday = date_range.normalize().isin(holiday_dates).astype(int)
    
    # Reduce traffic on holidays
    holiday_impact = is_holiday * -800
    traffic_volume = np.maximum(traffic_volume + holiday_impact, 300)
    
    # Create DataFrame
    df = pd.DataFrame({
        'date_time': date_range,
        'traffic_volume': traffic_volume.astype(int),
        'temp': temp,
        'rain_1h': rain,
        'snow_1h': snow,
        'clouds_all': clouds,
        'weather_main': weather_conditions,
        'holiday': np.where(is_holiday == 1, 'Holiday', 'None')
    })
    
    print(f"  ✓ Generated realistic traffic dataset")
    print(f"  ✓ Date range: {df['date_time'].min()} to {df['date_time'].max()}")
    print(f"  ✓ Traffic volume range: {df['traffic_volume'].min()} to {df['traffic_volume'].max()}")
    
    return df


def fetch_weather_api_data(lat: float = 44.9778, lon: float = -93.2650,
                           start_date: str = "2020-01-01",
                           end_date: str = "2022-01-01") -> pd.DataFrame:
    """
    Fetch historical weather data from Open-Meteo API.
    
    Args:
        lat: Latitude (default: Minneapolis, MN)
        lon: Longitude (default: Minneapolis, MN)
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        
    Returns:
        DataFrame containing weather data from API
    """
    print("\n" + "="*70)
    print("  FETCHING WEATHER DATA FROM OPEN-METEO API")
    print("="*70)
    
    try:
        # Open-Meteo API endpoint
        url = "https://archive-api.open-meteo.com/v1/archive"
        
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": "temperature_2m,precipitation,windspeed_10m,cloudcover",
            "timezone": "America/Chicago"
        }
        
        print(f"  Requesting data for coordinates: ({lat}, {lon})")
        print(f"  Date range: {start_date} to {end_date}")
        
        response = requests.get(url, params=params, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            
            # Parse the response
            hourly_data = data['hourly']
            
            df = pd.DataFrame({
                'date_time': pd.to_datetime(hourly_data['time']),
                'temperature_2m': hourly_data['temperature_2m'],
                'precipitation': hourly_data['precipitation'],
                'windspeed_10m': hourly_data['windspeed_10m'],
                'cloudcover': hourly_data['cloudcover']
            })
            
            print(f"  ✓ Successfully fetched {len(df):,} hourly weather records")
            return df
        else:
            print(f"  ✗ API request failed with status code: {response.status_code}")
            print(f"  Creating synthetic weather data as fallback...")
            return create_synthetic_weather_data()
            
    except Exception as e:
        print(f"  ✗ Error fetching weather data: {e}")
        print(f"  Creating synthetic weather data as fallback...")
        return create_synthetic_weather_data()


def create_synthetic_weather_data() -> pd.DataFrame:
    """
    Create synthetic weather data as fallback when API is unavailable.
    
    Returns:
        DataFrame with synthetic weather data
    """
    date_range = pd.date_range(start='2020-01-01', end='2022-01-01', freq='H')
    n_samples = len(date_range)
    
    month = date_range.month.values
    
    # Temperature in Celsius
    base_temp = 10 + 15 * np.sin((month - 3) * np.pi / 6)
    temperature = base_temp + np.random.normal(0, 5, n_samples)
    
    # Precipitation
    precipitation = np.where(np.random.random(n_samples) < 0.15,
                            np.random.exponential(2, n_samples), 0)
    
    # Wind speed
    windspeed = np.random.gamma(2, 2, n_samples)
    
    # Cloud cover
    cloudcover = np.random.beta(2, 2, n_samples) * 100
    
    df = pd.DataFrame({
        'date_time': date_range,
        'temperature_2m': temperature,
        'precipitation': precipitation,
        'windspeed_10m': windspeed,
        'cloudcover': cloudcover
    })
    
    return df


def simulate_event_data() -> pd.DataFrame:
    """
    Simulate event data (concerts, sports, public events).
    In a real scenario, this would come from city event calendars.
    
    Returns:
        DataFrame containing event information
    """
    print("\n" + "="*70)
    print("  SIMULATING EVENT DATA")
    print("="*70)
    
    # Create random events throughout the period
    event_dates = pd.date_range(start='2020-01-01', end='2022-01-01', freq='W')
    
    # Randomly select some dates for events
    n_events = len(event_dates) // 3
    selected_dates = np.random.choice(event_dates, size=n_events, replace=False)
    
    event_types = ['Concert', 'Sports', 'Festival', 'Conference', 'Fair']
    event_sizes = ['Small', 'Medium', 'Large']
    
    events = []
    for date in selected_dates:
        events.append({
            'date': date,
            'event_type': np.random.choice(event_types),
            'event_size': np.random.choice(event_sizes),
            'expected_attendance': np.random.randint(1000, 50000)
        })
    
    df = pd.DataFrame(events)
    print(f"  ✓ Simulated {len(df)} events")
    
    return df


def integrate_datasets(traffic_df: pd.DataFrame, weather_df: pd.DataFrame, 
                       events_df: pd.DataFrame) -> pd.DataFrame:
    """
    Integrate traffic, weather, and event data into a single dataset.
    
    Args:
        traffic_df: Traffic volume data
        weather_df: Weather data
        events_df: Event data
        
    Returns:
        Integrated DataFrame
    """
    print("\n" + "="*70)
    print("  INTEGRATING DATASETS")
    print("="*70)
    
    # Merge traffic and weather data
    print("  Merging traffic and weather data...")
    integrated = traffic_df.merge(weather_df, on='date_time', how='left')
    
    # Add event data
    print("  Adding event information...")
    integrated['date'] = integrated['date_time'].dt.normalize()
    events_df['date'] = pd.to_datetime(events_df['date']).dt.normalize()
    
    integrated = integrated.merge(events_df, on='date', how='left')
    
    # Fill NaN values for non-event days
    integrated['event_type'] = integrated['event_type'].fillna('None')
    integrated['event_size'] = integrated['event_size'].fillna('None')
    integrated['expected_attendance'] = integrated['expected_attendance'].fillna(0)
    
    integrated = integrated.drop('date', axis=1)
    
    print(f"  ✓ Integration complete")
    print(f"  ✓ Final dataset shape: {integrated.shape}")
    print(f"  ✓ Columns: {list(integrated.columns)}")
    
    return integrated


def save_raw_data(df: pd.DataFrame, filename: str = 'traffic_data_raw.csv') -> None:
    """
    Save raw integrated data to file.
    
    Args:
        df: DataFrame to save
        filename: Output filename
    """
    output_dir = os.path.join('data', 'raw')
    os.makedirs(output_dir, exist_ok=True)
    
    filepath = os.path.join(output_dir, filename)
    df.to_csv(filepath, index=False)
    
    print(f"\n  ✓ Data saved to: {filepath}")


def main():
    """Main execution function."""
    print("\n" + "="*70)
    print("  TRAFFIC CONGESTION DATA COLLECTION")
    print("="*70)
    
    # Step 1: Download/generate traffic data
    traffic_data = download_traffic_data()
    
    # Step 2: Fetch weather data from API
    weather_data = fetch_weather_api_data()
    
    # Step 3: Simulate event data
    event_data = simulate_event_data()
    
    # Step 4: Integrate all datasets
    integrated_data = integrate_datasets(traffic_data, weather_data, event_data)
    
    # Step 5: Save raw data
    save_raw_data(integrated_data)
    
    # Print summary statistics
    print("\n" + "="*70)
    print("  DATA COLLECTION SUMMARY")
    print("="*70)
    print(f"  Total records: {len(integrated_data):,}")
    print(f"  Date range: {integrated_data['date_time'].min()} to {integrated_data['date_time'].max()}")
    print(f"  Missing values:\n{integrated_data.isnull().sum()}")
    print("\n" + "="*70)
    print("  ✓ DATA COLLECTION COMPLETE")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
