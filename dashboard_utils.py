"""
Dashboard Utility Functions for Traffic Prediction Dashboard

This module provides helper functions for the Streamlit dashboard including:
- Model loading and caching
- Data preprocessing
- Feature engineering
- Visualization utilities
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime, timedelta
import os


@st.cache_resource
def load_model(model_path):
    """Load a trained model from pickle file with caching"""
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


@st.cache_data
def load_data(file_path):
    """Load dataset with caching"""
    try:
        df = pd.read_csv(file_path)
        if 'date_time' in df.columns:
            df['date_time'] = pd.to_datetime(df['date_time'])
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None


def get_available_models():
    """Get list of available models in models directory"""
    models_dir = 'models'
    regression_models = {}
    classification_models = {}
    
    if os.path.exists(models_dir):
        for file in os.listdir(models_dir):
            if file.endswith('_reg.pkl'):
                name = file.replace('_reg.pkl', '').replace('_', ' ').title()
                regression_models[name] = os.path.join(models_dir, file)
            elif file.endswith('_clf.pkl'):
                name = file.replace('_clf.pkl', '').replace('_', ' ').title()
                classification_models[name] = os.path.join(models_dir, file)
    
    return regression_models, classification_models


def create_temporal_features(dt):
    """Create temporal features from datetime matching training data exactly"""
    features = {
        'hour': dt.hour,
        'day_of_week': dt.weekday(),
        'day_of_month': dt.day,  # ADDED - required by model
        'month': dt.month,
        'year': dt.year,  # ADDED - required by model
        'week_of_year': dt.isocalendar()[1],
        'is_weekend': 1 if dt.weekday() >= 5 else 0,
        'is_rush_hour': 1 if (7 <= dt.hour <= 9) or (16 <= dt.hour <= 19) else 0,
    }
    
    # Time of day
    if 0 <= dt.hour < 6:
        time_of_day = 'Night'
    elif 6 <= dt.hour < 12:
        time_of_day = 'Morning'
    elif 12 <= dt.hour < 18:
        time_of_day = 'Afternoon'
    else:
        time_of_day = 'Evening'
    
    # Season
    month = dt.month
    if month in [12, 1, 2]:
        season = 'Winter'
    elif month in [3, 4, 5]:
        season = 'Spring'
    elif month in [6, 7, 8]:
        season = 'Summer'
    else:
        season = 'Fall'
    
    return features, time_of_day, season


def create_weather_features(temp_celsius, precipitation, weather_main):
    """Create weather-related features matching training data exactly"""
    # Approximate rain/snow split based on precip and weather
    rain_1h = precipitation if weather_main in ['Rain', 'Partly Cloudy'] else 0
    snow_1h = precipitation if weather_main == 'Snow' else 0
    
    # Cloud coverage approximation based on weather condition
    clouds_map = {'Clouds': 75, 'Partly Cloudy': 50, 'Rain': 90, 'Snow': 90, 'Clear': 0}
    clouds_all = clouds_map.get(weather_main, 50)
    cloudcover = clouds_all  # Same as clouds_all
    
    features = {
        'temp': temp_celsius,  # Original temp column
        'rain_1h': rain_1h,  # ADDED - required by model
        'snow_1h': snow_1h,  # ADDED - required by model   
        'clouds_all': clouds_all,  # ADDED - required by model
        'temperature_2m': temp_celsius,  # ADDED - required by model (same as temp)
        'precipitation': precipitation,  # ADDED - required by model
        'windspeed_10m': 5.0,  # ADDED - required by model (default approximation)
        'cloudcover': cloudcover,  # ADDED - required by model
        'temp_celsius': temp_celsius,  # Keep original
        'total_precipitation': precipitation,  # Keep original
        'bad_weather': 1 if precipitation > 0.5 else 0,
    }
    return features


def create_prediction_input(dt, temp, precip, weather_main, holiday, event_type, 
                            event_size, traffic_prev_hour=3500, traffic_prev_day=3500):
    """
    Create feature vector for prediction matching training data exactly
    
    Parameters:
    -----------
    dt : datetime
        Date and time for prediction
    temp : float
        Temperature in Celsius
    precip : float
        Precipitation in mm
    weather_main : str
        Weather condition
    holiday : bool
        Is it a holiday
    event_type : str
        Type of event (Conference, Fair, Festival, Sports, or None)
    event_size : str
        Size of event (Small, Medium, or None)
    traffic_prev_hour : float
        Traffic volume from previous hour (default: average)
    traffic_prev_day : float
        Traffic volume from same hour previous day (default: average)
    """
    
    # Temporal features
    temporal, time_of_day, season = create_temporal_features(dt)
    
    # Weather features
    weather = create_weather_features(temp, precip, weather_main)
    
    # CRITICAL: Features must be in EXACT order as training data
    # Training column order (excluding date_time, traffic_volume, congestion_level*):
    # temp,rain_1h,snow_1h,clouds_all,temperature_2m,precipitation,windspeed_10m,cloudcover,
    # expected_attendance,hour,day_of_week,day_of_month,month,year,week_of_year,is_weekend,
    # is_holiday,is_rush_hour,temp_celsius,total_precipitation,bad_weather,traffic_prev_hour,
    # traffic_prev_day,traffic_prev_week,traffic_rolling_mean_6h,traffic_rolling_std_6h,
    # traffic_rolling_mean_24h,weather_main_Clouds,weather_main_Partly Cloudy,weather_main_Rain,
    # weather_main_Snow,time_of_day_Morning,time_of_day_Afternoon,time_of_day_Evening,
    # season_Spring,season_Summer,season_Fall,event_type_Conference,event_type_Fair,
    # event_type_Festival,event_type_Sports,event_size_Medium,event_size_Small
    
    # Expected attendance
    attendance_map = {'None': 0, 'Small': 2000, 'Medium': 10000}
    expected_attendance = attendance_map.get(event_size, 0)
    
    # Build features in EXACT training order
    features = {
        # Weather features (from create_weather_features)
        'temp': weather['temp'],
        'rain_1h': weather['rain_1h'],
        'snow_1h': weather['snow_1h'],
        'clouds_all': weather['clouds_all'],
        'temperature_2m': weather['temperature_2m'],
        'precipitation': weather['precipitation'],
        'windspeed_10m': weather['windspeed_10m'],
        'cloudcover': weather['cloudcover'],
        
        # Event features
        'expected_attendance': expected_attendance,
        
        # Temporal features (from create_temporal_features)
        'hour': temporal['hour'],
        'day_of_week': temporal['day_of_week'],
        'day_of_month': temporal['day_of_month'],
        'month': temporal['month'],
        'year': temporal['year'],
        'week_of_year': temporal['week_of_year'],
        'is_weekend': temporal['is_weekend'],
        'is_holiday': 1 if holiday else 0,
        'is_rush_hour': temporal['is_rush_hour'],
        
        # Additional weather features
        'temp_celsius': weather['temp_celsius'],
        'total_precipitation': weather['total_precipitation'],
        'bad_weather': weather['bad_weather'],
        
        # Traffic history features
        'traffic_prev_hour': traffic_prev_hour,
        'traffic_prev_day': traffic_prev_day,
        'traffic_prev_week': traffic_prev_day,  # Approximation
        'traffic_rolling_mean_6h': traffic_prev_hour,
        'traffic_rolling_std_6h': traffic_prev_hour * 0.2,
        'traffic_rolling_mean_24h': traffic_prev_day,
        
        # One-hot: weather_main (Clouds, Partly Cloudy, Rain, Snow)
        'weather_main_Clouds': 1 if weather_main == 'Clouds' else 0,
        'weather_main_Partly Cloudy': 1 if weather_main == 'Partly Cloudy' else 0,
        'weather_main_Rain': 1 if weather_main == 'Rain' else 0,
        'weather_main_Snow': 1 if weather_main == 'Snow' else 0,
        
        # One-hot: time_of_day (Morning, Afternoon, Evening)
        'time_of_day_Morning': 1 if time_of_day == 'Morning' else 0,
        'time_of_day_Afternoon': 1 if time_of_day == 'Afternoon' else 0,
        'time_of_day_Evening': 1 if time_of_day == 'Evening' else 0,
        
        # One-hot: season (Spring, Summer, Fall)
        'season_Spring': 1 if season == 'Spring' else 0,
        'season_Summer': 1 if season == 'Summer' else 0,
        'season_Fall': 1 if season == 'Fall' else 0,
        
        # One-hot: event_type (Conference, Fair, Festival, Sports)
        'event_type_Conference': 1 if event_type == 'Conference' else 0,
        'event_type_Fair': 1 if event_type == 'Fair' else 0,
        'event_type_Festival': 1 if event_type == 'Festival' else 0,
        'event_type_Sports': 1 if event_type == 'Sports' else 0,
        
        # One-hot: event_size (Medium, Small)
        'event_size_Medium': 1 if event_size == 'Medium' else 0,
        'event_size_Small': 1 if event_size == 'Small' else 0,
    }
    
    return features


def classify_congestion(traffic_volume):
    """Classify traffic volume into congestion levels"""
    if traffic_volume < 2500:
        return 'Low', '🟢'
    elif traffic_volume < 4500:
        return 'Medium', '🟡'
    else:
        return 'High', '🔴'


def get_color_scheme():
    """Return color scheme for consistent styling"""
    return {
        'primary': '#1f77b4',
        'secondary': '#ff7f0e',
        'success': '#2ca02c',
        'warning': '#ffbb00',
        'danger': '#d62728',
        'low': '#2ca02c',
        'medium': '#ffbb00',
        'high': '#d62728',
    }


def format_metric(value, metric_type='volume'):
    """Format metrics for display"""
    if metric_type == 'volume':
        return f"{value:,.0f} vehicles/hour"
    elif metric_type == 'percentage':
        return f"{value:.1f}%"
    elif metric_type == 'decimal':
        return f"{value:.3f}"
    else:
        return str(value)


def get_recommendations(congestion_level, weather_main, is_rush_hour):
    """Generate recommendations based on predictions"""
    recommendations = []
    
    if congestion_level == 'High':
        recommendations.append("🚦 **High congestion expected** - Consider alternative routes")
        recommendations.append("🚍 Use public transportation to avoid delays")
        if is_rush_hour:
            recommendations.append("⏰ Peak rush hour - expect 15-20 minute delays")
    elif congestion_level == 'Medium':
        recommendations.append("⚠️ **Moderate traffic** - Some delays possible")
        recommendations.append("🚗 Allow extra 5-10 minutes for travel")
    else:
        recommendations.append("✅ **Low congestion** - Clear traffic conditions")
    
    if weather_main in ['Rain', 'Snow']:
        recommendations.append(f"🌧️ **{weather_main} conditions** - Reduce speed, increase following distance")
        recommendations.append("⚠️ Weather may cause additional delays")
    
    return recommendations


def calculate_model_metrics(y_true, y_pred, task='regression'):
    """Calculate performance metrics"""
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    if task == 'regression':
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        return {
            'RMSE': rmse,
            'MAE': mae,
            'R² Score': r2,
            'MAPE (%)': mape
        }
    else:
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        return {
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-Score': f1
        }
