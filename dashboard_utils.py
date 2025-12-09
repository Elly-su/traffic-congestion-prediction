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
    
    # Combine all features - matching exact training data structure
    features = {
        **temporal,
        **weather,
        'is_holiday': 1 if holiday else 0,
        'traffic_prev_hour': traffic_prev_hour,
        'traffic_prev_day': traffic_prev_day,
        'traffic_prev_week': traffic_prev_day,  # Approximation
        'traffic_rolling_mean_6h': traffic_prev_hour,
        'traffic_rolling_std_6h': traffic_prev_hour * 0.2,  # Approximation
        'traffic_rolling_mean_24h': traffic_prev_day,
    }
    
    # One-hot encode weather - ONLY categories seen during training
    # Training data has: Clouds, Partly Cloudy, Rain, Snow
    weather_categories = ['Clouds', 'Partly Cloudy', 'Rain', 'Snow']
    for cat in weather_categories:
        features[f'weather_main_{cat}'] = 1 if weather_main == cat else 0
    
    # One-hot encode time of day - ONLY categories seen during training
    # Training data has: Morning, Afternoon, Evening (no Night)
    time_categories = ['Morning', 'Afternoon', 'Evening']
    for cat in time_categories:
        features[f'time_of_day_{cat}'] = 1 if time_of_day == cat else 0
    
    # One-hot encode season - ONLY categories seen during training
    # Training data has: Spring, Summer, Fall (no Winter in training period)
    season_categories = ['Spring', 'Summer', 'Fall']
    for cat in season_categories:
        features[f'season_{cat}'] = 1 if season == cat else 0
    
    # One-hot encode event types - ONLY categories seen during training
    # Training data has: Conference, Fair, Festival, Sports (no Concert, no Large size)
    event_type_categories = ['Conference', 'Fair', 'Festival', 'Sports']
    for cat in event_type_categories:
        features[f'event_type_{cat}'] = 1 if event_type == cat else 0
    
    # One-hot encode event sizes - ONLY categories seen during training
    # Training data has: Medium, Small (no Large, no None as separate category)
    event_size_categories = ['Medium', 'Small']
    for cat in event_size_categories:
        features[f'event_size_{cat}'] = 1 if event_size == cat else 0
    
    # Expected attendance based on event size
    attendance_map = {'None': 0, 'Small': 2000, 'Medium': 10000}
    features['expected_attendance'] = attendance_map.get(event_size, 0)
    
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
