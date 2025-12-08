"""
Data Preprocessing Module for Traffic Congestion Prediction
Handles data cleaning, feature engineering, and preparation for modeling.
"""

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from typing import Tuple
import pickle


class TrafficDataPreprocessor:
    """Class to handle all preprocessing operations for traffic data."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def load_raw_data(self, filepath: str = 'data/raw/traffic_data_raw.csv') -> pd.DataFrame:
        """Load raw data from CSV."""
        print("\n" + "="*70)
        print("  LOADING RAW DATA")
        print("="*70)
        
        df = pd.read_csv(filepath)
        df['date_time'] = pd.to_datetime(df['date_time'])
        
        print(f"  ✓ Loaded {len(df):,} records")
        print(f"  ✓ Columns: {list(df.columns)}")
        
        return df
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values using appropriate strategies.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with missing values handled
        """
        print("\n" + "="*70)
        print("  HANDLING MISSING VALUES")
        print("="*70)
        
        print(f"\n  Missing values before:")
        missing_before = df.isnull().sum()
        print(missing_before[missing_before > 0])
        
        # Strategy 1: Forward fill for weather data (carries last observation forward)
        weather_cols = ['temp', 'rain_1h', 'snow_1h', 'clouds_all', 
                       'temperature_2m', 'precipitation', 'windspeed_10m', 'cloudcover']
        
        for col in weather_cols:
            if col in df.columns:
                df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
        
        # Strategy 2: Fill categorical with mode
        categorical_cols = ['weather_main', 'holiday', 'event_type', 'event_size']
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')
        
        # Strategy 3: Fill numeric event attendance with 0
        if 'expected_attendance' in df.columns:
            df['expected_attendance'] = df['expected_attendance'].fillna(0)
        
        print(f"\n  Missing values after:")
        missing_after = df.isnull().sum()
        print(missing_after[missing_after > 0] if missing_after.sum() > 0 else "  ✓ No missing values")
        
        return df
    
    def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create new features from existing data.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with engineered features
        """
        print("\n" + "="*70)
        print("  FEATURE ENGINEERING")
        print("="*70)
        
        # Temporal features
        print("  Creating temporal features...")
        df['hour'] = df['date_time'].dt.hour
        df['day_of_week'] = df['date_time'].dt.dayofweek  # 0=Monday, 6=Sunday
        df['day_of_month'] = df['date_time'].dt.day
        df['month'] = df['date_time'].dt.month
        df['year'] = df['date_time'].dt.year
        df['week_of_year'] = df['date_time'].dt.isocalendar().week
        
        # Boolean features
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_holiday'] = (df['holiday'] != 'None').astype(int)
        df['is_rush_hour'] = (
            ((df['hour'] >= 7) & (df['hour'] <= 9)) |
            ((df['hour'] >= 16) & (df['hour'] <= 19))
        ).astype(int)
        
        # Time of day categories
        df['time_of_day'] = pd.cut(df['hour'], 
                                    bins=[-1, 6, 12, 18, 24],
                                    labels=['Night', 'Morning', 'Afternoon', 'Evening'])
        
        # Season
        df['season'] = pd.cut(df['month'],
                             bins=[0, 3, 6, 9, 12],
                             labels=['Winter', 'Spring', 'Summer', 'Fall'])
        
        # Weather features
        print("  Creating weather features...")
        
        # Temperature in Celsius (convert from Kelvin if needed)
        if 'temp' in df.columns:
            df['temp_celsius'] = df['temp'] - 273.15  # Assuming Kelvin
        
        # Combine precipitation
        df['total_precipitation'] = df['rain_1h'] + df['snow_1h']
        
        # Weather severity
        df['bad_weather'] = (
            (df['rain_1h'] > 2) | 
            (df['snow_1h'] > 1) |
            (df['total_precipitation'] > 3)
        ).astype(int)
        
        # Lagged features (previous hour traffic)
        print("  Creating lagged features...")
        df = df.sort_values('date_time')
        df['traffic_prev_hour'] = df['traffic_volume'].shift(1)
        df['traffic_prev_day'] = df['traffic_volume'].shift(24)
        df['traffic_prev_week'] = df['traffic_volume'].shift(24*7)
        
        # Rolling statistics
        df['traffic_rolling_mean_6h'] = df['traffic_volume'].rolling(window=6, min_periods=1).mean()
        df['traffic_rolling_std_6h'] = df['traffic_volume'].rolling(window=6, min_periods=1).std()
        df['traffic_rolling_mean_24h'] = df['traffic_volume'].rolling(window=24, min_periods=1).mean()
        
        # Fill NaN from rolling/shift operations
        df = df.fillna(method='bfill')
        
        print(f"  ✓ Created {len(df.columns)} total features")
        
        return df
    
    def create_congestion_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create congestion level categories for classification.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with congestion labels
        """
        print("\n" + "="*70)
        print("  CREATING CONGESTION LABELS")
        print("="*70)
        
        # Define thresholds based on quantiles
        q33 = df['traffic_volume'].quantile(0.33)
        q67 = df['traffic_volume'].quantile(0.67)
        
        print(f"  Low congestion threshold: < {q33:.0f}")
        print(f"  Medium congestion: {q33:.0f} - {q67:.0f}")
        print(f"  High congestion: > {q67:.0f}")
        
        df['congestion_level'] = pd.cut(
            df['traffic_volume'],
            bins=[-np.inf, q33, q67, np.inf],
            labels=['Low', 'Medium', 'High']
        )
        
        # Distribution
        print(f"\n  Congestion level distribution:")
        print(df['congestion_level'].value_counts().sort_index())
        
        return df
    
    def encode_categorical(self, df: pd.DataFrame, 
                          categorical_cols: list = None) -> pd.DataFrame:
        """
        Encode categorical variables.
        
        Args:
            df: Input DataFrame
            categorical_cols: List of categorical columns to encode
            
        Returns:
            DataFrame with encoded categorical variables
        """
        print("\n" + "="*70)
        print("  ENCODING CATEGORICAL VARIABLES")
        print("="*70)
        
        if categorical_cols is None:
            categorical_cols = ['weather_main', 'time_of_day', 'season', 'event_type', 'event_size']
        
        # Drop original holiday column (we created is_holiday boolean)
        if 'holiday' in df.columns:
            df = df.drop('holiday', axis=1)
        
        # One-hot encoding
        print(f"  One-hot encoding: {categorical_cols}")
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
        
        # Label encoding for target (if classification)
        if 'congestion_level' in df.columns:
            print(f"  Label encoding: congestion_level")
            le = LabelEncoder()
            df['congestion_level_encoded'] = le.fit_transform(df['congestion_level'])
            self.label_encoders['congestion_level'] = le
        
        print(f"  ✓ Total columns after encoding: {len(df.columns)}")
        
        return df
    
    def normalize_features(self, df: pd.DataFrame, 
                          numerical_cols: list = None) -> pd.DataFrame:
        """
        Normalize numerical features using StandardScaler.
        
        Args:
            df: Input DataFrame
            numerical_cols: List of numerical columns to normalize
            
        Returns:
            DataFrame with normalized features
        """
        print("\n" + "="*70)
        print("  NORMALIZING NUMERICAL FEATURES")
        print("="*70)
        
        if numerical_cols is None:
            # Auto-detect numerical columns (excluding target and identifiers)
            exclude_cols = ['date_time', 'traffic_volume', 'congestion_level', 
                          'congestion_level_encoded']
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            numerical_cols = [col for col in numerical_cols if col not in exclude_cols]
        
        print(f"  Normalizing {len(numerical_cols)} numerical features...")
        
        df[numerical_cols] = self.scaler.fit_transform(df[numerical_cols])
        
        print(f"  ✓ Normalization complete")
        
        return df
    
    def train_test_split_temporal(self, df: pd.DataFrame, 
                                 test_size: float = 0.2,
                                 val_size: float = 0.1) -> Tuple:
        """
        Split data temporally to prevent data leakage.
        
        Args:
            df: Input DataFrame
            test_size: Proportion for test set
            val_size: Proportion for validation set
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        print("\n" + "="*70)
        print("  SPLITTING DATA (TEMPORAL)")
        print("="*70)
        
        df = df.sort_values('date_time')
        n = len(df)
        
        train_end = int(n * (1 - test_size - val_size))
        val_end = int(n * (1 - test_size))
        
        train_df = df.iloc[:train_end].copy()
        val_df = df.iloc[train_end:val_end].copy()
        test_df = df.iloc[val_end:].copy()
        
        print(f"  Train set: {len(train_df):,} records ({len(train_df)/n*100:.1f}%)")
        print(f"    Date range: {train_df['date_time'].min()} to {train_df['date_time'].max()}")
        
        print(f"  Validation set: {len(val_df):,} records ({len(val_df)/n*100:.1f}%)")
        print(f"    Date range: {val_df['date_time'].min()} to {val_df['date_time'].max()}")
        
        print(f"  Test set: {len(test_df):,} records ({len(test_df)/n*100:.1f}%)")
        print(f"    Date range: {test_df['date_time'].min()} to {test_df['date_time'].max()}")
        
        return train_df, val_df, test_df
    
    def save_processed_data(self, train_df: pd.DataFrame, 
                           val_df: pd.DataFrame,
                           test_df: pd.DataFrame,
                           preprocessor: 'TrafficDataPreprocessor') -> None:
        """Save processed datasets and preprocessor."""
        print("\n" + "="*70)
        print("  SAVING PROCESSED DATA")
        print("="*70)
        
        output_dir = 'data/processed'
        os.makedirs(output_dir, exist_ok=True)
        
        # Save datasets
        train_df.to_csv(f'{output_dir}/train_data.csv', index=False)
        val_df.to_csv(f'{output_dir}/val_data.csv', index=False)
        test_df.to_csv(f'{output_dir}/test_data.csv', index=False)
        
        # Save preprocessor
        with open(f'{output_dir}/preprocessor.pkl', 'wb') as f:
            pickle.dump(preprocessor, f)
        
        print(f"  ✓ Saved train_data.csv")
        print(f"  ✓ Saved val_data.csv")
        print(f"  ✓ Saved test_data.csv")
        print(f"  ✓ Saved preprocessor.pkl")


def main():
    """Main preprocessing pipeline."""
    print("\n" + "="*70)
    print("  TRAFFIC DATA PREPROCESSING PIPELINE")
    print("="*70)
    
    # Initialize preprocessor
    preprocessor = TrafficDataPreprocessor()
    
    # Load raw data
    df = preprocessor.load_raw_data()
    
    # Handle missing values
    df = preprocessor.handle_missing_values(df)
    
    # Feature engineering
    df = preprocessor.feature_engineering(df)
    
    # Create congestion labels
    df = preprocessor.create_congestion_labels(df)
    
    # Encode categorical variables
    df = preprocessor.encode_categorical(df)
    
    # Split data temporally
    train_df, val_df, test_df = preprocessor.train_test_split_temporal(df)
    
    # Normalize features (fit on train only)
    train_df = preprocessor.normalize_features(train_df)
    
    # Apply same normalization to val and test
    numerical_cols = train_df.select_dtypes(include=[np.number]).columns
    exclude_cols = ['traffic_volume', 'congestion_level_encoded']
    numerical_cols = [col for col in numerical_cols if col not in exclude_cols]
    
    val_df[numerical_cols] = preprocessor.scaler.transform(val_df[numerical_cols])
    test_df[numerical_cols] = preprocessor.scaler.transform(test_df[numerical_cols])
    
    # Save processed data
    preprocessor.save_processed_data(train_df, val_df, test_df, preprocessor)
    
    print("\n" + "="*70)
    print("  ✓ PREPROCESSING COMPLETE")
    print("="*70)
    print(f"\n  Final feature count: {len(train_df.columns)}")
    print(f"  Ready for modeling!\n")


if __name__ == "__main__":
    main()
