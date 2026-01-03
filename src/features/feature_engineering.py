import pandas as pd
import numpy as np


def add_rolling_features(df, window=24):
    """
    Add rolling window features.
    
    Args:
        df: pandas DataFrame with weather data
        window: Rolling window size in hours (default 24)
        
    Returns:
        pandas DataFrame: DataFrame with rolling features added
    """
    df_features = df.copy()
    
    # 24-hour rolling mean for temperature
    df_features['temp_rolling_mean_24h'] = df_features['temperature'].rolling(
        window=window, min_periods=1
    ).mean()
    
    # 24-hour rolling std for temperature (variability)
    df_features['temp_rolling_std_24h'] = df_features['temperature'].rolling(
        window=window, min_periods=1
    ).std()
    
    # Fill NaN values that might appear at the start
    df_features['temp_rolling_std_24h'] = df_features['temp_rolling_std_24h'].fillna(0)
    
    return df_features


def add_change_features(df):
    """
    Add rate of change features.
    
    Args:
        df: pandas DataFrame with weather data
        
    Returns:
        pandas DataFrame: DataFrame with change features added
    """
    df_features = df.copy()
    
    # Temperature change over last hour
    df_features['temp_change_1h'] = df_features['temperature'].diff()
    
    # Pressure change over last hour
    df_features['pressure_change_1h'] = df_features['pressure'].diff()
    
    # Humidity change over last hour
    df_features['humidity_change_1h'] = df_features['humidity'].diff()
    
    # Fill first NaN with 0 (no previous value)
    df_features['temp_change_1h'] = df_features['temp_change_1h'].fillna(0)
    df_features['pressure_change_1h'] = df_features['pressure_change_1h'].fillna(0)
    df_features['humidity_change_1h'] = df_features['humidity_change_1h'].fillna(0)
    
    return df_features


def add_time_features(df):
    """
    Add time-based features from timestamp.
    
    Args:
        df: pandas DataFrame with timestamp column
        
    Returns:
        pandas DataFrame: DataFrame with time features added
    """
    df_features = df.copy()
    
    # Extract hour of day
    df_features['hour'] = df_features['timestamp'].dt.hour
    
    # Extract day of week (0=Monday, 6=Sunday)
    df_features['day_of_week'] = df_features['timestamp'].dt.dayofweek
    
    # Is weekend flag
    df_features['is_weekend'] = (df_features['day_of_week'] >= 5).astype(int)
    
    return df_features


def create_features(df):
    """
    Apply all feature engineering steps.
    
    Args:
        df: pandas DataFrame with cleaned weather data
        
    Returns:
        tuple: (featured DataFrame, dict with feature stats)
    """
    initial_columns = len(df.columns)
    
    # Add all features
    df_featured = add_rolling_features(df, window=24)
    df_featured = add_change_features(df_featured)
    df_featured = add_time_features(df_featured)
    
    final_columns = len(df_featured.columns)
    features_added = final_columns - initial_columns
    
    # Get list of new feature names
    new_features = [col for col in df_featured.columns if col not in df.columns]
    
    stats = {
        'features_created': new_features,
        'total_features': final_columns,
        'new_feature_count': features_added
    }
    
    return df_featured, stats
