import pandas as pd
import numpy as np


def remove_outliers(df, column, method='iqr'):
    """
    Remove outliers from a column using IQR method.
    
    Args:
        df: pandas DataFrame
        column: Column name to clean
        method: Method to use ('iqr' for Interquartile Range)
        
    Returns:
        pandas DataFrame: DataFrame with outliers removed
    """
    if column not in df.columns:
        return df
    
    # Calculate Q1, Q3, and IQR
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    # Define outlier bounds
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Filter out outliers
    df_clean = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    return df_clean


def fill_missing_values(df, method='forward'):
    """
    Fill missing values in DataFrame.
    
    Args:
        df: pandas DataFrame
        method: Method to use ('forward' for forward fill, 'mean' for mean)
        
    Returns:
        pandas DataFrame: DataFrame with filled values
    """
    df_filled = df.copy()
    
    if method == 'forward':
        # Forward fill (use last valid observation)
        df_filled = df_filled.ffill()
        # Backward fill for any remaining NaN at start
        df_filled = df_filled.bfill()
    elif method == 'mean':
        # Fill with column mean
        numeric_cols = df_filled.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df_filled[col].fillna(df_filled[col].mean(), inplace=True)
    
    return df_filled


def clean_data(df):
    """
    Apply all cleaning steps to weather data.
    
    Args:
        df: pandas DataFrame with raw weather data
        
    Returns:
        tuple: (cleaned DataFrame, dict with cleaning stats)
    """
    initial_rows = len(df)
    
    # Remove outliers from numerical columns
    df_clean = remove_outliers(df, 'temperature')
    df_clean = remove_outliers(df_clean, 'humidity')
    df_clean = remove_outliers(df_clean, 'pressure')
    
    rows_after_outliers = len(df_clean)
    rows_removed = initial_rows - rows_after_outliers
    
    # Count missing values before filling
    missing_before = df_clean.isnull().sum().sum()
    
    # Fill missing values
    df_clean = fill_missing_values(df_clean, method='forward')
    
    # Count missing values after filling
    missing_after = df_clean.isnull().sum().sum()
    missing_filled = missing_before - missing_after
    
    # Create stats dict
    stats = {
        'rows_removed': rows_removed,
        'missing_values_filled': int(missing_filled),
        'final_row_count': len(df_clean)
    }
    
    return df_clean, stats
