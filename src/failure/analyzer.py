import pandas as pd
import numpy as np


def identify_failures(predictions_df, threshold_percentile=90):
    """
    Identify prediction failures based on error threshold.
    
    Args:
        predictions_df: DataFrame with predictions and errors
        threshold_percentile: Percentile for defining failures
        
    Returns:
        tuple: (failures DataFrame, threshold value)
    """
    # Calculate error threshold
    error_threshold = np.percentile(predictions_df['abs_error'], threshold_percentile)
    
    # Identify failures (predictions with large errors)
    failures = predictions_df[predictions_df['abs_error'] > error_threshold].copy()
    
    return failures, error_threshold


def categorize_failures(failures_df):
    """
    Categorize failures into different types.
    
    Args:
        failures_df: DataFrame with failure cases
        
    Returns:
        DataFrame with failure categories added
    """
    failures = failures_df.copy()
    
    # Categorize based on error characteristics
    def categorize_error(row):
        error = row['error']
        abs_error = row['abs_error']
        
        if abs_error > 1.5:
            return 'extreme_error'
        elif error > 0:
            return 'overestimation'
        elif error < 0:
            return 'underestimation'
        else:
            return 'unknown'
    
    failures['failure_type'] = failures.apply(categorize_error, axis=1)
    
    return failures


def analyze_failure_patterns(failures_df):
    """
    Analyze patterns in failures.
    
    Args:
        failures_df: DataFrame with categorized failures
        
    Returns:
        dict: Failure pattern statistics
    """
    if len(failures_df) == 0:
        return {
            'total_failures': 0,
            'failure_types': {},
            'avg_failure_magnitude': 0.0
        }
    
    # Count failure types
    failure_types = failures_df['failure_type'].value_counts().to_dict()
    
    # Calculate statistics
    patterns = {
        'total_failures': len(failures_df),
        'failure_types': failure_types,
        'avg_failure_magnitude': float(failures_df['abs_error'].mean()),
        'max_failure_magnitude': float(failures_df['abs_error'].max()),
        'failure_timestamps': failures_df['timestamp'].tolist()
    }
    
    return patterns


def add_failure_context(failures_df, original_df):
    """
    Add contextual information to failures.
    
    Args:
        failures_df: DataFrame with failures
        original_df: Original DataFrame with all features
        
    Returns:
        DataFrame with context added
    """
    failures_enriched = failures_df.copy()
    
    # Extract features that might contribute to failures
    feature_cols = ['humidity', 'pressure', 'temp_change_1h', 'pressure_change_1h']
    
    # Merge with original data to get features
    failures_enriched = failures_enriched.merge(
        original_df[['timestamp'] + feature_cols],
        on='timestamp',
        how='left'
    )
    
    # Create contributing features summary
    def get_contributing_features(row):
        contributors = []
        
        # Check for extreme values
        if 'humidity' in row and row['humidity'] > 90:
            contributors.append('high_humidity')
        if 'pressure_change_1h' in row and abs(row['pressure_change_1h']) > 2:
            contributors.append('rapid_pressure_change')
        if 'temp_change_1h' in row and abs(row['temp_change_1h']) > 2:
            contributors.append('rapid_temp_change')
        
        return ', '.join(contributors) if contributors else 'normal_conditions'
    
    failures_enriched['contributing_features'] = failures_enriched.apply(
        get_contributing_features, axis=1
    )
    
    return failures_enriched


def create_failure_summary(failures_df, threshold, patterns):
    """
    Create a summary of failure analysis.
    
    Args:
        failures_df: DataFrame with analyzed failures
        threshold: Error threshold used
        patterns: Failure patterns dict
        
    Returns:
        dict: Summary statistics
    """
    summary = {
        'error_threshold': float(threshold),
        'total_failures': patterns['total_failures'],
        'failure_rate': float(patterns['total_failures'] / len(failures_df) * 100) if len(failures_df) > 0 else 0.0,
        'failure_types': patterns['failure_types'],
        'avg_failure_magnitude': patterns['avg_failure_magnitude'],
        'max_failure_magnitude': patterns.get('max_failure_magnitude', 0.0)
    }
    
    return summary
