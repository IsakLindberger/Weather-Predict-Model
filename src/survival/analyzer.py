import pandas as pd
import numpy as np
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test


def prepare_survival_data(predictions_df, error_threshold):
    """
    Prepare data for survival analysis.
    
    Args:
        predictions_df: DataFrame with predictions and errors
        error_threshold: Threshold for defining failures
        
    Returns:
        pandas DataFrame with survival data
    """
    survival_df = predictions_df.copy()
    
    # Define event (1 = failure/large error, 0 = censored/good prediction)
    survival_df['event'] = (survival_df['abs_error'] > error_threshold).astype(int)
    
    # Calculate duration (time index - sequential order)
    survival_df['duration'] = range(1, len(survival_df) + 1)
    
    # Extract station_id if available
    if 'station_id' not in survival_df.columns:
        survival_df['station_id'] = 'STATION_001'
    
    return survival_df


def fit_kaplan_meier(survival_df):
    """
    Fit Kaplan-Meier survival curve.
    
    Args:
        survival_df: DataFrame with survival data
        
    Returns:
        KaplanMeierFitter object
    """
    kmf = KaplanMeierFitter()
    kmf.fit(
        durations=survival_df['duration'],
        event_observed=survival_df['event'],
        label='Model Performance'
    )
    
    return kmf


def calculate_survival_statistics(kmf, survival_df):
    """
    Calculate survival statistics.
    
    Args:
        kmf: Fitted KaplanMeierFitter
        survival_df: Original survival DataFrame
        
    Returns:
        dict: Survival statistics
    """
    # Get survival function values
    survival_function = kmf.survival_function_
    
    # Calculate median survival time (if exists)
    try:
        median_survival = kmf.median_survival_time_
    except:
        median_survival = None
    
    # Count events
    total_observations = len(survival_df)
    events_count = survival_df['event'].sum()
    censored_count = total_observations - events_count
    
    stats = {
        'total_observations': int(total_observations),
        'events_count': int(events_count),
        'censored_count': int(censored_count),
        'event_rate': float(events_count / total_observations * 100),
        'median_survival_time': float(median_survival) if median_survival else None,
        'final_survival_probability': float(survival_function.iloc[-1].values[0]) if len(survival_function) > 0 else None
    }
    
    return stats


def calculate_hazard_ratio(survival_df):
    """
    Calculate hazard ratios for different conditions.
    
    Args:
        survival_df: DataFrame with survival data
        
    Returns:
        DataFrame with hazard ratios
    """
    # Simple hazard calculation: events per time period
    hazard_df = survival_df.copy()
    
    # Calculate cumulative hazard (approximation)
    events_cumsum = hazard_df['event'].cumsum()
    at_risk = len(hazard_df) - hazard_df.index
    
    # Avoid division by zero
    hazard_df['hazard_ratio'] = np.where(
        at_risk > 0,
        events_cumsum / at_risk,
        0
    )
    
    return hazard_df


def create_survival_summary(survival_df, stats):
    """
    Create summary DataFrame for output.
    
    Args:
        survival_df: DataFrame with survival data and hazard ratios
        stats: Survival statistics dict
        
    Returns:
        DataFrame: Summary for saving
    """
    # Select relevant columns
    output_cols = ['timestamp', 'duration', 'event', 'hazard_ratio']
    
    # Add station_id if available
    if 'station_id' in survival_df.columns:
        output_cols.insert(0, 'station_id')
    
    summary_df = survival_df[output_cols].copy()
    
    return summary_df


def analyze_survival_by_group(survival_df, group_column=None):
    """
    Perform survival analysis by groups (if applicable).
    
    Args:
        survival_df: DataFrame with survival data
        group_column: Column to group by (optional)
        
    Returns:
        dict: Group-wise statistics
    """
    if group_column is None or group_column not in survival_df.columns:
        return None
    
    # Fit KM for each group
    groups = survival_df[group_column].unique()
    group_stats = {}
    
    for group in groups:
        group_data = survival_df[survival_df[group_column] == group]
        kmf = KaplanMeierFitter()
        kmf.fit(
            durations=group_data['duration'],
            event_observed=group_data['event']
        )
        group_stats[str(group)] = {
            'observations': len(group_data),
            'events': int(group_data['event'].sum())
        }
    
    return group_stats
