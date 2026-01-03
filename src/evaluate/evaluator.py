import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error


def load_model(model_path):
    """
    Load a trained model from disk.
    
    Args:
        model_path: Path to saved model file
        
    Returns:
        Loaded model
    """
    model_path = Path(model_path)
    model = joblib.load(model_path)
    return model


def prepare_evaluation_data(df, target_column='temperature'):
    """
    Prepare data for evaluation.
    
    Args:
        df: pandas DataFrame with features
        target_column: Name of target variable
        
    Returns:
        tuple: (X, y, feature_columns)
    """
    # Drop columns that shouldn't be features
    exclude_cols = ['timestamp', 'station_id', target_column]
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols]
    y = df[target_column]
    
    return X, y, feature_cols


def evaluate_predictions(y_true, y_pred):
    """
    Calculate evaluation metrics.
    
    Args:
        y_true: True target values
        y_pred: Predicted target values
        
    Returns:
        dict: Evaluation metrics
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    # Calculate MAPE, handle zero values
    try:
        mape = mean_absolute_percentage_error(y_true, y_pred)
    except:
        mape = None
    
    metrics = {
        'mae': float(mae),
        'rmse': float(rmse),
        'r2_score': float(r2),
        'mape': float(mape) if mape is not None else None
    }
    
    return metrics


def create_predictions_dataframe(df, y_true, y_pred):
    """
    Create DataFrame with predictions and errors.
    
    Args:
        df: Original DataFrame with timestamp
        y_true: True target values
        y_pred: Predicted target values
        
    Returns:
        pandas DataFrame with predictions
    """
    predictions_df = pd.DataFrame({
        'timestamp': df['timestamp'].values,
        'actual': y_true.values,
        'predicted': y_pred,
        'error': y_true.values - y_pred,
        'abs_error': np.abs(y_true.values - y_pred)
    })
    
    return predictions_df


def analyze_errors(predictions_df, threshold_percentile=90):
    """
    Analyze prediction errors to identify failures.
    
    Args:
        predictions_df: DataFrame with predictions and errors
        threshold_percentile: Percentile for defining large errors
        
    Returns:
        dict: Error analysis statistics
    """
    # Calculate error threshold (90th percentile)
    error_threshold = np.percentile(predictions_df['abs_error'], threshold_percentile)
    
    # Identify large errors (potential failures)
    large_errors = predictions_df[predictions_df['abs_error'] > error_threshold]
    
    analysis = {
        'error_threshold': float(error_threshold),
        'total_predictions': len(predictions_df),
        'large_error_count': len(large_errors),
        'large_error_percentage': float(len(large_errors) / len(predictions_df) * 100),
        'mean_error': float(predictions_df['error'].mean()),
        'std_error': float(predictions_df['error'].std())
    }
    
    return analysis
