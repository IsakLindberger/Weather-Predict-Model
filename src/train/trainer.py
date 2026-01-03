import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from pathlib import Path


def prepare_data(df, target_column='temperature', test_size=0.2, random_state=42):
    """
    Prepare data for training by splitting features and target.
    
    Args:
        df: pandas DataFrame with features
        target_column: Name of target variable to predict
        test_size: Fraction of data for validation
        random_state: Random seed for reproducibility
        
    Returns:
        tuple: (X_train, X_val, y_train, y_val, feature_columns)
    """
    # Drop columns that shouldn't be features
    exclude_cols = ['timestamp', 'station_id', target_column]
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Prepare features and target
    X = df[feature_cols]
    y = df[target_column]
    
    # Split into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    return X_train, X_val, y_train, y_val, feature_cols


def train_model(X_train, y_train, model_type='RandomForest', **hyperparameters):
    """
    Train a regression model.
    
    Args:
        X_train: Training features
        y_train: Training target
        model_type: Type of model to train
        **hyperparameters: Model hyperparameters
        
    Returns:
        trained model
    """
    if model_type == 'RandomForest':
        # Set default hyperparameters if not provided
        default_params = {
            'n_estimators': 100,
            'max_depth': 10,
            'random_state': 42,
            'n_jobs': -1
        }
        # Update with provided hyperparameters
        default_params.update(hyperparameters)
        
        model = RandomForestRegressor(**default_params)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Train the model
    model.fit(X_train, y_train)
    
    return model


def evaluate_model(model, X_val, y_val):
    """
    Evaluate model performance on validation set.
    
    Args:
        model: Trained model
        X_val: Validation features
        y_val: Validation target
        
    Returns:
        dict: Evaluation metrics
    """
    # Make predictions
    y_pred = model.predict(X_val)
    
    # Calculate metrics
    mae = mean_absolute_error(y_val, y_pred)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    r2 = r2_score(y_val, y_pred)
    
    metrics = {
        'mae': float(mae),
        'rmse': float(rmse),
        'r2_score': float(r2)
    }
    
    return metrics


def save_model(model, output_path):
    """
    Save trained model to disk.
    
    Args:
        model: Trained model
        output_path: Path to save model file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save model using joblib
    joblib.dump(model, output_path)
