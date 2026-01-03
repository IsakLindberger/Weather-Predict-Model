import pandas as pd
from pathlib import Path
from datetime import datetime
import sys
sys.path.append(str(Path(__file__).parent.parent))

from .trainer import prepare_data, train_model, evaluate_model, save_model
from ingest.logger import save_metadata, log_info


def train_weather_model(input_path, output_dir, target='temperature', model_type='RandomForest'):
    """
    Train a weather prediction model.
    
    Args:
        input_path: Path to featured parquet file
        output_dir: Directory to save model
        target: Target variable to predict
        model_type: Type of model to train
        
    Returns:
        tuple: (success: bool, model_path: str or None)
    """
    log_info(f"Starting model training from {input_path}")
    
    # Read featured parquet file
    df = pd.read_parquet(input_path)
    log_info(f"Read {len(df)} rows from featured data")
    
    # Prepare data for training
    X_train, X_val, y_train, y_val, feature_cols = prepare_data(
        df, target_column=target, test_size=0.2
    )
    log_info(f"Train samples: {len(X_train)}, Validation samples: {len(X_val)}")
    log_info(f"Using {len(feature_cols)} features: {', '.join(feature_cols)}")
    
    # Train model
    hyperparameters = {
        'n_estimators': 100,
        'max_depth': 10,
        'random_state': 42
    }
    log_info(f"Training {model_type} model with params: {hyperparameters}")
    model = train_model(X_train, y_train, model_type=model_type, **hyperparameters)
    log_info("Model training complete")
    
    # Evaluate model
    metrics = evaluate_model(model, X_val, y_val)
    log_info(f"Validation metrics - MAE: {metrics['mae']:.4f}, RMSE: {metrics['rmse']:.4f}, R²: {metrics['r2_score']:.4f}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save model with date
    date_str = datetime.now().strftime('%Y%m%d')
    model_path = output_path / f'model_{date_str}.pkl'
    save_model(model, model_path)
    log_info(f"Saved model to {model_path}")
    
    # Create and save metadata
    metadata = {
        'training_date': datetime.now().isoformat(),
        'model_type': model_type,
        'target_column': target,
        'feature_columns': feature_cols,
        'train_samples': len(X_train),
        'validation_samples': len(X_val),
        'hyperparameters': hyperparameters,
        'metrics': metrics,
        'model_path': str(model_path)
    }
    
    metadata_path = model_path.with_suffix('.json')
    save_metadata(metadata, metadata_path)
    log_info(f"Saved metadata to {metadata_path}")
    
    return (True, str(model_path))


if __name__ == "__main__":
    # Test the training
    input_file = "data/processed/weather_features_20260103.parquet"
    output_dir = "models"
    success, path = train_weather_model(input_file, output_dir)
    if success:
        print(f"✅ Success! Model saved to: {path}")
    else:
        print("❌ Training failed")
