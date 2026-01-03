import pandas as pd
from pathlib import Path
from datetime import datetime
import sys
sys.path.append(str(Path(__file__).parent.parent))

from .evaluator import (
    load_model, 
    prepare_evaluation_data, 
    evaluate_predictions, 
    create_predictions_dataframe,
    analyze_errors
)
from ingest.logger import save_metadata, log_info


def evaluate_model(model_path, data_path, output_dir, target='temperature'):
    """
    Evaluate a trained model on test data.
    
    Args:
        model_path: Path to trained model file
        data_path: Path to featured test data
        output_dir: Directory to save evaluation results
        target: Target variable name
        
    Returns:
        tuple: (success: bool, output_path: str or None)
    """
    log_info(f"Starting model evaluation")
    log_info(f"Model: {model_path}")
    log_info(f"Data: {data_path}")
    
    # Load model
    model = load_model(model_path)
    log_info("Model loaded successfully")
    
    # Load test data
    df = pd.read_parquet(data_path)
    log_info(f"Read {len(df)} rows for evaluation")
    
    # Prepare evaluation data
    X, y_true, feature_cols = prepare_evaluation_data(df, target_column=target)
    log_info(f"Using {len(feature_cols)} features for evaluation")
    
    # Make predictions
    y_pred = model.predict(X)
    log_info("Predictions complete")
    
    # Evaluate performance
    metrics = evaluate_predictions(y_true, y_pred)
    log_info(f"Evaluation metrics - MAE: {metrics['mae']:.4f}, RMSE: {metrics['rmse']:.4f}, R²: {metrics['r2_score']:.4f}")
    if metrics['mape'] is not None:
        log_info(f"MAPE: {metrics['mape']:.4f}")
    
    # Create predictions DataFrame
    predictions_df = create_predictions_dataframe(df, y_true, y_pred)
    
    # Analyze errors
    error_analysis = analyze_errors(predictions_df)
    log_info(f"Large errors (>90th percentile): {error_analysis['large_error_count']} ({error_analysis['large_error_percentage']:.2f}%)")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save predictions with date
    date_str = datetime.now().strftime('%Y%m%d')
    predictions_path = output_path / f'evaluation_results_{date_str}.parquet'
    predictions_df.to_parquet(predictions_path, index=False)
    log_info(f"Saved predictions to {predictions_path}")
    
    # Create and save metadata
    metadata = {
        'evaluation_date': datetime.now().isoformat(),
        'model_path': str(model_path),
        'test_samples': len(df),
        'metrics': metrics,
        'error_analysis': error_analysis,
        'file_path': str(predictions_path)
    }
    
    metadata_path = predictions_path.with_suffix('.json')
    save_metadata(metadata, metadata_path)
    log_info(f"Saved metadata to {metadata_path}")
    
    return (True, str(predictions_path))


if __name__ == "__main__":
    # Test the evaluation
    model_file = "models/model_20260103.pkl"
    data_file = "data/processed/weather_features_20260103.parquet"
    output_dir = "data/processed"
    
    success, path = evaluate_model(model_file, data_file, output_dir)
    if success:
        print(f"✅ Success! Evaluation results saved to: {path}")
    else:
        print("❌ Evaluation failed")
