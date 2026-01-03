import pandas as pd
from pathlib import Path
from datetime import datetime
import sys
sys.path.append(str(Path(__file__).parent.parent))

from .analyzer import (
    identify_failures,
    categorize_failures,
    analyze_failure_patterns,
    add_failure_context,
    create_failure_summary
)
from ingest.logger import save_metadata, log_info


def analyze_failures(predictions_path, features_path, output_dir, threshold_percentile=90):
    """
    Analyze model prediction failures.
    
    Args:
        predictions_path: Path to evaluation results parquet
        features_path: Path to original features data
        output_dir: Directory to save failure analysis
        threshold_percentile: Percentile for failure threshold
        
    Returns:
        tuple: (success: bool, output_path: str or None)
    """
    log_info(f"Starting failure analysis")
    log_info(f"Predictions: {predictions_path}")
    log_info(f"Features: {features_path}")
    
    # Load predictions
    predictions_df = pd.read_parquet(predictions_path)
    log_info(f"Loaded {len(predictions_df)} predictions")
    
    # Load original features for context
    features_df = pd.read_parquet(features_path)
    log_info(f"Loaded original features")
    
    # Identify failures
    failures_df, threshold = identify_failures(predictions_df, threshold_percentile)
    log_info(f"Identified {len(failures_df)} failures with threshold: {threshold:.4f}")
    
    if len(failures_df) == 0:
        log_info("No failures detected - model performs well across all predictions")
        return (True, None)
    
    # Categorize failures
    failures_df = categorize_failures(failures_df)
    log_info("Categorized failure types")
    
    # Analyze patterns
    patterns = analyze_failure_patterns(failures_df)
    log_info(f"Failure types: {patterns['failure_types']}")
    
    # Add context
    failures_df = add_failure_context(failures_df, features_df)
    log_info("Added contextual information to failures")
    
    # Create summary
    summary = create_failure_summary(predictions_df, threshold, patterns)
    log_info(f"Failure rate: {summary['failure_rate']:.2f}%")
    log_info(f"Average failure magnitude: {summary['avg_failure_magnitude']:.4f}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save failure analysis with date
    date_str = datetime.now().strftime('%Y%m%d')
    analysis_path = output_path / f'failure_analysis_{date_str}.parquet'
    
    # Select relevant columns for output
    output_columns = [
        'timestamp', 'actual', 'predicted', 'error', 'abs_error',
        'failure_type', 'contributing_features'
    ]
    failures_df[output_columns].to_parquet(analysis_path, index=False)
    log_info(f"Saved failure analysis to {analysis_path}")
    
    # Create and save metadata
    metadata = {
        'analysis_date': datetime.now().isoformat(),
        'total_failures': summary['total_failures'],
        'failure_threshold': summary['error_threshold'],
        'failure_rate': summary['failure_rate'],
        'failure_types': summary['failure_types'],
        'avg_failure_magnitude': summary['avg_failure_magnitude'],
        'file_path': str(analysis_path)
    }
    
    metadata_path = analysis_path.with_suffix('.json')
    save_metadata(metadata, metadata_path)
    log_info(f"Saved metadata to {metadata_path}")
    
    return (True, str(analysis_path))


if __name__ == "__main__":
    # Test the failure analysis
    predictions_file = "data/processed/evaluation_results_20260103.parquet"
    features_file = "data/processed/weather_features_20260103.parquet"
    output_dir = "data/processed"
    
    success, path = analyze_failures(predictions_file, features_file, output_dir)
    if success:
        if path:
            print(f"✅ Success! Failure analysis saved to: {path}")
        else:
            print("✅ Success! No failures detected - model performs well")
    else:
        print("❌ Failure analysis failed")
