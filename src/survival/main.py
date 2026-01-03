import pandas as pd
from pathlib import Path
from datetime import datetime
import sys
sys.path.append(str(Path(__file__).parent.parent))

from .analyzer import (
    prepare_survival_data,
    fit_kaplan_meier,
    calculate_survival_statistics,
    calculate_hazard_ratio,
    create_survival_summary
)
from ingest.logger import save_metadata, log_info


def perform_survival_analysis(predictions_path, output_dir, error_threshold=0.7):
    """
    Perform survival analysis on model predictions.
    
    Args:
        predictions_path: Path to evaluation results parquet
        output_dir: Directory to save survival analysis
        error_threshold: Threshold for defining failure events
        
    Returns:
        tuple: (success: bool, output_path: str or None)
    """
    log_info(f"Starting survival analysis")
    log_info(f"Predictions: {predictions_path}")
    log_info(f"Error threshold: {error_threshold}")
    
    # Load predictions
    predictions_df = pd.read_parquet(predictions_path)
    log_info(f"Loaded {len(predictions_df)} predictions")
    
    # Prepare survival data
    survival_df = prepare_survival_data(predictions_df, error_threshold)
    log_info(f"Prepared survival data with {survival_df['event'].sum()} events")
    
    # Fit Kaplan-Meier survival curve
    kmf = fit_kaplan_meier(survival_df)
    log_info("Fitted Kaplan-Meier survival curve")
    
    # Calculate survival statistics
    stats = calculate_survival_statistics(kmf, survival_df)
    log_info(f"Event rate: {stats['event_rate']:.2f}%")
    log_info(f"Events: {stats['events_count']}, Censored: {stats['censored_count']}")
    if stats['median_survival_time']:
        log_info(f"Median survival time: {stats['median_survival_time']:.2f}")
    
    # Calculate hazard ratios
    survival_df = calculate_hazard_ratio(survival_df)
    log_info("Calculated hazard ratios")
    
    # Create summary
    summary_df = create_survival_summary(survival_df, stats)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save survival analysis with date
    date_str = datetime.now().strftime('%Y%m%d')
    analysis_path = output_path / f'survival_analysis_{date_str}.parquet'
    summary_df.to_parquet(analysis_path, index=False)
    log_info(f"Saved survival analysis to {analysis_path}")
    
    # Create and save metadata
    metadata = {
        'analysis_date': datetime.now().isoformat(),
        'total_observations': stats['total_observations'],
        'events_count': stats['events_count'],
        'censored_count': stats['censored_count'],
        'event_rate': stats['event_rate'],
        'median_survival_time': stats['median_survival_time'],
        'final_survival_probability': stats['final_survival_probability'],
        'error_threshold': error_threshold,
        'file_path': str(analysis_path)
    }
    
    metadata_path = analysis_path.with_suffix('.json')
    save_metadata(metadata, metadata_path)
    log_info(f"Saved metadata to {metadata_path}")
    
    return (True, str(analysis_path))


if __name__ == "__main__":
    # Test the survival analysis
    predictions_file = "data/processed/evaluation_results_20260103.parquet"
    output_dir = "data/processed"
    
    # Use 90th percentile threshold (about 0.7 based on previous analysis)
    success, path = perform_survival_analysis(predictions_file, output_dir, error_threshold=0.7)
    if success:
        print(f"✅ Success! Survival analysis saved to: {path}")
    else:
        print("❌ Survival analysis failed")
