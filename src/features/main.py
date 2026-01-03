import pandas as pd
from pathlib import Path
from datetime import datetime
import sys
sys.path.append(str(Path(__file__).parent.parent))

from .validator import load_schema, validate_dataframe
from .feature_engineering import create_features
from ingest.logger import create_metadata, save_metadata, log_info


def engineer_features(input_path, output_dir, station_id='STATION_001'):
    """
    Create engineered features from cleaned weather data.
    
    Args:
        input_path: Path to cleaned parquet file
        output_dir: Directory to save featured data
        station_id: Weather station identifier
        
    Returns:
        tuple: (success: bool, output_path: str or None)
    """
    log_info(f"Starting feature engineering from {input_path}")
    
    # Read cleaned parquet file
    df = pd.read_parquet(input_path)
    log_info(f"Read {len(df)} rows from cleaned data")
    
    # Create features
    df_featured, stats = create_features(df)
    log_info(f"Created {stats['new_feature_count']} new features")
    log_info(f"New features: {', '.join(stats['features_created'])}")
    
    # Load and validate against features schema
    schema_path = Path(__file__).parent.parent.parent / 'schemas' / 'features.yaml'
    schema = load_schema(schema_path)
    
    is_valid, errors = validate_dataframe(df_featured, schema)
    if not is_valid:
        log_info(f"Validation failed: {errors}")
        return (False, None)
    log_info("Validation passed")
    
    # Create output path with date
    date_str = datetime.now().strftime('%Y%m%d')
    output_path = Path(output_dir) / f'weather_features_{date_str}.parquet'
    
    # Save as parquet format
    df_featured.to_parquet(output_path, index=False)
    log_info(f"Saved featured data to {output_path}")
    
    # Create and save metadata
    metadata = create_metadata(df_featured, station_id, str(output_path))
    # Add feature engineering stats to metadata
    metadata['features_created'] = stats['features_created']
    metadata['feature_engineering_date'] = datetime.now().isoformat()
    
    metadata_path = output_path.with_suffix('.json')
    save_metadata(metadata, metadata_path)
    log_info(f"Saved metadata to {metadata_path}")
    
    return (True, str(output_path))


if __name__ == "__main__":
    # Test the feature engineering
    input_file = "data/processed/weather_cleaned_20260103.parquet"
    output_dir = "data/processed"
    success, path = engineer_features(input_file, output_dir)
    if success:
        print(f"✅ Success! Featured data saved to: {path}")
    else:
        print("❌ Feature engineering failed")
