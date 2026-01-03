import pandas as pd
from pathlib import Path
from datetime import datetime
import sys
sys.path.append(str(Path(__file__).parent.parent))

from .validator import load_schema, validate_dataframe
from .cleaner import clean_data
from ingest.logger import create_metadata, save_metadata, log_info


def clean_weather_data(input_path, output_dir, station_id='STATION_001'):
    """
    Clean weather data from parquet format.
    
    Args:
        input_path: Path to input parquet file
        output_dir: Directory to save cleaned data
        station_id: Weather station identifier
        
    Returns:
        tuple: (success: bool, output_path: str or None)
    """
    log_info(f"Starting cleaning from {input_path}")
    
    # Read parquet file
    df = pd.read_parquet(input_path)
    log_info(f"Read {len(df)} rows from input")
    
    # Clean the data
    df_clean, stats = clean_data(df)
    log_info(f"Removed {stats['rows_removed']} outlier rows")
    log_info(f"Filled {stats['missing_values_filled']} missing values")
    log_info(f"Final dataset has {stats['final_row_count']} rows")
    
    # Load and validate against clean schema
    schema_path = Path(__file__).parent.parent.parent / 'schemas' / 'clean.yaml'
    schema = load_schema(schema_path)
    
    is_valid, errors = validate_dataframe(df_clean, schema)
    if not is_valid:
        log_info(f"Validation failed: {errors}")
        return (False, None)
    log_info("Validation passed")
    
    # Create output path with date
    date_str = datetime.now().strftime('%Y%m%d')
    output_path = Path(output_dir) / f'weather_cleaned_{date_str}.parquet'
    
    # Save as parquet format
    df_clean.to_parquet(output_path, index=False)
    log_info(f"Saved cleaned data to {output_path}")
    
    # Create and save metadata
    metadata = create_metadata(df_clean, station_id, str(output_path))
    # Add cleaning stats to metadata
    metadata['rows_removed'] = stats['rows_removed']
    metadata['missing_values_filled'] = stats['missing_values_filled']
    metadata['cleaning_date'] = datetime.now().isoformat()
    
    metadata_path = output_path.with_suffix('.json')
    save_metadata(metadata, metadata_path)
    log_info(f"Saved metadata to {metadata_path}")
    
    return (True, str(output_path))


if __name__ == "__main__":
    # Test the cleaning
    input_file = "data/raw/weather_20251230.parquet"
    output_dir = "data/processed"
    success, path = clean_weather_data(input_file, output_dir)
    if success:
        print(f"✅ Success! Cleaned data saved to: {path}")
    else:
        print("❌ Cleaning failed")
