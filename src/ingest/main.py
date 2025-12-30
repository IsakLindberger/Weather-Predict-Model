import pandas as pd
from pathlib import Path
from datetime import datetime
from .validator import load_schema, validate_dataframe
from .logger import create_metadata, save_metadata, log_info

def ingest_data(source_path, output_dir, station_id):
    """
    Ingest weather data from CSV to parquet format.

    Args:
        source_path: Path to source CSV file
        output_dir: Directory to save output files
        station_dir: Weather station identifier

    Returns:
        tuple: (success: bool, output_path: str or None)
    """

    log_info(f"Starting ingestion from {source_path}")

    df = pd.read_csv(source_path, parse_dates=['timestamp'])
    log_info(f"Read {len(df)} rows from source")

    schema_path = Path(__file__).parent.parent.parent / 'schemas' / 'ingest.yaml'
    schema = load_schema(schema_path)

    is_valid, errors = validate_dataframe(df, schema)
    if not is_valid:
        log_info(f"Validation failed: {errors}")
        return (False, None)
    log_info("Validation passed")

    date_str = datetime.now().strftime('%Y%m%d')
    output_path = Path(output_dir) / f'weather_{date_str}.parquet'

    df.to_parquet(output_path, index=False)
    log_info(f"Saved data to {output_path}")

    metadata = create_metadata(df, station_id, str(output_path))
    metadata_path = output_path.with_suffix('.json')
    save_metadata(metadata, metadata_path)
    log_info(f"Saved metadata to {metadata_path}")

    return (True, str(output_path))


if __name__ == "__main__":
    source = "data/raw/sample_weather.csv"
    output = "data/raw"
    success, path = ingest_data(source, output, "STATION_001")
    if success:
        print(f"Success! Data saved to: {path}")
    else:
        print("Ingestion failed")
