import json
from datetime import datetime
from pathlib import Path

def create_metadata(df, station_id, file_path):
    """
    Create metadata dict for ingested data.

    Args:
        df: pandas DataFrame
        station_id: str, weather station identifier
        file_path: str, path where data will be saved

    Returns:
        dict: Metadata according to contract
    """

    metadata = {
        'ingestion_date' : datetime.now().isoformat(),
        'source_station' : station_id,
        'total_rows' : len(df),
        'file_path' : str(file_path)
    }

    return metadata


def save_metadata(metadata, output_path):
    """
    Save metafata to JSON file.

    Args:
        metadata: dict with metadata
        output_path: Path where to save JSON
    """

    output_path = Path(output_path) # convert output_path

    with open(output_path, 'w') as f: # open file in write mode
        json.dump(metadata, f, indent=2) # save dict as JSON with indentation


def log_info(message):
    """Simple console logging."""
    # Create timestamp-str
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Write formated messagge
    print(f"[{timestamp}] INFO: {message}")