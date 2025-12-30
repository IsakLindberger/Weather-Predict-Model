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
    # Build metadata dict according to schema contract
    metadata = {
        'ingestion_date': datetime.now().isoformat(),
        'source_station': station_id,
        'total_rows': len(df),
        'file_path': str(file_path)
    }
    
    return metadata


def save_metadata(metadata, output_path):
    """
    Save metadata to JSON file.

    Args:
        metadata: dict with metadata
        output_path: Path where to save JSON
    """
    output_path = Path(output_path)
    
    # Write metadata as formatted JSON
    with open(output_path, 'w') as f:
        json.dump(metadata, f, indent=2)


def log_info(message):
    """Simple console logging with timestamp."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] INFO: {message}")