"""
Pytest configuration and shared fixtures for Weather Predict Model tests.
"""
import pytest
import pandas as pd
import yaml
from pathlib import Path
from datetime import datetime


@pytest.fixture
def project_root():
    """Return the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def schemas_dir(project_root):
    """Return the schemas directory."""
    return project_root / "schemas"


@pytest.fixture
def data_raw_dir(project_root, tmp_path):
    """Return temporary raw data directory for tests."""
    return tmp_path / "data" / "raw"


@pytest.fixture
def data_processed_dir(project_root, tmp_path):
    """Return temporary processed data directory for tests."""
    return tmp_path / "data" / "processed"


@pytest.fixture
def load_schema(schemas_dir):
    """Factory fixture to load schema files."""
    def _load_schema(schema_name):
        schema_path = schemas_dir / f"{schema_name}.yaml"
        with open(schema_path, 'r') as f:
            return yaml.safe_load(f)
    return _load_schema


@pytest.fixture
def sample_weather_data():
    """Create sample weather data for testing."""
    return pd.DataFrame({
        'timestamp': pd.date_range('2025-01-01', periods=24, freq='H'),
        'station_id': ['STATION_001'] * 24,
        'temperature': [15.5, 16.0, 16.5, 17.0, 17.5, 18.0, 18.5, 19.0,
                       19.5, 20.0, 20.5, 21.0, 21.5, 22.0, 22.5, 23.0,
                       22.5, 22.0, 21.5, 21.0, 20.5, 20.0, 19.5, 19.0],
        'humidity': [65.0, 64.0, 63.0, 62.0, 61.0, 60.0, 59.0, 58.0,
                    57.0, 56.0, 55.0, 54.0, 53.0, 52.0, 51.0, 50.0,
                    51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0],
        'pressure': [1013.25] * 24
    })


@pytest.fixture
def sample_metadata():
    """Create sample metadata for testing."""
    return {
        'ingestion_date': datetime.now().isoformat(),
        'source_station': 'STATION_001',
        'total_rows': 24,
        'file_path': 'data/raw/weather_20250101.parquet'
    }
