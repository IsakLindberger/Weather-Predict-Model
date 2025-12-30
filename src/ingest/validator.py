import yaml
from pathlib import Path
import pandas as pd

def load_schema(schema_path):
    """
    Load schema definition from YAML file.

    Args:
        schema_path: Path to schema YAML file

    Returns:
        dict: Schema definition
    """

    schema_path = Path(schema_path) # Convert to Path-obejct

    with open(schema_path, 'r') as f:
        return yaml.safe_load(f) # Read YAML and return
    

def check_required_columns(df, schema):
    """
    Check if all required columns exist in DataFrame.

    Args:
        df: pandas DataFrame to validate
        schema: Schema dict from load_schema()

    Returns:
        list: Missing required column names (empty if all present)
    """

    missing = []

    for col_name, col_info in schema['columns'].items():
        if col_info.get('required', False):
            if col_name not in df.columns:
                missing.append(col_name)
    
    return missing


def validate_dataframe(df, schema):
    """
    Validate DataFrame against schema.

    Args:
        df: pandas DataFrame to validate
        schema: Schema dict from load_schema()

    Returns:
        tuple: (is_valid: bool, errors: list of str)
    """

    errors = []

    missing = check_required_columns(df, schema)

    if missing:
        errors.append(f"Missing required columns: {missing}")

    is_valid = len(errors) == 0

    return (is_valid, errors)