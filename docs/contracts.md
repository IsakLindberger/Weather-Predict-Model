# Pipeline Contracts

This document defines the contracts between different steps in the ML pipeline.

## Overview

Each module in the pipeline has a clearly defined contract for input and output:

```
ingest → clean → features → train → evaluate → failure/survival
```

## Data Flow

### 1. Ingest
**Output:** `data/raw/weather_YYYYMMDD.parquet`  
**Schema:** [schemas/ingest.yaml](../schemas/ingest.yaml)

- Fetches raw data from weather station
- Saves data with minimal processing
- Logs metadata about data collection

**Metadata:**
- `ingestion_date` - When data was fetched
- `source_station` - Which weather station
- `total_rows` - Number of observations
- `file_path` - Path to saved file

---

### 2. Clean
**Input:** `data/raw/weather_YYYYMMDD.parquet`  
**Output:** `data/processed/weather_cleaned_YYYYMMDD.parquet`  
**Schema:** [schemas/clean.yaml](../schemas/clean.yaml)

- Validates that input matches ingest schema
- Removes outliers and invalid values
- Fills missing values
- Ensures all required columns exist

**Metadata:**
- `cleaning_date` - When data was cleaned
- `source_station` - Source station
- `total_rows` - Number of rows after cleaning
- `rows_removed` - Number of removed rows
- `missing_values_filled` - Number of filled values
- `file_path` - Path to saved file

---

### 3. Features
**Input:** `data/processed/weather_cleaned_YYYYMMDD.parquet`  
**Output:** `data/processed/weather_features_YYYYMMDD.parquet`  
**Schema:** [schemas/features.yaml](../schemas/features.yaml)

- Creates engineered features
- Calculates rolling windows
- Adds derived features

**Metadata:**
- `feature_engineering_date` - When features were created
- `source_station` - Source station
- `total_rows` - Number of rows
- `features_created` - List of new features
- `file_path` - Path to saved file

---

### 4. Train
**Input:** `data/processed/weather_features_YYYYMMDD.parquet`  
**Output:** `models/model_YYYYMMDD.pkl`  
**Schema:** [schemas/train.yaml](../schemas/train.yaml)

- Trains model on features
- Saves trained model
- Logs hyperparameters and configuration

**Metadata:**
- `training_date` - When model was trained
- `total_samples` - Number of training samples
- `features_used` - Which features were used
- `model_path` - Path to model file

---

### 5. Evaluate
**Input:** `models/model_YYYYMMDD.pkl`  
**Output:** `data/processed/evaluation_results_YYYYMMDD.parquet`  
**Schema:** [schemas/evaluate.yaml](../schemas/evaluate.yaml)

- Evaluates model performance
- Calculates metrics (MAE, RMSE, R²)
- Saves predictions vs actual values

**Metadata:**
- `evaluation_date` - When evaluation was performed
- `model_path` - Which model was evaluated
- `test_samples` - Number of test samples
- `file_path` - Path to results

---

### 6. Failure
**Input:** `data/processed/evaluation_results_YYYYMMDD.parquet`  
**Output:** `data/processed/failure_analysis_YYYYMMDD.parquet`  
**Schema:** [schemas/failure.yaml](../schemas/failure.yaml)

- Analyzes when the model fails
- Categorizes types of errors
- Identifies patterns in failures

**Metadata:**
- `analysis_date` - When analysis was performed
- `total_failures` - Number of identified failures
- `failure_threshold` - Threshold value for failure
- `file_path` - Path to analysis

---

### 7. Survival
**Input:** `data/processed/evaluation_results_YYYYMMDD.parquet`  
**Output:** `data/processed/survival_analysis_YYYYMMDD.parquet`  
**Schema:** [schemas/survival.yaml](../schemas/survival.yaml)

- Survival analysis of model performance
- Calculates hazard ratios
- Analyzes time to event/failure

**Metadata:**
- `analysis_date` - When analysis was performed
- `total_observations` - Number of observations
- `events_count` - Number of events
- `censored_count` - Number of censored
- `file_path` - Path to analysis

---

## Validation

Each module should:

1. **Validate input** against expected schema
2. **Produce output** according to its schema
3. **Log metadata** according to specification
4. **Raise exception** if contract is broken

## Usage Example

```python
# Each module validates its input
def validate_input(df, schema):
    # Check that all required columns exist
    # Check data types
    # Return True/False

# Each module saves according to contract
def save_output(df, metadata, filepath):
    # Save parquet file
    # Save metadata as JSON
```
