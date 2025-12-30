# Testing

## Setup

Install test dependencies:
```bash
pip install -r requirements-dev.txt
```

## Running Tests

Run all tests:
```bash
pytest
```

Run specific module tests:
```bash
pytest tests/test_ingest/
pytest tests/test_clean/
```

Run with coverage:
```bash
pytest --cov=src --cov-report=html
```

## Test Structure

- `tests/conftest.py` - Shared fixtures and configuration
- `tests/test_<module>/` - Tests for each pipeline module
- `pytest.ini` - Pytest configuration

## Writing Tests

Each module should test:
1. Schema validation
2. Data transformation logic
3. Metadata logging
4. Error handling
