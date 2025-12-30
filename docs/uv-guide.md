# uv - Fast Python Package Manager

## Installation

Install uv:
```bash
pip install uv
```

Or on Windows with pipx:
```bash
pipx install uv
```

## Setup Project

Install dependencies:
```bash
uv sync
```

Install with dev dependencies:
```bash
uv sync --extra dev
```

Install with ML dependencies:
```bash
uv sync --extra ml
```

Install all extras:
```bash
uv sync --all-extras
```

## Usage

Run Python with uv:
```bash
uv run python src/ingest/main.py
```

Run tests:
```bash
uv run pytest
```

Add a new package:
```bash
uv add package-name
```

Add a dev package:
```bash
uv add --dev package-name
```

## Virtual Environment

uv automatically manages a virtual environment in `.venv/`

Activate manually if needed:
```bash
# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate
```

## Lock File

`uv.lock` ensures reproducible builds - commit this file to git.
