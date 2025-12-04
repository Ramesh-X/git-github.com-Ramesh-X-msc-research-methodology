#!/bin/bash

# Script to run ruff linting and formatting on the project

set -e  # Exit on any error

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "Error: Virtual environment '.venv' not found. Please create it first."
    exit 1
fi

echo "Running ruff check"
./.venv/bin/python3 -m ruff check --fix .
./.venv/bin/python3 -m ruff check --select I --fix .

echo "Running ruff format"
./.venv/bin/python3 -m ruff format .

echo "Ruff checks and formatting completed successfully."
