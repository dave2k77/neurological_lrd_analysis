#!/bin/bash
# Setup script for the neurological_lrd_analysis project
# Creates and configures the 'neurological_env' virtual environment

set -e  # Exit on any error

echo "Setting up neurological_lrd_analysis development environment..."
echo "=============================================================="

# Check if Python 3.11+ is available
python_version=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
required_version="3.11"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "Error: Python 3.11+ is required, but found Python $python_version"
    echo "Please install Python 3.11 or later"
    exit 1
fi

echo "✓ Python version check passed: $python_version"

# Create virtual environment
echo "Creating virtual environment 'neurological_env'..."
if [ -d "neurological_env" ]; then
    echo "Virtual environment 'neurological_env' already exists. Removing it..."
    rm -rf neurological_env
fi

python3 -m venv neurological_env
echo "✓ Virtual environment created"

# Activate virtual environment
echo "Activating virtual environment..."
source neurological_env/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install project dependencies
echo "Installing project dependencies..."
pip install -e .

# Install development dependencies
echo "Installing development dependencies..."
pip install pytest pytest-cov pytest-html pytest-json-report pytest-timeout black isort flake8 mypy

# Install optional dependencies for enhanced functionality
echo "Installing optional dependencies..."
pip install jax jaxlib numba scikit-learn matplotlib seaborn pywavelets myst-parser sphinx sphinx-rtd-theme

echo ""
echo "Setup completed successfully!"
echo "============================="
echo ""
echo "To activate the environment in the future, run:"
echo "  source neurological_env/bin/activate"
echo ""
echo "To deactivate the environment, run:"
echo "  deactivate"
echo ""
echo "To run tests:"
echo "  python -m pytest tests/ -v"
echo ""
echo "To run the demo:"
echo "  python scripts/biomedical_scenarios_demo.py"
echo ""

