# Environment Setup Guide

This guide explains how to set up the development environment for the **Neurological LRD Analysis** project.

## Quick Setup

Run the automated setup script:

```bash
bash scripts/setup_venv.sh
```

This script will:
- Check for Python 3.11+ compatibility
- Create a virtual environment named `neurological_env`
- Install all required dependencies
- Install development and optional dependencies

## Manual Setup

If you prefer to set up the environment manually:

### 1. Create Virtual Environment

```bash
python3 -m venv neurological_env
```

### 2. Activate Virtual Environment

**Linux/macOS:**
```bash
source neurological_env/bin/activate
```

**Windows:**
```bash
neurological_env\Scripts\activate
```

### 3. Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install project in development mode
pip install -e .

# Install development dependencies
pip install pytest pytest-cov pytest-html pytest-json-report pytest-timeout black isort flake8 mypy

# Install optional dependencies for enhanced functionality
pip install jax jaxlib numba scikit-learn matplotlib seaborn pywavelets
```

## Environment Management

### Activating the Environment

Before working on the project, always activate the virtual environment:

```bash
source neurological_env/bin/activate  # Linux/macOS
# or
neurological_env\Scripts\activate     # Windows
```

You should see `(neurological_env)` at the beginning of your command prompt when the environment is active.

### Deactivating the Environment

To deactivate the virtual environment:

```bash
deactivate
```

### Checking Environment Status

To verify the environment is working correctly:

```bash
# Check Python version (should be 3.11+)
python --version

# Check installed packages
pip list

# Run tests to verify installation
python -m pytest tests/ -v
```

## Project Structure

The virtual environment (`neurological_env/`) should be created in the project root directory:

```
neurological_lrd_analysis/
├── neurological_env/          # Virtual environment (created by setup)
├── neurological_lrd_analysis/ # Main package directory
├── docs/                      # Documentation
├── scripts/                   # Setup and demo scripts
├── tests/                     # Test suite
├── pyproject.toml             # Project configuration
└── README.md                  # Project overview
```

## Troubleshooting

### Python Version Issues

If you encounter Python version issues:

1. Check your Python version: `python3 --version`
2. Ensure you have Python 3.11 or later installed
3. If using a different Python version, specify it explicitly:
   ```bash
   python3.11 -m venv neurological_env
   ```

### Permission Issues

If you encounter permission issues with the setup script:

```bash
chmod +x scripts/setup_venv.sh
bash scripts/setup_venv.sh
```

### Dependency Installation Issues

If some dependencies fail to install:

1. Update pip: `pip install --upgrade pip`
2. Try installing dependencies individually:
   ```bash
   pip install numpy scipy pandas
   pip install jax jaxlib
   pip install numba
   ```

### Environment Not Found

If the virtual environment is not found:

1. Ensure you're in the project root directory
2. Check if the `neurological_env/` directory exists
3. Recreate the environment if necessary:
   ```bash
   rm -rf neurological_env
   bash scripts/setup_venv.sh
   ```

## Development Workflow

1. **Activate environment**: `source neurological_env/bin/activate`
2. **Make changes** to the code
3. **Run tests**: `python -m pytest tests/ -v`
4. **Test functionality**: `python scripts/biomedical_scenarios_demo.py`
5. **Deactivate when done**: `deactivate`

## IDE Integration

### VS Code

1. Open the project in VS Code
2. Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on macOS)
3. Type "Python: Select Interpreter"
4. Choose the interpreter from `./neurological_env/bin/python`

### PyCharm

1. Open the project in PyCharm
2. Go to File → Settings → Project → Python Interpreter
3. Click the gear icon → Add
4. Choose "Existing environment"
5. Select `./neurological_env/bin/python`

## Notes

- The virtual environment name `neurological_env` is appropriate for this project
- All dependencies are specified in `pyproject.toml`
- The project uses Python 3.11+ for compatibility with advanced ML and JAX libraries
- Lazy imports are implemented to reduce startup time and memory usage

