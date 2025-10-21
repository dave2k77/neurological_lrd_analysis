#!/usr/bin/env python3
"""
Test Coverage Report Generator

This script generates a comprehensive test coverage report for the
Neurological LRD Analysis library.

Author: Davian R. Chin (PhD Candidate in Biomedical Engineering, University of Reading, UK)
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def run_command(cmd, description):
    """Run a command and return the result."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {cmd}")
    print(f"{'='*60}")
    
    start_time = time.time()
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        execution_time = time.time() - start_time
        
        print(f"Exit code: {result.returncode}")
        print(f"Execution time: {execution_time:.2f} seconds")
        
        if result.stdout:
            print(f"\nSTDOUT:\n{result.stdout}")
        if result.stderr:
            print(f"\nSTDERR:\n{result.stderr}")
            
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        print(f"Error running command: {e}")
        return False, "", str(e)

def analyze_test_coverage():
    """Analyze test coverage for the library."""
    print("🧪 NEUROLOGICAL LRD ANALYSIS - TEST COVERAGE REPORT")
    print("=" * 70)
    print("Author: Davian R. Chin (PhD Candidate in Biomedical Engineering, University of Reading, UK)")
    print("=" * 70)
    
    # Change to project directory
    project_dir = Path(__file__).parent.parent
    os.chdir(project_dir)
    
    print(f"\n📁 Project Directory: {project_dir}")
    
    # 1. Count test files and lines
    print("\n📊 TEST FILE ANALYSIS")
    print("-" * 40)
    
    test_files = list(Path("tests").glob("*.py"))
    print(f"Test files: {len(test_files)}")
    
    total_test_lines = 0
    for test_file in test_files:
        with open(test_file, 'r') as f:
            lines = len(f.readlines())
            total_test_lines += lines
            print(f"  {test_file.name}: {lines} lines")
    
    print(f"Total test lines: {total_test_lines}")
    
    # 2. Count source files and lines
    print("\n📊 SOURCE FILE ANALYSIS")
    print("-" * 40)
    
    source_files = list(Path("neurological_lrd_analysis").rglob("*.py"))
    print(f"Source files: {len(source_files)}")
    
    total_source_lines = 0
    for source_file in source_files:
        with open(source_file, 'r') as f:
            lines = len(f.readlines())
            total_source_lines += lines
            try:
                rel_path = source_file.relative_to(project_dir)
            except ValueError:
                rel_path = source_file
            print(f"  {rel_path}: {lines} lines")
    
    print(f"Total source lines: {total_source_lines}")
    
    # 3. Run tests and collect results
    print("\n🧪 TEST EXECUTION")
    print("-" * 40)
    
    # Run ML baselines tests
    success, stdout, stderr = run_command(
        "python -m pytest tests/test_ml_baselines.py -v",
        "ML Baselines Tests"
    )
    
    if success:
        ml_tests = len([line for line in stdout.split('\n') if 'PASSED' in line])
        print(f"✅ ML Baselines: {ml_tests} tests passed")
    else:
        print(f"❌ ML Baselines: Tests failed")
    
    # Run pretrained models tests
    success, stdout, stderr = run_command(
        "python -m pytest tests/test_pretrained_models.py -v",
        "Pretrained Models Tests"
    )
    
    if success:
        pretrained_tests = len([line for line in stdout.split('\n') if 'PASSED' in line])
        print(f"✅ Pretrained Models: {pretrained_tests} tests passed")
    else:
        print(f"❌ Pretrained Models: Tests failed")
    
    # Run core tests
    success, stdout, stderr = run_command(
        "python -m pytest tests/test_accuracy.py tests/test_backends.py tests/test_bench.py tests/test_registry.py -v",
        "Core Functionality Tests"
    )
    
    if success:
        core_tests = len([line for line in stdout.split('\n') if 'PASSED' in line])
        print(f"✅ Core Functionality: {core_tests} tests passed")
    else:
        print(f"❌ Core Functionality: Tests failed")
    
    # 4. Test coverage analysis
    print("\n📈 TEST COVERAGE ANALYSIS")
    print("-" * 40)
    
    coverage_ratio = (total_test_lines / total_source_lines) * 100 if total_source_lines > 0 else 0
    print(f"Test-to-Source Ratio: {coverage_ratio:.1f}%")
    
    # 5. Test quality metrics
    print("\n🎯 TEST QUALITY METRICS")
    print("-" * 40)
    
    print(f"Test Files: {len(test_files)}")
    print(f"Source Files: {len(source_files)}")
    print(f"Test Lines: {total_test_lines}")
    print(f"Source Lines: {total_source_lines}")
    print(f"Coverage Ratio: {coverage_ratio:.1f}%")
    
    # 6. Test categories
    print("\n📋 TEST CATEGORIES")
    print("-" * 40)
    
    test_categories = {
        "ML Baselines": "tests/test_ml_baselines.py",
        "Pretrained Models": "tests/test_pretrained_models.py", 
        "Benchmark Comparison": "tests/test_benchmark_comparison.py",
        "Core Accuracy": "tests/test_accuracy.py",
        "Backend Selection": "tests/test_backends.py",
        "Benchmarking Core": "tests/test_bench.py",
        "Registry System": "tests/test_registry.py"
    }
    
    for category, test_file in test_categories.items():
        if Path(test_file).exists():
            with open(test_file, 'r') as f:
                lines = len(f.readlines())
            print(f"  {category}: {lines} lines")
        else:
            print(f"  {category}: Not found")
    
    # 7. Recommendations
    print("\n💡 RECOMMENDATIONS")
    print("-" * 40)
    
    if coverage_ratio >= 10:
        print("✅ Excellent test coverage!")
    elif coverage_ratio >= 5:
        print("✅ Good test coverage")
    else:
        print("⚠️ Consider adding more tests")
    
    print("\n🎉 TEST COVERAGE REPORT COMPLETE!")
    print("=" * 70)
    
    return {
        "test_files": len(test_files),
        "source_files": len(source_files),
        "test_lines": total_test_lines,
        "source_lines": total_source_lines,
        "coverage_ratio": coverage_ratio
    }

if __name__ == "__main__":
    try:
        results = analyze_test_coverage()
        print(f"\n📊 Final Results: {results}")
    except Exception as e:
        print(f"❌ Error generating coverage report: {e}")
        sys.exit(1)
