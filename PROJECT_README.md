
# Our Work

## Modified Files

The following files are additions to the original codebase:

- **`test_all_optimizations.py`**: Comprehensive test suite that benchmarks baseline, I1, I2, and I1+I2 implementations with customizable input data and K.
- **`data_simulation/data.csv`**: Sample CSV file containing 200 data points for testing the changepoint detection algorithm.
- **`core_dp_si/fixed_k_dp_optimized_1.py`**: Optimized implementation with Improvement 1 (selective constraint storage) that reduces memory usage by storing only competitive alternatives.
- **`core_dp_si/fixed_k_dp_optimized_2.py`**: Optimized implementation with both Improvement 1 and Improvement 2 (vectorized SSE computation) that combines selective storage with vectorized operations for maximum performance.


# Test Suite Usage Guide

## Overview

`test_all_optimizations.py` is a comprehensive test suite that evaluates the performance of different optimization levels for the dynamic programming changepoint detection algorithm. It compares:

- **Baseline**: Original implementation (no optimizations)
- **Improvement 1 (I1)**: Selective constraint storage only
- **Improvement 2 (I2)**: Vectorized SSE computation only
- **I1+I2 Combined**: Both optimizations together

## Usage

### Method 1: Command-Line with Custom Input

Test with your own data file or data array:

```bash
# Using a CSV file
python test_all_optimizations.py data.csv 3 5

# Using comma-separated values directly
python test_all_optimizations.py "1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0" 2 3
```

**Arguments:**
- `data_file_or_array`: Path to CSV file OR comma-separated values as a string
- `K`: Number of segments (integer)
- `num_trials`: (Optional) Number of trials per implementation (default: 5)

**Example:**
```bash
# Test with data.csv, 3 segments, 5 trials each
python test_all_optimizations.py data.csv 3 5

# Test with data.csv, 4 segments, default 5 trials
python test_all_optimizations.py data.csv 4
```

### Method 2: Python API

Use the test function directly in Python:

```python
from test_all_optimizations import test_custom_input
import numpy as np

# Create or load your data
data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
K = 3
num_trials = 5

# Run tests
results = test_custom_input(data, K, num_trials=num_trials, save_output=True)

# Access results
print(f"Baseline time: {results['baseline']['avg_time']:.4f}s")
print(f"I1 speedup: {results['summary']['i1_speedup']:.2f}x")
print(f"I2 speedup: {results['summary']['i2_speedup']:.2f}x")
print(f"I1+I2 speedup: {results['summary']['i1_i2_speedup']:.2f}x")
```

### Method 3: Default Comprehensive Test Suite

Run the default test suite with predefined test cases (no arguments):

```bash
python test_all_optimizations.py
```

This runs three test cases:
- **Small**: n=50, K=3
- **Medium**: n=100, K=3
- **Large**: n=200, K=4

## Input Data Format

### CSV File Format

The test suite accepts CSV files with comma-separated numeric values. Two formats are supported:

#### Format 1: Single Row (Recommended)
```csv
1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0
```
- One row with all values separated by commas
- Loaded directly as a 1D array

#### Format 2: Multiple Rows
```csv
1.0,2.0,3.0,4.0
5.0,6.0,7.0,8.0
9.0,10.0,11.0,12.0
```
- Multiple rows, each with comma-separated values
- Automatically flattened into a 1D array (row by row)

### Data Requirements

- **Type**: Floating-point numbers (integers are also accepted)
- **Format**: Comma-separated values (CSV)
- **Encoding**: Standard text encoding (UTF-8)
- **No headers**: The CSV should contain only numeric data, no column names or headers

### Example CSV File

```csv
2.283,1.456,0.852,1.373,2.228,1.333,1.407,1.652,0.660,1.709
```

Or with scientific notation:
```csv
2.283349757704839167e+00,-5.206912058464294724e-01,1.036117548532579091e-01
```

### Loading Data from File

The test suite uses `numpy.loadtxt()` to read CSV files:

```python
# The code automatically handles:
data = np.loadtxt('data.csv', delimiter=',')
# Flattens multi-dimensional arrays if needed
if data.ndim > 1:
    data = data.flatten()
```

## Output

### Output Files

Test results are saved to the `outputs/` folder:

- **Custom tests**: `outputs/test_results_custom_{timestamp}.txt`
- **Comprehensive suite**: `outputs/test_results_{timestamp}.txt`

The `outputs/` folder is created automatically if it doesn't exist.

### Output Format

Each test result file contains:

1. **Test Information**:
   - Data size (n)
   - Number of segments (K)
   - Number of trials
   - Test date and time

2. **Results for Each Implementation**:
   - Execution time (average ± standard deviation)
   - Number of constraints generated
   - Estimated memory usage
   - Detected changepoints

3. **Summary Table**:
   - Comparison of all implementations
   - Speedup relative to baseline
   - Constraint counts
   - Memory usage

### Example Output

```
======================================================================
TEST SUITE: All Optimization Improvement Levels
======================================================================
Data size: n=200
Number of segments: K=4
Number of trials per implementation: 5
======================================================================

======================================================================
TEST 1: Original Implementation (Baseline)
======================================================================
Original
----------------------------------------------------------------------
  Trials: 5
  Execution time: 13.7493 ± 0.0894 seconds
  Changepoints detected: [50, 100, 150]
  Number of constraints: 39798
  Estimated constraint storage: 12145.69 MB

======================================================================
SUMMARY: Performance Comparison
======================================================================

Implementation                      Time (s)        Speedup    Constraints  Memory (MB) 
----------------------------------------------------------------------
Original                             13.7493±0.0894   1.00×    39798        12145.69
Improvement 1 (I1 - competitive 1%)  8.1482±0.0556   1.69×    445            136.11
Improvement 2 (I2 - vectorized)     12.5059±0.2449   1.10×    39798        12145.69
I1+I2 Combined                       0.2633±0.0176  52.22×    445            136.11
```

## Return Value (Python API)

When using the Python API, `test_custom_input()` returns a dictionary:

```python
results = {
    'baseline': {
        'name': 'Original',
        'avg_time': 0.0338,
        'std_time': 0.0050,
        'changepoints': [...],
        'num_constraints': 1273,
        'memory': {...}
    },
    'i1': {...},      # Improvement 1 results
    'i2': {...},      # Improvement 2 results
    'i1_i2': {...},   # Combined results
    'summary': {
        'data_size': 200,
        'num_segments': 4,
        'baseline_time': 13.7493,
        'i1_time': 8.1482,
        'i2_time': 12.5059,
        'i1_i2_time': 0.2633,
        'i1_speedup': 1.69,
        'i2_speedup': 1.10,
        'i1_i2_speedup': 52.22,
        'baseline_constraints': 39798,
        'i1_constraints': 445,
        'i2_constraints': 39798,
        'i1_i2_constraints': 445
    },
    'output_file': 'outputs/test_results_custom_20260103_163229.txt'
}
```

## What Gets Tested

### Baseline (Original Implementation)
- No optimizations
- Stores all constraint matrices
- Sequential SSE computation
- Reference for correctness and performance

### Improvement 1 (I1) - Selective Constraint Storage
- Stores only competitive alternatives (within 1% threshold of optimal)
- Reduces memory usage significantly
- Same time complexity (all matrices still computed)

### Improvement 2 (I2) - Vectorization
- Vectorized SSE computation using NumPy
- Faster execution due to optimized C-level operations
- Stores all constraints (for comparison)

### I1+I2 Combined
- Both optimizations together
- Maximum performance improvement
- Reduced memory and faster execution

## Correctness Verification

All optimized implementations are verified against the baseline:
- ✅ Changepoint positions match
- ✅ Segment assignments match
- ✅ Constraint matrices match (when same count)

## Requirements

- Python 3.7.6 or higher
- NumPy 1.15.0 or higher
- The `core_dp_si` module with all implementations

## Examples

### Example 1: Test with CSV file
```bash
python test_all_optimizations.py data_simulation/data.csv 3 5
```

### Example 2: Test with inline data
```bash
python test_all_optimizations.py "1,2,3,4,5,6,7,8,9,10" 2 3
```

### Example 3: Python script
```python
import numpy as np
from test_all_optimizations import test_custom_input

# Load your data
data = np.loadtxt('my_data.csv', delimiter=',')

# Run tests
results = test_custom_input(data, K=4, num_trials=10)

# Print summary
s = results['summary']
print(f"Best speedup: {s['i1_i2_speedup']:.2f}x")
print(f"Memory reduction: {(1 - s['i1_i2_constraints']/s['baseline_constraints'])*100:.1f}%")
```

## Notes

- The test suite measures **actual execution time**, not estimated performance
- Memory usage is **estimated** based on data structure sizes
- All tests use the same random seed for reproducibility (when using `generate_test_data`)
- Results are saved automatically to the `outputs/` folder
- Custom tests and comprehensive suite use different output filenames to avoid confusion

