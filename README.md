# Kernel Matrix Condition Number Growth and Density Recovery from Option Prices

This repository implements a comprehensive suite of experiments examining the ill-conditioning of kernel matrices in option pricing and demonstrating a robust density recovery method using truncated SVD and L1 regularization.

## Overview

The experiments reproduce results from a quantitative finance research paper investigating:
- How kernel matrix condition numbers grow with the number of option strikes
- Singular value decay patterns in option pricing kernel matrices
- Density recovery from synthetic and real option prices using constrained optimization
- Arbitrage-free interpolation and extrapolation of implied volatility smiles

## Experiments

### Experiment 1: Kernel Matrix Condition Number Growth vs Number of Strikes
Demonstrates that condition number C grows approximately as M^2 (quadratically) with the number of strikes M, motivating the use of truncated SVD.

### Experiment 2: Singular Value Decay of Kernel Matrix vs Grid Resolution
Characterizes the rapid decay of singular values (initial power-law exponent ~2.7), showing that only a small number of singular values carry significant information.

### Experiment 3: Density Recovery from Bachelier (Normal) Option Prices
Validates the full pipeline on synthetic Bachelier option prices, demonstrating chi^2 plateau behavior and optimal lambda selection.

### Experiment 4: Density Recovery from Black-Scholes (Log-Normal) Option Prices
Tests the method on log-normal densities with solver convergence analysis.

### Experiment 5: Density Recovery from Multimodal Mixture Option Prices
Demonstrates recovery of complex three-component mixture densities without parametric assumptions.

### Experiment 6: Density Recovery with Arbitrage-Contaminated Option Prices
Shows graceful handling of arbitrage-contaminated prices by returning valid de-arbitraged densities.

### Experiment 7: Real SPX 500 Data Analysis (Feb 5, 2018)
Applies the method to real market data, reproducing implied volatility smiles with characteristic kinks and demonstrating stable extrapolation.

## Repository Structure

```
.
├── src/
│   ├── __init__.py
│   ├── utils.py              # Core utility functions (kernel matrix, SVD, pricing formulas)
│   ├── exp_1.py              # Experiment 1 implementation
│   ├── exp_2.py              # Experiment 2 implementation
│   ├── exp_3.py              # Experiment 3 implementation
│   ├── exp_4.py              # Experiment 4 implementation
│   ├── exp_5.py              # Experiment 5 implementation
│   ├── exp_6.py              # Experiment 6 implementation
│   └── exp_7.py              # Experiment 7 implementation
├── tests/
│   ├── test_utils.py         # Unit tests for utilities
│   ├── test_exp_1.py         # Tests for Experiment 1
│   ├── test_exp_2.py         # Tests for Experiment 2
│   ├── test_exp_3.py         # Tests for Experiment 3
│   ├── test_exp_4.py         # Tests for Experiment 4
│   ├── test_exp_5.py         # Tests for Experiment 5
│   ├── test_exp_6.py         # Tests for Experiment 6
│   └── test_exp_7.py         # Tests for Experiment 7
├── results/
│   └── RESULTS.md            # Experiment results and metrics
├── data/                     # Data files (if needed)
├── run_all_experiments.py    # Main script to run all experiments
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run all experiments:
```bash
python run_all_experiments.py
```

Run individual experiments:
```bash
python -m src.exp_1
python -m src.exp_2
# ... etc
```

Run tests:
```bash
pytest tests/
```

## Results

All experimental results, metrics, and plots are saved in the `results/` directory. See `results/RESULTS.md` for a comprehensive summary.

## Requirements

- Python 3.8+
- NumPy, SciPy, pandas
- CVXPY (with ECOS and SCS solvers)
- Matplotlib, Seaborn
- pytest (for testing)

## Methodology Adherence

This implementation strictly follows the paper's methodology:
- Exact parameter values from the paper are used
- Custom algorithms are implemented from scratch (no standard library substitutions)
- Mathematical formulas are coded step-by-step as described
- All data processing steps follow the paper precisely
- Trapezoidal integration rules are applied consistently

## Author

Implemented by QCA Agent for quantitative finance research.

## License

MIT License
