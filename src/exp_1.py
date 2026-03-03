"""
Experiment 1: Kernel Matrix Condition Number Growth vs Number of Strikes

Investigates how the condition number of the kernel matrix G grows as a function
of the number of strikes M. Reproduces Figure 1 from the paper.

Research objectives:
- Demonstrate ill-conditioning of kernel matrix G
- Quantify growth rate of condition number as function of M
- Verify quadratic growth (exponent k ≈ 2)
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List
from src.utils import build_kernel_matrix


def compute_condition_number_vs_strikes(
    M_values: np.ndarray,
    N: int = 10000,
    x_min: float = -1.0,
    x_max: float = 1.0,
    r: float = 0.05,
    tau: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute condition numbers of kernel matrices for varying numbers of strikes.
    
    Parameters:
    -----------
    M_values : np.ndarray
        Array of M values (number of strikes) to test
    N : int
        Number of density grid points (fixed at 10000)
    x_min : float
        Minimum grid value
    x_max : float
        Maximum grid value
    r : float
        Risk-free rate
    tau : float
        Time to maturity
    
    Returns:
    --------
    M_values : np.ndarray
        Input M values
    condition_numbers : np.ndarray
        Condition numbers for each M
    """
    # Build density grid (fixed)
    x_grid = np.linspace(x_min, x_max, N)
    delta_x = (x_max - x_min) / (N - 1)
    
    condition_numbers = np.zeros(len(M_values))
    
    for idx, M in enumerate(M_values):
        # Create M equidistant strikes
        strikes = np.linspace(x_min, x_max, M)
        
        # Build kernel matrix (all calls)
        G = build_kernel_matrix(
            strikes=strikes,
            x_grid=x_grid,
            delta_x=delta_x,
            r=r,
            tau=tau,
            option_types=None  # All calls
        )
        
        # Compute SVD
        U, S, Vh = np.linalg.svd(G, full_matrices=False)
        
        # Condition number = largest / smallest non-zero singular value
        # G has shape (M, N) with M << N, so rank is at most M
        # Filter out very small (near-zero) singular values
        tol = 1e-10 * S[0]
        S_nonzero = S[S > tol]
        
        if len(S_nonzero) > 0:
            cond_number = S_nonzero[0] / S_nonzero[-1]
        else:
            cond_number = np.inf
        
        condition_numbers[idx] = cond_number
        
        print(f"M = {M:4d}, Condition Number = {cond_number:.2e}")
    
    return M_values, condition_numbers


def fit_power_law(
    M_values: np.ndarray,
    condition_numbers: np.ndarray
) -> Tuple[float, float]:
    """
    Fit power law: C = A * M^k  =>  log(C) = k*log(M) + log(A)
    
    Parameters:
    -----------
    M_values : np.ndarray
        Number of strikes
    condition_numbers : np.ndarray
        Condition numbers
    
    Returns:
    --------
    k : float
        Power law exponent
    A : float
        Power law coefficient
    """
    # Linear fit in log-log space
    log_M = np.log10(M_values)
    log_C = np.log10(condition_numbers)
    
    coeffs = np.polyfit(log_M, log_C, 1)
    k = coeffs[0]  # Slope
    log_A = coeffs[1]  # Intercept
    A = 10**log_A
    
    return k, A


def plot_condition_number_vs_strikes(
    M_values: np.ndarray,
    condition_numbers: np.ndarray,
    k: float,
    A: float,
    save_path: str
) -> None:
    """
    Plot condition number vs number of strikes in log-log scale with fitted line.
    
    Parameters:
    -----------
    M_values : np.ndarray
        Number of strikes
    condition_numbers : np.ndarray
        Condition numbers
    k : float
        Fitted power law exponent
    A : float
        Fitted power law coefficient
    save_path : str
        Path to save the figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot data
    ax.loglog(M_values, condition_numbers, 'o', markersize=8, label='Computed')
    
    # Plot fitted line
    M_fit = np.logspace(np.log10(M_values.min()), np.log10(M_values.max()), 100)
    C_fit = A * M_fit**k
    ax.loglog(M_fit, C_fit, 'r--', linewidth=2, label=f'Fit: C ∝ M^{k:.2f}')
    
    ax.set_xlabel('Number of Strikes (M)', fontsize=12)
    ax.set_ylabel('Condition Number (C)', fontsize=12)
    ax.set_title('Kernel Matrix Condition Number vs Number of Strikes', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, which='both', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Figure saved to {save_path}")


def run_experiment_1(save_dir: str = 'results') -> dict:
    """
    Run Experiment 1: Condition number growth vs number of strikes.
    
    Parameters:
    -----------
    save_dir : str
        Directory to save results
    
    Returns:
    --------
    results : dict
        Dictionary containing M_values, condition_numbers, k, A
    """
    print("=" * 70)
    print("EXPERIMENT 1: Kernel Matrix Condition Number Growth vs Number of Strikes")
    print("=" * 70)
    
    # Parameters from paper
    N = 10000
    x_min = -1.0
    x_max = 1.0
    r = 0.05
    tau = 1.0
    
    # Log-spaced M values from 10 to ~1000
    M_values = np.unique(np.logspace(np.log10(10), np.log10(979), 18).astype(int))
    
    print(f"\nParameters:")
    print(f"  Density grid: N = {N}, x in [{x_min}, {x_max}]")
    print(f"  Risk-free rate: r = {r}")
    print(f"  Time to maturity: tau = {tau}")
    print(f"  Number of M values: {len(M_values)}")
    print(f"  M range: [{M_values.min()}, {M_values.max()}]")
    print()
    
    # Compute condition numbers
    M_values, condition_numbers = compute_condition_number_vs_strikes(
        M_values=M_values,
        N=N,
        x_min=x_min,
        x_max=x_max,
        r=r,
        tau=tau
    )
    
    # Fit power law
    k, A = fit_power_law(M_values, condition_numbers)
    
    print(f"\n" + "=" * 70)
    print(f"RESULTS:")
    print(f"  Fitted power law exponent k = {k:.3f}")
    print(f"  Fitted coefficient A = {A:.3e}")
    print(f"  Expected: k ≈ 2 (quadratic growth)")
    print(f"  Deviation from quadratic: {abs(k - 2.0):.3f}")
    print("=" * 70)
    
    # Plot
    import os
    os.makedirs(save_dir, exist_ok=True)
    plot_path = os.path.join(save_dir, 'exp_1_condition_number_vs_strikes.png')
    plot_condition_number_vs_strikes(M_values, condition_numbers, k, A, plot_path)
    
    results = {
        'M_values': M_values,
        'condition_numbers': condition_numbers,
        'k': k,
        'A': A,
        'parameters': {
            'N': N,
            'x_min': x_min,
            'x_max': x_max,
            'r': r,
            'tau': tau
        }
    }
    
    return results


if __name__ == '__main__':
    results = run_experiment_1()
    print("\nExperiment 1 completed successfully!")
