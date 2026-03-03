"""
Experiment 2: Singular Value Decay of Kernel Matrix vs Grid Resolution N

Investigates how singular values of the kernel matrix G decay as a function of
their index, for several choices of density grid resolution N, with fixed M=25.
Reproduces Figure 2 from the paper.

Research objectives:
- Characterize rapid decay of singular values
- Show normalized singular values decay rapidly regardless of N
- Estimate initial power-law decay exponent (paper finds ≈2.7)
- Justify truncation based on rapid decay
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict
from src.utils import build_kernel_matrix


def compute_singular_values_vs_grid_resolution(
    M: int,
    N_values: List[int],
    x_min: float = -1.0,
    x_max: float = 1.0,
    r: float = 0.05,
    tau: float = 1.0
) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """
    Compute singular values of kernel matrices for varying grid resolutions N.
    
    Parameters:
    -----------
    M : int
        Number of strikes (fixed at 25)
    N_values : List[int]
        List of N values (grid resolutions) to test
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
    results : Dict[int, Tuple[np.ndarray, np.ndarray]]
        Dictionary mapping N -> (indices, normalized_singular_values)
    """
    # Fixed strikes
    strikes = np.linspace(x_min, x_max, M)
    
    results = {}
    
    for N in N_values:
        # Build density grid
        x_grid = np.linspace(x_min, x_max, N)
        delta_x = (x_max - x_min) / (N - 1)
        
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
        
        # Normalize by largest singular value
        S_normalized = S / S[0]
        
        # Indices (1-based for plotting)
        indices = np.arange(1, len(S) + 1)
        
        results[N] = (indices, S_normalized)
        
        print(f"N = {N:4d}: {len(S)} singular values, "
              f"s_10/s_1 = {S_normalized[9]:.2e}, s_20/s_1 = {S_normalized[19] if len(S) >= 20 else np.nan:.2e}")
    
    return results


def fit_power_law_decay(
    indices: np.ndarray,
    singular_values: np.ndarray,
    fit_range: Tuple[int, int] = (1, 10)
) -> float:
    """
    Fit power law decay: s_i/s_1 = C * i^(-alpha)  =>  log(s_i/s_1) = -alpha*log(i) + log(C)
    
    Parameters:
    -----------
    indices : np.ndarray
        Singular value indices
    singular_values : np.ndarray
        Normalized singular values (s_i / s_1)
    fit_range : Tuple[int, int]
        Range of indices to fit (inclusive, 1-based)
    
    Returns:
    --------
    alpha : float
        Power law decay exponent
    """
    # Select fitting range
    start_idx = fit_range[0] - 1  # Convert to 0-based
    end_idx = fit_range[1]  # Exclusive in slicing
    
    indices_fit = indices[start_idx:end_idx]
    sv_fit = singular_values[start_idx:end_idx]
    
    # Filter out zeros or near-zeros
    mask = sv_fit > 1e-15
    indices_fit = indices_fit[mask]
    sv_fit = sv_fit[mask]
    
    if len(indices_fit) < 2:
        return np.nan
    
    # Linear fit in log-log space
    log_i = np.log10(indices_fit)
    log_s = np.log10(sv_fit)
    
    coeffs = np.polyfit(log_i, log_s, 1)
    alpha = -coeffs[0]  # Negative slope
    
    return alpha


def plot_singular_value_decay(
    results: Dict[int, Tuple[np.ndarray, np.ndarray]],
    alpha: float,
    save_path: str
) -> None:
    """
    Plot normalized singular values vs index for different N values.
    
    Parameters:
    -----------
    results : Dict[int, Tuple[np.ndarray, np.ndarray]]
        Dictionary mapping N -> (indices, normalized_singular_values)
    alpha : float
        Fitted power law decay exponent
    save_path : str
        Path to save the figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot data for each N
    for N in sorted(results.keys()):
        indices, sv_normalized = results[N]
        ax.semilogy(indices, sv_normalized, 'o-', markersize=6, label=f'N = {N}')
    
    # Plot fitted power law (use first N's indices as reference)
    N_ref = list(sorted(results.keys()))[0]
    indices_ref, _ = results[N_ref]
    i_fit = np.linspace(1, 15, 100)
    sv_fit = i_fit**(-alpha)
    ax.semilogy(i_fit, sv_fit, 'k--', linewidth=2, label=f'Fit: i^(-{alpha:.2f})')
    
    ax.set_xlabel('Singular Value Index (i)', fontsize=12)
    ax.set_ylabel('Normalized Singular Value (s_i / s_1)', fontsize=12)
    ax.set_title('Singular Value Decay vs Grid Resolution N (M=25)', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, which='both', alpha=0.3)
    ax.set_xlim(0.8, min(26, indices_ref[-1] + 1))
    ax.set_ylim(1e-16, 2)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Figure saved to {save_path}")


def run_experiment_2(save_dir: str = 'results') -> dict:
    """
    Run Experiment 2: Singular value decay vs grid resolution.
    
    Parameters:
    -----------
    save_dir : str
        Directory to save results
    
    Returns:
    --------
    results : dict
        Dictionary containing singular value results and alpha
    """
    print("=" * 70)
    print("EXPERIMENT 2: Singular Value Decay vs Grid Resolution N")
    print("=" * 70)
    
    # Parameters from paper
    M = 25
    N_values = [25, 40, 50, 200]
    x_min = -1.0
    x_max = 1.0
    r = 0.05
    tau = 1.0
    
    print(f"\nParameters:")
    print(f"  Number of strikes: M = {M}")
    print(f"  Grid resolutions: N = {N_values}")
    print(f"  Grid range: x in [{x_min}, {x_max}]")
    print(f"  Risk-free rate: r = {r}")
    print(f"  Time to maturity: tau = {tau}")
    print()
    
    # Compute singular values
    sv_results = compute_singular_values_vs_grid_resolution(
        M=M,
        N_values=N_values,
        x_min=x_min,
        x_max=x_max,
        r=r,
        tau=tau
    )
    
    # Fit power law decay (use N=200 for fitting)
    indices_200, sv_200 = sv_results[200]
    alpha = fit_power_law_decay(indices_200, sv_200, fit_range=(2, 10))
    
    print(f"\n" + "=" * 70)
    print(f"RESULTS:")
    print(f"  Fitted power law decay exponent alpha = {alpha:.3f}")
    print(f"  Expected: alpha ≈ 2.7")
    print(f"  Deviation: {abs(alpha - 2.7):.3f}")
    print("=" * 70)
    
    # Plot
    import os
    os.makedirs(save_dir, exist_ok=True)
    plot_path = os.path.join(save_dir, 'exp_2_singular_value_decay.png')
    plot_singular_value_decay(sv_results, alpha, plot_path)
    
    results = {
        'singular_values': sv_results,
        'alpha': alpha,
        'parameters': {
            'M': M,
            'N_values': N_values,
            'x_min': x_min,
            'x_max': x_max,
            'r': r,
            'tau': tau
        }
    }
    
    return results


if __name__ == '__main__':
    results = run_experiment_2()
    print("\nExperiment 2 completed successfully!")
