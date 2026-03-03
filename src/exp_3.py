"""
Experiment 3: Density Recovery from Bachelier (Normal) Option Prices

Tests the full density recovery pipeline on synthetic option prices generated
from a Bachelier model. Validates chi^2 and Bhattacharyya distance vs lambda.
Reproduces the Normal/Bachelier synthetic example from the paper.

Research objectives:
- Validate truncated SVD + L1 regularization recovers correct density
- Demonstrate chi^2 plateau for small lambda and elbow behavior
- Show overfitting at very small lambda and over-smoothing at large lambda
- Identify optimal lambda near chi^2 elbow
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from typing import Tuple, Dict, List
from src.utils import (
    build_kernel_matrix, truncated_svd, bachelier_call_price,
    bachelier_put_price, solve_density_recovery, bhattacharyya_distance,
    trapezoidal_integral, normalize_density_on_grid
)


def generate_bachelier_prices(
    strikes_call: np.ndarray,
    strikes_put: np.ndarray,
    S0: float,
    sigma: float,
    r: float,
    tau: float
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Generate Bachelier option prices.
    
    Parameters:
    -----------
    strikes_call : np.ndarray
        Call strike prices
    strikes_put : np.ndarray
        Put strike prices
    S0 : float
        Spot price
    sigma : float
        Normal volatility
    r : float
        Risk-free rate
    tau : float
        Time to maturity
    
    Returns:
    --------
    prices_call : np.ndarray
        Call prices
    prices_put : np.ndarray
        Put prices
    F : float
        Forward price
    """
    F = S0 * np.exp(r * tau)
    
    prices_call = np.array([
        bachelier_call_price(K, F, sigma, r, tau) for K in strikes_call
    ])
    
    prices_put = np.array([
        bachelier_put_price(K, F, sigma, r, tau) for K in strikes_put
    ])
    
    return prices_call, prices_put, F


def run_experiment_3(save_dir: str = 'results') -> dict:
    """
    Run Experiment 3: Bachelier density recovery.
    
    Parameters:
    -----------
    save_dir : str
        Directory to save results
    
    Returns:
    --------
    results : dict
        Dictionary containing experiment results
    """
    print("=" * 70)
    print("EXPERIMENT 3: Density Recovery from Bachelier Option Prices")
    print("=" * 70)
    
    # Parameters from paper
    S0 = 0.1
    sigma = 0.1
    r = 0.05
    tau = 1.0
    
    # Generate option prices
    num_calls = 200
    num_puts = 200
    strikes_call = np.linspace(-0.7, 0.7, num_calls)
    strikes_put = np.linspace(-0.7, 0.7, num_puts)
    
    prices_call, prices_put, F = generate_bachelier_prices(
        strikes_call, strikes_put, S0, sigma, r, tau
    )
    
    # Combine strikes and prices
    strikes = np.concatenate([strikes_call, strikes_put])
    prices = np.concatenate([prices_call, prices_put])
    option_types = np.array(['call'] * num_calls + ['put'] * num_puts)
    M = len(prices)
    
    # Density grid
    x_min = -0.9
    x_max = 0.9
    N = 1000
    x_grid = np.linspace(x_min, x_max, N)
    delta_x = (x_max - x_min) / (N - 1)
    
    # True density (Normal)
    mu_true = F
    sigma_true = sigma * np.sqrt(tau)
    phi_true = norm.pdf(x_grid, loc=mu_true, scale=sigma_true)
    phi_true = normalize_density_on_grid(phi_true, delta_x)
    
    print(f"\nParameters:")
    print(f"  S0 = {S0}, sigma = {sigma}, r = {r}, tau = {tau}")
    print(f"  Forward F = {F:.6f}")
    print(f"  True density: Normal(mu={mu_true:.6f}, sigma={sigma_true:.6f})")
    print(f"  Strikes: {num_calls} calls + {num_puts} puts = {M} total")
    print(f"  Strike range: [{strikes.min():.2f}, {strikes.max():.2f}]")
    print(f"  Density grid: N = {N}, x in [{x_min}, {x_max}]")
    print()
    
    # Build kernel matrix
    G = build_kernel_matrix(strikes, x_grid, delta_x, r, tau, option_types)
    
    # Truncated SVD
    Q = 150
    U_tilde, S_tilde, V_tilde = truncated_svd(G, Q)
    print(f"Truncated SVD: Q = {Q}")
    print(f"  U_tilde shape: {U_tilde.shape}")
    print(f"  S_tilde shape: {S_tilde.shape}")
    print(f"  V_tilde shape: {V_tilde.shape}")
    print()
    
    # Lambda scan
    lambda_values = np.logspace(-12, 0, 60)
    
    # Store results
    chi2_values = []
    db_values = []
    phi_recovered = {}
    lambda_highlights = [1e-12, 10**(-7.5)]
    
    print("Running optimization for different lambda values...")
    for i, lam in enumerate(lambda_values):
        phi, chi2, success = solve_density_recovery(
            prices, U_tilde, S_tilde, V_tilde, lam, delta_x
        )
        
        if success:
            chi2_values.append(chi2)
            db = bhattacharyya_distance(phi, phi_true, delta_x)
            db_values.append(db)
            
            # Store highlighted densities
            if any(abs(lam - lh) / lh < 0.05 for lh in lambda_highlights):
                phi_recovered[lam] = phi
            
            if (i + 1) % 10 == 0:
                print(f"  lambda = {lam:.2e}: chi^2 = {chi2:.2e}, d_B = {db:.4f}")
        else:
            print(f"  lambda = {lam:.2e}: Optimization FAILED")
            chi2_values.append(np.nan)
            db_values.append(np.nan)
    
    chi2_values = np.array(chi2_values)
    db_values = np.array(db_values)
    
    # Filter out NaNs
    valid_mask = ~np.isnan(chi2_values)
    lambda_valid = lambda_values[valid_mask]
    chi2_valid = chi2_values[valid_mask]
    db_valid = db_values[valid_mask]
    
    print(f"\n" + "=" * 70)
    print(f"RESULTS:")
    print(f"  Successful optimizations: {valid_mask.sum()}/{len(lambda_values)}")
    print(f"  Min chi^2: {chi2_valid.min():.2e} at lambda = {lambda_valid[chi2_valid.argmin()]:.2e}")
    print(f"  Min d_B: {db_valid.min():.4f} at lambda = {lambda_valid[db_valid.argmin()]:.2e}")
    print("=" * 70)
    
    # Plotting
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    # Plot 1: chi^2 vs lambda
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.loglog(lambda_valid, chi2_valid, 'b-', linewidth=2)
    for lh in lambda_highlights:
        ax.axvline(lh, color='r', linestyle='--', alpha=0.7, label=f'λ = {lh:.2e}')
    ax.set_xlabel('Regularization Parameter (λ)', fontsize=12)
    ax.set_ylabel('Chi-squared (χ²)', fontsize=12)
    ax.set_title('Chi-squared vs Lambda (Bachelier)', fontsize=14)
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'exp_3_chi2_vs_lambda.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: d_B vs lambda
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.semilogx(lambda_valid, db_valid, 'g-', linewidth=2)
    for lh in lambda_highlights:
        ax.axvline(lh, color='r', linestyle='--', alpha=0.7, label=f'λ = {lh:.2e}')
    ax.set_xlabel('Regularization Parameter (λ)', fontsize=12)
    ax.set_ylabel('Bhattacharyya Distance (d_B)', fontsize=12)
    ax.set_title('Bhattacharyya Distance vs Lambda (Bachelier)', fontsize=14)
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'exp_3_db_vs_lambda.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 3: Recovered densities
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for idx, lam in enumerate(sorted(phi_recovered.keys())):
        ax = axes[idx] if idx < 2 else None
        if ax is None:
            continue
        
        phi = phi_recovered[lam]
        ax.plot(x_grid, phi_true, 'k-', linewidth=2, label='True')
        ax.plot(x_grid, phi, 'r--', linewidth=2, label='Recovered')
        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel('Density φ(x)', fontsize=12)
        ax.set_title(f'λ = {lam:.2e}', fontsize=13)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'exp_3_recovered_densities.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nFigures saved to {save_dir}")
    
    results = {
        'lambda_values': lambda_valid,
        'chi2_values': chi2_valid,
        'db_values': db_valid,
        'phi_true': phi_true,
        'phi_recovered': phi_recovered,
        'parameters': {
            'S0': S0, 'sigma': sigma, 'r': r, 'tau': tau,
            'F': F, 'Q': Q, 'N': N, 'M': M
        }
    }
    
    return results


if __name__ == '__main__':
    results = run_experiment_3()
    print("\nExperiment 3 completed successfully!")
