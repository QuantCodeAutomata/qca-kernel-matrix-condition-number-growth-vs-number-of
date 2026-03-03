"""
Experiment 5: Density Recovery from Multimodal Mixture Option Prices

Tests density recovery on a challenging multimodal mixture density consisting
of three normal components. Reproduces Example 3 from the paper.

Research objectives:
- Validate recovery of complex multimodal density
- Demonstrate method can recover multi-peaked shapes without parametric assumptions
- Optimal lambda ≈ 10^-8.5
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from typing import Tuple, Dict
from src.utils import (
    build_kernel_matrix, truncated_svd, bachelier_call_price,
    bachelier_put_price, solve_density_recovery, bhattacharyya_distance,
    normalize_density_on_grid
)


def generate_mixture_prices(
    strikes_call: np.ndarray,
    strikes_put: np.ndarray,
    components: list,
    r: float,
    tau: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate option prices from a mixture of Bachelier models.
    
    Parameters:
    -----------
    strikes_call, strikes_put : np.ndarray
        Strike prices
    components : list of dict
        Each dict has keys: 'weight', 'mu', 'sigma'
    r, tau : float
        Risk-free rate and maturity
    
    Returns:
    --------
    prices_call, prices_put : np.ndarray
        Option prices
    """
    prices_call = np.zeros(len(strikes_call))
    prices_put = np.zeros(len(strikes_put))
    
    for comp in components:
        c = comp['weight']
        mu = comp['mu']
        sigma = comp['sigma']
        
        for i, K in enumerate(strikes_call):
            prices_call[i] += c * bachelier_call_price(K, mu, sigma, r, tau)
        
        for i, K in enumerate(strikes_put):
            prices_put[i] += c * bachelier_put_price(K, mu, sigma, r, tau)
    
    return prices_call, prices_put


def run_experiment_5(save_dir: str = 'results') -> dict:
    """Run Experiment 5: Multimodal mixture density recovery."""
    print("=" * 70)
    print("EXPERIMENT 5: Density Recovery from Multimodal Mixture")
    print("=" * 70)
    
    # Parameters from Table I in paper
    components = [
        {'weight': 0.50, 'mu': -0.20, 'sigma': 0.10},
        {'weight': 0.45, 'mu': 0.15, 'sigma': 0.15},
        {'weight': 0.05, 'mu': 0.55, 'sigma': 0.05}
    ]
    r = 0.05
    tau = 1.0
    
    # Verify weights sum to 1
    total_weight = sum(c['weight'] for c in components)
    assert abs(total_weight - 1.0) < 1e-10, f"Weights sum to {total_weight}, not 1.0"
    
    # Generate option prices
    num_calls = 200
    num_puts = 200
    strikes_call = np.linspace(-0.7, 0.7, num_calls)
    strikes_put = np.linspace(-0.7, 0.7, num_puts)
    
    prices_call, prices_put = generate_mixture_prices(
        strikes_call, strikes_put, components, r, tau
    )
    
    # Combine
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
    
    # True density (mixture of normals)
    phi_true = np.zeros(N)
    for comp in components:
        c = comp['weight']
        mu = comp['mu']
        sigma = comp['sigma']
        phi_true += c * norm.pdf(x_grid, loc=mu, scale=sigma)
    
    phi_true = normalize_density_on_grid(phi_true, delta_x)
    
    print(f"\nParameters:")
    print(f"  Mixture components:")
    for i, comp in enumerate(components):
        print(f"    Component {i+1}: weight={comp['weight']}, mu={comp['mu']}, sigma={comp['sigma']}")
    print(f"  r = {r}, tau = {tau}")
    print(f"  Strikes: {M} total")
    print(f"  Density grid: N = {N}, x in [{x_min}, {x_max}]")
    print()
    
    # Build kernel matrix
    G = build_kernel_matrix(strikes, x_grid, delta_x, r, tau, option_types)
    
    # Truncated SVD
    Q = 150
    U_tilde, S_tilde, V_tilde = truncated_svd(G, Q)
    print(f"Truncated SVD: Q = {Q}\n")
    
    # Lambda scan
    lambda_values = np.logspace(-12, 0, 70)
    lambda_highlights = [10**(-8.5), 1e-4]
    
    # Store results
    chi2_values = []
    db_values = []
    phi_recovered = {}
    
    print("Running optimization...")
    for i, lam in enumerate(lambda_values):
        phi, chi2, success = solve_density_recovery(
            prices, U_tilde, S_tilde, V_tilde, lam, delta_x
        )
        
        if success:
            chi2_values.append(chi2)
            db = bhattacharyya_distance(phi, phi_true, delta_x)
            db_values.append(db)
            
            for lh in lambda_highlights:
                if abs(lam - lh) / lh < 0.05:
                    phi_recovered[lam] = phi
            
            if (i + 1) % 15 == 0:
                print(f"  lambda = {lam:.2e}: chi^2 = {chi2:.2e}, d_B = {db:.4f}")
        else:
            chi2_values.append(np.nan)
            db_values.append(np.nan)
    
    chi2_values = np.array(chi2_values)
    db_values = np.array(db_values)
    
    # Filter valid
    valid_mask = ~np.isnan(chi2_values)
    lambda_valid = lambda_values[valid_mask]
    chi2_valid = chi2_values[valid_mask]
    db_valid = db_values[valid_mask]
    
    print(f"\n" + "=" * 70)
    print(f"RESULTS:")
    print(f"  Successful: {valid_mask.sum()}/{len(lambda_values)}")
    print(f"  Min chi^2: {chi2_valid.min():.2e}")
    print(f"  Min d_B: {db_valid.min():.4f}")
    print("=" * 70)
    
    # Plotting
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    # Chi^2 plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.loglog(lambda_valid, chi2_valid, 'b-', linewidth=2)
    for lh in lambda_highlights:
        ax.axvline(lh, color='r', linestyle='--', alpha=0.7, label=f'λ={lh:.2e}')
    ax.set_xlabel('λ', fontsize=12)
    ax.set_ylabel('χ²', fontsize=12)
    ax.set_title('Chi-squared vs Lambda (Multimodal Mixture)', fontsize=14)
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'exp_5_chi2_vs_lambda.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # d_B plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.semilogx(lambda_valid, db_valid, 'g-', linewidth=2)
    for lh in lambda_highlights:
        ax.axvline(lh, color='r', linestyle='--', alpha=0.7, label=f'λ={lh:.2e}')
    ax.set_xlabel('λ', fontsize=12)
    ax.set_ylabel('d_B', fontsize=12)
    ax.set_title('Bhattacharyya Distance vs Lambda (Multimodal)', fontsize=14)
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'exp_5_db_vs_lambda.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Density plots
    if phi_recovered:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        for idx, lam in enumerate(sorted(phi_recovered.keys())):
            if idx >= 2:
                break
            ax = axes[idx]
            phi = phi_recovered[lam]
            ax.plot(x_grid, phi_true, 'k-', linewidth=2, label='True')
            ax.plot(x_grid, phi, 'r--', linewidth=2, label='Recovered')
            ax.set_xlabel('x', fontsize=12)
            ax.set_ylabel('φ(x)', fontsize=12)
            ax.set_title(f'λ = {lam:.2e}', fontsize=13)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'exp_5_recovered_densities.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"\nFigures saved to {save_dir}")
    
    results = {
        'lambda_values': lambda_valid,
        'chi2_values': chi2_valid,
        'db_values': db_valid,
        'phi_true': phi_true,
        'phi_recovered': phi_recovered,
        'parameters': {'components': components, 'r': r, 'tau': tau, 'Q': Q, 'N': N, 'M': M}
    }
    
    return results


if __name__ == '__main__':
    results = run_experiment_5()
    print("\nExperiment 5 completed successfully!")
