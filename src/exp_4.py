"""
Experiment 4: Density Recovery from Black-Scholes (Log-Normal) Option Prices

Tests the density recovery pipeline on synthetic option prices generated from
the Black-Scholes model. Validates with solver convergence analysis.
Reproduces the Log-Normal/Black-Scholes synthetic example from the paper.

Research objectives:
- Validate density recovery from BS prices
- Demonstrate optimal lambda ≈ 10^-7.5 for log-normal case
- Document solver convergence issues at very small lambda
- Verify non-negativity and normalization
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import lognorm
from typing import Tuple, Dict
from src.utils import (
    build_kernel_matrix, truncated_svd, black_scholes_call_price,
    black_scholes_put_price, solve_density_recovery, bhattacharyya_distance,
    trapezoidal_integral, normalize_density_on_grid
)


def generate_black_scholes_prices(
    strikes_call: np.ndarray,
    strikes_put: np.ndarray,
    S0: float,
    sigma: float,
    r: float,
    tau: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate Black-Scholes option prices."""
    prices_call = np.array([
        black_scholes_call_price(K, S0, sigma, r, tau) for K in strikes_call
    ])
    
    prices_put = np.array([
        black_scholes_put_price(K, S0, sigma, r, tau) for K in strikes_put
    ])
    
    return prices_call, prices_put


def run_experiment_4(save_dir: str = 'results') -> dict:
    """Run Experiment 4: Black-Scholes density recovery."""
    print("=" * 70)
    print("EXPERIMENT 4: Density Recovery from Black-Scholes Option Prices")
    print("=" * 70)
    
    # Parameters from paper
    S0 = 0.5
    sigma = 0.2
    r = 0.0
    tau = 1.0
    
    # Generate option prices
    num_calls = 200
    num_puts = 200
    strikes_call = np.linspace(0.01, 1.0, num_calls)
    strikes_put = np.linspace(0.01, 1.0, num_puts)
    
    prices_call, prices_put = generate_black_scholes_prices(
        strikes_call, strikes_put, S0, sigma, r, tau
    )
    
    # Combine
    strikes = np.concatenate([strikes_call, strikes_put])
    prices = np.concatenate([prices_call, prices_put])
    option_types = np.array(['call'] * num_calls + ['put'] * num_puts)
    M = len(prices)
    
    # Density grid
    x_min = 0.0
    x_max = 1.5
    N = 1000
    x_grid = np.linspace(x_min, x_max, N)
    delta_x = (x_max - x_min) / (N - 1)
    
    # True density (log-normal)
    mu_ln = np.log(S0) + (r - 0.5 * sigma**2) * tau
    sigma_ln = sigma * np.sqrt(tau)
    
    phi_true = np.zeros(N)
    for i in range(N):
        if x_grid[i] > 0:
            phi_true[i] = (1 / (x_grid[i] * sigma_ln * np.sqrt(2 * np.pi))) * \
                          np.exp(-0.5 * ((np.log(x_grid[i]) - mu_ln) / sigma_ln)**2)
    
    phi_true = normalize_density_on_grid(phi_true, delta_x)
    
    print(f"\nParameters:")
    print(f"  S0 = {S0}, sigma = {sigma}, r = {r}, tau = {tau}")
    print(f"  True density: Log-Normal(mu_ln={mu_ln:.6f}, sigma_ln={sigma_ln:.6f})")
    print(f"  Strikes: {M} total ({num_calls} calls + {num_puts} puts)")
    print(f"  Density grid: N = {N}, x in [{x_min}, {x_max}]")
    print()
    
    # Build kernel matrix
    G = build_kernel_matrix(strikes, x_grid, delta_x, r, tau, option_types)
    
    # Truncated SVD
    Q = 150
    U_tilde, S_tilde, V_tilde = truncated_svd(G, Q)
    print(f"Truncated SVD: Q = {Q}\n")
    
    # Lambda scan
    lambda_values = np.logspace(-12, 0, 60)
    
    # Store results
    chi2_values = []
    db_values = []
    phi_recovered = {}
    lambda_highlight = 10**(-7.5)
    failed_lambdas = []
    
    print("Running optimization...")
    for lam in lambda_values:
        phi, chi2, success = solve_density_recovery(
            prices, U_tilde, S_tilde, V_tilde, lam, delta_x
        )
        
        if success:
            chi2_values.append(chi2)
            db = bhattacharyya_distance(phi, phi_true, delta_x)
            db_values.append(db)
            
            if abs(lam - lambda_highlight) / lambda_highlight < 0.05:
                phi_recovered[lam] = phi
        else:
            failed_lambdas.append(lam)
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
    print(f"  Failed lambdas: {len(failed_lambdas)}")
    if failed_lambdas:
        print(f"  Failed at: {failed_lambdas[:5]}..." if len(failed_lambdas) > 5 else f"  Failed at: {failed_lambdas}")
    print(f"  Min chi^2: {chi2_valid.min():.2e}")
    print(f"  Min d_B: {db_valid.min():.4f}")
    print("=" * 70)
    
    # Plotting
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    # Chi^2 plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.loglog(lambda_valid, chi2_valid, 'b-', linewidth=2)
    ax.axvline(lambda_highlight, color='r', linestyle='--', label=f'λ = {lambda_highlight:.2e}')
    ax.set_xlabel('λ', fontsize=12)
    ax.set_ylabel('χ²', fontsize=12)
    ax.set_title('Chi-squared vs Lambda (Black-Scholes)', fontsize=14)
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'exp_4_chi2_vs_lambda.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # d_B plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.semilogx(lambda_valid, db_valid, 'g-', linewidth=2)
    ax.axvline(lambda_highlight, color='r', linestyle='--', label=f'λ = {lambda_highlight:.2e}')
    ax.set_xlabel('λ', fontsize=12)
    ax.set_ylabel('d_B', fontsize=12)
    ax.set_title('Bhattacharyya Distance vs Lambda (Black-Scholes)', fontsize=14)
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'exp_4_db_vs_lambda.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Density plot
    if phi_recovered:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(x_grid, phi_true, 'k-', linewidth=2, label='True')
        for lam, phi in phi_recovered.items():
            ax.plot(x_grid, phi, '--', linewidth=2, label=f'λ={lam:.2e}')
        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel('φ(x)', fontsize=12)
        ax.set_title('Recovered Density (Black-Scholes)', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'exp_4_recovered_density.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"\nFigures saved to {save_dir}")
    
    results = {
        'lambda_values': lambda_valid,
        'chi2_values': chi2_valid,
        'db_values': db_valid,
        'phi_true': phi_true,
        'phi_recovered': phi_recovered,
        'failed_lambdas': failed_lambdas,
        'parameters': {'S0': S0, 'sigma': sigma, 'r': r, 'tau': tau, 'Q': Q, 'N': N, 'M': M}
    }
    
    return results


if __name__ == '__main__':
    results = run_experiment_4()
    print("\nExperiment 4 completed successfully!")
