"""
Core utility functions for kernel matrix operations and option pricing.

This module implements the fundamental building blocks used across all experiments:
- Kernel matrix construction for call and put options
- SVD computation and truncation
- Option pricing formulas (Bachelier, Black-Scholes)
- Density metrics (Bhattacharyya distance)
- Integration utilities (trapezoidal rule)
"""

import numpy as np
from scipy.stats import norm
from typing import Tuple, Optional
import cvxpy as cp


def build_kernel_matrix(
    strikes: np.ndarray,
    x_grid: np.ndarray,
    delta_x: float,
    r: float,
    tau: float,
    option_types: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Build the kernel matrix G linking discretized terminal density to option prices.
    
    For call options: g_{ij} = Delta_x * exp(-r*tau) * max(0, x_j - K_i)
    For put options: g_{ij} = Delta_x * exp(-r*tau) * max(0, K_i - x_j)
    
    Applies trapezoidal endpoint correction: multiply first and last columns by 0.5.
    
    Parameters:
    -----------
    strikes : np.ndarray
        Array of strike prices, shape (M,)
    x_grid : np.ndarray
        Density grid points, shape (N,)
    delta_x : float
        Grid spacing
    r : float
        Risk-free rate
    tau : float
        Time to maturity
    option_types : Optional[np.ndarray]
        Array of option types ('call' or 'put'), shape (M,). If None, all calls.
    
    Returns:
    --------
    G : np.ndarray
        Kernel matrix, shape (M, N)
    """
    M = len(strikes)
    N = len(x_grid)
    G = np.zeros((M, N))
    
    discount = np.exp(-r * tau)
    
    if option_types is None:
        option_types = np.array(['call'] * M)
    
    for i in range(M):
        for j in range(N):
            if option_types[i] == 'call':
                G[i, j] = delta_x * discount * max(0.0, x_grid[j] - strikes[i])
            else:  # put
                G[i, j] = delta_x * discount * max(0.0, strikes[i] - x_grid[j])
    
    # Trapezoidal endpoint correction
    G[:, 0] *= 0.5
    G[:, -1] *= 0.5
    
    return G


def truncated_svd(G: np.ndarray, Q: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute truncated SVD of kernel matrix G, retaining Q leading singular values.
    
    G = U @ diag(S) @ Vh
    
    Parameters:
    -----------
    G : np.ndarray
        Kernel matrix, shape (M, N)
    Q : int
        Number of singular values to retain
    
    Returns:
    --------
    U_tilde : np.ndarray
        Left singular vectors, shape (M, Q)
    S_tilde : np.ndarray
        Singular values, shape (Q,)
    V_tilde : np.ndarray
        Right singular vectors (transposed), shape (Q, N)
    """
    U, S, Vh = np.linalg.svd(G, full_matrices=False)
    
    # Truncate to Q components
    U_tilde = U[:, :Q]
    S_tilde = S[:Q]
    V_tilde = Vh[:Q, :]
    
    return U_tilde, S_tilde, V_tilde


def bachelier_call_price(K: float, F: float, sigma: float, r: float, tau: float) -> float:
    """
    Compute Bachelier (normal model) call option price.
    
    C = exp(-r*tau) * [(F-K)*Phi(m) + sigma*sqrt(tau)*phi(m)]
    where m = (F - K) / (sigma * sqrt(tau))
    
    Parameters:
    -----------
    K : float
        Strike price
    F : float
        Forward price
    sigma : float
        Volatility (normal model)
    r : float
        Risk-free rate
    tau : float
        Time to maturity
    
    Returns:
    --------
    price : float
        Call option price
    """
    sigma_tau = sigma * np.sqrt(tau)
    
    if sigma_tau < 1e-10:
        # Degenerate case: zero volatility
        return np.exp(-r * tau) * max(0.0, F - K)
    
    m = (F - K) / sigma_tau
    
    call_price = np.exp(-r * tau) * (
        (F - K) * norm.cdf(m) + sigma_tau * norm.pdf(m)
    )
    
    return call_price


def bachelier_put_price(K: float, F: float, sigma: float, r: float, tau: float) -> float:
    """
    Compute Bachelier (normal model) put option price.
    
    P = exp(-r*tau) * [(K-F)*Phi(-m) + sigma*sqrt(tau)*phi(m)]
    where m = (F - K) / (sigma * sqrt(tau))
    
    Parameters:
    -----------
    K : float
        Strike price
    F : float
        Forward price
    sigma : float
        Volatility (normal model)
    r : float
        Risk-free rate
    tau : float
        Time to maturity
    
    Returns:
    --------
    price : float
        Put option price
    """
    sigma_tau = sigma * np.sqrt(tau)
    
    if sigma_tau < 1e-10:
        # Degenerate case: zero volatility
        return np.exp(-r * tau) * max(0.0, K - F)
    
    m = (F - K) / sigma_tau
    
    put_price = np.exp(-r * tau) * (
        (K - F) * norm.cdf(-m) + sigma_tau * norm.pdf(m)
    )
    
    return put_price


def black_scholes_call_price(K: float, S0: float, sigma: float, r: float, tau: float) -> float:
    """
    Compute Black-Scholes call option price.
    
    C = S0*Phi(d1) - K*exp(-r*tau)*Phi(d2)
    where d1 = (ln(S0/K) + (r + sigma^2/2)*tau) / (sigma*sqrt(tau))
          d2 = d1 - sigma*sqrt(tau)
    
    Parameters:
    -----------
    K : float
        Strike price
    S0 : float
        Spot price
    sigma : float
        Volatility (log-normal model)
    r : float
        Risk-free rate
    tau : float
        Time to maturity
    
    Returns:
    --------
    price : float
        Call option price
    """
    if sigma * np.sqrt(tau) < 1e-10:
        # Degenerate case
        return max(0.0, S0 * np.exp(r * tau) - K) * np.exp(-r * tau)
    
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)
    
    call_price = S0 * norm.cdf(d1) - K * np.exp(-r * tau) * norm.cdf(d2)
    
    return call_price


def black_scholes_put_price(K: float, S0: float, sigma: float, r: float, tau: float) -> float:
    """
    Compute Black-Scholes put option price.
    
    P = K*exp(-r*tau)*Phi(-d2) - S0*Phi(-d1)
    
    Parameters:
    -----------
    K : float
        Strike price
    S0 : float
        Spot price
    sigma : float
        Volatility (log-normal model)
    r : float
        Risk-free rate
    tau : float
        Time to maturity
    
    Returns:
    --------
    price : float
        Put option price
    """
    if sigma * np.sqrt(tau) < 1e-10:
        # Degenerate case
        return max(0.0, K - S0 * np.exp(r * tau)) * np.exp(-r * tau)
    
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)
    
    put_price = K * np.exp(-r * tau) * norm.cdf(-d2) - S0 * norm.cdf(-d1)
    
    return put_price


def black_call_price(K: float, F: float, sigma: float, r: float, tau: float) -> float:
    """
    Compute Black (1976) model call option price (used for forward-based pricing).
    
    C = exp(-r*tau) * [F*Phi(d1) - K*Phi(d2)]
    where d1 = (ln(F/K) + 0.5*sigma^2*tau) / (sigma*sqrt(tau))
          d2 = d1 - sigma*sqrt(tau)
    
    Parameters:
    -----------
    K : float
        Strike price
    F : float
        Forward price
    sigma : float
        Volatility
    r : float
        Risk-free rate
    tau : float
        Time to maturity
    
    Returns:
    --------
    price : float
        Call option price
    """
    if sigma * np.sqrt(tau) < 1e-10:
        return np.exp(-r * tau) * max(0.0, F - K)
    
    d1 = (np.log(F / K) + 0.5 * sigma**2 * tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)
    
    call_price = np.exp(-r * tau) * (F * norm.cdf(d1) - K * norm.cdf(d2))
    
    return call_price


def black_put_price(K: float, F: float, sigma: float, r: float, tau: float) -> float:
    """
    Compute Black (1976) model put option price.
    
    P = exp(-r*tau) * [K*Phi(-d2) - F*Phi(-d1)]
    
    Parameters:
    -----------
    K : float
        Strike price
    F : float
        Forward price
    sigma : float
        Volatility
    r : float
        Risk-free rate
    tau : float
        Time to maturity
    
    Returns:
    --------
    price : float
        Put option price
    """
    if sigma * np.sqrt(tau) < 1e-10:
        return np.exp(-r * tau) * max(0.0, K - F)
    
    d1 = (np.log(F / K) + 0.5 * sigma**2 * tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)
    
    put_price = np.exp(-r * tau) * (K * norm.cdf(-d2) - F * norm.cdf(-d1))
    
    return put_price


def trapezoidal_integral(f: np.ndarray, delta_x: float) -> float:
    """
    Compute trapezoidal integral of function values on uniform grid.
    
    Integral = (0.5*f[0] + sum(f[1:-1]) + 0.5*f[-1]) * delta_x
    
    Parameters:
    -----------
    f : np.ndarray
        Function values on grid
    delta_x : float
        Grid spacing
    
    Returns:
    --------
    integral : float
        Approximate integral
    """
    integral = (0.5 * f[0] + np.sum(f[1:-1]) + 0.5 * f[-1]) * delta_x
    return integral


def bhattacharyya_distance(
    phi1: np.ndarray,
    phi2: np.ndarray,
    delta_x: float
) -> float:
    """
    Compute Bhattacharyya distance between two discrete probability densities.
    
    d_B = -log(integral(sqrt(phi1 * phi2) * dx))
    
    Uses trapezoidal rule for integration.
    
    Parameters:
    -----------
    phi1 : np.ndarray
        First density, shape (N,)
    phi2 : np.ndarray
        Second density, shape (N,)
    delta_x : float
        Grid spacing
    
    Returns:
    --------
    distance : float
        Bhattacharyya distance (can be negative if densities not normalized)
    """
    # Element-wise sqrt of product
    sqrt_product = np.sqrt(np.maximum(0.0, phi1 * phi2))
    
    # Trapezoidal integral
    overlap = trapezoidal_integral(sqrt_product, delta_x)
    
    # Distance
    # Note: can be negative if overlap > 1 (happens when densities not normalized)
    distance = -np.log(overlap) if overlap > 0 else np.inf
    
    return distance


def solve_density_recovery(
    prices: np.ndarray,
    U_tilde: np.ndarray,
    S_tilde: np.ndarray,
    V_tilde: np.ndarray,
    lambda_reg: float,
    delta_x: float,
    solver: str = 'ECOS',
    fallback_solver: str = 'SCS'
) -> Tuple[Optional[np.ndarray], Optional[float], bool]:
    """
    Solve density recovery optimization problem using CVXPY.
    
    minimize: (1/2)*||Pr_prime - S_tilde @ phi_prime||_2^2 + lambda*||phi_prime||_1
    subject to: V_tilde.T @ phi_prime >= 0  (non-negativity)
                integral(V_tilde.T @ phi_prime) == 1  (normalization)
    
    Parameters:
    -----------
    prices : np.ndarray
        Option prices, shape (M,)
    U_tilde : np.ndarray
        Left singular vectors, shape (M, Q)
    S_tilde : np.ndarray
        Singular values, shape (Q,)
    V_tilde : np.ndarray
        Right singular vectors (transposed), shape (Q, N)
    lambda_reg : float
        L1 regularization parameter
    delta_x : float
        Grid spacing for normalization constraint
    solver : str
        Primary CVXPY solver
    fallback_solver : str
        Fallback solver if primary fails
    
    Returns:
    --------
    phi : Optional[np.ndarray]
        Recovered density on grid, shape (N,). None if optimization fails.
    chi2 : Optional[float]
        Chi-squared value. None if optimization fails.
    success : bool
        Whether optimization succeeded
    """
    Q = len(S_tilde)
    N = V_tilde.shape[1]
    
    # Transformed prices
    Pr_prime = U_tilde.T @ prices
    
    # Decision variable
    phi_prime = cp.Variable(Q)
    
    # Reconstructed density
    phi = V_tilde.T @ phi_prime
    
    # Objective function
    S_diag = np.diag(S_tilde)
    objective = cp.Minimize(
        0.5 * cp.sum_squares(Pr_prime - S_diag @ phi_prime) +
        lambda_reg * cp.norm1(phi_prime)
    )
    
    # Constraints
    constraints = [
        phi >= 0,  # Non-negativity
        (0.5 * (phi[0] + phi[-1]) + cp.sum(phi[1:-1])) * delta_x == 1  # Normalization
    ]
    
    # Solve
    problem = cp.Problem(objective, constraints)
    
    try:
        problem.solve(solver=solver)
        if problem.status not in ['optimal', 'optimal_inaccurate']:
            raise Exception(f"Solver status: {problem.status}")
    except:
        # Try fallback solver
        try:
            problem.solve(solver=fallback_solver)
            if problem.status not in ['optimal', 'optimal_inaccurate']:
                return None, None, False
        except:
            return None, None, False
    
    # Extract solution
    phi_prime_opt = phi_prime.value
    if phi_prime_opt is None:
        return None, None, False
    
    phi_opt = V_tilde.T @ phi_prime_opt
    
    # Compute chi^2
    chi2 = 0.5 * np.sum((Pr_prime - S_diag @ phi_prime_opt)**2)
    
    return phi_opt, chi2, True


def normalize_density_on_grid(phi: np.ndarray, delta_x: float) -> np.ndarray:
    """
    Normalize a density on a uniform grid to integrate to 1.
    
    Parameters:
    -----------
    phi : np.ndarray
        Unnormalized density values
    delta_x : float
        Grid spacing
    
    Returns:
    --------
    phi_normalized : np.ndarray
        Normalized density
    """
    integral = trapezoidal_integral(phi, delta_x)
    
    if integral <= 0:
        raise ValueError("Cannot normalize: integral is non-positive")
    
    return phi / integral
