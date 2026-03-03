"""
Tests for core utility functions.
"""

import numpy as np
import pytest
from src.utils import (
    build_kernel_matrix, truncated_svd, bachelier_call_price,
    bachelier_put_price, black_scholes_call_price, black_scholes_put_price,
    black_call_price, black_put_price, trapezoidal_integral,
    bhattacharyya_distance, normalize_density_on_grid
)


def test_kernel_matrix_shape():
    """Test that kernel matrix has correct shape."""
    strikes = np.array([0.8, 0.9, 1.0, 1.1])
    x_grid = np.linspace(0.5, 1.5, 100)
    delta_x = x_grid[1] - x_grid[0]
    
    G = build_kernel_matrix(strikes, x_grid, delta_x, r=0.05, tau=1.0)
    
    assert G.shape == (len(strikes), len(x_grid))


def test_kernel_matrix_call_payoff():
    """Test that kernel matrix correctly represents call payoff."""
    strikes = np.array([1.0])
    x_grid = np.array([0.8, 1.0, 1.2])
    delta_x = 0.2
    r = 0.0
    tau = 1.0
    
    G = build_kernel_matrix(strikes, x_grid, delta_x, r, tau)
    
    # For call: max(x - K, 0) with K=1.0
    # x=0.8: payoff=0, x=1.0: payoff=0, x=1.2: payoff=0.2
    # With trapezoidal correction: first and last columns multiplied by 0.5
    # G[0,0] = 0.5 * delta_x * exp(-r*tau) * max(0.8-1.0, 0) = 0
    # G[0,1] = delta_x * exp(-r*tau) * max(1.0-1.0, 0) = 0
    # G[0,2] = 0.5 * delta_x * exp(-r*tau) * max(1.2-1.0, 0) = 0.5 * 0.2 * 1.0 * 0.2 = 0.02
    
    assert G[0, 0] == 0.0
    assert G[0, 1] == 0.0
    assert np.isclose(G[0, 2], 0.02)


def test_kernel_matrix_put_payoff():
    """Test that kernel matrix correctly represents put payoff."""
    strikes = np.array([1.0])
    x_grid = np.array([0.8, 1.0, 1.2])
    delta_x = 0.2
    r = 0.0
    tau = 1.0
    option_types = np.array(['put'])
    
    G = build_kernel_matrix(strikes, x_grid, delta_x, r, tau, option_types)
    
    # For put: max(K - x, 0) with K=1.0
    # x=0.8: payoff=0.2, x=1.0: payoff=0, x=1.2: payoff=0
    # With trapezoidal correction:
    # G[0,0] = 0.5 * 0.2 * 1.0 * 0.2 = 0.02
    # G[0,1] = 0.2 * 1.0 * 0.0 = 0.0
    # G[0,2] = 0.5 * 0.2 * 1.0 * 0.0 = 0.0
    
    assert np.isclose(G[0, 0], 0.02)
    assert G[0, 1] == 0.0
    assert G[0, 2] == 0.0


def test_truncated_svd_dimensions():
    """Test that truncated SVD returns correct dimensions."""
    M, N, Q = 10, 50, 5
    G = np.random.randn(M, N)
    
    U_tilde, S_tilde, V_tilde = truncated_svd(G, Q)
    
    assert U_tilde.shape == (M, Q)
    assert S_tilde.shape == (Q,)
    assert V_tilde.shape == (Q, N)


def test_truncated_svd_reconstruction():
    """Test that truncated SVD approximately reconstructs G."""
    M, N, Q = 10, 50, 10
    G = np.random.randn(M, N)
    
    U_tilde, S_tilde, V_tilde = truncated_svd(G, Q)
    
    # Reconstruct
    G_reconstructed = U_tilde @ np.diag(S_tilde) @ V_tilde
    
    # Should be close (exact if Q = min(M, N))
    assert np.allclose(G, G_reconstructed)


def test_bachelier_call_put_parity():
    """Test Bachelier call-put parity: C - P = exp(-r*tau) * (F - K)."""
    K = 100.0
    F = 105.0
    sigma = 10.0
    r = 0.05
    tau = 1.0
    
    C = bachelier_call_price(K, F, sigma, r, tau)
    P = bachelier_put_price(K, F, sigma, r, tau)
    
    parity_lhs = C - P
    parity_rhs = np.exp(-r * tau) * (F - K)
    
    assert np.isclose(parity_lhs, parity_rhs, rtol=1e-10)


def test_black_scholes_call_put_parity():
    """Test BS call-put parity: C - P = S0 - K*exp(-r*tau)."""
    K = 100.0
    S0 = 105.0
    sigma = 0.2
    r = 0.05
    tau = 1.0
    
    C = black_scholes_call_price(K, S0, sigma, r, tau)
    P = black_scholes_put_price(K, S0, sigma, r, tau)
    
    parity_lhs = C - P
    parity_rhs = S0 - K * np.exp(-r * tau)
    
    assert np.isclose(parity_lhs, parity_rhs, rtol=1e-10)


def test_black_call_put_parity():
    """Test Black model call-put parity: C - P = exp(-r*tau) * (F - K)."""
    K = 100.0
    F = 105.0
    sigma = 0.2
    r = 0.05
    tau = 1.0
    
    C = black_call_price(K, F, sigma, r, tau)
    P = black_put_price(K, F, sigma, r, tau)
    
    parity_lhs = C - P
    parity_rhs = np.exp(-r * tau) * (F - K)
    
    assert np.isclose(parity_lhs, parity_rhs, rtol=1e-10)


def test_trapezoidal_integral():
    """Test trapezoidal integration on simple function."""
    # Integrate f(x) = x on [0, 1] with uniform grid
    # Exact answer: 0.5
    x = np.linspace(0, 1, 1001)
    f = x
    delta_x = x[1] - x[0]
    
    integral = trapezoidal_integral(f, delta_x)
    
    assert np.isclose(integral, 0.5, atol=1e-6)


def test_bhattacharyya_distance_identical_densities():
    """Test that Bhattacharyya distance is zero for identical densities."""
    phi1 = np.array([0.1, 0.2, 0.3, 0.2, 0.2])
    phi1 = phi1 / (phi1.sum() * 0.1)  # Normalize
    delta_x = 0.1
    
    # Normalize properly using trapezoidal rule
    phi1 = normalize_density_on_grid(phi1, delta_x)
    
    distance = bhattacharyya_distance(phi1, phi1, delta_x)
    
    assert np.isclose(distance, 0.0, atol=1e-10)


def test_normalize_density_on_grid():
    """Test that normalization produces unit integral."""
    phi = np.exp(-np.linspace(-2, 2, 1001)**2)  # Gaussian-like
    delta_x = 4.0 / 1000
    
    phi_normalized = normalize_density_on_grid(phi, delta_x)
    integral = trapezoidal_integral(phi_normalized, delta_x)
    
    assert np.isclose(integral, 1.0, atol=1e-10)


def test_bachelier_price_non_negative():
    """Test that Bachelier prices are non-negative."""
    strikes = np.linspace(50, 150, 20)
    F = 100.0
    sigma = 10.0
    r = 0.05
    tau = 1.0
    
    for K in strikes:
        C = bachelier_call_price(K, F, sigma, r, tau)
        P = bachelier_put_price(K, F, sigma, r, tau)
        assert C >= 0.0
        assert P >= 0.0


def test_black_scholes_price_non_negative():
    """Test that BS prices are non-negative."""
    strikes = np.linspace(50, 150, 20)
    S0 = 100.0
    sigma = 0.2
    r = 0.05
    tau = 1.0
    
    for K in strikes:
        C = black_scholes_call_price(K, S0, sigma, r, tau)
        P = black_scholes_put_price(K, S0, sigma, r, tau)
        assert C >= 0.0
        assert P >= 0.0


def test_bachelier_intrinsic_value():
    """Test that Bachelier call price >= max(0, F - K)*exp(-r*tau)."""
    K = 90.0
    F = 100.0
    sigma = 10.0
    r = 0.05
    tau = 1.0
    
    C = bachelier_call_price(K, F, sigma, r, tau)
    intrinsic = max(0, F - K) * np.exp(-r * tau)
    
    assert C >= intrinsic - 1e-10


def test_black_scholes_atm_price_properties():
    """Test BS ATM call has certain properties."""
    K = 100.0
    S0 = 100.0
    sigma = 0.2
    r = 0.05
    tau = 1.0
    
    C = black_scholes_call_price(K, S0, sigma, r, tau)
    
    # ATM call should be worth more than intrinsic
    # For r > 0, forward > spot, so slightly ITM
    assert C > 0.0
    # Rough sanity check: should be a few % of spot
    assert 0.01 * S0 < C < 0.5 * S0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
