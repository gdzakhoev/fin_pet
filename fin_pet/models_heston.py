"""Heston stochastic volatility model.

Typed, documented stubs for characteristic-function-based pricing (Carr–Madan,
Lewis) and parameter calibration utilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Sequence

import numpy as np

from .validation import ensure_positive


Method = Literal["carr-madan", "lewis"]


@dataclass
class HestonModel:
    """Heston model pricer (stub).

    Parameters
    ----------
    method : {"carr-madan","lewis"}
        Choice of Fourier pricing formulation.
    """

    method: Method = "carr-madan"

    def price(
        self,
        option_type: str,
        spot: float,
        strike: float,
        maturity: float,
        rate: float,
        dividend_yield: float,
        kappa: float,
        theta: float,
        sigma_v: float,
        rho: float,
        v0: float,
    ) -> float:
        """Price a European option under Heston dynamics.

        Parameters
        ----------
        option_type : str
            "call" or "put".
        spot : float
            Current spot price S0 > 0.
        strike : float
            Strike price K > 0.
        maturity : float
            Time to maturity T > 0.
        rate : float
            Risk-free rate r.
        dividend_yield : float
            Continuous dividend yield q.
        kappa : float
            Mean reversion speed > 0.
        theta : float
            Long-term variance > 0.
        sigma_v : float
            Volatility of variance > 0.
        rho : float
            Correlation between spot and variance in [-1, 1].
        v0 : float
            Initial variance > 0.

        Returns
        -------
        float
            Option price under Heston model.
        """

        ensure_positive(spot, "spot")
        ensure_positive(strike, "strike")
        ensure_positive(maturity, "maturity")
        ensure_positive(theta, "theta")
        ensure_positive(sigma_v, "sigma_v")
        ensure_positive(v0, "v0")

        if self.method == "carr-madan":
            return self._price_carr_madan(option_type, spot, strike, maturity, rate, dividend_yield, kappa, theta, sigma_v, rho, v0)
        elif self.method == "lewis":
            return self._price_lewis(option_type, spot, strike, maturity, rate, dividend_yield, kappa, theta, sigma_v, rho, v0)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _price_carr_madan(
        self,
        option_type: str,
        spot: float,
        strike: float,
        maturity: float,
        rate: float,
        dividend_yield: float,
        kappa: float,
        theta: float,
        sigma_v: float,
        rho: float,
        v0: float,
    ) -> float:
        """Carr-Madan FFT method for Heston pricing."""

        # Damping parameter
        alpha = 1.25

        # Integration bounds
        N = 4096
        eta = 0.25
        lambda_param = 2 * np.pi / (N * eta)
        b = N * lambda_param / 2

        # Log strike grid
        k_grid = np.arange(-b, b, lambda_param)

        # Characteristic function evaluation
        u_grid = np.arange(0, N) * eta
        phi_grid = np.zeros(N, dtype=complex)

        for i, u in enumerate(u_grid):
            phi_grid[i] = self._heston_characteristic_function(
                u - (alpha + 1) * 1j, spot, strike, maturity, rate, dividend_yield, kappa, theta, sigma_v, rho, v0
            )

        # FFT computation
        from scipy.fft import fft
        fft_result = fft(phi_grid)

        # Extract option prices
        option_prices = np.real(fft_result) * np.exp(-alpha * k_grid) / np.pi

        # Interpolate to find price at log(strike/spot)
        log_moneyness = np.log(strike / spot)
        price = np.interp(log_moneyness, k_grid, option_prices)

        if option_type == "put":
            # Put-call parity adjustment
            forward = spot * np.exp((rate - dividend_yield) * maturity)
            price = price + strike * np.exp(-rate * maturity) - forward * np.exp(-dividend_yield * maturity)

        return float(price)

    def _price_lewis(
        self,
        option_type: str,
        spot: float,
        strike: float,
        maturity: float,
        rate: float,
        dividend_yield: float,
        kappa: float,
        theta: float,
        sigma_v: float,
        rho: float,
        v0: float,
    ) -> float:
        """Lewis method for Heston pricing."""

        # Integration parameters
        N = 1000
        upper_bound = 100.0

        def integrand(k):
            """Integrand for Lewis formula."""
            phi = self._heston_characteristic_function(k, spot, strike, maturity, rate, dividend_yield, kappa, theta, sigma_v, rho, v0)
            return np.real(phi / (k * (k - 1j)))

        # Numerical integration
        from scipy.integrate import quad
        integral, _ = quad(integrand, 0, upper_bound, limit=N)

        # Lewis formula
        forward = spot * np.exp((rate - dividend_yield) * maturity)
        price = forward - strike * np.exp(-rate * maturity) * (0.5 + integral / np.pi)

        if option_type == "put":
            # Put-call parity
            price = price + strike * np.exp(-rate * maturity) - forward * np.exp(-dividend_yield * maturity)

        return float(price)

    def _heston_characteristic_function(
        self,
        u: complex,
        spot: float,
        strike: float,
        maturity: float,
        rate: float,
        dividend_yield: float,
        kappa: float,
        theta: float,
        sigma_v: float,
        rho: float,
        v0: float,
    ) -> complex:
        """Heston characteristic function."""

        # Heston parameters
        d = np.sqrt((kappa - rho * sigma_v * u * 1j) ** 2 + sigma_v**2 * (u * 1j + u**2))
        g = (kappa - rho * sigma_v * u * 1j - d) / (kappa - rho * sigma_v * u * 1j + d)

        # Characteristic function components
        C = (rate - dividend_yield) * u * 1j * maturity + (kappa * theta / sigma_v**2) * (
            (kappa - rho * sigma_v * u * 1j - d) * maturity - 2 * np.log((1 - g * np.exp(-d * maturity)) / (1 - g))
        )

        D = ((kappa - rho * sigma_v * u * 1j - d) / sigma_v**2) * ((1 - np.exp(-d * maturity)) / (1 - g * np.exp(-d * maturity)))

        # Final characteristic function
        phi = np.exp(C + D * v0 + u * 1j * np.log(spot))

        return phi



