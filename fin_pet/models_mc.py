"""Monte Carlo pricing engines.

Typed and documented stubs for path simulation under GBM and extensions, with
hooks for variance reduction (antithetic variates, control variates).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np

from .validation import ensure_positive, ensure_numeric


Payoff = Callable[[np.ndarray], float]


@dataclass
class MonteCarloEngine:
    """Generic Monte Carlo engine (stub) for GBM paths.

    Parameters
    ----------
    num_paths : int
        Number of Monte Carlo paths to simulate (>= 1).
    num_steps : int
        Time steps per path (>= 1).
    antithetic : bool
        If True, use antithetic variates.
    seed : int, optional
        Random seed for reproducibility.
    """

    num_paths: int = 10000
    num_steps: int = 252
    antithetic: bool = True
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        if int(self.num_paths) < 1:
            raise ValueError("num_paths must be >= 1")
        if int(self.num_steps) < 1:
            raise ValueError("num_steps must be >= 1")

    def price(
        self,
        spot: float,
        rate: float,
        dividend_yield: float,
        volatility: float,
        maturity: float,
        payoff: Payoff,
        control_variate: Optional[Callable[[np.ndarray], float]] = None,
    ) -> float:
        """Price via MC by simulating GBM paths and discounting expected payoff.

        Parameters
        ----------
        spot : float
            Initial spot price S0 > 0.
        rate : float
            Risk-free rate r.
        dividend_yield : float
            Continuous dividend yield q.
        volatility : float
            Annualized volatility sigma > 0.
        maturity : float
            Time to maturity T > 0.
        payoff : Callable[[ndarray], float]
            Payoff function mapping final spot prices to payoffs.
        control_variate : Callable[[ndarray], float], optional
            Control variate function for variance reduction.

        Returns
        -------
        float
            Estimated option price.
        """

        ensure_positive(spot, "spot")
        ensure_positive(volatility, "volatility")
        ensure_positive(maturity, "maturity")
        ensure_numeric(rate, "rate")
        ensure_numeric(dividend_yield, "dividend_yield")

        # Set random seed if provided
        if self.seed is not None:
            np.random.seed(self.seed)

        # Generate paths
        paths = self._generate_gbm_paths(spot, rate, dividend_yield, volatility, maturity)

        # Compute payoffs
        payoffs = np.array([payoff(path) for path in paths])

        # Apply antithetic variates if enabled
        if self.antithetic:
            payoffs = self._apply_antithetic_variates(payoffs)

        # Apply control variate if provided
        if control_variate is not None:
            payoffs = self._apply_control_variate(payoffs, paths, control_variate, spot, rate, dividend_yield, volatility, maturity)

        # Discount and average
        discount_factor = np.exp(-rate * maturity)
        return float(discount_factor * np.mean(payoffs))

    def _generate_gbm_paths(
        self,
        spot: float,
        rate: float,
        dividend_yield: float,
        volatility: float,
        maturity: float,
    ) -> np.ndarray:
        """Generate GBM paths under risk-neutral measure."""

        dt = maturity / self.num_steps
        drift = (rate - dividend_yield - 0.5 * volatility**2) * dt
        diffusion = volatility * np.sqrt(dt)

        # Generate random shocks
        if self.antithetic:
            # Generate half the paths, then create antithetic pairs
            half_paths = self.num_paths // 2
            z_half = np.random.standard_normal((half_paths, self.num_steps))
            z_antithetic = -z_half  # Antithetic variates
            z = np.concatenate([z_half, z_antithetic], axis=0)
            # If num_paths is odd, add one more path
            if self.num_paths % 2 == 1:
                z_extra = np.random.standard_normal((1, self.num_steps))
                z = np.concatenate([z, z_extra], axis=0)
        else:
            z = np.random.standard_normal((self.num_paths, self.num_steps))

        # Generate log returns
        log_returns = drift + diffusion * z

        # Convert to price paths
        log_prices = np.cumsum(log_returns, axis=1)
        log_prices = np.concatenate([np.zeros((self.num_paths, 1)), log_prices], axis=1)
        paths = spot * np.exp(log_prices)

        return paths

    def _apply_antithetic_variates(self, payoffs: np.ndarray) -> np.ndarray:
        """Apply antithetic variates variance reduction."""

        if not self.antithetic:
            return payoffs

        # For antithetic, we have pairs of paths, so we average them
        if len(payoffs) % 2 == 0:
            half = len(payoffs) // 2
            return (payoffs[:half] + payoffs[half:]) / 2.0
        else:
            # If odd number, just return as is
            return payoffs

    def _apply_control_variate(
        self,
        payoffs: np.ndarray,
        paths: np.ndarray,
        control_variate: Callable[[np.ndarray], float],
        spot: float,
        rate: float,
        dividend_yield: float,
        volatility: float,
        maturity: float,
    ) -> np.ndarray:
        """Apply control variate variance reduction."""

        # Compute control variate values
        control_values = np.array([control_variate(path) for path in paths])

        # Estimate optimal coefficient (beta) via regression
        # For European options, we can use the analytical solution as control
        # Here we use a simple correlation-based approach
        if len(payoffs) > 1:
            correlation = np.corrcoef(payoffs, control_values)[0, 1]
            if not np.isnan(correlation):
                beta = correlation * np.std(payoffs) / np.std(control_values)
                # Adjust payoffs
                control_mean = np.mean(control_values)
                payoffs = payoffs - beta * (control_values - control_mean)

        return payoffs



