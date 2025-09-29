"""Longstaff–Schwartz method for early-exercise options.

Documented stubs for implementing the LSM algorithm with a flexible regression
basis to estimate continuation values for American and Bermudan options.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np

from .instruments_standard import Option
from .validation import ensure_positive


BasisFunction = Callable[[np.ndarray], np.ndarray]


@dataclass
class LongstaffSchwartz:
    """LSM pricer (stub) using polynomial basis by default.

    Parameters
    ----------
    basis_degree : int
        Degree of polynomial basis for regression (>= 1).
    """

    basis_degree: int = 2

    def __post_init__(self) -> None:
        if int(self.basis_degree) < 1:
            raise ValueError("basis_degree must be >= 1")

    def price(
        self,
        instrument: Option,
        spot_paths: np.ndarray,
        discount_factors: np.ndarray,
        payoff_fn: Callable[[np.ndarray], np.ndarray],
        exercise_times_idx: Sequence[int],
        basis: Optional[Sequence[BasisFunction]] = None,
    ) -> float:
        """Price an early-exercise option via LSM.

        Parameters
        ----------
        instrument : Option
            American or Bermudan option spec.
        spot_paths : ndarray, shape (num_paths, num_steps+1)
            Simulated spots on a grid including t0 and maturity.
        discount_factors : ndarray, shape (num_steps+1,)
            Discount factors aligned with the grid.
        payoff_fn : Callable[[np.ndarray], np.ndarray]
            Vectorized payoff function at each path state.
        exercise_times_idx : Sequence[int]
            Indices into the time grid where early exercise is allowed.
        basis : sequence of basis functions, optional
            Functions mapping state vector to features for regression.

        Returns
        -------
        float
            Estimated present value.

        Raises
        ------
        NotImplementedError
            This is a stub. Implement regression of continuation values and optimal stopping.
        """

        if spot_paths.ndim != 2:
            raise ValueError("spot_paths must be 2D: (num_paths, num_steps+1)")
        if discount_factors.ndim != 1:
            raise ValueError("discount_factors must be 1D and align with the time grid")
        if len(exercise_times_idx) == 0:
            raise ValueError("exercise_times_idx must not be empty for early-exercise options")

        num_paths, num_steps = spot_paths.shape
        num_steps -= 1  # Adjust for t0

        # Initialize exercise decisions
        exercise_decisions = np.zeros((num_paths, num_steps + 1), dtype=bool)
        cash_flows = np.zeros((num_paths, num_steps + 1))

        # Terminal payoff
        terminal_payoffs = payoff_fn(spot_paths[:, -1])
        cash_flows[:, -1] = terminal_payoffs
        exercise_decisions[:, -1] = terminal_payoffs > 0

        # Backward induction
        for t_idx in reversed(exercise_times_idx[:-1]):  # Skip terminal time
            # Get in-the-money paths
            current_spots = spot_paths[:, t_idx]
            current_payoffs = payoff_fn(current_spots)
            itm_mask = current_payoffs > 0

            if not np.any(itm_mask):
                continue

            # Get continuation values for ITM paths
            continuation_values = self._compute_continuation_values(
                spot_paths[itm_mask, t_idx:],
                cash_flows[itm_mask, t_idx + 1:],
                discount_factors[t_idx + 1:],
                basis
            )

            # Exercise decision: exercise if intrinsic > continuation
            exercise_mask = current_payoffs[itm_mask] > continuation_values
            exercise_decisions[itm_mask, t_idx] = exercise_mask

            # Update cash flows
            cash_flows[itm_mask, t_idx] = np.where(
                exercise_mask,
                current_payoffs[itm_mask],
                cash_flows[itm_mask, t_idx]
            )

        # Compute option value
        option_values = np.zeros(num_paths)
        for path_idx in range(num_paths):
            # Find first exercise time
            exercise_times = np.where(exercise_decisions[path_idx])[0]
            if len(exercise_times) > 0:
                first_exercise = exercise_times[0]
                payoff = cash_flows[path_idx, first_exercise]
                discount = discount_factors[first_exercise]
                option_values[path_idx] = payoff * discount

        return float(np.mean(option_values))

    def _compute_continuation_values(
        self,
        spot_paths: np.ndarray,
        future_cash_flows: np.ndarray,
        discount_factors: np.ndarray,
        basis: Optional[Sequence[BasisFunction]] = None,
    ) -> np.ndarray:
        """Compute continuation values using least squares regression."""

        if basis is None:
            basis = self._default_basis_functions()

        # Prepare features for regression
        features = self._build_features(spot_paths[:, 0], basis)  # Current spot prices

        # Prepare target values (discounted future cash flows)
        target_values = np.zeros(len(spot_paths))
        for i in range(len(spot_paths)):
            # Find first non-zero cash flow in the future
            future_flows = future_cash_flows[i]
            future_discounts = discount_factors
            for j, (flow, discount) in enumerate(zip(future_flows, future_discounts)):
                if flow > 0:
                    target_values[i] = flow * discount
                    break

        # Perform regression
        from sklearn.linear_model import LinearRegression
        regressor = LinearRegression()
        regressor.fit(features, target_values)

        # Predict continuation values
        return regressor.predict(features)

    def _default_basis_functions(self) -> list[BasisFunction]:
        """Default polynomial basis functions."""

        def constant(x: np.ndarray) -> np.ndarray:
            return np.ones_like(x)

        def linear(x: np.ndarray) -> np.ndarray:
            return x

        def quadratic(x: np.ndarray) -> np.ndarray:
            return x**2

        basis = [constant, linear]
        if self.basis_degree >= 2:
            basis.append(quadratic)
        if self.basis_degree >= 3:
            basis.append(lambda x: x**3)

        return basis

    def _build_features(self, spots: np.ndarray, basis: Sequence[BasisFunction]) -> np.ndarray:
        """Build feature matrix from spot prices and basis functions."""

        features = []
        for basis_func in basis:
            features.append(basis_func(spots))
        return np.column_stack(features)



