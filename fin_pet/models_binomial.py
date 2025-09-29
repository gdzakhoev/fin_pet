"""Binomial and trinomial tree pricing models.

This module declares typed, documented stubs for binomial models to price
European, American, and Bermudan options, including early exercise features.
Concrete implementations (CRR, Jarrow–Rudd, Leisen–Reimer) should extend
the base interfaces below.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Sequence

import numpy as np

from .instruments_standard import Option
from .validation import ensure_positive


TreeType = Literal["crr", "jr", "leisen-reimer", "trinomial"]


@dataclass
class BinomialTreePricer:
    """Configurable binomial tree pricer (stub).

    Parameters
    ----------
    steps : int
        Number of time steps in the tree; must be >= 1.
    tree : {"crr","jr","leisen-reimer","trinomial"}
        Choice of tree construction.

    Notes
    -----
    - Use risk-neutral probability with continuous compounding (r, q).
    - For Bermudan options, exercise dates must align to the tree grid.
    """

    steps: int = 200
    tree: TreeType = "crr"

    def __post_init__(self) -> None:
        if int(self.steps) < 1:
            raise ValueError("steps must be >= 1")

    def price(
        self,
        instrument: Option,
        spot: float,
        volatility: float,
    ) -> float:
        """Price an option using the configured binomial/trinomial tree.

        Parameters
        ----------
        instrument : Option
            Contract specification (European, American, Bermudan).
        spot : float
            Current underlying spot price (S0 > 0).
        volatility : float
            Annualized volatility (sigma > 0).

        Returns
        -------
        float
            Present value of the option.
        """

        # Validate basic numeric inputs to fail-fast
        ensure_positive(spot, "spot")
        ensure_positive(volatility, "volatility")

        if self.tree == "crr":
            return self._price_crr(instrument, spot, volatility)
        else:
            raise NotImplementedError(f"Tree type '{self.tree}' not implemented yet")

    def _price_crr(
        self,
        instrument: Option,
        spot: float,
        volatility: float,
    ) -> float:
        """Cox-Ross-Rubinstein binomial tree implementation."""

        # CRR parameters
        dt = instrument.maturity / self.steps
        u = np.exp(volatility * np.sqrt(dt))
        d = 1.0 / u
        p = (np.exp((instrument.risk_free_rate - instrument.dividend_yield) * dt) - d) / (u - d)

        # Ensure risk-neutral probability is valid
        if p <= 0.0 or p >= 1.0:
            raise ValueError(f"Invalid risk-neutral probability p={p:.6f}. Adjust steps or parameters.")

        # Build stock price tree
        stock_tree = np.zeros((self.steps + 1, self.steps + 1))
        for i in range(self.steps + 1):
            for j in range(i + 1):
                stock_tree[j, i] = spot * (u ** (i - j)) * (d ** j)

        # Build option value tree (backward induction)
        option_tree = np.zeros((self.steps + 1, self.steps + 1))

        # Terminal payoff
        for j in range(self.steps + 1):
            stock_price = stock_tree[j, self.steps]
            if instrument.option_type == "call":
                option_tree[j, self.steps] = max(0.0, stock_price - instrument.strike)
            else:
                option_tree[j, self.steps] = max(0.0, instrument.strike - stock_price)

        # Backward induction with early exercise
        discount_factor = np.exp(-instrument.risk_free_rate * dt)
        exercise_times = self._get_exercise_times(instrument)

        for i in range(self.steps - 1, -1, -1):
            for j in range(i + 1):
                # Continuation value
                continuation = discount_factor * (p * option_tree[j, i + 1] + (1 - p) * option_tree[j + 1, i + 1])

                # Early exercise check
                if self._can_exercise_early(instrument, i, exercise_times):
                    stock_price = stock_tree[j, i]
                    if instrument.option_type == "call":
                        intrinsic = max(0.0, stock_price - instrument.strike)
                    else:
                        intrinsic = max(0.0, instrument.strike - stock_price)
                    option_tree[j, i] = max(continuation, intrinsic)
                else:
                    option_tree[j, i] = continuation

        return float(option_tree[0, 0])

    def _get_exercise_times(self, instrument: Option) -> list[int]:
        """Get exercise time indices for Bermudan options."""

        if instrument.style != "bermudan" or instrument.bermudan_exercise_times is None:
            return []

        exercise_times = []
        dt = instrument.maturity / self.steps

        for exercise_time in instrument.bermudan_exercise_times:
            # Find closest grid point
            idx = round(exercise_time / dt)
            if 0 <= idx <= self.steps:
                exercise_times.append(idx)

        return sorted(exercise_times)

    def _can_exercise_early(self, instrument: Option, time_idx: int, exercise_times: list[int]) -> bool:
        """Check if early exercise is allowed at the given time index."""

        if instrument.style == "european":
            return False
        elif instrument.style == "american":
            return time_idx < self.steps  # Can exercise at any time except maturity
        elif instrument.style == "bermudan":
            return time_idx in exercise_times
        else:
            return False



