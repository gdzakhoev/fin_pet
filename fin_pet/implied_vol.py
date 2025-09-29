"""Implied volatility solvers.

Provides robust root-finding utilities to back out volatility from option prices.
"""

from dataclasses import dataclass
from typing import Callable, Dict, Literal, Optional, Tuple

import numpy as np
from scipy.optimize import brentq

from .models_bsm import BlackScholesMerton as BSM
from .validation import FinPetValueError, ensure_positive, ensure_numeric, ensure_str_in


OptionType = Literal["call", "put"]


@dataclass
class ImpliedVolatility:
    """Solve implied volatility for European options under BSM using Brent's method.

    The solver is robust to edge cases by bracketing volatility in [v_min, v_max].
    """

    v_min: float = 1e-6
    v_max: float = 5.0
    tol: float = 1e-8
    max_iter: int = 100

    def _objective(self, vol: float, option_type: OptionType, spot: float, strike: float, maturity: float, rate: float, dividend_yield: float, price: float) -> float:
        """Objective function: model_price(vol) - market_price."""
        model_price = BSM.price(option_type, spot, strike, maturity, rate, dividend_yield, vol)
        return model_price - price

    def solve(
        self,
        option_type: OptionType,
        spot: float,
        strike: float,
        maturity: float,
        rate: float,
        dividend_yield: float,
        price: float,
        bracket: Optional[Tuple[float, float]] = None,
    ) -> float:
        """Compute implied volatility.

        Parameters are BSM inputs and the observed option price. If `bracket` is not provided,
        defaults to [v_min, v_max].
        """

        option = ensure_str_in(option_type, "option_type", ("call", "put"))
        s = ensure_positive(spot, "spot")
        k = ensure_positive(strike, "strike")
        t = ensure_positive(maturity, "maturity")
        r = ensure_numeric(rate, "rate")
        q = ensure_numeric(dividend_yield, "dividend_yield")
        p = ensure_positive(price, "price", allow_zero=False)

        v_lo, v_hi = bracket if bracket is not None else (self.v_min, self.v_max)
        v_lo = ensure_positive(v_lo, "bracket[0]")
        v_hi = ensure_positive(v_hi, "bracket[1]")
        if v_lo >= v_hi:
            raise FinPetValueError("bracket must satisfy v_lo < v_hi")

        f_lo = self._objective(v_lo, option, s, k, t, r, q, p)
        f_hi = self._objective(v_hi, option, s, k, t, r, q, p)

        # If not properly bracketed, expand v_hi up to a cap
        expand_factor = 2.0
        attempts = 0
        while f_lo * f_hi > 0.0 and v_hi < 100.0 and attempts < 10:
            v_hi *= expand_factor
            f_hi = self._objective(v_hi, option, s, k, t, r, q, p)
            attempts += 1

        if f_lo * f_hi > 0.0:
            raise FinPetValueError("Implied volatility not bracketed; price may be outside model bounds")

        return float(
            brentq(
                lambda v: self._objective(v, option, s, k, t, r, q, p),
                v_lo,
                v_hi,
                xtol=self.tol,
                maxiter=self.max_iter,
            )
        )

    def solve_callable(
        self,
        pricer: Callable[..., float],
        params: Dict[str, float],
        target_price: float,
        vol_key: str = "volatility",
        bracket: Optional[Tuple[float, float]] = None,
    ) -> float:
        """Implied volatility (or any volatility-like parameter) via an arbitrary pricing callable.

        Parameters
        ----------
        pricer : Callable[..., float]
            Pricing function that accepts keyword params and returns price.
        params : Dict[str, float]
            Base parameters. One of the keys must be `vol_key`.
        target_price : float
            Observed market price to match.
        vol_key : str
            Name of the volatility-like parameter to solve for. Default "volatility".
        bracket : tuple[float, float], optional
            Search bracket for the volatility-like parameter.
        """

        if vol_key not in params:
            raise FinPetValueError(f"params must contain '{vol_key}' to solve implied value")
        p = ensure_positive(target_price, "target_price")
        v_lo, v_hi = bracket if bracket is not None else (self.v_min, self.v_max)
        v_lo = ensure_positive(v_lo, "bracket[0]")
        v_hi = ensure_positive(v_hi, "bracket[1]")
        if v_lo >= v_hi:
            raise FinPetValueError("bracket must satisfy v_lo < v_hi")

        def f(x: float) -> float:
            local = dict(params)
            local[vol_key] = x
            value = float(pricer(**local))
            if not np.isfinite(value):
                raise FinPetValueError("Pricing callable returned non-finite value")
            return value - p

        f_lo = f(v_lo)
        f_hi = f(v_hi)
        attempts = 0
        while f_lo * f_hi > 0.0 and v_hi < 100.0 and attempts < 10:
            v_hi *= 2.0
            f_hi = f(v_hi)
            attempts += 1
        if f_lo * f_hi > 0.0:
            raise FinPetValueError("Implied value not bracketed; target price may be outside model bounds")
        return float(brentq(f, v_lo, v_hi, xtol=self.tol, maxiter=self.max_iter))


