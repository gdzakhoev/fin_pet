"""Black-Scholes-Merton model utilities.

Provides pricing for European options and helper functions (d1, d2, N, n).
Greeks are implemented in `greeks.py` but reuse these primitives.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.stats import norm

from .validation import ensure_positive, ensure_numeric, ensure_str_in


OptionType = Literal["call", "put"]


def _d1_d2(spot: float, strike: float, maturity: float, rate: float, dividend_yield: float, volatility: float) -> tuple[float, float]:
    s = ensure_positive(spot, "spot", allow_zero=False)
    k = ensure_positive(strike, "strike", allow_zero=False)
    t = ensure_positive(maturity, "maturity", allow_zero=False)
    v = ensure_positive(volatility, "volatility", allow_zero=False)
    r = ensure_numeric(rate, "rate")
    q = ensure_numeric(dividend_yield, "dividend_yield")

    sigma_sqrt_t = v * np.sqrt(t)
    if sigma_sqrt_t == 0.0:
        raise ValueError("volatility * sqrt(maturity) must be > 0")
    ln_sk = np.log(s / k)
    d1 = (ln_sk + (r - q + 0.5 * v * v) * t) / sigma_sqrt_t
    d2 = d1 - sigma_sqrt_t
    return d1, d2


@dataclass
class BlackScholesMerton:
    """Black-Scholes-Merton pricing engine.

    Methods
    -------
    price(option_type, spot, strike, maturity, rate, dividend_yield, volatility)
        Compute the European option price.
    """

    @staticmethod
    def price(
        option_type: OptionType,
        spot: float,
        strike: float,
        maturity: float,
        rate: float,
        dividend_yield: float,
        volatility: float,
    ) -> float:
        """Price a European option using BSM closed form.

        Parameters are standard BSM inputs (continuous compounding).
        """

        option = ensure_str_in(option_type, "option_type", ("call", "put"))
        d1, d2 = _d1_d2(spot, strike, maturity, rate, dividend_yield, volatility)
        s = spot
        k = strike
        t = maturity
        r = rate
        q = dividend_yield
        df_r = np.exp(-r * t)
        df_q = np.exp(-q * t)

        if option == "call":
            return float(df_q * s * norm.cdf(d1) - df_r * k * norm.cdf(d2))
        return float(df_r * k * norm.cdf(-d2) - df_q * s * norm.cdf(-d1))

    @staticmethod
    def d1(spot: float, strike: float, maturity: float, rate: float, dividend_yield: float, volatility: float) -> float:
        return _d1_d2(spot, strike, maturity, rate, dividend_yield, volatility)[0]

    @staticmethod
    def d2(spot: float, strike: float, maturity: float, rate: float, dividend_yield: float, volatility: float) -> float:
        return _d1_d2(spot, strike, maturity, rate, dividend_yield, volatility)[1]


