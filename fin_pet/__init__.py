"""fin_pet package

A robust Python library for derivative pricing, greeks (first and second order), implied volatility,
and risk metrics for multiple option styles and models.

This package exposes a high-level OOP API while keeping modules cohesive:
- models_bsm: Black-Scholes-Merton pricing utilities
- models_binomial: Binomial tree models (placeholder)
- models_mc: Monte Carlo engines (placeholder)
- models_lsm: Longstaff-Schwartz algorithm for American/Bermudan (placeholder)
- models_heston: Heston stochastic volatility model (placeholder)
- instruments_standard: Standard European, American, Bermudan options
- instruments_combined: Spreads and chooser options (placeholder)
- instruments_exotic: Barrier, Asian, Lookback, Forward-start, and Forwards (placeholder)
- greeks: Greeks computations (1st and 2nd order)
- implied_vol: Implied volatility solvers
- risk: Risk metrics (VaR, ES, Sharpe, Sortino, Max Drawdown)
- validation: Input validation utilities and custom error types

"""

from .validation import (
    FinPetError,
    FinPetTypeError,
    FinPetValueError,
    FinPetKeyError,
)
from .instruments_standard import Option, EuropeanOption, AmericanOption, BermudanOption
from .instruments_combined import (
    CombinedInstrument,
    BullCallSpread,
    BearCallSpread,
    BullPutSpread,
    BearPutSpread,
    ButterflySpread,
    BoxSpread,
    ChooserOption,
)
from .instruments_exotic import (
    ForwardContract,
    BarrierOption,
    ForwardStartOption,
    BinaryOption,
    AsianOption,
    LookbackOption,
)
from .models_bsm import BlackScholesMerton
from .greeks import Greeks, NumericalGreeks
from .implied_vol import ImpliedVolatility
from .risk import RiskMetrics

__all__ = [
    "FinPetError",
    "FinPetTypeError",
    "FinPetValueError",
    "FinPetKeyError",
    "Option",
    "EuropeanOption",
    "AmericanOption",
    "BermudanOption",
    "BlackScholesMerton",
    "Greeks",
    "NumericalGreeks",
    "ImpliedVolatility",
    "RiskMetrics",
    # Combined instruments
    "CombinedInstrument",
    "BullCallSpread",
    "BearCallSpread",
    "BullPutSpread",
    "BearPutSpread",
    "ButterflySpread",
    "BoxSpread",
    "ChooserOption",
    # Exotic instruments
    "ForwardContract",
    "BarrierOption",
    "ForwardStartOption",
    "BinaryOption",
    "AsianOption",
    "LookbackOption",
]

