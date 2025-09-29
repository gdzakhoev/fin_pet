"""Standard option instruments for fin_pet.

Defines base `Option` and standard styles: `EuropeanOption`, `AmericanOption`, and `BermudanOption`.
These classes encapsulate contract specifications but delegate pricing and greeks to model modules.
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence

from .validation import ensure_positive, ensure_probability, ensure_str_in


ALLOWED_OPTION_TYPES = ("call", "put")
ALLOWED_STYLES = ("european", "american", "bermudan")


@dataclass(frozen=True)
class Option:
    """Base option contract specification.

    Parameters
    ----------
    option_type : str
        Either "call" or "put".
    strike : float
        Strike price K > 0.
    maturity : float
        Time to maturity T in years, T > 0.
    risk_free_rate : float
        Continuously compounded risk-free rate r.
    dividend_yield : float
        Continuous dividend yield q (default 0.0).
    style : str
        One of "european", "american", "bermudan".
    bermudan_exercise_times : Optional[Sequence[float]]
        For Bermudan options only: sorted exercise times in years within (0, T].
    """

    option_type: str
    strike: float
    maturity: float
    risk_free_rate: float
    dividend_yield: float = 0.0
    style: str = "european"
    bermudan_exercise_times: Optional[Sequence[float]] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "option_type", ensure_str_in(self.option_type, "option_type", ALLOWED_OPTION_TYPES))
        object.__setattr__(self, "style", ensure_str_in(self.style, "style", ALLOWED_STYLES))
        object.__setattr__(self, "strike", ensure_positive(self.strike, "strike"))
        object.__setattr__(self, "maturity", ensure_positive(self.maturity, "maturity"))
        # risk_free_rate and dividend_yield can be any finite float; reuse ensure_probability is not suitable
        # Validation is handled in pricing functions for bounds if needed.

        if self.style == "bermudan":
            if self.bermudan_exercise_times is None:
                raise ValueError("bermudan_exercise_times must be provided for Bermudan options")
            times: List[float] = []
            for t in self.bermudan_exercise_times:
                times.append(ensure_positive(t, "bermudan_exercise_times element", allow_zero=False))
            times_sorted = sorted(times)
            if times_sorted[-1] > self.maturity:
                raise ValueError("All Bermudan exercise times must be <= maturity")
            object.__setattr__(self, "bermudan_exercise_times", times_sorted)


class EuropeanOption(Option):
    """European option (exercise only at maturity)."""

    def __init__(self, option_type: str, strike: float, maturity: float, risk_free_rate: float, dividend_yield: float = 0.0) -> None:
        super().__init__(option_type=option_type, strike=strike, maturity=maturity, risk_free_rate=risk_free_rate, dividend_yield=dividend_yield, style="european")


class AmericanOption(Option):
    """American option (exercise any time up to maturity)."""

    def __init__(self, option_type: str, strike: float, maturity: float, risk_free_rate: float, dividend_yield: float = 0.0) -> None:
        super().__init__(option_type=option_type, strike=strike, maturity=maturity, risk_free_rate=risk_free_rate, dividend_yield=dividend_yield, style="american")


class BermudanOption(Option):
    """Bermudan option (exercise at specified dates)."""

    def __init__(self, option_type: str, strike: float, maturity: float, risk_free_rate: float, dividend_yield: float, bermudan_exercise_times: Sequence[float]) -> None:
        super().__init__(
            option_type=option_type,
            strike=strike,
            maturity=maturity,
            risk_free_rate=risk_free_rate,
            dividend_yield=dividend_yield,
            style="bermudan",
            bermudan_exercise_times=bermudan_exercise_times,
        )


