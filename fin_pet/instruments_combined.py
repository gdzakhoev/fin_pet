"""Combined option strategies.

Defines reusable classes for common spreads and chooser options. These classes compose
standard `Option` instruments and expose a generic pricing hook compatible with any
pricing engine. Greeks/IV can be computed numerically via `NumericalGreeks`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Sequence, Tuple

from .instruments_standard import AmericanOption, BermudanOption, EuropeanOption, Option
from .validation import ensure_positive, ensure_str_in


Pricer = Callable[..., float]


@dataclass(frozen=True)
class CombinedInstrument:
    """A linear combination of option legs.

    Each leg is a tuple of (instrument, quantity). Quantity can be positive (long) or
    negative (short). Pricing is the sum of quantity * price(leg).
    """

    legs: Sequence[Tuple[Option, float]]

    def price(self, pricer: Pricer, **kwargs) -> float:
        """Compute total price via provided pricing callable.

        Parameters
        ----------
        pricer : Callable[..., float]
            Function of signature pricer(instrument=..., **kwargs) -> float
        kwargs : dict
            Additional parameters required by the pricing model (spot, volatility, etc.).
        """
        total = 0.0
        for inst, qty in self.legs:
            total += float(qty) * float(pricer(instrument=inst, **kwargs))
        return float(total)


def _make_option(style: str, option_type: str, strike: float, maturity: float, rate: float, dividend_yield: float, bermudan_exercise_times: Sequence[float] | None) -> Option:
    """Factory to create an `Option` of the requested style."""
    style = ensure_str_in(style, "style", ("european", "american", "bermudan"))
    if style == "european":
        return EuropeanOption(option_type, strike, maturity, rate, dividend_yield)
    if style == "american":
        return AmericanOption(option_type, strike, maturity, rate, dividend_yield)
    if bermudan_exercise_times is None:
        raise ValueError("bermudan_exercise_times must be provided for Bermudan spreads")
    return BermudanOption(option_type, strike, maturity, rate, dividend_yield, bermudan_exercise_times)


@dataclass(frozen=True)
class VerticalSpread(CombinedInstrument):
    """Base class for vertical spreads with two strikes K1 and K2 (K1 < K2)."""

    style: str
    option_type: str
    k1: float
    k2: float
    maturity: float
    rate: float
    dividend_yield: float = 0.0
    bermudan_exercise_times: Sequence[float] | None = None

    def __post_init__(self) -> None:
        ensure_str_in(self.option_type, "option_type", ("call", "put"))
        ensure_str_in(self.style, "style", ("european", "american", "bermudan"))
        k1 = ensure_positive(self.k1, "k1")
        k2 = ensure_positive(self.k2, "k2")
        if k1 >= k2:
            raise ValueError("Require k1 < k2 for a vertical spread")


@dataclass(frozen=True)
class BullCallSpread(VerticalSpread):
    """Long call at K1, short call at K2 (K1 < K2)."""

    def __init__(self, style: str, k1: float, k2: float, maturity: float, rate: float, dividend_yield: float = 0.0, bermudan_exercise_times: Sequence[float] | None = None) -> None:
        legs = (
            (_make_option(style, "call", k1, maturity, rate, dividend_yield, bermudan_exercise_times), 1.0),
            (_make_option(style, "call", k2, maturity, rate, dividend_yield, bermudan_exercise_times), -1.0),
        )
        object.__setattr__(self, "legs", legs)
        object.__setattr__(self, "style", style)
        object.__setattr__(self, "option_type", "call")
        object.__setattr__(self, "k1", k1)
        object.__setattr__(self, "k2", k2)
        object.__setattr__(self, "maturity", maturity)
        object.__setattr__(self, "rate", rate)
        object.__setattr__(self, "dividend_yield", dividend_yield)
        object.__setattr__(self, "bermudan_exercise_times", bermudan_exercise_times)
        self.__post_init__()


@dataclass(frozen=True)
class BearCallSpread(VerticalSpread):
    """Short call at K1, long call at K2 (K1 < K2)."""

    def __init__(self, style: str, k1: float, k2: float, maturity: float, rate: float, dividend_yield: float = 0.0, bermudan_exercise_times: Sequence[float] | None = None) -> None:
        legs = (
            (_make_option(style, "call", k1, maturity, rate, dividend_yield, bermudan_exercise_times), -1.0),
            (_make_option(style, "call", k2, maturity, rate, dividend_yield, bermudan_exercise_times), 1.0),
        )
        object.__setattr__(self, "legs", legs)
        object.__setattr__(self, "style", style)
        object.__setattr__(self, "option_type", "call")
        object.__setattr__(self, "k1", k1)
        object.__setattr__(self, "k2", k2)
        object.__setattr__(self, "maturity", maturity)
        object.__setattr__(self, "rate", rate)
        object.__setattr__(self, "dividend_yield", dividend_yield)
        object.__setattr__(self, "bermudan_exercise_times", bermudan_exercise_times)
        self.__post_init__()


@dataclass(frozen=True)
class BullPutSpread(VerticalSpread):
    """Short put at K1, long put at K2 (K1 < K2)."""

    def __init__(self, style: str, k1: float, k2: float, maturity: float, rate: float, dividend_yield: float = 0.0, bermudan_exercise_times: Sequence[float] | None = None) -> None:
        legs = (
            (_make_option(style, "put", k1, maturity, rate, dividend_yield, bermudan_exercise_times), -1.0),
            (_make_option(style, "put", k2, maturity, rate, dividend_yield, bermudan_exercise_times), 1.0),
        )
        object.__setattr__(self, "legs", legs)
        object.__setattr__(self, "style", style)
        object.__setattr__(self, "option_type", "put")
        object.__setattr__(self, "k1", k1)
        object.__setattr__(self, "k2", k2)
        object.__setattr__(self, "maturity", maturity)
        object.__setattr__(self, "rate", rate)
        object.__setattr__(self, "dividend_yield", dividend_yield)
        object.__setattr__(self, "bermudan_exercise_times", bermudan_exercise_times)
        self.__post_init__()


@dataclass(frozen=True)
class BearPutSpread(VerticalSpread):
    """Long put at K1, short put at K2 (K1 < K2)."""

    def __init__(self, style: str, k1: float, k2: float, maturity: float, rate: float, dividend_yield: float = 0.0, bermudan_exercise_times: Sequence[float] | None = None) -> None:
        legs = (
            (_make_option(style, "put", k1, maturity, rate, dividend_yield, bermudan_exercise_times), 1.0),
            (_make_option(style, "put", k2, maturity, rate, dividend_yield, bermudan_exercise_times), -1.0),
        )
        object.__setattr__(self, "legs", legs)
        object.__setattr__(self, "style", style)
        object.__setattr__(self, "option_type", "put")
        object.__setattr__(self, "k1", k1)
        object.__setattr__(self, "k2", k2)
        object.__setattr__(self, "maturity", maturity)
        object.__setattr__(self, "rate", rate)
        object.__setattr__(self, "dividend_yield", dividend_yield)
        object.__setattr__(self, "bermudan_exercise_times", bermudan_exercise_times)
        self.__post_init__()


@dataclass(frozen=True)
class ButterflySpread(CombinedInstrument):
    """Butterfly spread (call or put): +1 at K1, -2 at Kmid, +1 at K2."""

    def __init__(self, style: str, option_type: str, k1: float, kmid: float, k2: float, maturity: float, rate: float, dividend_yield: float = 0.0, bermudan_exercise_times: Sequence[float] | None = None) -> None:
        ensure_str_in(option_type, "option_type", ("call", "put"))
        if not (k1 < kmid < k2):
            raise ValueError("Require k1 < kmid < k2 for a butterfly spread")
        legs = (
            (_make_option(style, option_type, k1, maturity, rate, dividend_yield, bermudan_exercise_times), 1.0),
            (_make_option(style, option_type, kmid, maturity, rate, dividend_yield, bermudan_exercise_times), -2.0),
            (_make_option(style, option_type, k2, maturity, rate, dividend_yield, bermudan_exercise_times), 1.0),
        )
        object.__setattr__(self, "legs", legs)


@dataclass(frozen=True)
class BoxSpread(CombinedInstrument):
    """Box spread: bull call spread + bear put spread (synthetic loan)."""

    def __init__(self, style: str, k1: float, k2: float, maturity: float, rate: float, dividend_yield: float = 0.0, bermudan_exercise_times: Sequence[float] | None = None) -> None:
        if k1 >= k2:
            raise ValueError("Require k1 < k2 for a box spread")
        legs = (
            (_make_option(style, "call", k1, maturity, rate, dividend_yield, bermudan_exercise_times), 1.0),
            (_make_option(style, "call", k2, maturity, rate, dividend_yield, bermudan_exercise_times), -1.0),
            (_make_option(style, "put", k1, maturity, rate, dividend_yield, bermudan_exercise_times), -1.0),
            (_make_option(style, "put", k2, maturity, rate, dividend_yield, bermudan_exercise_times), 1.0),
        )
        object.__setattr__(self, "legs", legs)


@dataclass(frozen=True)
class ChooserOption:
    """Chooser option: at choice_time, holder chooses a call or put with same K, T.

    Supports styles: european, american, bermudan. For bermudan, provide exercise times.
    Pricing is delegated to the provided pricer via `price(pricer, **kwargs)`.
    """

    strike: float
    maturity: float
    rate: float
    dividend_yield: float
    choice_time: float
    style: str = "european"
    bermudan_exercise_times: Sequence[float] | None = None

    def __post_init__(self) -> None:
        ensure_positive(self.strike, "strike")
        ensure_positive(self.maturity, "maturity")
        ct = ensure_positive(self.choice_time, "choice_time")
        if ct >= self.maturity:
            raise ValueError("choice_time must be < maturity")
        ensure_str_in(self.style, "style", ("european", "american", "bermudan"))
        if self.style == "bermudan":
            if self.bermudan_exercise_times is None:
                raise ValueError("bermudan_exercise_times must be provided for Bermudan chooser options")
            times = sorted(float(t) for t in self.bermudan_exercise_times)
            if len(times) == 0 or times[0] <= 0.0 or times[-1] > self.maturity:
                raise ValueError("bermudan_exercise_times must be within (0, maturity]")

    def price(self, pricer: Pricer, **kwargs) -> float:
        return float(pricer(instrument=self, **kwargs))



