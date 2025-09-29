"""Exotic instruments.

Defines contracts for barrier, forward-start, binary, lookback, Asian options, and forward contracts.
Pricing is delegated to the provided pricing engine via a generic `price(pricer, **kwargs)` API.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Sequence

from .validation import ensure_positive, ensure_str_in


Pricer = Callable[..., float]


@dataclass(frozen=True)
class ForwardContract:
    """Forward contract specification with delivery at maturity."""

    maturity: float
    delivery_price: float
    rate: float
    dividend_yield: float = 0.0

    def __post_init__(self) -> None:
        ensure_positive(self.maturity, "maturity")
        ensure_positive(self.delivery_price, "delivery_price")

    def price(self, pricer: Pricer, **kwargs) -> float:
        """Delegate pricing to the provided callable."""
        return float(pricer(instrument=self, **kwargs))


BarrierType = Literal[
    "up-and-in",
    "up-and-out",
    "down-and-in",
    "down-and-out",
]


@dataclass(frozen=True)
class BarrierOption:
    """Barrier option with standard knock-in/knock-out styles."""

    option_type: str
    barrier_type: BarrierType
    barrier_level: float
    rebate: float
    strike: float
    maturity: float
    rate: float
    dividend_yield: float = 0.0

    def __post_init__(self) -> None:
        ensure_str_in(self.option_type, "option_type", ("call", "put"))
        ensure_str_in(self.barrier_type, "barrier_type", ("up-and-in", "up-and-out", "down-and-in", "down-and-out"))
        ensure_positive(self.barrier_level, "barrier_level")
        ensure_positive(self.strike, "strike")
        ensure_positive(self.maturity, "maturity")
        if self.rebate < 0.0:
            raise ValueError("rebate must be >= 0")

    def price(self, pricer: Pricer, **kwargs) -> float:
        """Delegate pricing to the provided callable."""
        return float(pricer(instrument=self, **kwargs))


@dataclass(frozen=True)
class ForwardStartOption:
    """Forward-start option: strike is set at the forward-start time as a multiple of spot."""

    option_type: str
    set_time: float
    maturity: float
    strike_multiple: float
    rate: float
    dividend_yield: float = 0.0

    def __post_init__(self) -> None:
        ensure_str_in(self.option_type, "option_type", ("call", "put"))
        ensure_positive(self.set_time, "set_time")
        ensure_positive(self.maturity, "maturity")
        if not (0.0 < self.set_time < self.maturity):
            raise ValueError("set_time must be in (0, maturity)")
        ensure_positive(self.strike_multiple, "strike_multiple")

    def price(self, pricer: Pricer, **kwargs) -> float:
        """Delegate pricing to the provided callable."""
        return float(pricer(instrument=self, **kwargs))


@dataclass(frozen=True)
class BinaryOption:
    """Cash-or-nothing digital option."""

    option_type: str
    strike: float
    maturity: float
    payout: float
    rate: float
    dividend_yield: float = 0.0

    def __post_init__(self) -> None:
        ensure_str_in(self.option_type, "option_type", ("call", "put"))
        ensure_positive(self.strike, "strike")
        ensure_positive(self.maturity, "maturity")
        ensure_positive(self.payout, "payout")

    def price(self, pricer: Pricer, **kwargs) -> float:
        """Delegate pricing to the provided callable."""
        return float(pricer(instrument=self, **kwargs))


AverageType = Literal["arithmetic", "geometric"]


@dataclass(frozen=True)
class AsianOption:
    """Asian option with arithmetic or geometric averaging (fixed strike variant)."""

    option_type: str
    strike: float
    maturity: float
    rate: float
    dividend_yield: float
    average_type: AverageType = "arithmetic"
    observation_times: Sequence[float] | None = None

    def __post_init__(self) -> None:
        ensure_str_in(self.option_type, "option_type", ("call", "put"))
        ensure_str_in(self.average_type, "average_type", ("arithmetic", "geometric"))
        ensure_positive(self.strike, "strike")
        ensure_positive(self.maturity, "maturity")
        if self.observation_times is not None:
            times = sorted(float(t) for t in self.observation_times)
            if len(times) == 0 or times[0] <= 0.0 or times[-1] > self.maturity:
                raise ValueError("observation_times must be within (0, maturity]")

    def price(self, pricer: Pricer, **kwargs) -> float:
        """Delegate pricing to the provided callable."""
        return float(pricer(instrument=self, **kwargs))


@dataclass(frozen=True)
class LookbackOption:
    """Lookback option with fixed strike (payoff depends on min/max of the path)."""

    option_type: str
    strike: float
    maturity: float
    rate: float
    dividend_yield: float

    def __post_init__(self) -> None:
        ensure_str_in(self.option_type, "option_type", ("call", "put"))
        ensure_positive(self.strike, "strike")
        ensure_positive(self.maturity, "maturity")

    def price(self, pricer: Pricer, **kwargs) -> float:
        """Delegate pricing to the provided callable."""
        return float(pricer(instrument=self, **kwargs))



