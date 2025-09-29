"""Greeks computations for options.

Provides first- and second-order Greeks under the Black-Scholes-Merton model,
and model-agnostic numerical Greeks for arbitrary instruments via a pricing callable.
"""

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Optional

import numpy as np
from scipy.stats import norm

from .models_bsm import BlackScholesMerton as BSM
from .validation import ensure_positive, ensure_numeric, ensure_str_in, FinPetValueError


@dataclass
class Greeks:
    """Compute Greeks for European options using BSM.

    Methods return floats. Second-order Greeks include Vomma (Volga), Vanna, Charm, and Veta.
    """

    @staticmethod
    def delta(option_type: str, spot: float, strike: float, maturity: float, rate: float, dividend_yield: float, volatility: float) -> float:
        """Delta of a European option under BSM.

        Parameters
        ----------
        option_type : str
            "call" or "put".
        spot, strike, maturity, rate, dividend_yield, volatility : float
            Standard BSM inputs.

        Returns
        -------
        float
            Sensitivity of price to spot.
        """
        option = ensure_str_in(option_type, "option_type", ("call", "put"))
        d1 = BSM.d1(spot, strike, maturity, rate, dividend_yield, volatility)
        df_q = float(np.exp(-ensure_numeric(dividend_yield, "dividend_yield") * ensure_positive(maturity, "maturity")))
        if option == "call":
            return float(df_q * norm.cdf(d1))
        return float(df_q * (norm.cdf(d1) - 1.0))

    @staticmethod
    def gamma(spot: float, strike: float, maturity: float, rate: float, dividend_yield: float, volatility: float) -> float:
        """Gamma of a European option under BSM.

        Returns second derivative w.r.t. spot.
        """
        d1 = BSM.d1(spot, strike, maturity, rate, dividend_yield, volatility)
        s = ensure_positive(spot, "spot")
        v = ensure_positive(volatility, "volatility")
        t = ensure_positive(maturity, "maturity")
        df_q = float(np.exp(-ensure_numeric(dividend_yield, "dividend_yield") * t))
        return float(df_q * norm.pdf(d1) / (s * v * np.sqrt(t)))

    @staticmethod
    def vega(spot: float, strike: float, maturity: float, rate: float, dividend_yield: float, volatility: float) -> float:
        """Vega of a European option under BSM.

        Sensitivity to volatility (per 1.0 unit of vol).
        """
        d1 = BSM.d1(spot, strike, maturity, rate, dividend_yield, volatility)
        s = ensure_positive(spot, "spot")
        t = ensure_positive(maturity, "maturity")
        df_q = float(np.exp(-ensure_numeric(dividend_yield, "dividend_yield") * t))
        return float(df_q * s * norm.pdf(d1) * np.sqrt(t))

    @staticmethod
    def theta(option_type: str, spot: float, strike: float, maturity: float, rate: float, dividend_yield: float, volatility: float) -> float:
        """Theta of a European option under BSM.

        Sensitivity to passage of time (dPrice/dt).
        """
        option = ensure_str_in(option_type, "option_type", ("call", "put"))
        d1 = BSM.d1(spot, strike, maturity, rate, dividend_yield, volatility)
        d2 = BSM.d2(spot, strike, maturity, rate, dividend_yield, volatility)
        s = ensure_positive(spot, "spot")
        k = ensure_positive(strike, "strike")
        t = ensure_positive(maturity, "maturity")
        r = ensure_numeric(rate, "rate")
        q = ensure_numeric(dividend_yield, "dividend_yield")
        v = ensure_positive(volatility, "volatility")
        df_q = float(np.exp(-q * t))
        df_r = float(np.exp(-r * t))
        term1 = -df_q * s * norm.pdf(d1) * v / (2.0 * np.sqrt(t))
        if option == "call":
            term2 = q * df_q * s * norm.cdf(d1)
            term3 = -r * df_r * k * norm.cdf(d2)
        else:
            term2 = q * df_q * s * norm.cdf(-d1)
            term3 = -r * df_r * k * norm.cdf(-d2)
        return float(term1 + term2 + term3)

    @staticmethod
    def rho(option_type: str, spot: float, strike: float, maturity: float, rate: float, dividend_yield: float, volatility: float) -> float:
        """Rho of a European option under BSM.

        Sensitivity to interest rate.
        """
        option = ensure_str_in(option_type, "option_type", ("call", "put"))
        d2 = BSM.d2(spot, strike, maturity, rate, dividend_yield, volatility)
        t = ensure_positive(maturity, "maturity")
        r = ensure_numeric(rate, "rate")
        k = ensure_positive(strike, "strike")
        df_r = float(np.exp(-r * t))
        if option == "call":
            return float(t * k * df_r * norm.cdf(d2))
        return float(-t * k * df_r * norm.cdf(-d2))

    # Second-order Greeks
    @staticmethod
    def vomma(spot: float, strike: float, maturity: float, rate: float, dividend_yield: float, volatility: float) -> float:
        """Vomma (Volga): sensitivity of Vega to volatility."""
        d1 = BSM.d1(spot, strike, maturity, rate, dividend_yield, volatility)
        d2 = BSM.d2(spot, strike, maturity, rate, dividend_yield, volatility)
        v = ensure_positive(volatility, "volatility")
        return float(Greeks.vega(spot, strike, maturity, rate, dividend_yield, volatility) * d1 * d2 / v)

    @staticmethod
    def vanna(spot: float, strike: float, maturity: float, rate: float, dividend_yield: float, volatility: float) -> float:
        """Vanna: cross-derivative w.r.t. spot and volatility."""
        d1 = BSM.d1(spot, strike, maturity, rate, dividend_yield, volatility)
        t = ensure_positive(maturity, "maturity")
        q = ensure_numeric(dividend_yield, "dividend_yield")
        df_q = float(np.exp(-q * t))
        return float(df_q * norm.pdf(d1) * np.sqrt(t) * (1.0 - d1 / (volatility * np.sqrt(t))))

    @staticmethod
    def charm(option_type: str, spot: float, strike: float, maturity: float, rate: float, dividend_yield: float, volatility: float) -> float:
        """Charm: time decay of Delta."""
        option = ensure_str_in(option_type, "option_type", ("call", "put"))
        d1 = BSM.d1(spot, strike, maturity, rate, dividend_yield, volatility)
        d2 = BSM.d2(spot, strike, maturity, rate, dividend_yield, volatility)
        t = ensure_positive(maturity, "maturity")
        q = ensure_numeric(dividend_yield, "dividend_yield")
        df_q = float(np.exp(-q * t))
        sign = 1.0 if option == "call" else -1.0
        return float(
            df_q * (norm.pdf(d1) * (2.0 * q * t - d2 * volatility * np.sqrt(t)) / (2.0 * t * volatility * np.sqrt(t)) - sign * q * norm.cdf(sign * d1))
        )

    @staticmethod
    def veta(spot: float, strike: float, maturity: float, rate: float, dividend_yield: float, volatility: float) -> float:
        """Veta: time decay of Vega."""
        d1 = BSM.d1(spot, strike, maturity, rate, dividend_yield, volatility)
        t = ensure_positive(maturity, "maturity")
        q = ensure_numeric(dividend_yield, "dividend_yield")
        df_q = float(np.exp(-q * t))
        s = ensure_positive(spot, "spot")
        v = ensure_positive(volatility, "volatility")
        return float(-df_q * s * norm.pdf(d1) * np.sqrt(t) * (d1 / (2.0 * t) + q / v))


@dataclass
class NumericalGreeks:
    """Model-agnostic numerical Greeks using finite differences.

    Provide a pricing callable and a base parameter dict. The callable should accept the
    parameters as keyword arguments. Only the parameters available will be perturbed.

    Supported keys to perturb (if present in params):
    - spot (Delta)
    - strike (not used directly for standard Greeks, but kept for symmetry)
    - maturity (Theta)
    - rate (Rho)
    - dividend_yield (appears in some definitions; not a Greek here)
    - volatility (Vega)

    For Gamma, we use the second derivative wrt spot.
    """

    rel_step: float = 1e-5
    abs_min_step: float = 1e-6

    def _step(self, x: float) -> float:
        """Compute a finite-difference step size based on relative and absolute caps."""
        h = max(abs(x) * self.rel_step, self.abs_min_step)
        return h

    def _price(self, pricer: Callable[..., float], params: Dict[str, float]) -> float:
        """Evaluate the pricing callable and ensure the result is finite."""
        value = pricer(**params)
        if not np.isfinite(value):
            raise FinPetValueError("Pricing callable returned a non-finite value")
        return float(value)

    def delta(self, pricer: Callable[..., float], params: Dict[str, float]) -> float:
        """Central-difference estimate of Delta for an arbitrary instrument.

        Parameters
        ----------
        pricer : Callable[..., float]
            Pricing function accepting keyword params (must include 'spot').
        params : dict
            Base parameters for pricing.
        """
        if "spot" not in params:
            raise FinPetValueError("'spot' is required in params to compute Delta")
        s = float(params["spot"])
        h = self._step(s)
        p_up = self._price(pricer, {**params, "spot": s + h})
        p_dn = self._price(pricer, {**params, "spot": s - h})
        return (p_up - p_dn) / (2.0 * h)

    def gamma(self, pricer: Callable[..., float], params: Dict[str, float]) -> float:
        """Second derivative w.r.t. spot via three-point stencil."""
        if "spot" not in params:
            raise FinPetValueError("'spot' is required in params to compute Gamma")
        s = float(params["spot"])
        h = self._step(s)
        p_up = self._price(pricer, {**params, "spot": s + h})
        p = self._price(pricer, params)
        p_dn = self._price(pricer, {**params, "spot": s - h})
        return (p_up - 2.0 * p + p_dn) / (h * h)

    def vega(self, pricer: Callable[..., float], params: Dict[str, float]) -> float:
        """Central-difference estimate w.r.t. volatility-like parameter 'volatility'."""
        if "volatility" not in params:
            raise FinPetValueError("'volatility' is required in params to compute Vega")
        v = float(params["volatility"])
        h = self._step(v)
        p_up = self._price(pricer, {**params, "volatility": v + h})
        p_dn = self._price(pricer, {**params, "volatility": v - h})
        return (p_up - p_dn) / (2.0 * h)

    def theta(self, pricer: Callable[..., float], params: Dict[str, float]) -> float:
        """Central-difference estimate w.r.t. maturity (time decay)."""
        if "maturity" not in params:
            raise FinPetValueError("'maturity' is required in params to compute Theta")
        t = float(params["maturity"])
        # Ensure maturity stays positive
        h = min(self._step(t), 0.5 * max(t, self.abs_min_step))
        p_up = self._price(pricer, {**params, "maturity": t + h})
        p_dn = self._price(pricer, {**params, "maturity": max(t - h, self.abs_min_step)})
        return (p_up - p_dn) / (2.0 * h)

    def rho(self, pricer: Callable[..., float], params: Dict[str, float]) -> float:
        """Central-difference estimate w.r.t. interest rate."""
        if "rate" not in params:
            raise FinPetValueError("'rate' is required in params to compute Rho")
        r = float(params["rate"])
        h = self._step(r if r != 0.0 else 1.0)
        p_up = self._price(pricer, {**params, "rate": r + h})
        p_dn = self._price(pricer, {**params, "rate": r - h})
        return (p_up - p_dn) / (2.0 * h)

    def compute(self, pricer: Callable[..., float], params: Dict[str, float], which: Optional[Iterable[str]] = None) -> Dict[str, float]:
        """Compute a selection of numerical Greeks.

        Parameters
        ----------
        pricer : Callable[..., float]
            Pricing function for the instrument.
        params : dict
            Base parameters; should include keys relevant to chosen Greeks.
        which : Iterable[str], optional
            Subset of {"delta","gamma","vega","theta","rho"}; all by default.
        """
        to_compute = list(which) if which is not None else ["delta", "gamma", "vega", "theta", "rho"]
        result: Dict[str, float] = {}
        for name in to_compute:
            if name == "delta":
                result["delta"] = self.delta(pricer, params)
            elif name == "gamma":
                result["gamma"] = self.gamma(pricer, params)
            elif name == "vega":
                result["vega"] = self.vega(pricer, params)
            elif name == "theta":
                result["theta"] = self.theta(pricer, params)
            elif name == "rho":
                result["rho"] = self.rho(pricer, params)
            else:
                raise FinPetValueError(f"Unknown greek '{name}'")
        return result


