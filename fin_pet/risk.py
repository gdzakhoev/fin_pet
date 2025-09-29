"""Risk metrics for portfolios and return series.

Implements Value at Risk (VaR), Expected Shortfall (ES, CVaR), Sharpe ratio,
Sortino ratio, and Maximum Drawdown. Inputs can be scalars or array-like;
scalars are treated as length-1 arrays.
"""

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np

from .validation import ensure_array_like, ensure_bounds, ensure_numeric


Method = Literal["historical", "gaussian"]


@dataclass
class RiskMetrics:
    """Compute common risk metrics from return series."""

    @staticmethod
    def var(returns, alpha: float = 0.95, method: Method = "historical", mean: Optional[float] = None, std: Optional[float] = None) -> float:
        """Value at Risk (VaR) at confidence level alpha.

        Parameters
        ----------
        returns : array-like or float
            Return series (percentage or log-returns) or a scalar.
        alpha : float
            Confidence level in [0,1].
        method : {"historical","gaussian"}
            Historical quantile or parametric normal approximation.
        mean, std : float, optional
            If provided with method="gaussian", overrides sample estimates.
        """
        a = ensure_bounds(alpha, "alpha", 0.0, 1.0)
        data = ensure_array_like(returns, "returns")
        if method == "historical":
            return float(-np.quantile(data, 1.0 - a))
        # Gaussian parametric
        mu = ensure_numeric(mean if mean is not None else float(np.mean(data)), "mean")
        sigma = float(np.std(data, ddof=1)) if std is None else float(std)
        if sigma < 0.0:
            raise ValueError("Standard deviation must be >= 0")
        from scipy.stats import norm

        q = norm.ppf(1.0 - a)
        return float(-(mu + q * sigma))

    @staticmethod
    def expected_shortfall(returns, alpha: float = 0.95, method: Method = "historical", mean: Optional[float] = None, std: Optional[float] = None) -> float:
        """Expected Shortfall (ES, CVaR) at confidence level alpha.

        Same parameters as `var`.
        """
        a = ensure_bounds(alpha, "alpha", 0.0, 1.0)
        data = ensure_array_like(returns, "returns")
        if method == "historical":
            cutoff = np.quantile(data, 1.0 - a)
            tail = data[data <= cutoff]
            if tail.size == 0:
                return float(-cutoff)
            return float(-np.mean(tail))
        # Gaussian parametric ES
        from scipy.stats import norm

        mu = ensure_numeric(mean if mean is not None else float(np.mean(data)), "mean")
        sigma = float(np.std(data, ddof=1)) if std is None else float(std)
        if sigma <= 0.0:
            return float(-mu)
        z = norm.ppf(1.0 - a)
        es = mu - sigma * norm.pdf(z) / (1.0 - a)
        return float(-es)

    @staticmethod
    def sharpe_ratio(returns, risk_free_rate: float = 0.0, periods_per_year: Optional[int] = None) -> float:
        """Sharpe ratio of a return series.

        If periods_per_year is provided, annualizes the ratio.
        """
        data = ensure_array_like(returns, "returns")
        excess = data - ensure_numeric(risk_free_rate, "risk_free_rate")
        mean_excess = float(np.mean(excess))
        std_excess = float(np.std(excess, ddof=1))
        if std_excess == 0.0:
            return 0.0
        sr = mean_excess / std_excess
        if periods_per_year:
            return float(np.sqrt(periods_per_year) * sr)
        return float(sr)

    @staticmethod
    def sortino_ratio(returns, risk_free_rate: float = 0.0, periods_per_year: Optional[int] = None) -> float:
        """Sortino ratio using downside deviation.

        If periods_per_year is provided, annualizes the ratio.
        """
        data = ensure_array_like(returns, "returns")
        rf = ensure_numeric(risk_free_rate, "risk_free_rate")
        excess = data - rf
        downside = ExcessDownsideCalculator.downside_deviation(excess)
        if downside == 0.0:
            return 0.0
        sr = float(np.mean(excess)) / downside
        if periods_per_year:
            return float(np.sqrt(periods_per_year) * sr)
        return float(sr)

    @staticmethod
    def max_drawdown(values) -> float:
        """Maximum drawdown of a cumulative PnL/return series.

        Interprets `values` as period returns and computes drawdown of cumulative sum.
        """
        series = ensure_array_like(values, "values")
        if series.size == 0:
            return 0.0
        cumulative = np.cumsum(series)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = running_max - cumulative
        return float(np.max(drawdowns))


class ExcessDownsideCalculator:
    """Utility to compute downside deviation for Sortino."""

    @staticmethod
    def downside_deviation(excess_returns) -> float:
        """Downside deviation (square root of mean squared negative excess returns)."""
        data = ensure_array_like(excess_returns, "excess_returns")
        downside = np.where(data < 0.0, data, 0.0)
        return float(np.sqrt(np.mean(downside**2)))


