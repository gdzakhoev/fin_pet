"""Benchmarking utilities against external libraries and analytical solutions.

Attempts to compare outputs with QuantLib and similar libraries if installed.
No hard dependency is introduced; this module will skip comparisons if imports fail.
"""

import math
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy.stats import norm

from .models_bsm import BlackScholesMerton as BSM
from .models_binomial import BinomialTreePricer
from .models_mc import MonteCarloEngine
from .models_heston import HestonModel
from .instruments_standard import EuropeanOption, AmericanOption
from .greeks import Greeks
from .implied_vol import ImpliedVolatility
from .risk import RiskMetrics


def analytical_bsm_call(spot: float, strike: float, maturity: float, rate: float, dividend_yield: float, volatility: float) -> float:
    """Analytical BSM call price for reference."""
    d1 = (math.log(spot / strike) + (rate - dividend_yield + 0.5 * volatility**2) * maturity) / (volatility * math.sqrt(maturity))
    d2 = d1 - volatility * math.sqrt(maturity)
    
    call_price = spot * math.exp(-dividend_yield * maturity) * norm.cdf(d1) - strike * math.exp(-rate * maturity) * norm.cdf(d2)
    return call_price


def compare_bsm_call() -> Dict[str, Any]:
    """Compare BSM call pricing across different implementations."""
    s, k, t, r, q, v = 100.0, 100.0, 1.0, 0.02, 0.01, 0.2
    
    results = {}
    
    # Our implementation
    results["fin_pet"] = BSM.price("call", s, k, t, r, q, v)
    
    # Analytical reference
    results["analytical"] = analytical_bsm_call(s, k, t, r, q, v)
    
    # Binomial tree
    tree = BinomialTreePricer(steps=1000, tree="crr")
    euro_opt = EuropeanOption("call", k, t, r, q)
    results["binomial_1000"] = tree.price(euro_opt, s, v)
    
    # Monte Carlo
    mc = MonteCarloEngine(num_paths=50000, num_steps=252, antithetic=False, seed=42)
    def call_payoff(path):
        return max(0.0, path[-1] - k)
    results["monte_carlo"] = mc.price(s, r, q, v, t, call_payoff)
    
    # External libraries
    try:
        import mpmath as mp  # type: ignore
        def norm_cdf_mp(x: float) -> float:
            return 0.5 * (1.0 + mp.erf(x / mp.sqrt(2)))
        
        d1 = (math.log(s / k) + (r - q + 0.5 * v * v) * t) / (v * math.sqrt(t))
        d2 = d1 - v * math.sqrt(t)
        ref = math.exp(-q * t) * s * norm_cdf_mp(d1) - math.exp(-r * t) * k * norm_cdf_mp(d2)
        results["mpmath_ref"] = float(ref)
    except Exception:
        results["mpmath_ref"] = None

    try:
        import QuantLib as ql  # type: ignore
        day_count = ql.Actual365Fixed()
        calendar = ql.NullCalendar()
        settlement_date = ql.Date.todaysDate()
        payoff = ql.PlainVanillaPayoff(ql.Option.Call, k)
        exercise = ql.EuropeanExercise(settlement_date + int(t * 365))
        option = ql.VanillaOption(payoff, exercise)
        spot_handle = ql.QuoteHandle(ql.SimpleQuote(s))
        flat_ts = ql.YieldTermStructureHandle(ql.FlatForward(settlement_date, r, day_count))
        dividend_yield = ql.YieldTermStructureHandle(ql.FlatForward(settlement_date, q, day_count))
        vol_ts = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(settlement_date, calendar, v, day_count))
        process = ql.BlackScholesMertonProcess(spot_handle, dividend_yield, flat_ts, vol_ts)
        engine = ql.AnalyticEuropeanEngine(process)
        option.setPricingEngine(engine)
        results["quantlib"] = option.NPV()
    except Exception:
        results["quantlib"] = None

    return results


def compare_greeks() -> Dict[str, Any]:
    """Compare Greeks calculations."""
    s, k, t, r, q, v = 100.0, 100.0, 1.0, 0.02, 0.01, 0.2
    
    results = {}
    
    # Analytical Greeks
    d1 = (math.log(s / k) + (r - q + 0.5 * v * v) * t) / (v * math.sqrt(t))
    d2 = d1 - v * math.sqrt(t)
    
    # Delta
    analytical_delta = math.exp(-q * t) * norm.cdf(d1)
    results["delta_analytical"] = analytical_delta
    results["delta_fin_pet"] = Greeks.delta("call", s, k, t, r, q, v)
    
    # Gamma
    analytical_gamma = math.exp(-q * t) * norm.pdf(d1) / (s * v * math.sqrt(t))
    results["gamma_analytical"] = analytical_gamma
    results["gamma_fin_pet"] = Greeks.gamma(s, k, t, r, q, v)
    
    # Vega
    analytical_vega = s * math.exp(-q * t) * norm.pdf(d1) * math.sqrt(t)
    results["vega_analytical"] = analytical_vega
    results["vega_fin_pet"] = Greeks.vega(s, k, t, r, q, v)
    
    return results


def compare_implied_volatility() -> Dict[str, Any]:
    """Test implied volatility solver."""
    s, k, t, r, q, v_true = 100.0, 100.0, 1.0, 0.02, 0.01, 0.2
    
    # Get market price
    market_price = BSM.price("call", s, k, t, r, q, v_true)
    
    # Solve for implied volatility
    iv_solver = ImpliedVolatility()
    iv_solved = iv_solver.solve("call", s, k, t, r, q, market_price)
    
    return {
        "true_volatility": v_true,
        "market_price": market_price,
        "implied_volatility": iv_solved,
        "error": abs(iv_solved - v_true),
        "relative_error": abs(iv_solved - v_true) / v_true * 100
    }


def compare_american_vs_european() -> Dict[str, Any]:
    """Compare American vs European option prices."""
    s, k, t, r, q, v = 100.0, 100.0, 1.0, 0.02, 0.01, 0.2
    
    # European
    euro_opt = EuropeanOption("put", k, t, r, q)
    euro_price = BSM.price("put", s, k, t, r, q, v)
    
    # American (using binomial tree)
    american_opt = AmericanOption("put", k, t, r, q)
    tree = BinomialTreePricer(steps=1000, tree="crr")
    american_price = tree.price(american_opt, s, v)
    
    return {
        "european_price": euro_price,
        "american_price": american_price,
        "early_exercise_premium": american_price - euro_price,
        "relative_premium": (american_price - euro_price) / euro_price * 100
    }


def compare_risk_metrics() -> Dict[str, Any]:
    """Test risk metrics calculations."""
    # Generate sample returns
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, 1000)  # 0.1% daily return, 2% daily vol
    
    results = {}
    
    # VaR and ES
    results["var_95"] = RiskMetrics.var(returns, alpha=0.95, method="historical")
    results["var_99"] = RiskMetrics.var(returns, alpha=0.99, method="historical")
    results["es_95"] = RiskMetrics.expected_shortfall(returns, alpha=0.95, method="historical")
    
    # Sharpe and Sortino
    results["sharpe"] = RiskMetrics.sharpe_ratio(returns, risk_free_rate=0.0)
    results["sortino"] = RiskMetrics.sortino_ratio(returns, risk_free_rate=0.0)
    
    # Max drawdown
    results["max_drawdown"] = RiskMetrics.max_drawdown(returns)
    
    return results


def run_comprehensive_benchmark() -> Dict[str, Any]:
    """Run all benchmark tests."""
    results = {}
    
    print("Running BSM pricing benchmark...")
    results["bsm_pricing"] = compare_bsm_call()
    
    print("Running Greeks benchmark...")
    results["greeks"] = compare_greeks()
    
    print("Running implied volatility benchmark...")
    results["implied_vol"] = compare_implied_volatility()
    
    print("Running American vs European benchmark...")
    results["american_vs_european"] = compare_american_vs_european()
    
    print("Running risk metrics benchmark...")
    results["risk_metrics"] = compare_risk_metrics()
    
    return results


if __name__ == "__main__":
    results = run_comprehensive_benchmark()
    
    print("\n" + "="*60)
    print("COMPREHENSIVE BENCHMARK RESULTS")
    print("="*60)
    
    for test_name, test_results in results.items():
        print(f"\n{test_name.upper().replace('_', ' ')}:")
        print("-" * 40)
        for key, value in test_results.items():
            if isinstance(value, float):
                print(f"{key}: {value:.6f}")
            else:
                print(f"{key}: {value}")


