# Black-Scholes valuation, shared by market instruments and VaR.

import math

import numpy as np
from scipy.stats import distributions

from ..price import Price

N = distributions.norm.cdf
fi = distributions.norm.pdf


def _black_scholes(
    S: Price | float | np.ndarray,
    K: Price | float,
    r: float,
    volatility: float,
    Time: float,
    d: float,
    *,
    call: bool,
) -> dict:
    """European BSM price and Greeks for a scalar or a NumPy spot array.

    Scalar expiry premiums retain the historical Price return type. Scenario
    arrays intentionally remain unrounded so Monte Carlo losses are not
    distorted by per-scenario price tick rounding.
    """
    scalar = np.ndim(S) == 0
    spot = np.asarray(S, dtype=float)
    strike = float(K)
    rate = float(r)
    sigma = float(volatility)
    years = float(Time)
    dividend = float(d)

    if (
        not np.isfinite(spot).all()
        or (spot < 0).any()
        or not all(math.isfinite(x) for x in (strike, rate, sigma, years, dividend))
        or strike <= 0
        or sigma < 0
        or years < 0
    ):
        raise ValueError("Invalid Black-Scholes inputs")

    zeros = np.zeros_like(spot, dtype=float)
    if years == 0:
        payoff = spot - strike if call else strike - spot
        price = np.maximum(payoff, 0)
        if call:
            delta = np.where(spot > strike, 1.0, np.where(spot == strike, 0.5, 0.0))
        else:
            delta = np.where(spot < strike, -1.0, np.where(spot == strike, -0.5, 0.0))
        gamma = vega = theta = rho = zeros
        intrinsic = price
    else:
        spot_discount = math.exp(-dividend * years)
        strike_discount = math.exp(-rate * years)
        discounted_spot = spot * spot_discount
        discounted_strike = strike * strike_discount
        forward_moneyness = discounted_spot - discounted_strike
        intrinsic = np.maximum(forward_moneyness if call else -forward_moneyness, 0)
        if sigma == 0:
            # Zero-variance (deterministic) BSM limit. At the payoff kink,
            # select the midpoint delta as a documented convention.
            is_itm = forward_moneyness > 0 if call else forward_moneyness < 0
            is_atm = forward_moneyness == 0
            probability = np.where(is_itm, 1.0, np.where(is_atm, 0.5, 0.0))
            price = intrinsic
            delta = spot_discount * probability * (1 if call else -1)
            gamma = vega = zeros
            theta = np.where(
                is_itm,
                (dividend * discounted_spot - rate * discounted_strike) * (1 if call else -1),
                0.0,
            )
            rho = strike * years * strike_discount * probability * (1 if call else -1)
        else:
            sqrt_years = math.sqrt(years)
            vol_time = sigma * sqrt_years
            with np.errstate(divide="ignore", invalid="ignore"):
                d1 = (np.log(spot / strike) + (rate - dividend + sigma**2 / 2) * years) / vol_time
            d2 = d1 - vol_time
            pdf = fi(d1)
            if call:
                price = discounted_spot * N(d1) - discounted_strike * N(d2)
                delta = spot_discount * N(d1)
                theta_carry = dividend * discounted_spot * N(d1) - rate * discounted_strike * N(d2)
                rho = strike * years * strike_discount * N(d2)
            else:
                price = discounted_strike * N(-d2) - discounted_spot * N(-d1)
                delta = -spot_discount * N(-d1)
                theta_carry = -dividend * discounted_spot * N(-d1) + rate * discounted_strike * N(
                    -d2
                )
                rho = -strike * years * strike_discount * N(-d2)
            # At spot=0, the limiting gamma is zero; no division by zero.
            with np.errstate(divide="ignore", invalid="ignore"):
                gamma = np.where(spot > 0, spot_discount * pdf / (spot * vol_time), 0.0)
            theta = -discounted_spot * pdf * sigma / (2 * sqrt_years) + theta_carry
            vega = discounted_spot * pdf * sqrt_years

    time_value = price - intrinsic
    values = {
        "price": price,
        "delta": delta,
        "gamma": gamma,
        "theta": theta,
        "vega": vega,
        "rho": rho,
        "intrinsicValue": intrinsic,
        "timeValue": time_value,
    }
    if scalar:
        values = {name: float(value) for name, value in values.items()}
        if years == 0:
            for name in ("price", "intrinsicValue", "timeValue"):
                values[name] = Price(values[name])
    return values


def BSCall(
    S: Price | float | np.ndarray,
    K: Price | float,
    r: float,
    volatility: float,
    Time: float,
    d: float = 0.0,
) -> dict:
    """Black-Scholes European call; accepts a scalar or scenario spot array."""
    return _black_scholes(S, K, r, volatility, Time, d, call=True)


def BSPut(
    S: Price | float | np.ndarray,
    K: Price | float,
    r: float,
    volatility: float,
    Time: float,
    d: float = 0.0,
) -> dict:
    """Black-Scholes European put; accepts a scalar or scenario spot array."""
    return _black_scholes(S, K, r, volatility, Time, d, call=False)


# function calculating the volatility implied in the   prices of options
def impliedVolatilityInCall(C, S, K, Time, r, d=0.0):
    # if C > S - math.exp(r*Time) * K:
    # bullshit: raise RuntimeError("Cannot calculate implied volatility! - There is opportunity of arbitrage by writing a call, buying stock for borrowed money!")
    # if C < S - math.exp(r*Time) * K:
    # bullshit either:raise RuntimeError("Here is arbitrage opportunity: short sell stocks, buy calls, "
    if C > S:
        print("Impossible C, S, K, Time, r, d: ", C, S, K, Time, r, d)
        raise RuntimeError(
            "Impossible! (arbitrage opportunity - call is more expensive than spot!)"
        )
    # if C < math.exp(r*Time) * S - K: #that was wrong
    if C < S - K * math.exp(-r * Time):
        print("Impossible C, S, K, Time, r, d: ", C, S, K, Time, r, d)
        raise RuntimeError(
            "Impossible! (arbitrage opportunity - call to cheap! call cheaper than intrinsic option value.)"
        )

    # a simple artefact function
    # TODO
    # def innerCall(vega):
    #   return BSCall(S, K, r, vega, Time, d)
    #
    # return ChrustSolver.solveNonlinear(C, innerCall, 0.0005, (0.001, 50))


def impliedVolatilityInPut(P, S, K, Time, r, d=0.0):
    if P >= math.exp(r * Time) * K:
        print("Impossible P, S, K, Time, r, d: ", P, S, K, Time, r, d)
        raise RuntimeError("Impossible! (arbitrage opportunity - put too expensive!)")
    if P < math.exp(-r * Time) * K - S:
        print("Impossible P, S, K, Time, r, d: ", P, S, K, Time, r, d)
        raise RuntimeError("Impossible! (arbitrage opportunity - put too cheap!)")

    # simple artefact function:
    def innerPut(vega):
        return BSPut(S, K, r, vega, Time, d)

    # TODO
    # return ChrustSolver.solveNonlinear(P, innerPut, 0.0005, (0.001, 50))


# function calculates whether there is an arbitrage opportunity (and points it) on the futures and options market
# we should include the bid-ask spread here, ey?  - the C, P, Futures and maybe even r should passed as tuples
def findOptionArbitrage(C, P, K, Futures, Time, r):
    if isinstance(r, float):
        FVoptionSpread = ((C[1] - P[0]) * math.exp(r * Time), (C[0] - P[1]) * math.exp(r * Time))
    elif isinstance(r, tuple) and len(r) == 2:
        FVoptionSpread = (
            (C[1] - P[0]) * math.exp(r[0] * Time),
            (C[0] - P[1]) * math.exp(r[1] * Time),
        )
    else:
        raise ValueError(
            "Invalid interest rate format! It should be float or two-element tuple, is %s."
            % r.__class__
        )
    if FVoptionSpread[0] <= Futures[1] - K and FVoptionSpread[1] >= Futures[0] - K:  # check it!
        return False  # no arbitrage opportunity
    else:
        if FVoptionSpread[0] > Futures[1] - K:  # ckeck it
            print("Arbirage opportunity: buy Put, sell Call and take long Forward position.")
        else:
            print("Arbitrage opportunity: buy Call, sell Put and take short Forward position.")
        return True  # arbitrage opportunity as printed above


def findOptionArbitrage2(C1, P1, K1, C2, P2, K2, r, T):
    if (C1[1] - P1[0]) - (C2[0] - P2[1]) < (K1 - K2) * math.exp(
        -r * T
    ):  # comparing synthetic long position on K1 with synthetic short on K2
        print(
            "Arbitrage opportunity: create synthetic long on %f and synthetic short on %f expiring %f."
            % (K1, K2, T)
        )
        return True
    if (C1[0] - P1[1]) - (C2[1] - P2[0]) > (K2 - K1) * math.exp(
        -r * T
    ):  # check it! - should be ok, as it is a mirror from the above equation
        print(
            "Arbitrage opportunity: create synthetic short on %f and synthethic long on %f (both expiring %f)."
            % (K1, K2, T)
        )
        return True
    return False
