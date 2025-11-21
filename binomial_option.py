"""Binomial option pricing model implemented in Python.

Provides a reusable function ``binomial_option_price`` that prices European or
American call/put options using the Cox-Ross-Rubinstein binomial tree.
"""
from __future__ import annotations

import math
from typing import Literal


OptionType = Literal["call", "put"]


def binomial_option_price(
    spot: float,
    strike: float,
    maturity: float,
    risk_free_rate: float,
    volatility: float,
    steps: int,
    option_type: OptionType = "call",
    american: bool = False,
) -> float:
    """Price an option with the Cox-Ross-Rubinstein binomial tree.

    Args:
        spot: Current underlying price (S_0).
        strike: Strike price (K).
        maturity: Time to maturity in years (T).
        risk_free_rate: Continuously compounded annual risk-free rate (r).
        volatility: Annualized volatility of the underlying (sigma).
        steps: Number of time steps in the tree (N). Must be positive.
        option_type: "call" for a call option or "put" for a put option.
        american: Whether to allow early exercise (American-style). Defaults
            to ``False`` for European-style options.

    Returns:
        The option price at the root of the tree.

    Raises:
        ValueError: If an unsupported ``option_type`` is provided or ``steps``
            is not positive.
    """
    if steps <= 0:
        raise ValueError("steps must be positive")

    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'")

    dt = maturity / steps
    up = math.exp(volatility * math.sqrt(dt))
    down = 1 / up
    discount = math.exp(-risk_free_rate * dt)
    p = (math.exp(risk_free_rate * dt) - down) / (up - down)

    # Initialize terminal payoffs.
    payoffs = []
    for i in range(steps + 1):
        spot_t = spot * (up ** i) * (down ** (steps - i))
        if option_type == "call":
            payoff = max(spot_t - strike, 0.0)
        else:
            payoff = max(strike - spot_t, 0.0)
        payoffs.append(payoff)

    # Backward induction through the tree.
    for step in range(steps - 1, -1, -1):
        for i in range(step + 1):
            continuation = discount * (p * payoffs[i + 1] + (1 - p) * payoffs[i])
            if american:
                spot_t = spot * (up ** i) * (down ** (step - i))
                if option_type == "call":
                    exercise = max(spot_t - strike, 0.0)
                else:
                    exercise = max(strike - spot_t, 0.0)
                payoffs[i] = max(continuation, exercise)
            else:
                payoffs[i] = continuation

    return payoffs[0]


if __name__ == "__main__":
    price = binomial_option_price(
        spot=100,
        strike=100,
        maturity=1.0,
        risk_free_rate=0.05,
        volatility=0.2,
        steps=200,
        option_type="call",
        american=False,
    )
    print(f"European call price: {price:.4f}")
