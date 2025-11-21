"""Binomial option pricing model implemented in Python.

Provides a reusable function ``binomial_option_price`` that prices European or
American call/put options using the Cox-Ross-Rubinstein binomial tree.
"""
from __future__ import annotations

import math
import random
from typing import Callable, Literal, NamedTuple, Sequence


OptionType = Literal["call", "put"]


class MonteCarloResult(NamedTuple):
    price: float
    standard_error: float


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


def monte_carlo_option_price(
    payoff: Callable[[Sequence[float]], float],
    *,
    spot: float,
    maturity: float,
    risk_free_rate: float,
    volatility: float,
    steps: int = 1,
    paths: int = 10_000,
    seed: int | None = None,
) -> MonteCarloResult:
    """Price an option via Monte Carlo simulation with a custom payoff.

    Simulates geometric Brownian motion paths and evaluates a user-supplied
    payoff callable on each simulated price path. The payoff function receives
    a sequence of spot prices starting at ``spot`` and ending at maturity.

    Args:
        payoff: Callable accepting a price path and returning the payoff for
            that path.
        spot: Current underlying price (S_0).
        maturity: Time to maturity in years (T).
        risk_free_rate: Continuously compounded annual risk-free rate (r).
        volatility: Annualized volatility of the underlying (sigma).
        steps: Number of time steps for path discretization. Defaults to 1
            (terminal payoff only).
        paths: Number of Monte Carlo paths to simulate. Defaults to 10,000.
        seed: Optional random seed for reproducibility.

    Returns:
        ``MonteCarloResult`` containing the discounted price estimate and its
        standard error.

    Raises:
        ValueError: If ``steps`` or ``paths`` are not positive.
    """
    if steps <= 0:
        raise ValueError("steps must be positive")
    if paths <= 0:
        raise ValueError("paths must be positive")

    rng = random.Random(seed)
    dt = maturity / steps
    drift = (risk_free_rate - 0.5 * volatility ** 2) * dt
    diffusion = volatility * math.sqrt(dt)
    discount = math.exp(-risk_free_rate * maturity)

    payoff_sum = 0.0
    payoff_sq_sum = 0.0

    for _ in range(paths):
        path = [spot]
        for _ in range(steps):
            z = rng.gauss(0.0, 1.0)
            next_spot = path[-1] * math.exp(drift + diffusion * z)
            path.append(next_spot)

        payoff_val = payoff(path)
        payoff_sum += payoff_val
        payoff_sq_sum += payoff_val ** 2

    mean_payoff = payoff_sum / paths
    variance = max(payoff_sq_sum / paths - mean_payoff ** 2, 0.0)
    standard_error = math.sqrt(variance / paths)

    price = discount * mean_payoff
    return MonteCarloResult(price=price, standard_error=discount * standard_error)


if __name__ == "__main__":
    binomial_price = binomial_option_price(
        spot=100,
        strike=100,
        maturity=1.0,
        risk_free_rate=0.05,
        volatility=0.2,
        steps=200,
        option_type="call",
        american=False,
    )
    print(f"European call price (binomial): {binomial_price:.4f}")

    def asian_call_payoff(path: Sequence[float]) -> float:
        average_price = sum(path) / len(path)
        return max(average_price - 100, 0.0)

    mc_result = monte_carlo_option_price(
        payoff=asian_call_payoff,
        spot=100,
        maturity=1.0,
        risk_free_rate=0.05,
        volatility=0.2,
        steps=252,
        paths=20_000,
        seed=7,
    )
    print(
        "Asian call price (Monte Carlo): "
        f"{mc_result.price:.4f} ± {1.96 * mc_result.standard_error:.4f} (95% CI)"
    )
