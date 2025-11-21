# 20251121

Simple Python implementation of option pricing models.

## Usage

Run the module directly to see a sample European call price:

```bash
python binomial_option.py
```

Or import the function for custom parameters:

```python
from binomial_option import binomial_option_price, monte_carlo_option_price
from typing import Sequence

price = binomial_option_price(
    spot=100,
    strike=105,
    maturity=1.0,
    risk_free_rate=0.03,
    volatility=0.25,
    steps=300,
    option_type="put",
    american=True,
)
print(price)


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
)
print("Asian call (Monte Carlo):", mc_result.price)
```
