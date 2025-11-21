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

### Command-line Monte Carlo with a custom payoff expression

You can also price options from the command line by supplying a Python payoff
expression. The expression receives a ``path`` variable (list of simulated spot
prices) and can use ``math``, ``max``, ``min``, ``sum``, and ``len``. For
example, to price an Asian call using 20,000 paths and 252 steps:

```bash
python binomial_option.py monte-carlo \
  --payoff-expr "max(sum(path)/len(path) - 100, 0)" \
  --spot 100 --maturity 1.0 --risk-free-rate 0.05 --volatility 0.2 \
  --steps 252 --paths 20000 --seed 7
```

For a plain European call payoff, you could pass
``--payoff-expr "max(path[-1] - 100, 0)"``.
