"""Simple Flask web server for Monte Carlo option pricing."""
from __future__ import annotations

import math
from typing import Callable, Sequence

from flask import Flask, Request, render_template_string, request

from binomial_option import MonteCarloResult, monte_carlo_option_price


def _build_payoff(
    payoff_mode: str,
    strike: float,
    *,
    custom_expr: str | None,
) -> Callable[[Sequence[float]], float]:
    """Return a payoff function based on user selection."""
    if payoff_mode == "call":
        return lambda path: max(path[-1] - strike, 0.0)
    if payoff_mode == "put":
        return lambda path: max(strike - path[-1], 0.0)

    allowed_globals = {"math": math, "max": max, "min": min, "sum": sum, "len": len}

    def custom_payoff(path: Sequence[float]) -> float:
        expression = custom_expr or ""
        return float(eval(expression, allowed_globals, {"path": path}))

    return custom_payoff


app = Flask(__name__)


def _float_from_form(req: Request, name: str, default: float) -> float:
    value = req.form.get(name)
    if value is None or value == "":
        return default
    return float(value)


def _int_from_form(req: Request, name: str, default: int) -> int:
    value = req.form.get(name)
    if value is None or value == "":
        return default
    return int(value)


FORM_TEMPLATE = """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Monte Carlo Option Pricer</title>
    <style>
      body { font-family: Arial, sans-serif; margin: 2rem auto; max-width: 960px; line-height: 1.5; }
      header { margin-bottom: 1.5rem; }
      form { background: #f8f9fa; padding: 1rem 1.5rem; border-radius: 8px; border: 1px solid #e1e5ea; }
      fieldset { border: none; padding: 0; margin-bottom: 1rem; }
      legend { font-weight: bold; margin-bottom: 0.5rem; }
      label { display: block; margin-top: 0.4rem; font-weight: 600; }
      input, select, textarea { width: 100%; padding: 0.5rem; font-size: 1rem; margin-top: 0.2rem; box-sizing: border-box; }
      textarea { resize: vertical; min-height: 60px; }
      .two-col { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; }
      .actions { margin-top: 1rem; }
      button { padding: 0.6rem 1rem; font-size: 1rem; cursor: pointer; }
      .result { margin-top: 1.5rem; padding: 1rem; background: #e8f5e9; border: 1px solid #c8e6c9; border-radius: 6px; }
      .error { margin-top: 1.5rem; padding: 1rem; background: #fdecea; border: 1px solid #f5c2c7; border-radius: 6px; color: #8a1c1c; }
      code { background: #eef; padding: 0.1rem 0.3rem; border-radius: 4px; }
    </style>
  </head>
  <body>
    <header>
      <h1>Monte Carlo Option Pricer</h1>
      <p>Simulate option payoffs using geometric Brownian motion paths.</p>
    </header>

    <form method="post" action="/">
      <fieldset>
        <legend>Market parameters</legend>
        <div class="two-col">
          <label>Spot price
            <input type="number" step="0.01" name="spot" value="{{spot}}" required>
          </label>
          <label>Strike price
            <input type="number" step="0.01" name="strike" value="{{strike}}" required>
          </label>
          <label>Maturity (years)
            <input type="number" step="0.01" name="maturity" value="{{maturity}}" required>
          </label>
          <label>Risk-free rate
            <input type="number" step="0.001" name="risk_free_rate" value="{{risk_free_rate}}" required>
          </label>
          <label>Volatility
            <input type="number" step="0.001" name="volatility" value="{{volatility}}" required>
          </label>
        </div>
      </fieldset>

      <fieldset>
        <legend>Simulation settings</legend>
        <div class="two-col">
          <label>Steps per path
            <input type="number" name="steps" value="{{steps}}" min="1" required>
          </label>
          <label>Number of paths
            <input type="number" name="paths" value="{{paths}}" min="1" required>
          </label>
          <label>Random seed (optional)
            <input type="number" name="seed" value="{{seed or ''}}">
          </label>
        </div>
      </fieldset>

      <fieldset>
        <legend>Payoff</legend>
        <div class="two-col">
          <label>Payoff type
            <select name="payoff_mode">
              <option value="call" {% if payoff_mode == 'call' %}selected{% endif %}>European call (max(S_T - K, 0))</option>
              <option value="put" {% if payoff_mode == 'put' %}selected{% endif %}>European put (max(K - S_T, 0))</option>
              <option value="custom" {% if payoff_mode == 'custom' %}selected{% endif %}>Custom expression</option>
            </select>
          </label>
          <label>Custom payoff expression (uses <code>path</code>)
            <textarea name="payoff_expr" placeholder="max(sum(path)/len(path) - 100, 0)">{{payoff_expr or ''}}</textarea>
          </label>
        </div>
        <p>Custom expressions can reference <code>path</code>, <code>math</code>, <code>max</code>, <code>min</code>, <code>sum</code>, and <code>len</code>.</p>
      </fieldset>

      <div class="actions">
        <button type="submit">Run simulation</button>
      </div>
    </form>

    {% if result %}
    <div class="result">
      <strong>Price:</strong> {{"{:.4f}".format(result.price)}}<br>
      <strong>95% CI:</strong> ± {{"{:.4f}".format(1.96 * result.standard_error)}}
    </div>
    {% endif %}

    {% if error %}
    <div class="error">
      <strong>Error:</strong> {{error}}
    </div>
    {% endif %}
  </body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def index():
    defaults = {
        "spot": 100.0,
        "strike": 100.0,
        "maturity": 1.0,
        "risk_free_rate": 0.05,
        "volatility": 0.2,
        "steps": 252,
        "paths": 20_000,
        "seed": None,
        "payoff_mode": "call",
        "payoff_expr": "",
    }

    context: dict[str, float | int | str | None | MonteCarloResult] = dict(defaults)
    result: MonteCarloResult | None = None
    error: str | None = None

    if request.method == "POST":
        try:
            context["spot"] = _float_from_form(request, "spot", defaults["spot"])
            context["strike"] = _float_from_form(request, "strike", defaults["strike"])
            context["maturity"] = _float_from_form(request, "maturity", defaults["maturity"])
            context["risk_free_rate"] = _float_from_form(
                request, "risk_free_rate", defaults["risk_free_rate"]
            )
            context["volatility"] = _float_from_form(request, "volatility", defaults["volatility"])
            context["steps"] = _int_from_form(request, "steps", defaults["steps"])
            context["paths"] = _int_from_form(request, "paths", defaults["paths"])
            seed_val = request.form.get("seed")
            context["seed"] = int(seed_val) if seed_val else None
            context["payoff_mode"] = request.form.get("payoff_mode", "call")
            context["payoff_expr"] = request.form.get("payoff_expr", "")

            payoff_fn = _build_payoff(
                str(context["payoff_mode"]),
                float(context["strike"]),
                custom_expr=str(context["payoff_expr"]),
            )

            result = monte_carlo_option_price(
                payoff=payoff_fn,
                spot=float(context["spot"]),
                maturity=float(context["maturity"]),
                risk_free_rate=float(context["risk_free_rate"]),
                volatility=float(context["volatility"]),
                steps=int(context["steps"]),
                paths=int(context["paths"]),
                seed=context["seed"],
            )
            context["result"] = result
        except Exception as exc:  # noqa: BLE001 - surface errors to the UI
            error = str(exc)

    context["result"] = result

    return render_template_string(FORM_TEMPLATE, **context, error=error)


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
