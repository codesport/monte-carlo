"""
Python starter script for running a Monte Carlo simulation on equitiess
assets (e.g., NVDA, APPL). Uses Geometric Brownian Motion (GBM) and 
pulls real price data via an API

filename: base-mmcs-stocks.py
author: Marcos A.B. (https://github.com/codesport)
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta

# === STEP 1: Fetch historical price data for NVIDIA (using yfinance) ===
def fetch_stock_data(ticker, period):
    stock = yf.Ticker(ticker)
    hist = stock.history(period=period)
    return hist["Close"], stock

nvda_prices, dat = fetch_stock_data("NVDA", "7d")
log_returns = np.log(nvda_prices / nvda_prices.shift(1)).dropna()

# === STEP 2: Compute log returns to estimate drift (mu) and vol (sigma) ===
mu = log_returns.mean()
sigma = log_returns.std()

print(f"Estimated Daily Drift (mu): {mu:.6f}")
print(f"Estimated Daily Volatility (sigma): {sigma:.6f}")

# === STEP 3: Monte Carlo Simulation ===
S0 = nvda_prices.iloc[-1]  # last known NVIDIA price

# Simulation parameters
T = 7              # number of days to simulate
simulations = 5000  # number of price paths
target_gte_price = 200   # example: NVIDIA reaching $1500
target_lte_price = 165    # example: NVIDIA dropping to $900

paths = np.zeros((T, simulations))
paths[0] = S0
for t in range(1, T):
    Z = np.random.standard_normal(simulations)
    paths[t] = paths[t-1] * np.exp((mu - 0.5 * sigma**2) + sigma * Z)

# === STEP 4: Probability of reaching price targets ===
hits = np.sum(np.any(paths >= target_gte_price, axis=0))
probability = hits / simulations

downside_hits = np.sum(np.any(paths <= target_lte_price, axis=0))
downside_probability = downside_hits / simulations

print(f"Current NVDA Price: ${S0:.2f}")
print(f"Probability NVDA reaches at least ${target_gte_price} in {T} days: {probability:.2%}")
print(f"Probability NVDA falls to ${target_lte_price} in {T} days: {downside_probability:.2%}")

print(
dat.info,
dat.calendar,
dat.analyst_price_targets,
dat.quarterly_income_stmt,
dat.history(period='7d'),
)

# === STEP 5: Confidence intervals ===
p5 = np.percentile(paths, 5, axis=1)
p50 = np.percentile(paths, 50, axis=1)
p95 = np.percentile(paths, 95, axis=1)

# === STEP 6: Plot simulation ===
plt.figure(figsize=(10,6))

# Plot 20 sample paths
for i in range(20):
    plt.plot(paths[:, i], lw=0.8, alpha=0.6)

# Add labels
final_value = p50[-1]
start_value = p50[0]
final_p5 = p5[-1]
final_p95 = p95[-1]

plt.text(T-1, final_value, f"${final_value:,.0f}", color="blue",
         fontsize=10, ha="right", va="bottom", fontweight="bold")
plt.text(0, start_value, f"${start_value:,.0f}", color="blue",
         fontsize=10, ha="left", va="bottom", fontweight="bold")

plt.text(T, final_p95, f"95th %: ${final_p95:,.0f}", color="green",
         fontsize=9, ha="left", va="center", fontweight="bold")
plt.text(T, final_p5, f"5th %: ${final_p5:,.0f}", color="red",
         fontsize=9, ha="left", va="center", fontweight="bold")

plt.plot(p50, label="Median (50th Percentile)", color="blue", lw=2)
plt.fill_between(range(T), p5, p95, color="lightblue", alpha=0.4, label="5th–95th Percentile")

plt.title(f"Monte Carlo Simulation of NVDA Price (Next {T} Days)")
plt.xlabel("Day")
plt.ylabel("NVDA Price (USD)")
plt.legend()
plt.grid(True)
plt.show()
