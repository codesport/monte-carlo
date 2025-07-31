[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/codesport/monte-carlo/HEAD?urlpath=%2Fdoc%2Ftree%2Fbase-mcs-stocks.py) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/codesport/monte-carlo/blob/master/nb-base-mcs-stocks.ipynb)

# Author Credits

This writeup was researched, tested, and compiled by Marcos A.B. (https://github.com/codesport). 

He may be reached through [Code Sport's contact us page](https://codesport.io/contact-us)

# 📊 Monte Carlo Simulations in Financial Risk Managemnt


 **Monte Carlo simulations** allow us to model uncertainty, stress‑test financial portfolios, and make informed risk decisions.  
This tutorial explains Monte Carlo methods through the lens of **DeFi liquidation risk** and **equity option pricing**, with equations, Python code, and **charts** for intuition.  

We’ll use **Ethereum (ETH)** as the consistent underlying asset

---

## 1. What is a Monte Carlo Simulation?

A **Monte Carlo simulation** is a method for estimating the probability distribution of outcomes by generating a large number of random trials.  

### 1a. Definition and Geometric Brownian Motion (GBM)


In finance, asset prices are often modeled using **Geometric Brownian Motion (GBM)**:

$$
S_{t+\Delta t} = S_t \cdot \exp\Big( (\mu - \tfrac{1}{2}\sigma^2)\Delta t + \sigma \sqrt{\Delta t}\cdot Z \Big)
$$

**Where:**
- $S_t$: Asset price at time t  
- $\mu$: Drift (expected return per step)  
- $\sigma$: Volatility (standard deviation of returns)  
- $\Delta t$: Time increment  
- $Z \sim \mathcal{N}(0,1)$: Standard normal random variable  

**Python Setup**

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests

# Fetch ETH historical price data
def fetch_crypto_data(coin_id="ethereum", vs_currency="usd", days=365):
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {"vs_currency": vs_currency, "days": days}
    data = requests.get(url, params=params).json()
    prices = [p[1] for p in data['prices']]
    return pd.Series(prices)

eth_prices = fetch_crypto_data()
log_returns = np.log(eth_prices / eth_prices.shift(1)).dropna()
mu = log_returns.mean()
sigma = log_returns.std()
S0 = eth_prices.iloc[-1]
```

### 1b. Monte Carlo for Option Pricing

Monte Carlo methods are often used to price options. American options can be exercised at any time before expiration. European options cannot.


Monte Carlo methods are widely used to price complex derivatives like American or path‑dependent options (e.g., Asian options, barrier options).

The key idea:

* Simulate many possible stock paths under GBM.

* Compute the payoff of the option along each path.

* Discount the average payoff back to today.


#### European Call Option

The Black–Scholes formula for a **European call option** is:

$$
C = S_0 N(d_1) - K e^{-rT} N(d_2)
$$

**Where:**
- $C$: Call price  
- $S_0$: Current asset price  
- $K$: Strike price  
- $T$: Time to maturity (in years)  
- $r$: Risk-free interest rate  
- $N(\cdot)$: Cumulative distribution function of the standard normal  
- $d_1 = \frac{\ln(S_0/K) + (r + 0.5\sigma^2)T}{\sigma \sqrt{T}}$  
- $d_2 = d_1 - \sigma\sqrt{T}$

#### 💻 Python Implementation for European Call Option

```python
import numpy as np
from scipy.stats import norm

def european_call_price(S0, K, T, r, sigma):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

# Example: Pricing ETH European Call Option
S0 = 2000   # Current ETH price
K = 2100    # Strike price
T = 0.5     # Time to maturity (0.5 years)
r = 0.03    # Risk-free rate
sigma = 0.5 # Volatility (50%)

price = european_call_price(S0, K, T, r, sigma)
print(f"European Call Option Price: {price:.2f}")
```

#### American Call Options

American options allow early exercise, making closed-form solutions more complex.
One numerical method is the Longstaff–Schwartz Monte Carlo algorithm, which uses regression to estimate the continuation value.


The valuation relies on the Longstaff–Schwartz least squares method, which estimates the continuation value via regression. The value of an American call option can be expressed as:

$$
C_0 = \max_{t \leq T} \; \mathbb{E}\Big[ e^{-r t} \cdot \max(S_t - K, 0) \,\Big|\, 	ext{optimal exercise policy} \Big]
$$

Where:

- $C_0$ = value of the American call  
- $S_t$ = simulated stock price at time $t$  
- $K$ = strike price  
- $r$ = risk free interest rate  
- $T$ = expiration time  
- “optimal exercise policy” = decision rule from regression continuation value  

#### 💻 American Call Option Pricing via Least Squares Monte Carlo (LSM)

```python
import numpy as np

# --- Step 1: Simulate Geometric Brownian Motion paths ---
def simulate_gbm(S0, r, sigma, T, steps, sims):
    dt = T / steps
    paths = np.zeros((steps+1, sims))
    paths[0] = S0
    for t in range(1, steps+1):
        Z = np.random.standard_normal(sims)
        paths[t] = paths[t-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
    return paths

# --- Step 2: Price American Call using LSM ---
def price_american_call(S0, K, r, sigma, T, steps=50, sims=5000):
    dt = T / steps
    paths = simulate_gbm(S0, r, sigma, T, steps, sims)
    payoffs = np.maximum(paths - K, 0)

    V = payoffs[-1]  # option values at maturity

    for t in reversed(range(1, steps)):
        itm = payoffs[t] > 0  # in-the-money paths
        if np.any(itm):
            X = paths[t, itm]
            Y = V[itm] * np.exp(-r * dt)  # discounted continuation values

            # Regression to estimate continuation value
            coeffs = np.polyfit(X, Y, 2)
            continuation = np.polyval(coeffs, X)

            # Exercise if immediate payoff better than continuation
            exercise = payoffs[t, itm]
            V[itm] = np.where(exercise > continuation, exercise, V[itm] * np.exp(-r * dt))
        V[~itm] = V[~itm] * np.exp(-r * dt)

    return np.mean(V) * np.exp(-r * dt)

S0 = 3700    # ETH starting price
K = 3800     # Strike price
r = 0.02     # 2% risk-free rate
sigma = 0.5  # 50% annual volatility
T = 0.25     # 3 months to maturity

price = price_american_call(S0, K, r, sigma, T)
print(f"American Call Option Price: ${price:.2f}")


```
---
## 2. Number of Simulations: 500 vs 5000

Monte Carlo simulations rely on generating many random paths for the underlying asset price.  
The number of simulations chosen directly affects both the **accuracy** and the **computational cost**.

- **500 simulations**
  - Faster to run, less accurate. Higher variance in estimates
  - Less accurate results due to higher sampling error
  - Useful for quick estimates or testing

- **5000 simulations**
  - Slower, but results converge toward true distribution (Law of Large Numbers).
  - Smoother distributions of outcomes
  - More accurate estimates of tail risks (e.g., Value-at-Risk)
  - Higher computational cost but often necessary for risk-sensitive decisions

 **Pro Tip:** Use more paths until results stabilize, balancing speed vs accuracy.

### Visualization Example

```python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sim_counts = [500, 5000]
colors = ["red", "blue"]

for sims, color in zip(sim_counts, colors):
    paths = np.zeros((30, sims))
    paths[0] = S0
    for t in range(1, 30):
        Z = np.random.standard_normal(sims)
        paths[t] = paths[t-1] * np.exp((mu - 0.5*sigma**2) + sigma*Z)
    plt.plot(paths[:, :10], color=color, alpha=0.4, label=f"{sims} sims" if sims==500 else "")

plt.title("ETH Monte Carlo Simulation: 500 vs 5000 Simulations")
plt.xlabel("Day")
plt.ylabel("ETH Price (USD)")
plt.legend()
plt.grid(True)
plt.show()
```
---
## 3. Probability of Touching a Price

In finance, we often need to estimate the probability that an asset’s price touches (falls below or rises above) a critical level within a given time horizon.

For example, in lending protocols, liquidation may occur if the asset price drops below a certain threshold.

### Mathematical Formulation

The probability that ETH touches a barrier $B$ at least once within horizon $T$ is:

$$
P(\text{touch}) = \frac{\left| \{ \text{simulated paths where } \min_{t \leq T} S_t \leq B \} \right|}{\text{Total Paths}}
$$

> [!TIP]
> In probability/statistics, instead of `#` for “number of,” use cardinality notation with absolute values: 
> #{sumlatioins} becomes |simulations|

### Python Implementation

```python

import numpy as np

def probability_of_touch(S0, mu, sigma, T, barrier, simulations=5000):
    paths = np.zeros((T, simulations))
    paths[0] = S0
    for t in range(1, T):
        Z = np.random.standard_normal(simulations)
        paths[t] = paths[t-1] * np.exp((mu - 0.5*sigma**2) + sigma*Z)
    
    # Check if barrier touched in each simulation
    touched = np.any(paths <= barrier, axis=0)
    return touched.mean()

# Example: Probability ETH touches $1800 in 30 days
barrier = 1800
T = 30
prob_touch = probability_of_touch(S0, mu, sigma, T, barrier)
print(f"Probability ETH touches ${barrier} in {T} days: {prob_touch:.2%}")
```

#### Intuition
If the barrier is far below the current price or LTV, the probability of touch will be low.

If the barrier is close to or above the current price or LTV, the probability increases sharply.

This technique is widely used in barrier option pricing and in estimating probabilty of liquidation (risk assessment) in DeFi lending protocols.

---
## 4. Liquidation Risk Model

In DeFi lending, borrowers provide collateral (e.g., ETH) to take a loan in stablecoins or another asset.  
The **Loan-to-Value (LTV)** ratio measures the risk of liquidation:

$$
\text{LTV}_t = \frac{L}{C \cdot S_t}
$$

**Where:**
- $L$: Loan value (USD)  
- $C$: Collateral amount (ETH)  
- $S_t$: ETH price at time $t$  

A liquidation occurs if:

$$
S_t \leq \frac{L}{\text{LTV}_{crit} \cdot C}
$$


### Python Implementation

```python
import numpy as np

def probability_of_liquidation(S0, mu, sigma, T, loan_usd, collateral_eth, ltv_crit, simulations=5000):
    paths = np.zeros((T, simulations))
    paths[0] = S0
    for t in range(1, T):
        Z = np.random.standard_normal(simulations)
        paths[t] = paths[t-1] * np.exp((mu - 0.5*sigma**2) + sigma*Z)
    
    # Compute LTV paths
    collateral_values = paths * collateral_eth
    ltv_paths = loan_usd / collateral_values
    
    # Check if liquidation threshold breached
    liquidations = np.any(ltv_paths >= ltv_crit, axis=0)
    return liquidations.mean()

# Example: Liquidation risk
loan_usd = 25000
collateral_eth = 10
ltv_crit = 0.8
T = 30

prob_liq = probability_of_liquidation(S0, mu, sigma, T, loan_usd, collateral_eth, ltv_crit)
print(f"Probability of liquidation in {T} days: {prob_liq:.2%}")
```

### Intuition

- **Higher collateral** or **lower loan amount** reduces liquidation risk.  
- **Lower ETH price** or **higher volatility** increases risk.  
- DeFi Protocols set liquidation thresholds ($\text{LTV}_{crit}$) to protect lenders

---

## 5. Value-at-Risk (VaR)

VaR answers: “What’s the worst I can lose with 95% confidence in T days?”

**Value-at-Risk (VaR)** is a widely used risk measure that estimates the maximum potential loss of a portfolio within a given time horizon at a specified confidence level.

For example, a **95% VaR** represents the maximum loss you would expect **95% of the time** over the period considered.

### Formula

For a Monte Carlo simulation:

$$
VaR_{95} = \text{Percentile}_{5\%}(\text{Portfolio Returns})
$$

**Where:**
- $VaR_{95}$: Value-at-Risk at 95% confidence  
- $\text{Percentile}_{5\%}$: 5th percentile of simulated returns or portfolio values  

### Python Implementation

```python

import numpy as np

def value_at_risk(S0, mu, sigma, T, loan_usd, collateral_eth, simulations=5000, percentile=5):
    paths = np.zeros((T, simulations))
    paths[0] = S0
    for t in range(1, T):
        Z = np.random.standard_normal(simulations)
        paths[t] = paths[t-1] * np.exp((mu - 0.5*sigma**2) + sigma*Z)

    final_prices = paths[-1]
    portfolio_values = final_prices * collateral_eth - loan_usd
    var_value = np.percentile(portfolio_values, percentile)
    return var_value

loan_usd = 25000
collateral_eth = 10
T = 30

var_95 = value_at_risk(S0, mu, sigma, T, loan_usd, collateral_eth)
print(f"95% Value-at-Risk over {T} days: ${var_95:,.0f}")
```

### Why the 5th Percentile?

We examine the **5th percentile** of outcomes because:

- It represents a **worst-case scenario** within the 95% confidence band.  
- It helps quantify **tail risk** (rare but severe losses).  
- It provides a benchmark for determining **capital requirements** and **collateral safety margins** in DeFi lending.  

---

## 6. Drift (μ) and Volatility (σ)

Monte Carlo simulations of asset prices using **Geometric Brownian Motion (GBM)** require two critical parameters: **drift (μ)** and **volatility (σ)**.

---

### 6a. Interpretation

- **Drift (μ)**:  
  **average expected log return per time step** (e.g., per day).  
  Over $n$ days:

  $$
  \mu_n = \mu_{daily} \cdot n
  $$

- **Volatility (σ)**:  
  Measures the **uncertainty or dispersion** of returns.  
  Over $n$ days:

  $$
  \sigma_n = \sigma_{daily} \cdot \sqrt{n}
  $$

**Where:**
- $\mu_{daily}$: mean daily log return  
- $\sigma_{daily}$: standard deviation of daily log returns  
- $n$: number of days in the forecast horizon  

In other words:  
- Drift gives the **directional tendency** of ETH’s price.  
- Volatility gives the **scale of randomness** (how wide the distribution of possible outcomes is).

---

### 6b. Volatility Estimation

Volatility is typically estimated from **historical log returns**:

$$
\sigma = \sqrt{\frac{1}{N-1}\sum_{i=1}^N \big(r_i - \bar{r}\big)^2}
$$

**Where:**
- $r_i$: log return on day $i$  
- $\bar{r}$: mean of the log returns  
- $N$: number of historical observations  

**Python Implementation**

```python

import pandas as pd

# dummy data set
eth_prices = pd.Series([3000, 3020, 3050, 3010, 3100])
# Compute drift (mu) and volatility (sigma) from ETH daily log returns
log_returns = np.log(eth_prices / eth_prices.shift(1)).dropna()
mu = log_returns.mean()
sigma = log_returns.std()

print(f"Estimated daily drift (μ): {mu:.6f}")
print(f"Estimated daily volatility (σ): {sigma:.6f}")
```

### Which Historical Window Should You Use?

- **Short windows (e.g., last 30 days):**  
  Capture **recent market behavior** but may be noisy.  

- **Longer windows (e.g., 365 days):**  
  Provide **stability** in estimates but may lag in capturing regime changes.  

**Guideline:**  
For modeling ETH liquidation risk over 30 days, using **365 days of volatility data** is common, as it balances accuracy and stability.  
However, during highly volatile periods, **shorter windows** may better reflect the current environment.


**Pro Tip:** For our risk modeling we calculate volatilty using n-days of volatilty.
**Where:**
 n  =  intended holding period of the asset

Additionally, to account for seasonality we may also use pricing data over the same period but 1 year prior. For example, for n = 7-day holding period:

If we intend to hold asset from August 7, 2025 to August 14, 2025,  we'll compute σ using both August 7, 2024 to August 14, 2024 and July 31, 2025 to August 7, 2025.

---

## 7. Confidence Intervals and Percentiles

Monte Carlo simulations produce a distribution of possible asset prices.  
To interpret this distribution, we often use **confidence intervals** (percentiles).

---

### Key Percentiles

- **5th Percentile (P5):**  
  Represents a **bearish / worst-case scenario**.  
  In liquidation modeling, it helps quantify severe downside risk.  

- **50th Percentile (P50 / Median):**  
  Represents the **most likely outcome** in the middle of the distribution.  
  This is often plotted as the central “expected path.”  

- **95th Percentile (P95):**  
  Represents a **bullish / best-case scenario**, showing optimistic outcomes.

---

### Visualization Example

```python
import numpy as np
import matplotlib.pyplot as plt

# Calculate percentiles
p5 = np.percentile(paths, 5, axis=1)
p50 = np.percentile(paths, 50, axis=1)
p95 = np.percentile(paths, 95, axis=1)

plt.figure(figsize=(10,6))
plt.plot(p50, label="Median (50th Percentile)", color="blue")
plt.fill_between(range(T), p5, p95, color="lightblue", alpha=0.4, 
                 label="5th–95th Percentile Range")
plt.title(f"ETH Monte Carlo Simulation ({T} days)")
plt.xlabel("Day")
plt.ylabel("ETH Price (USD)")
plt.legend()
plt.grid(True)
plt.show()
```

### Interpretation in DeFi Context

- **Liquidation Risk:**  
  If the **5th percentile path** crosses your liquidation threshold,  there’s at least a **5% chance** you’ll be liquidated in the given time horizon.  

- **Protocol Risk Management:**  
  DeFi lending protocols may use **95% confidence bands** to set **safe collateral ratios**.  

- **Investor View:**  
  Traders can evaluate the **upside vs downside balance** by comparing the **95th and 5th percentile paths**.  

---

## 8. Business Requirement: Ensuring <5% Chance of Undercollateralization

A common business requirement in DeFi lending is to ensure that  
there is less than a **5% probability of undercollateralization** across all loans. 
(i.e., loan value exceeding collateral value) within a specified horizon.


For example, you run a DeFi protocol that wants to manage the risk of bad debt:

> “We want the probability of undercollateralization (liquidation) to stay below 5% in the next 30 days.”

---

### Mathematical Formulation

Undercollateralization occurs when:

$$
\text{LTV}_t \geq 1
$$

To maintain safety, protocols set a critical liquidation threshold ($\text{LTV}_{crit}$) such that:

$$
P(\text{LTV}_t \geq \text{LTV}_{crit} \text{ for some } t \leq T) < 5\%
$$

---

### 8b: Python Implementation Set 1

```python
import numpy as np

def probability_of_liquidation(S0, mu, sigma, T, loan_usd, collateral_eth, ltv_crit, simulations=5000):
    paths = np.zeros((T, simulations))
    paths[0] = S0
    for t in range(1, T):
        Z = np.random.standard_normal(simulations)
        paths[t] = paths[t-1] * np.exp((mu - 0.5*sigma**2) + sigma*Z)
    
    # Compute LTV paths
    collateral_values = paths * collateral_eth
    ltv_paths = loan_usd / collateral_values
    
    # Check if liquidation threshold breached
    liquidations = np.any(ltv_paths >= ltv_crit, axis=0)
    return liquidations.mean()

def safe_liquidation_threshold(S0, mu, sigma, T, loan_usd, collateral_eth, simulations=5000, target_prob=0.05):
    thresholds = np.linspace(0.5, 0.95, 20)
    for ltv_crit in thresholds:
        prob = probability_of_liquidation(S0, mu, sigma, T, loan_usd, collateral_eth, ltv_crit, simulations)
        if prob < target_prob:
            return ltv_crit, prob
    return None, None

loan_usd = 25000
collateral_eth = 10
T = 30

ltv_safe, prob_safe = safe_liquidation_threshold(S0, mu, sigma, T, loan_usd, collateral_eth)
if ltv_safe:
    print(f"Set LTV threshold at {ltv_safe:.0%} → Probability of liquidation: {prob_safe:.2%}")
else:
    print("No safe threshold found within tested range.")
```

### Explanation

- This function **iterates over candidate LTV thresholds**  
  to find the highest LTV level that keeps liquidation risk below 5%.  

- If no threshold is found, the protocol must require  
  **more collateral** or **smaller loans**.  

### Practical Implication

- **Borrowers:** Can assess how close they are to unsafe territory.  
- **Lenders/Protocols:** Can set thresholds to minimize systemic risk.  
- **Risk Managers:** Can justify risk frameworks with quantitative evidence.  



## 8c. Python Implementation Set 2

### Step 1: Define Liquidation Condition

$$
	ext{Liquidation if } S_t \leq frac{L}{	ext{LTV}_{crit} \cdot C}
$$

### Step 2: Iterate Over Thresholds

```python
for ltv in np.linspace(0.6, 0.9, 7):
    liq_price = loan_usd / (ltv * collateral_eth)
    liqs = np.sum(np.any(paths <= liq_price, axis=0))
    prob_liq = liqs / sims
    print(f"LTV {ltv:.0%}: Liquidation Probability = {prob_liq:.2%}")
    if prob_liq < 0.05:
        print(f"✅ Safe threshold: {ltv:.0%}")
```

### Step 3: Interpret Results

- At LTV = 80%, probability of liquidation = 12% → too risky.  
- At LTV = 65%, probability of liquidation = 3% → acceptable.

📈 Visualization:

```python
ltvs = np.linspace(0.6, 0.9, 7)
probs = []
for ltv in ltvs:
    liq_price = loan_usd / (ltv * collateral_eth)
    liqs = np.sum(np.any(paths <= liq_price, axis=0))
    probs.append(liqs / sims)

plt.plot(ltvs, probs, marker="o")
plt.axhline(0.05, color="red", linestyle="--", label="5% Risk Target")
plt.title("Probability of Liquidation vs. LTV Threshold")
plt.xlabel("LTV Threshold")
plt.ylabel("Probability of Liquidation")
plt.legend()
plt.show()
```

### 📊 Interpretation

- The curve shows how risk increases with higher LTV thresholds.  
- The red dashed line = 5% risk policy.  
- The intersection = maximum safe LTV.

---

## ✅ Conclusion

By combining math, Monte Carlo simulations, and visuals, we can:

- Understand ETH price uncertainty via GBM  
- Quantify liquidation risk under different thresholds  
- Set safe LTV levels that align with business risk appetite  

This approach works in DeFi, US equities, or derivatives pricing. It turns abstract math into actionable risk management.



