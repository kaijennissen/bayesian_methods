import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from pymc3 import (
    Model,
    NUTS,
    Deterministic,
    Exponential,
    get_data,
    StudentT,
    find_MAP,
    sample,
    summary,
    traceplot,
)
from pymc3.distributions.timeseries import GaussianRandomWalk

returns = pd.read_csv(get_data("SP500.csv"), index_col="Date")
returns = returns.loc[(returns.index >= "2008") & (returns.index <= "2009")]

sp500_model = Model()

with sp500_model:

    nu = Exponential("nu", 1.0 / 10, testval=5.0)

    sigma = Exponential("sigma", 1.0 / 0.02, testval=0.1)

    s = GaussianRandomWalk("s", sigma ** -2, shape=len(returns))

    volatility_process = Deterministic("volatility_process", np.exp(-2 * s))

    r = StudentT("r", nu, lam=1 / volatility_process, observed=returns["Close"])


with sp500_model:
    start = find_MAP(vars=[s], fmin=scipy.optimize.fmin_l_bfgs_b)

    step = NUTS(scaling=start)
    trace = sample(250, step, progressbar=True)

    # Start next run at the last sampled position.
    step = NUTS(scaling=trace[-1], gamma=0.25)
    trace = sample(2000, step, start=trace[-1], progressbar=True)


summary(trace["nu"])
summary(trace["sigma"])
summary(trace["s"])

#%%
from pymc3 import traceplot

traceplot(trace, [nu, sigma])

# %%
ig, ax = plt.subplots(figsize=(15, 8))
returns.plot(ax=ax)
ax.plot(returns.index, 1 / np.exp(trace["s", ::30].T), "r", alpha=0.03)
ax.set(title="volatility_process", xlabel="time", ylabel="volatility")
ax.legend(["S&P500", "stochastic volatility process"])
