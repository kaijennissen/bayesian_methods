# import matplotlib.pyplot as plt
import numpy as np
from pymc3 import (
    NUTS,
    HalfNormal,
    Model,
    Normal,
    find_MAP,
    sample,
    summary,
    traceplot,
)
from scipy import optimize

# Initialize random number generator
np.random.seed(123)

# True parameter values
alpha, sigma = 1, 1
beta = [1, 2.5]

# Size of dataset
size = 100

# Predictor variable
X1 = np.linspace(0, 1, size)
X2 = np.linspace(0, 0.2, size)

# Simulate outcome variable
Y = alpha + beta[0] * X1 + beta[1] * X2 + np.random.randn(size) * sigma

basic_model = Model()

with basic_model:

    # Priors for unknown model parameters
    alpha = Normal("alpha", mu=0, sd=10)
    beta = Normal("beta", mu=0, sd=10, shape=2)
    sigma = HalfNormal("sigma", sd=1)

    # Expected value of outcome
    mu = alpha + beta[0] * X1 + beta[1] * X2

    # Likelihood (sampling distribution) of observations
    Y_obs = Normal("Y_obs", mu=mu, sd=sigma, observed=Y)


map_estimate = find_MAP(model=basic_model)

print(map_estimate)

with basic_model:

    # obtain starting values via MAP
    start = find_MAP(fmin=optimize.fmin_powell)

    # instantiate sampler
    step = NUTS(scaling=start)

    # draw 2000 posterior samples
    trace = sample(2000, step, start=start)


traceplot(trace)


summary(trace["alpha"])
summary(trace["beta"])
