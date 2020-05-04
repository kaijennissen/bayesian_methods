from IPython import get_ipython
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pymc3 as pm
import theano.tensor as tt

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

from IPython.display import Image
from theano.compile.ops import as_op
from scipy.stats import norm

get_ipython().run_line_magic("matplotlib", "inline")

color = "#87ceeb"
f_dict = {"size": 14}


# Calculate Gamma shape and rate from mode and sd.
def gammaShRaFromModeSD(mode, sd):
    rate = (mode + np.sqrt(mode ** 2 + 4 * sd ** 2)) / (2 * sd ** 2)
    shape = 1 + mode * rate
    return (shape, rate)


data = pd.read_csv("https://osf.io/zftb3/download")
data.info()


data.head()


# Columns n1 - n5
y = data.iloc[:, 2:]


# Number of outcomes
nYlevels = y.columns.size
nYlevels


Ncases = y.index.size
Ncases


z = y.sum(1)
z.head()


gammaShRa = gammaShRaFromModeSD(3, 3)
gammaShRa


# Thresholds, masking the the inner two values.
thresh = [k + 0.5 for k in range(1, nYlevels)]
thresh_obs = np.ma.asarray(thresh)
thresh_obs[1:-1] = np.ma.masked

print("thresh:\t\t{}".format(thresh))
print("thresh_obs:\t{}".format(thresh_obs))


@as_op(itypes=[tt.dvector, tt.dvector, tt.dvector], otypes=[tt.dmatrix])
def outcome_probabilities(theta, mu, sigma):
    out = np.empty((nYlevels, Ncases), dtype=np.float64)
    n = norm(loc=mu, scale=sigma)
    lbound = np.repeat(0, Ncases)

    # Thresholded cumulative normal probabilities.
    # Four thresholds (theta values) define the 5 outcome probabilities.
    out[0, :] = n.cdf(theta[0])
    out[1, :] = np.max([lbound, n.cdf(theta[1]) - n.cdf(theta[0])], axis=0)
    out[2, :] = np.max([lbound, n.cdf(theta[2]) - n.cdf(theta[1])], axis=0)
    out[3, :] = np.max([lbound, n.cdf(theta[3]) - n.cdf(theta[2])], axis=0)
    out[4, :] = 1 - n.cdf(theta[3])

    return out


# Model a hierarchical sigma?
hierarchSD = False

with pm.Model() as ordinal_model_multi_groups:
    # Latent means (rating) of the movies
    mu = pm.Normal(
        "mu", mu=(1 + nYlevels) / 2.0, tau=1.0 / (nYlevels) ** 2, shape=Ncases
    )

    # Latent standard deviations of the ratings.
    if hierarchSD:
        sigmaSD = pm.Gamma("sigmaSD", gammaShRa[0], gammaShRa[1])
        sigmaMode = pm.Gamma("sigmaMode", gammaShRa[0], gammaShRa[1])
    else:
        sigmaSD = 3.0
        sigmaMode = 3.0
    sigmaRa = pm.Deterministic(
        "sigmaRa",
        (
            (sigmaMode + pm.math.sqrt(sigmaMode ** 2 + 4 * sigmaSD ** 2))
            / (2 * sigmaSD ** 2)
        ),
    )
    sigmaSh = pm.Deterministic("sigmaSh", 1 + sigmaMode * sigmaRa)
    sigma = pm.Gamma("sigma", sigmaSh, sigmaRa, shape=Ncases)

    # Latent thresholds between the ratings (ordinal values)
    theta = pm.Normal(
        "theta",
        mu=thresh,
        tau=1 / np.repeat(2 ** 2, len(thresh)),
        shape=len(thresh),
        observed=thresh_obs,
    )

    # Cumulative normal probabilities for ratings (ordinal values)
    pr = outcome_probabilities(theta, mu, sigma)

    # Likelihood
    out = pm.Multinomial("out", n=z, p=pr.T, observed=y.values)


with ordinal_model_multi_groups:
    trace = pm.sample(400, cores=4, progressbar=True)


fig1 = pm.traceplot(trace, ["mu", "sigma"], compact=True, combined=True)


fig2 = pm.traceplot(trace, ["mu", "sigma"], compact=False, combined=True)


plt.savefig(fig1, "traceplot.jpg")


pm.summary(trace, ["theta_missing"])
