import numpy as np
import logging
from warnings import catch_warnings, simplefilter
from sklearn.gaussian_process import GaussianProcessRegressor
import scipy

logging.basicConfig(level=logging.INFO)

# TODO:Add decorator
def objective_function(x, noise_value=0.1):

    if noise_value > 0:
        noise = 2 * np.random.rand(1) * noise_value - noise_value / 2
    else:
        noise = 0

    y = x**2 * np.sin(5 * np.pi * x)**6

    return y + noise


def surrogate_function(model: GaussianProcessRegressor, X: np.array):
    with catch_warnings():
        simplefilter("ignore")

        return model.predict(X, return_std=True)

# Probability of Improvement (PI).
# Expected Improvement (EI).
# Lower Confidence Bound (LCB).


def acquisition_function(X, X_samples, model):

    # Obtain results for the samples already processed
    y_hat, _ = surrogate_function(model, X)

    y_best = max(y_hat)

    # calculate mean and stdev
    mu, std = surrogate_function(model, X_samples)

    eps = 1e-10

    # Calculate the probability of improvement
    probs = scipy.stats.norm.cdf((mu - y_best) / (std + eps))

    return probs


def optimisation_acquisition(model, X):

    X_sample = np.random.rand(100).reshape(-1, 1)

    scores = acquisition_function(X, X_sample, model)

    # Obtain the samples that maximises the probability of having higher values
    idx_x_max = np.argmax(scores)
    return X_sample[idx_x_max]

