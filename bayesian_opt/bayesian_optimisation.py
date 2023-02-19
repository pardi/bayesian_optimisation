import logging
from warnings import catch_warnings, simplefilter

import numpy as np
import scipy
from sklearn.gaussian_process import GaussianProcessRegressor

logging.basicConfig(level=logging.INFO)


def surrogate_function(model: GaussianProcessRegressor, X: np.array):
    with catch_warnings():
        simplefilter("ignore")

        return model.predict(X.reshape(-1, 1), return_std=True)


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
