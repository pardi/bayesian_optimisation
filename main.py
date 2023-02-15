import numpy as np
import matplotlib.pyplot as plt
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


def run_optimisation():

    X_ideal = np.linspace(0, 1, 1000)
    noise_level = 0.01

    y_ideal = [objective_function(x, noise_value=0) for x in X_ideal]

    # Find best results
    optimal_x = np.argmax(y_ideal)
    logging.info('Optima: x=%.3f, y=%.3f', X_ideal[optimal_x], y_ideal[optimal_x])

    # GP Regressor
    model = GaussianProcessRegressor()

    # Define variables
    X = np.random.rand(1).reshape(-1, 1)

    # First iteration
    y_measured = objective_function(X, noise_value=noise_level)

    model.fit(X, y_measured.reshape(-1, 1))

    # Obtain predicted data
    y_pred, std_pred = surrogate_function(model=model, X=X_ideal.reshape(-1, 1))
    #
    # y_pred_lb = y_pred - 1 * std_pred
    # y_pred_ub = y_pred + 1 * std_pred
    # plt.plot(X_ideal, y_pred)
    # plt.plot(X_ideal, y_pred_lb, 'b')
    # plt.plot(X_ideal, y_pred_ub, 'b')
    # plt.show()
    # Loop for the optimisation process

    # Initialise y - outputs
    y = y_measured

    # Init tracking of the error
    err_array = np.array([])

    for _ in range(1000):
        # select the next sample
        x = optimisation_acquisition(model, X)

        # sample the point
        read_value = objective_function(x, noise_value=noise_level)

        # Obtain prediction for this optimisation
        prediction, _ = surrogate_function(model, np.array([x]))

        # Store data points
        X = np.append(X, [x]).reshape(-1, 1)
        y = np.append(y, [read_value]).reshape(-1, 1)

        # Update the model
        model.fit(X, y)

        # Error
        err = prediction - objective_function(x, noise_value=0)
        err_array = np.append(err_array, err)
        # print(f'Error: {err}')

    logging.info("Optimal value found: x=%.3f, y=%.3f", X[np.argmax(y)], np.max(y))

    # Plot real data
    plt.plot(X_ideal, y_ideal)

    y_pred, std_pred = surrogate_function(model=model, X=X_ideal.reshape(-1, 1))

    y_pred_lb = y_pred - 1 * std_pred
    y_pred_ub = y_pred + 1 * std_pred

    plt.plot(X_ideal, y_pred)
    plt.plot(X_ideal, y_pred_lb, 'b')
    plt.plot(X_ideal, y_pred_ub, 'b')





    plt.scatter(X, y)

    plt.show()


if __name__ == '__main__':
    run_optimisation()

