import logging
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor

from bayesian_opt import bayesian_optimisation as bo
from bayesian_opt.hidden_fcns import single_var_obj_fcn

logging.basicConfig(level=logging.INFO)


def single_optimisation(
    model: GaussianProcessRegressor, reads_record: np.array, objective_fcn: Callable
) -> np.array:

    try:
        # select the next sample
        x = bo.optimisation_acquisition(model, reads_record)

        # sample the point
        new_read = objective_fcn(x)

        # Update the model
        model.fit(reads_record, new_read)

        return new_read

    except ValueError as e:
        raise e


def run_optimisation():
    # X_ideal = np.linspace(0, 1, 1000)
    # noise_level = 0.01
    #
    # y_ideal = [bo.objective_function(x, noise_value=0) for x in X_ideal]
    #
    # # Find best results
    # optimal_x = np.argmax(y_ideal)
    # logging.info("Optima: x=%.3f, y=%.3f",
    #              X_ideal[optimal_x], y_ideal[optimal_x])
    #
    # GP Regressor
    model = GaussianProcessRegressor()
    #
    # Define variables
    reads_record = np.array([])
    #
    # # First iteration
    # y_measured = bo.objective_function(X, noise_value=noise_level)
    #
    # model.fit(X, y_measured.reshape(-1, 1))
    #
    # # Obtain predicted data
    # y_pred, std_pred = bo.surrogate_function(model=model,
    #                                          X=X_ideal.reshape(-1, 1))
    # #
    # # y_pred_lb = y_pred - 1 * std_pred
    # # y_pred_ub = y_pred + 1 * std_pred
    # # plt.plot(X_ideal, y_pred)
    # # plt.plot(X_ideal, y_pred_lb, 'b')
    # # plt.plot(X_ideal, y_pred_ub, 'b')
    # # plt.show()
    # # Loop for the optimisation process
    #
    # # Initialise y - outputs
    # y = y_measured
    #
    # # Init tracking of the error
    # err_array = np.array([])

    for _ in range(10):

        # Run single loop optimisation

        new_read = single_optimisation(
            model, reads_record=reads_record, objective_fcn=single_var_obj_fcn
        )
        #
        # # select the next sample
        # x = bo.optimisation_acquisition(model, reads_record)
        #
        # # sample the point
        # read_value = bo.objective_function(x, noise_value=noise_level)
        #
        # # Obtain prediction for this optimisation
        # prediction, _ = bo.surrogate_function(model, np.array([x]))

        # Store data points
        X = np.append(X, [x]).reshape(-1, 1)
        y = np.append(y, [read_value]).reshape(-1, 1)

        # Update the model
        model.fit(reads_record, reads_record)

        # Error
        err = prediction - bo.objective_function(x, noise_value=0)
        err_array = np.append(err_array, err)
        # print(f'Error: {err}')

    logging.info("Optimal value found: x=%.3f, y=%.3f", X[np.argmax(y)], np.max(y))

    # Plot real data
    plt.plot(X_ideal, y_ideal)

    y_pred, std_pred = bo.surrogate_function(model=model, X=X_ideal.reshape(-1, 1))

    y_pred_lb = y_pred - 1 * std_pred
    y_pred_ub = y_pred + 1 * std_pred

    plt.plot(X_ideal, y_pred)
    plt.plot(X_ideal, y_pred_lb, "b")
    plt.plot(X_ideal, y_pred_ub, "b")

    plt.scatter(X, y)

    plt.show()


if __name__ == "__main__":
    run_optimisation()
