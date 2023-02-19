import logging
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from tqdm import tqdm

from bayesian_opt import bayesian_optimisation as bo
from bayesian_opt.hidden_fcns import single_var_obj_fcn

logging.basicConfig(level=logging.INFO)


def single_optimisation(
    model: GaussianProcessRegressor,
    sample_records: np.array,
    readings_record: np.array,
    objective_fcn: Callable,
) -> (np.array, np.array):

    try:
        # select the next sample
        x = bo.optimisation_acquisition(model, sample_records)

        # sample the point
        new_reading = objective_fcn(x)

        readings_record = np.append(readings_record, new_reading).reshape(-1, 1)
        sample_records = np.append(sample_records, x).reshape(-1, 1)

        # Update the model
        model.fit(sample_records, readings_record)

        return sample_records, readings_record

    except ValueError as e:
        raise e


def run_optimisation(obj_func: Callable, noise_level: float = 0.1) -> None:

    # Obtain maximum value of the function

    # Ideal domain of the function
    X_ideal = np.linspace(0, 1, 1000)
    # Ideal functions
    y_ideal = obj_func(X_ideal, noise_value=0)

    # Optimal value
    optimal_idx = np.argmax(y_ideal)
    x_optim, y_optim = X_ideal[optimal_idx], y_ideal[optimal_idx]

    # Find best results
    optimal_x = np.argmax(y_ideal)
    logging.info("Optima: x=%.3f, y=%.3f", X_ideal[optimal_x], y_ideal[optimal_x])

    # GP Regressor
    model = GaussianProcessRegressor()

    # First iteration
    sample_records = np.random.rand(1)
    readings_record = obj_func(sample_records, noise_value=noise_level)

    # Fit model to the data measured
    model.fit(sample_records.reshape(-1, 1), readings_record.reshape(-1, 1))

    for _ in tqdm(range(100)):
        sample_records, readings_record = single_optimisation(
            model, sample_records, readings_record, obj_func
        )

    # Estimated Optimal value
    ext_optimal_idx = np.argmax(readings_record)
    est_x_optim, est_y_optim = (
        sample_records[ext_optimal_idx],
        readings_record[ext_optimal_idx],
    )

    # Evaluate predicted data
    y_pred, std_pred = bo.surrogate_function(model=model, X=X_ideal.reshape(-1, 1))

    # Plotting
    plt.plot(X_ideal, y_ideal)
    plt.plot(X_ideal, y_pred)
    plt.plot(x_optim, y_optim, "x")
    plt.plot(est_x_optim, est_y_optim, "o")

    plt.scatter(sample_records, readings_record)

    plt.show()


if __name__ == "__main__":
    # Create handle to the obj_func
    run_optimisation(single_var_obj_fcn)
