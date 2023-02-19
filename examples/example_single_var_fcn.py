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


def run_optimisation(obj_func: Callable) -> None:

    # Obtain maximum value of the function

    # Ideal domain of the function
    X_ideal = np.linspace(0, 1, 1000)
    # Ideal functions
    y_ideal = [obj_func(x, noise_value=0) for x in X_ideal]

    # Plotting
    plt.plot(X_ideal, y_ideal)

    plt.show()


if __name__ == "__main__":
    # Create handle to the obj_func
    run_optimisation(single_var_obj_fcn)
