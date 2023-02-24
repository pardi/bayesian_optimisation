import numpy as np
import pytest
from sklearn.gaussian_process import GaussianProcessRegressor

import bayesian_opt.hidden_fcns as hf
import examples.example_single_var_fcn as ex


@pytest.mark.parametrize(
    "sample_records, readings_record, length, expected_exception",
    [
        (np.array([0, 1, 2, 3, 4]), np.array([0, 1, 2, 3, 2]), 6, False),
        (np.array([0, 1, 2, 3, 4, 5]), np.array([0, 1, 2, 3, 2]), 6, True),
    ],
)
def test_single_optimisation(
    sample_records, readings_record, length, expected_exception
):

    model = GaussianProcessRegressor()

    objective_fcn = hf.single_var_obj_fcn

    try:
        sample_records, readings_record = ex.single_optimisation(
            model, sample_records, readings_record, objective_fcn
        )
    except ValueError as e:
        if expected_exception:
            assert True
            return
        else:
            raise e

    assert len(sample_records) == length, "sample_record should be size " + str(length)
    assert len(readings_record) == length, "readings_record should be size " + str(
        length
    )
