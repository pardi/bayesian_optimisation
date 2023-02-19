import numpy as np


# TODO:Add decorator
def single_var_obj_fcn(x: np.array, noise_value: float = 0.1) -> np.array:
    if noise_value > 0:
        noise = 2 * np.random.rand(1) * noise_value - noise_value / 2
    else:
        noise = 0

    y = x**2 * np.sin(5 * np.pi * x) ** 6

    return y + noise
