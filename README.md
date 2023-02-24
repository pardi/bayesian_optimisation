![example workflow](https://github.com/pardi/bayesian_optimisation/actions/workflows/python-app.yml/badge.svg)

# Bayesian Optimisation suite
This repo contains a collection of bayesian optimisation algorithms. 

## Examples

1. [Search maximum of a single variable function](examples/example_single_var_fcn.py)

This example uses the Gaussian Process Regression (GPR) to find the maximum value of a hidden single variable function.
Two functions are provided for testing, and they can be selected with the `option=[int]` argument when the function is defined.

The functions have the option to set additive noise with (mean, std) = (0.0, noise_level). 

Because bayesian optimisation do not have a defined stopping mechanism, we run the script for 100 iterations. The following 
picture shows the evolution of the best point found over iterations - `noise = 0.1`.

![](https://github.com/pardi/bayesian_optimisation/blob/main/bayesian_opt.gif)
