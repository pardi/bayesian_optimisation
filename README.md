![example branch parameter](https://github.com/pardi/bayesian_optimisation/actions/workflows/python-app.yml/badge.svg?branch=main)

# Bayesian Optimisation suite
This repo contains a collection of bayesian optimisation algorithms. 

## Install

Dependencies:
- python@3.9
- pipenv

The package uses pipenv as virtual environment but a `Makefile` is provided for simplify the installation.

The `Makefile` calls follows the usage:

`make [ARG]`

with 

``` 
[ARG] 
    install - to setup the environment
    format - to run PEP8 checking and format
    lint - to run the linting on the code
    test - to run the tests
```

During the PR process, the CI will execute all these calls to check if the code is ready for merging.

## Examples

1. [Search maximum of a single variable function](examples/example_single_var_fcn.py)

This example uses the Gaussian Process Regression (GPR) to find the maximum value of a hidden single variable function.
Two functions are provided for testing, and they can be selected with the `option=[int]` argument when the function is defined.

The functions have the option to set additive noise with (mean, std) = (0.0, noise_level). 

Because bayesian optimisation do not have a defined stopping mechanism, we run the script for 100 iterations. The following 
picture shows the evolution of the best point found over iterations - `noise = 0.1`.

![](https://github.com/pardi/bayesian_optimisation/blob/main/bayesian_opt.gif)
