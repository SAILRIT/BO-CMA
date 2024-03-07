Created by: Joshua Yancosek
# BEACON
This is python code for a falsification method that combines Bayesian Optimization with aspects of CMA-ES.
Given a controller and simulation, we aim to identify environmental paramters that model uncertainty in the system that violate one or multiple specifications.
BO-CMA is an optimization framework for falsification. BO-CMA searches subsets of the global uncertainty space. Within each subset, BO-CMA initializes a separate GP surrogate to model the objective function over this local search zone. Then, BO-CMA computes a new local search zone by looking at the covariance between the best environmental parameters simulated in the prior search zone.

# Test
Env_Prep.py provides a list of packages that should be installed in that order in order to use this framework.
Testing was done in Ubuntu 20.04 with anaconda spyder and Matlab 2021b.

There are 5 environments used for testing. Code to run BO-CMA and BO are located in the same file with code to run CMA-ES separate.
