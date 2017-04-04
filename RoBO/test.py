import numpy as np

from robo.fmin import bayesian_optimization
import logging

logging.basicConfig(level=logging.INFO)

def objective_function(x):
    y = np.sin(3 * x[0]) * 4 * (x[0] - 1) * (x[0] + 2)
    return y

lower = np.array([0])
upper = np.array([6])

results = bayesian_optimization(objective_function, lower, upper, num_iterations=50)

print(results["x_opt"])


