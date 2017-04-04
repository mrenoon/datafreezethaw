import setup_logger

import GPy
from robo.models.gpy_model import GPyModel
from robo.acquisition.ei import EI
from robo.maximizers.cmaes import CMAES
from robo.maximizers.direct import Direct
from robo.task.synthetic_functions.logsgd import LogSGD
from robo.solver.bayesian_optimization import BayesianOptimization


logsgd = LogSGD()

kernel = GPy.kern.Matern52(input_dim=logsgd.n_dims)
model = GPyModel(kernel)

acquisition_func = EI(model,
                     X_upper=logsgd.X_upper,
                     X_lower=logsgd.X_lower,
                     par=0.1)

maximizer = Direct(acquisition_func, logsgd.X_lower, logsgd.X_upper)

bo = BayesianOptimization(acquisition_func=acquisition_func,
                          model=model,
                          maximize_func=maximizer,
                          task=logsgd)

bo.run(10)