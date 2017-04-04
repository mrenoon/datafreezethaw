# import setup_logger
import gzip
import cPickle as pickle
from robo.acquisition.ei import EI
from robo.maximizers.cmaes import CMAES
import numpy as np
from robo.task.ml.lda_task_freeze import LDA
from robo.solver.freeze_thaw_bayesian_optimization import FreezeThawBO
from robo.models.freeze_thaw_model import FreezeThawGP
from robo.maximizers.direct import Direct
from robo.initial_design.init_random_uniform import init_random_uniform



lda_task = LDA()

freeze_thaw_model = FreezeThawGP(hyper_configs=14)
#freeze_thaw_model = FreezeThawGP(hyper_configs=14, economically=False)

acquisition_func = EI(freeze_thaw_model, X_upper=lda_task.X_upper, X_lower=lda_task.X_lower)

maximizer = Direct(acquisition_func, lda_task.X_lower, lda_task.X_upper)

bo = FreezeThawBO(acquisition_func=acquisition_func,
                          freeze_thaw_model=freeze_thaw_model,
                          maximize_func=maximizer,
                          task=lda_task, init_points=2)


#bo = FreezeThawBO(acquisition_func=acquisition_func,
#                          freeze_thaw_model=freeze_thaw_model,
#                          maximize_func=maximizer,
#                          task=lda_task, init_points=10,
#                          max_epochs=500, stop_epochs=True)

incumbent, incumbent_value = bo.run(5)