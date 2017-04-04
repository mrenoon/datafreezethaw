# import setup_logger
import gzip
import cPickle as pickle
from robo.acquisition.ei import EI
from robo.acquisition.information_gain_mc_data_freeze import InformationGainMC
from robo.maximizers.cmaes import CMAES
import numpy as np
import sys
sys.path.append('/data/ml/RoBO/robo')

# from robo.task.ml.lasagne_logrg_task_freeze_trial import LogisticRegression

# from robo.task.ml.var_size_data_freeze_convnet_cifar import VarSizeDataConvNetCifar
from task.ml.var_size_data_freeze_convnet_cifar_2para import VarSizeDataConvNetCifar


# from robo.solver.var_size_data_freeze_thaw_bayesian_optimization import VarSizeDataFreezeThawBO
from solver.var_size_data_freeze_thaw_bayesian_optimization_ver4 import VarSizeDataFreezeThawBO
# from solver.var_size_data_freeze_thaw_bayesian_optimization_ver2 import VarSizeDataFreezeThawBO

# from robo.models.var_size_data_freeze_thaw_model import VarSizeDataFreezeThawGP
from models.var_size_data_freeze_thaw_model_fixed import VarSizeDataFreezeThawGP

from robo.maximizers.direct import Direct
from robo.initial_design.init_random_uniform import init_random_uniform


convnetCifarTask = VarSizeDataConvNetCifar(num_epochs=30)


#logre.X_upper = np.array([np.log(1e-1), 1.0, 2000, 0.75, 20, 10])

freeze_thaw_model = VarSizeDataFreezeThawGP(hyper_configs=14)
#freeze_thaw_model = FreezeThawGP(hyper_configs=14, economically=False)


acquisition_func = InformationGainMC(freeze_thaw_model, X_upper=convnetCifarTask.X_upper, X_lower=convnetCifarTask.X_lower)

maximizer = Direct(acquisition_func, convnetCifarTask.X_lower, convnetCifarTask.X_upper)

bo = VarSizeDataFreezeThawBO(acquisition_func=acquisition_func,
                          freeze_thaw_model=freeze_thaw_model,
                          maximize_func=maximizer,
                          nr_epochs_inits=1,
                          nr_epochs_further=1,
                          task=convnetCifarTask, init_points=7)


#bo = FreezeThawBO(acquisition_func=acquisition_func,
#                          freeze_thaw_model=freeze_thaw_model,
#                          maximize_func=maximizer,
#                          task=logre, init_points=10,
#                          max_epochs=500, stop_epochs=True)

incumbent, incumbent_value = bo.run(500)

# print ":::   FINALLLLLLLLL   :::    incumbent = ", incumbent , " value  = " , incumbent_value
