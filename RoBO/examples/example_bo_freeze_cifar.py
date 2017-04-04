# import setup_logger
import gzip
import cPickle as pickle
from robo.acquisition.ei import EI
from robo.maximizers.cmaes import CMAES
import numpy as np
# from robo.task.ml.lasagne_logrg_task_freeze_trial import LogisticRegression
from robo.task.ml.convnet_cifar_freeze import ConvNetCifar

from robo.solver.freeze_thaw_bayesian_optimization import FreezeThawBO
from robo.models.freeze_thaw_model import FreezeThawGP
from robo.maximizers.direct import Direct
from robo.initial_design.init_random_uniform import init_random_uniform

def load_dataset():

    def load_mnist_images(filename):
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
         
        data = data.reshape(-1, 1, 28, 28)

        return data / np.float32(256)

    def load_mnist_labels(filename):
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        
        return data

    
    X_train = load_mnist_images('train-images-idx3-ubyte.gz')
    y_train = load_mnist_labels('train-labels-idx1-ubyte.gz')
    X_test = load_mnist_images('t10k-images-idx3-ubyte.gz')
    y_test = load_mnist_labels('t10k-labels-idx1-ubyte.gz')

    
    X_train, X_val = X_train[:-10000], X_train[-10000:]
    y_train, y_val = y_train[:-10000], y_train[-10000:]

    return X_train, y_train, X_val, y_val, X_test, y_test


X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()

convnetCifarTask = ConvNetCifar(num_epochs=3)


#logre.X_upper = np.array([np.log(1e-1), 1.0, 2000, 0.75, 20, 10])

freeze_thaw_model = FreezeThawGP(hyper_configs=14)
#freeze_thaw_model = FreezeThawGP(hyper_configs=14, economically=False)


acquisition_func = EI(freeze_thaw_model, X_upper=convnetCifarTask.X_upper, X_lower=convnetCifarTask.X_lower)

maximizer = Direct(acquisition_func, convnetCifarTask.X_lower, convnetCifarTask.X_upper)

bo = FreezeThawBO(acquisition_func=acquisition_func,
                          freeze_thaw_model=freeze_thaw_model,
                          maximize_func=maximizer,
                          nr_epochs_inits=1,
                          nr_epochs_further=1,
                          task=convnetCifarTask, init_points=3)


#bo = FreezeThawBO(acquisition_func=acquisition_func,
#                          freeze_thaw_model=freeze_thaw_model,
#                          maximize_func=maximizer,
#                          task=logre, init_points=10,
#                          max_epochs=500, stop_epochs=True)

incumbent, incumbent_value = bo.run(500)

print ":::   FINALLLLLLLLL   :::    incumbent = ", incumbent , " value  = " , incumbent_value
