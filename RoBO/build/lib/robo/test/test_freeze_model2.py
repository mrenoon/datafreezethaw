# -*- coding: utf-8 -*-

import george
import unittest
import numpy as np
from scipy.optimize import check_grad
from robo.models.gaussian_process import GaussianProcess
from robo.priors.default_priors import TophatPrior
from robo.initial_design.init_random_uniform import init_random_uniform
from robo.acquisition.ei import EI
from robo.acquisition.pi import PI
from robo.acquisition.lcb import LCB
from robo.acquisition.information_gain import InformationGain
from robo.incumbent.best_observation import BestObservation
from robo.incumbent.posterior_optimization import PosteriorMeanOptimization
from robo.incumbent.posterior_optimization import PosteriorMeanAndStdOptimization
from robo.models.freeze_model import FreezeThawGP


class TestGaussianProcess(unittest.TestCase):

    def test(self):
        X_lower = np.array([0])
        X_upper = np.array([1])
        X = init_random_uniform(X_lower, X_upper, 10)

        curves = np.zeros(len(X), dtype=object)
        for i in xrange(len(curves)):
            curves[i] = np.random.rand(3)

        model = FreezeThawGP(x_train=X, y_train=curves)
        model.train()

        x_test = init_random_uniform(X_lower, X_upper, 3)

        # Shape matching predict
        m, v = model.predict(x_test)

        assert len(m.shape) == 1
        assert m.shape[0] == x_test.shape[0]
        assert len(v.shape) == 1
        assert v.shape[0] == x_test.shape[0]

if __name__ == "__main__":
    unittest.main()
