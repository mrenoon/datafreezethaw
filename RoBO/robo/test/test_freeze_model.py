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
from robo.models.freeze_thaw_model import FreezeThawGP


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
        
        assert len(model.samples.shape)==2
        
        #"""
        x_test = init_random_uniform(X_lower, X_upper, 3)

        # Shape matching predict
        #m, v = model.predict(x_test)
        #m, v, _ = model.pred_hyper(x_test)
        m_asympt, v_asympt = model.predict(xprime=x_test, option='asympt')

        assert len(m_asympt.shape) == 2
        assert m_asympt.shape[0] == x_test.shape[0]
        assert len(v_asympt.shape) == 1
        assert v_asympt.shape[0] == x_test.shape[0]
        #"""

        m_old,v_old = model.predict(xprime=None, option='old', conf_nr=0, from_step=None, further_steps=1)

        assert len(m_old.shape) == 2
        assert m_old.shape[0] == 1
        assert len(v_old.shape) == 1
        assert v_old.shape[0] == 1

        m_new,v_new = model.predict(xprime=np.array([x_test[0]]), option='new', further_steps=1)

        assert len(m_new.shape) == 2
        assert m_new.shape[0] == np.array([x_test[0]]).shape[0]
        assert len(v_new.shape) == 1
        assert v_new.shape[0] == np.array([x_test[0]]).shape[0]

if __name__ == "__main__":
    unittest.main()
