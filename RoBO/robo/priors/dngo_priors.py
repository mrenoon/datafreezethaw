# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 18:06:59 2016

@author: kleinaa
"""
import numpy as np

from robo.priors.base_prior import BasePrior
from robo.priors.base_prior import LognormalPrior
from robo.priors.base_prior import HorseshoePrior

class DNGOPrior(BasePrior):

    def __init__(self, rng=None):
        """
        Abstract base class to define the interface for priors
        of GP hyperparameter.
        Parameters
        ----------

        """
        if rng is None:
            self.rng = np.random.RandomState(np.random.randint(0, 10000))
        else:
            self.rng = rng
            
        # Prior for the alpha
        self.ln_prior_alpha = LognormalPrior(sigma=0.1, mean=-10)
            
        # Prior for the sigma^2
        #self.ln_prior_beta = LognormalPrior(sigma=0.1, mean=2)
        self.horseshoe = HorseshoePrior(scale=0.1)    

    def lnprob(self, theta):
        """
        Returns the log probability of theta. Note: theta should
        be on a log scale.

        Parameters
        ----------
        theta : (D,) numpy array
            A hyperparameter configuration in log space.

        Returns
        -------
        float
            The log probability of theta
        """
        lp = 0
        lp += self.ln_prior_alpha.lnprob(theta[0])
        #lp += self.ln_prior_beta.lnprob(theta[-1])
        lp += self.horseshoe.lnprob(theta[-1])
        
        return lp        
        

    def sample_from_prior(self, n_samples):
        p0 = np.zeros([n_samples, 2])
        p0[:, 0] = self.ln_prior_alpha.sample_from_prior(n_samples)[:, 0]
        #p0[:, -1] = self.ln_prior_beta.sample_from_prior(n_samples)[:, 0]
        # Noise sigma^2
        sigmas = self.horseshoe.sample_from_prior(n_samples)[:, 0]
        # Betas
        p0[:, -1] = np.log(1 / np.exp(sigmas))

        return p0

    def gradient(self, theta):
        """
        Computes the gradient of the prior with
        respect to theta.

        Parameters
        ----------
        theta : (D,) numpy array
            Hyperparameter configuration in log space

        Returns
        -------
        (D) np.array
            The gradient of the prior at theta.
        """
        pass