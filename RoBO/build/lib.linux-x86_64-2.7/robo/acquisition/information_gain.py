# encoding=utf8

import sys
import logging

import scipy
import numpy as np
import emcee


from robo.acquisition.log_ei import LogEI
from robo.acquisition.base_acquisition import BaseAcquisitionFunction
from robo.util import epmgp

sq2 = np.sqrt(2)
l2p = np.log(2) + np.log(np.pi)
eps = np.finfo(np.float32).eps

logger = logging.getLogger(__name__)


class InformationGain(BaseAcquisitionFunction):

    def __init__(self, model, X_lower, X_upper,
            Nb=50, Np=400, sampling_acquisition=None,
            sampling_acquisition_kw={"par": 0.0},
            **kwargs):

        """
        The Information Gain acquisition function for Entropy Search [1].
        In a nutshell entropy search approximates the
        distribution pmin of the global minimum and tries to decrease its
        entropy. See Hennig and Schuler[1] for a more detailed view.

        [1] Hennig and C. J. Schuler
            Entropy search for information-efficient global optimization
            Journal of Machine Learning Research, 13, 2012

        Parameters
        ----------
        model: Model object
            A model that implements at least
                 - predict(X)
                 - predict_variances(X1, X2)
            If you want to calculate derivatives than it should also support
                 - predictive_gradients(X)
        X_lower: np.ndarray (D)
            Lower bounds of the input space
        X_upper: np.ndarray (D)
            Upper bounds of the input space
        Nb: int
            Number of representer points to define pmin.
        Np: int
            Number of hallucinated values to compute the innovations
            at the representer points
        sampling_acquisition: function
            Proposal measurement from which the representer points will
            be samples
        sampling_acquisition_kw: dist
            Additional keyword parameters that are passed to the
            acquisition function
        """

        self.Nb = Nb
        super(InformationGain, self).__init__(model, X_lower, X_upper)
        self.D = self.X_lower.shape[0]
        self.sn2 = None

        if sampling_acquisition is None:
            sampling_acquisition = LogEI
        self.sampling_acquisition = sampling_acquisition(
            model, self.X_lower, self.X_upper, **sampling_acquisition_kw)

        self.Np = Np

    def loss_function(self, logP, lmb, lPred, *args):

        H = -np.sum(np.multiply(np.exp(logP), (logP + lmb)))  # current entropy
        dHp = - np.sum(np.multiply(np.exp(lPred),
                                   np.add(lPred, lmb)), axis=0) - H
        return np.array([dHp])

    def compute(self, X, derivative=False, **kwargs):
        """
        Computes the information gain of X and its derivatives

        Parameters
        ----------
        X: np.ndarray(1, D), The input point where the acquisition function
            should be evaluate. The dimensionality of X is (N, D), with N as
            the number of points to evaluate at and D is the number of
            dimensions of one X.

        derivative: Boolean
            If is set to true also the derivative of the acquisition
            function at X is returned
            Not implemented yet!

        Returns
        -------
        np.ndarray(1,1)
            Log Expected Improvement of X
        np.ndarray(1,D)
            Derivative of Log Expected Improvement at X
            (only if derivative=True)

        """
        if X.shape[0] > 1:
            raise ValueError("Entropy is only for single test points")
        if np.any(X < self.X_lower) or np.any(X > self.X_upper):
            if derivative:
                f = 0
                df = np.zeros((1, X.shape[1]))
                return np.array([[f]]), np.array([df])
            else:
                return np.array([[0]])

        if derivative:
            acq, grad = self.dh_fun(X, derivative=True)

            if np.any(np.isnan(acq)) or np.any(acq == np.inf):
                return -sys.float_info.max
            return acq, grad
        else:
            acq = self.dh_fun(X, derivative=False)

            if np.any(np.isnan(acq)) or np.any(acq == np.inf):
                return -sys.float_info.max
            return acq

    def sampling_acquisition_wrapper(self, x):
        if np.any(x < self.X_lower) or np.any(x > self.X_upper):
            return -np.inf
        return self.sampling_acquisition(np.array([x]))[0]

    def sample_representer_points(self):
        self.sampling_acquisition.update(self.model)
        restarts = np.zeros((self.Nb, self.D))
        restarts[0:self.Nb, ] = self.X_lower + (self.X_upper - self.X_lower) \
                    * np.random.uniform(size=(self.Nb, self.D))
        sampler = emcee.EnsembleSampler(
            self.Nb, self.D, self.sampling_acquisition_wrapper)
        # zb are the representer points and lmb are their log EI values
        self.zb, self.lmb, _ = sampler.run_mcmc(restarts, 20)
        if len(self.zb.shape) == 1:
            self.zb = self.zb[:, None]
        if len(self.lmb.shape) == 1:
            self.lmb = self.lmb[:, None]

    def update(self, model):
        self.model = model

        self.sn2 = self.model.get_noise()
        self.sample_representer_points()
        mu, var = self.model.predict(np.array(self.zb), full_cov=True)

        self.logP, self.dlogPdMu, self.dlogPdSigma, self.dlogPdMudMu = \
                        epmgp.joint_min(mu, var, with_derivatives=True)

        self.W = scipy.stats.norm.ppf(np.linspace(1. / (self.Np + 1),
                                    1 - 1. / (self.Np + 1),
                                    self.Np))[np.newaxis, :]

        self.logP = np.reshape(self.logP, (self.logP.shape[0], 1))

    def _dh_fun(self, x):
        # Number of belief locations:
        N = self.logP.size

        # Evaluate innovation
        dMdx, dVdx = self.innovations(x, self.zb)
        # The transpose operator is there to make the array indexing equivalent
        # to matlab's
        dVdx = dVdx[np.triu(np.ones((N, N))).T.astype(bool), np.newaxis]

        dMM = dMdx.dot(dMdx.T)
        trterm = np.sum(np.sum(np.multiply(self.dlogPdMudMu, np.reshape(
                        dMM, (1, dMM.shape[0], dMM.shape[1]))), 2), 1)[
            :, np.newaxis]

        # add a second dimension to the arrays if necessary:
        logP = np.reshape(self.logP, (self.logP.shape[0], 1))

        # Deterministic part of change:
        detchange = self.dlogPdSigma.dot(dVdx) + 0.5 * trterm
        # Stochastic part of change:
        stochange = (self.dlogPdMu.dot(dMdx)).dot(self.W)
        # Predicted new logP:
        lPred = np.add(logP + detchange, stochange)
        _maxLPred = np.amax(lPred, axis=0)
        s = _maxLPred + np.log(np.sum(np.exp(lPred - _maxLPred), axis=0))
        lselP = _maxLPred if np.any(np.isinf(s)) else s

        # Normalise:
        lPred = np.subtract(lPred, lselP)

        # We maximize the information gain
        dHp = -self.loss_function(logP, self.lmb, lPred, self.zb)
        dH = np.mean(dHp)
        return dH

    def dh_fun(self, x, derivative=False):

        if not (np.all(np.isfinite(self.lmb))):
            logger.debug(self.zb[np.where(np.isinf(self.lmb))],
                        self.lmb[np.where(np.isinf(self.lmb))])
            raise Exception(
                "lmb should not be infinite.")

        D = x.shape[1]
        # If x is a vector, convert it to a matrix (some functions are
        # sensitive to this distinction)
        if len(x.shape) == 1:
            x = x[np.newaxis]

        if np.any(x < self.X_lower) or np.any(x > self.X_upper):
            dH = np.spacing(1)
            ddHdx = np.zeros((x.shape[1], 1))
            return np.array([[dH]]), np.array([[ddHdx]])

        dH = self._dh_fun(x)

        if not np.isreal(dH):
            raise Exception("dH is not real")
        # Numerical derivative, renormalisation makes analytical derivatives
        # unstable.
        e = 1.0e-5
        if derivative:
            ddHdx = np.zeros((1, D))
            for d in range(D):
                # ## First part:
                y = np.array(x)
                y[0, d] += e
                dHy1 = self._dh_fun(y)
                # ## Second part:
                y = np.array(x)
                y[0, d] -= e
                dHy2 = self._dh_fun(y)

                ddHdx[0, d] = np.divide((dHy1 - dHy2), 2 * e)
                ddHdx = -ddHdx
            # endfor
            if len(ddHdx.shape) == 3:
                return_df = ddHdx
            else:
                return_df = np.array([ddHdx])
            return np.array([[dH]]), return_df
        return np.array([[dH]])

    def innovations(self, x, rep):
        # Get the variance at x with noise
        _, v = self.model.predict(x)

        # Get the variance at x without noise
        v_ = v - self.sn2

        # Compute the variance between the test point x and
        # the representer points
        sigma_x_rep = self.model.predict_variance(rep, x)

        norm_cov = np.dot(sigma_x_rep, np.linalg.inv(v_))
        # Compute the stochastic innovation for the mean
        dm_rep = np.dot(norm_cov,
                    np.linalg.cholesky(v + 1e-10))

        # Compute the deterministic innovation for the variance
        dv_rep = -norm_cov.dot(sigma_x_rep.T)

        return dm_rep, dv_rep
