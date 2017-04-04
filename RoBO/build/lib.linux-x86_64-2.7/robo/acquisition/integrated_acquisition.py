'''
Created on Aug 11, 2015

@author: Aaron Klein
'''
import logging
import numpy as np

from copy import deepcopy
from robo.acquisition.base_acquisition import BaseAcquisitionFunction
from robo.models.dngo import DNGO
import lasagne

logger = logging.getLogger(__name__)


class IntegratedAcquisition(BaseAcquisitionFunction):

    def __init__(self, model, acquisition_func,
                 X_lower, X_upper, cost_model=None, is_dngo=False):
        '''
        Meta acquisition function that allows to marginalise the
        acquisition function over GP hyperparameter.

        Parameters
        ----------
        model: Model object
            The model of the objective function, it has to be an instance of
            GaussianProcessMCMC or GPyModelMCMC.
        acquisition_func: BaseAcquisitionFunction object
            The acquisition function that will be integrated.
        cost_model: Model object
            If the acquisition function also takes the cost into account, we
            have to specify here the model for the cost function. cost_model
            has to be an instance of GaussianProcessMCMC or GPyModelMCMC.
        is_dngo: boolean
            If you use DNGO as model set this flag to true in order to 
            apply first the basis functions to the input data point
        '''

        self.model = model
        self.is_dngo = is_dngo

        # Save also the cost model if the acquisition function needs it
        if cost_model is not None:

            self.cost_model = cost_model

        # Keep for each model an extra acquisition function module
        self.estimators = []
        for _ in range(self.model.n_hypers):
            # Copy the acquisition function for this model
            estimator = deepcopy(acquisition_func)
            estimator.model = None
            if cost_model is not None:
                estimator.cost_model = None
            self.estimators.append(estimator)

        super(IntegratedAcquisition, self).__init__(model, X_lower, X_upper)

    def update(self, model, cost_model=None, **kwargs):
        '''
        Updates each acquisition function object if the models
        have changed

        Parameters
        ----------
        model: Model object
            The model of the objective function, it has to be an instance of
            GaussianProcessMCMC or GPyModelMCMC.
        cost_model: Model object
            If the acquisition function also takes the cost into account, we
            have to specify here the model for the cost function. cost_model
            has to be an instance of GaussianProcessMCMC or GPyModelMCMC.
        '''

        self.model = model
        if cost_model is not None:
            self.cost_model = cost_model
        for i in range(self.model.n_hypers):
            if cost_model is not None:
                self.estimators[i].update(self.model.models[i],
                                          self.cost_model.models[i],
                                          **kwargs)
            else:
                self.estimators[i].update(self.model.models[i], **kwargs)

    def compute(self, X, derivative=False):
        """
        Integrates the acquisition function over the GP's hyperparameters by
        averaging the acquisition value for X of each hyperparameter sample.

        Parameters
        ----------
        X: np.ndarray(1, D), The input point where the acquisition function
            should be evaluate. The dimensionality of X is (N, D), with N as
            the number of points to evaluate at and D is the number of
            dimensions of one X.

        derivative: Boolean
            If is set to true also the derivative of the acquisition
            function at X is returned

        Returns
        -------
        np.ndarray(1,1)
            Integrated acquisition value of X
        np.ndarray(1,D)
            Derivative of the acquisition value at X (only if derivative=True)
        """
        acquisition_values = np.zeros([self.model.n_hypers])

        if self.is_dngo:
            import lasagne
            X_ = (X - self.model.X_mean) /  self.model.X_std 

            # Get features from the net
            layers = lasagne.layers.get_all_layers(self.model.network)        
            theta = lasagne.layers.get_output(layers[:-1], X_)[-1].eval()
        else:
            theta = X

        # Integrate over the acquisition values
        for i in range(self.model.n_hypers):
            acquisition_values[i] = self.estimators[i](theta,
                                                    derivative=derivative)
                                                 
        return np.array([[acquisition_values.mean()]])
