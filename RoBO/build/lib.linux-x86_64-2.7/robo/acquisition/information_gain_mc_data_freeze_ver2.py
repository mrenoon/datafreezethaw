import numpy as np
import emcee
import logging
from scipy.stats import norm

from robo.acquisition.log_ei import LogEI
from robo.acquisition.base_acquisition import BaseAcquisitionFunction
from robo.initial_design.init_random_uniform import init_random_uniform
from robo.util import mc_part
from copy import deepcopy

logger = logging.getLogger(__name__)

#Basically cloned from the information_gain_mc_freeze file
#Things to change:
#   - the information gain is over the distribution of asymptotic training loss, or dataset size = 1
#           => so the model.predict somewhere must be appropriate as to predict the asymptotic loss
#               BUTTT, the distribution to be obtained is from training some X at a specific S


#   - the information gain must be divided by the time taken to train the model

VU_PRINT = -1

class InformationGainMC(BaseAcquisitionFunction):
    # def __init__(self, model, X_lower, X_upper,
    #              Nb=100, Nf=500,
    #              sampling_acquisition=None,
    #              sampling_acquisition_kw={"par": 0.0},
    #              Np=50, **kwargs):

    def __init__(self, model, X_lower, X_upper,
                 Nb=50, Nf=500,
                 sampling_acquisition=None,
                 sampling_acquisition_kw={"par": 0.0},
                 Np=50, **kwargs):

        """
        The InformationGainMC computes the asymptotically exact, sampling
        based variant of the entropy search acquisition function [1] by
        approximating the distribution over the minimum with MC sampling.

        [1] Hennig and C. J. Schuler
            Entropy search for information-efficient global optimization
            Journal of Machine Learning Research, 13, 2012


        Parameters
        ----------
        model: Model object
            A model should have following methods:
            - predict(X)
            - predict_variance(X1, X2)
        X_lower: np.ndarray (D)
            Lower bounds of the input space
        X_upper: np.ndarray (D)
            Upper bounds of the input space
        Nb: int
            Number of representer points.
        Np: int
            Number of prediction points at X to calculate stochastic changes
            of the mean for the representer points
        Nf: int
            Number of functions that are sampled to approximate pmin
        sampling_acquisition: BaseAcquisitionFunction
            A function to be used in calculating the density that
            representer points are to be sampled from. It uses
        sampling_acquisition_kw: dict
            Additional keyword parameters to be passed to sampling_acquisition

        """

        self.Nb = Nb
        super(InformationGainMC, self).__init__(model, X_lower, X_upper)
        self.D = self.X_lower.shape[0]
        self.sn2 = None
        if sampling_acquisition is None:
            sampling_acquisition = LogEI

        #The sampling acquisition is only over the config space, not including the
        #datasize. The sampling acquisition will call freezeModel.predict(), which
        #return the asymptotic prediction
        self.sampling_acquisition = sampling_acquisition(
            model, self.X_lower[:-1], self.X_upper[:-1], **sampling_acquisition_kw)
        
        self.Nf = Nf
        self.Np = Np
        self.config_nr = -1
        self.overhead = 0

    # def compute(self, X, derivative=False, *args):

    #     if derivative:
    #         raise NotImplementedError
    #     # Compute the fantasized pmin if we would evaluate at x
    #     new_pmin = self.change_pmin_by_innovation(X)

    #     # Calculate the Kullback-Leibler divergence between the old and the
    #     # fantasised pmin
    #     H = -np.sum(np.multiply(np.exp(self.logP), (self.logP + self.lmb)))
    #     dHp = - np.sum(np.multiply(new_pmin,
    #                         np.add(np.log(new_pmin), self.lmb)), axis=0) - H
    #     # We maximize
    #     return -np.array([dHp])

    def computeNew(self, X=None, overhead=0, derivative=False, *args):
        # Return information gain per unit time from evaluating an X (already updated in model) 
        # with dataset size=S, compared to the current model

        if derivative:
            raise NotImplementedError

        assert self.old_H is not None, "must call ig_mc.update(true_update=True) before compute()"

        

        # Calculate the Kullback-Leibler divergence between the old and the
        # fantasised pmin

        # oldH = -np.sum(np.multiply(np.exp(self.logP), (self.logP + self.lmb)))
        
        #now we have to append the new fantasized input to the model
        m, v = self.model.predict(xprime=X, option='new')
        # if VU_PRINT >= 1:
        #     print "Predicted "

        fant_new = m

        newModel = deepcopy(self.model)

        # if VU_PRINT >= 1:
        #     print "X=", X, " X.shape=", X.shape
        #     print "newModel.X.shape = ", newModel.X.shape, " while X[np.newaxis,:-1].shape=", X[np.newaxis,:-1].shape


        newX = np.append(newModel.X, X[:,:-1], axis=0)
        #oh yeah the real config is just X[:-1]

        # if VU_PRINT >= 1:
        #     print ""



        ysNew = np.zeros(len(newModel.ys) + 1, dtype=object)
        for i in xrange(len(newModel.ys)): ##improve: do not use loop here, but some expansion
            ysNew[i] = newModel.ys[i]
        ysNew[-1] = np.array(fant_new[0])

        
        ssNew = np.zeros(len(newModel.ss) + 1, dtype=object)
        for i in xrange(len(newModel.ss)): ##improve: do not use loop here, but some expansion
            ssNew[i] = newModel.ss[i]
        ssNew[-1] = np.array(X[:,-1])

        newModel.train(X=newX, Y=ysNew, S=ssNew, do_optimize=False)
        # newModel.ys = newModel.y_train = ysNew

        # newModel.Y = np.append(newModel.Y, np.array([[fant_new]]), axis=0)
        
        newModel.C_samples = np.zeros(
                    (newModel.C_samples.shape[0], newModel.C_samples.shape[1] + 1, newModel.C_samples.shape[2] + 1))
        newModel.mu_samples = np.zeros(
            (newModel.mu_samples.shape[0], newModel.mu_samples.shape[1] + 1, 1))

        #So we dont retrain the model now ?????

        if VU_PRINT >= 1:
            print "after adding fantasized stuff, newModel.X is ", newModel.X
            print "newModel.ys = ", newModel.ys
            print "newModel.ss = ", newModel.ss
            print "dataset size = ", X[:,-1]

        H_gained_per_time = self.get_H_per_time(newModel, X[0,-1], overhead)

        # if VU_PRINT >= 1:
        #     print "In ig_mc, compute(), X = ", X, "H gained per time = ", H_gained_per_time
            # print "lmb = ",lmb

        return np.array([[H_gained_per_time]])

    # def compute(self, newS):

    def compute(self, X=None, derivative=False):
        newS = X[0][0]

        # print "in Compute, overhead  = ", self.overhead
        #compute the information gained per unit time if we train the old model for another iteration
        #iteration size = the first data size trained on this config
        freezeModel = deepcopy(self.model)

        config = freezeModel.X[self.config_nr]
        ys = freezeModel.ys[self.config_nr]
        ss = freezeModel.ss[self.config_nr]

        if newS <= ss[-1]:
            return np.array([[-1000000000]])
        # newS = min( ss[-1] + ss[0], 1.0)

        fantOld = freezeModel.predict(option='old', conf_nr=self.config_nr, datasize=newS)[0][0]


        ys = np.append(ys, np.array(fantOld), axis=0)

        # freezeModel.ys[config_id] = ys

        ss = np.append(ss, np.array([newS]), axis=0)

        if VU_PRINT >= 0:
            print "in IG_MC, computeOldModel, after Predicting for old model, fantOld = ", fantOld
            print "ys = ", ys
            print "ss after appending new shit = ", ss
            print "freezeModel.ss[config_id] = ", freezeModel.ss[self.config_nr]
            print "ss.shape = ", ss.shape, " while ss[config-id].shape = ", freezeModel.ss[self.config_nr].shape
            print "ys.shape = ", ys.shape, " while ys[config-id].shape = ", freezeModel.ys[self.config_nr].shape


        # freezeModel.train(Y=ys, S=ss, do_optimize=False)
        freezeModel.ys[self.config_nr] = freezeModel.y_train[self.config_nr] = ys

        freezeModel.ss[self.config_nr] = freezeModel.s_train[self.config_nr] = ss

        freezeModel.Y[self.config_nr, :] = ys[-1]

        #updated the freezeModel with new fantasized observation

        H_gained_per_time = self.get_H_per_time(freezeModel, newS, self.overhead, currentS=ss[-2])
        
        if VU_PRINT >= 1:
            print "Infor Gained for size ", newS, " of config #", self.config_nr, " = ", H_gained_per_time

        return np.asarray([[H_gained_per_time]])

    def get_H_per_time(self, model, datasize, overhead, currentS=0):
        
        assert self.old_H is not None, "ig_mc must be updated with true_update first"
        self.update(model, calc_repr=False)
        H = -np.sum(np.multiply(np.exp(self.logP), (self.logP + self.lmb)))

        H_per_time = (H-self.old_H) / self.get_training_time_old(datasize, overhead, currentS)

        return H_per_time

    def get_training_time_old(self, S, overhead, currentS):
        result = self.get_training_time(S, overhead) - self.get_training_time(currentS, 0)
        if currentS != 0:
            print "Estimate time taken to train old model size ", currentS, " to size ",S , " = ", result, " overhead = ", overhead
        return result

    def get_training_time(self, S, overhead):
        #dataset size 10    20     30     40,      50,       80,        120,        150
        timeBoundaries = [0,    10,    20,   30,     40,     50,   60,    80,    90,    100,    120,    150,      600]
        timeBenchmarks = [5,    10,    28,   66,     88,     110,   190,   260,   300,   350,    450,    600,      6500]

        if S==0:
            return 0

        Sp = S*480+20

        idex = 0
        while timeBoundaries[idex]<Sp:
            idex += 1

        #now idex is 0 if S <= 10, ...
        upperBound = timeBenchmarks[idex]
        lowerBound = timeBenchmarks[idex-1]

        difference = upperBound - lowerBound
        result = lowerBound + difference * (Sp-timeBoundaries[idex-1]) / ( timeBoundaries[idex] - timeBoundaries[idex-1] )
        result += overhead #this is for overhead

        # result = np.exp(S/20)
        if VU_PRINT >= 1:
            print "ESTIMATE training time for dataset size = ", Sp, " = ", result
        return result

    def sample_representer_points(self):
        self.sampling_acquisition.update(self.model)

        start_points = init_random_uniform(self.X_lower[:-1],
                                       self.X_upper[:-1],
                                       self.Nb)

        # Set all the representer points to have S = 1
        # Do I even need to ???
        # actually, the representer points should be the configs, and not including the dataset size

        if VU_PRINT >= 1:
            print "In ig_mc, sample_representer_points, start points = ", start_points
        # for i in range(self.Nb):
        #     start_points[i][-1] = 1
        
        if VU_PRINT >= 1:
            print "In ig_mc, sample_representer_points, start points after setting size = ", start_points
        
        # input("Please Enter")

        sampler = emcee.EnsembleSampler(self.Nb,
                                        self.D - 1,     # dimension - 1(dataset size)
                                        self.sampling_acquisition_wrapper)

        # zb are the representer points and lmb are their log EI values
        self.zb, self.lmb, _ = sampler.run_mcmc(start_points, 20)

        # input("DONE mcmc to sample points, Enter now")

        if VU_PRINT >= 1:
            print "In ig_mc, after sampling representer points, zb = ", self.zb
            # print "lmb = ",self.lmb


        if len(self.zb.shape) == 1:
            self.zb = self.zb[:, None]
        if len(self.lmb.shape) == 1:
            self.lmb = self.lmb[:, None]

    def actualize(self, zb, lmb):       #new
        self.zb = zb
        self.lmb = lmb


    def sampling_acquisition_wrapper(self, x):
        if np.any(x < self.X_lower[:-1]) or np.any(x > self.X_upper[:-1]):
            return -np.inf
        return self.sampling_acquisition(np.array([x]))[0]

    def update(self, model, calc_repr=False, true_update=False):   #Different
        
        backupModel = self.model

        self.model = model

        #self.sn2 = self.model.get_noise()

        # Sample representer points
        self.sampling_acquisition.update(model)
        if calc_repr:
            self.sample_representer_points()

        # Omega values which are needed for the innovations
        # by sampling from a uniform grid
        self.W = norm.ppf(np.linspace(1. / (self.Np + 1),
                                    1 - 1. / (self.Np + 1),
                                    self.Np))[np.newaxis, :]

        # Compute current posterior belief at the representer points
        self.Mb, self.Vb = self.model.predict(self.zb, full_cov=True)

        # print "zb = ", self.zb
        # print "Mb = ", self.Mb
        # input("Done predicting at zb, Enter now")
        #So Mb and Vb are the mean and variances of the representer points, due to the
        #fantasized model after evaluating this X.

        #So wait, maybe now I just need to make this Mb and Vb to be the mean and variance of 
        #the asymptotic training loss instead right ?? Yup !

        #What about I let all those representer points having data-size = 1 ??? That works too right ?


        #self.Mb, self.Vb = self.model.predict(self.zb)
        self.pmin = mc_part.joint_pmin(self.Mb, self.Vb, self.Nf)
        self.logP = np.log(self.pmin)
        if true_update:
            self.old_logP = self.logP
            self.old_H = -np.sum(np.multiply(np.exp(self.logP), (self.logP + self.lmb)))
        else:
            self.model = backupModel

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
        dm_rep = dm_rep.dot(self.W)

        # Compute the deterministic innovation for the variance
        dv_rep = -norm_cov.dot(sigma_x_rep.T)

        return dm_rep, dv_rep

    def change_pmin_by_innovation(self, x):

        # Compute the change of our posterior at the representer points for
        # different halluzinate function values of x
        dmdb, dvdb = self.innovations(x, self.zb)

        # Update mean and variance of the posterior (at the representer points)
        # by the innovations
        Mb_new = self.Mb + dmdb
        Vb_new = self.Vb + dvdb

        # Return the fantasized pmin
        return mc_part.joint_pmin(Mb_new, Vb_new, self.Nf)


#obsolete
def get_training_time(S):
    #dataset size 10    20     30     40,      50,       80,        120,        150
    timeBoundaries = [0,    10,       30,     40,     50,   60,    80,        120,        150,      600]
    timeBenchmarks = [5,    10,       40,     60,     110,   150,   300,      600,        900,      6500]

    Sp = S*500

    idex = 0
    while timeBoundaries[idex]<Sp:
        idex += 1

    #now idex is 0 if S <= 10, ...
    upperBound = timeBenchmarks[idex]
    lowerBound = timeBenchmarks[idex-1]

    difference = upperBound - lowerBound
    result = lowerBound + difference * (Sp-timeBoundaries[idex-1]) / ( timeBoundaries[idex] - timeBoundaries[idex-1] )
    result += 35 #this is for overhead

    # result = np.exp(S/20)
    if VU_PRINT >= 1:
        print "ESTIMATE training time for dataset size = ", Sp, " = ", result
    return result