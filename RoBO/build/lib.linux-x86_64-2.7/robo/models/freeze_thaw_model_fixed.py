# encoding=utf8
__author__ = "Tulio Paiva"
__email__ = "paivat@cs.uni-freiburg.de"

import scipy
import numpy as np
import logging
from scipy import optimize
from robo.models.base_model import BaseModel
from sklearn.metrics import mean_squared_error as mse
import time
import matplotlib.pyplot as pl
from numpy.linalg import inv
import emcee
import scipy.stats as sps
from scipy.optimize import minimize
from numpy.linalg import solve
from math import exp
from scipy.linalg import block_diag
from robo.priors.base_prior import BasePrior, TophatPrior, \
LognormalPrior, HorseshoePrior, UniformPrior

logger = logging.getLogger(__name__)

VU_PRINT = -5

class FreezeThawGP(BaseModel):

    def __init__(self,
                 x_train=None,
                 y_train=None,
                 x_test=None,
                 y_test=None,
                 sampleSet=None, 
                 hyper_configs=12, 
                 chain_length=100, 
                 burnin_steps=100,
                 invChol=True,
                 horse=True, 
                 samenoise=True,
                 lg=True):
        """
        Interface to the freeze-thawn GP library. The GP hyperparameter are obtained
        by integrating out the marginal loglikelihood over the GP hyperparameters.
        
        Parameters
        ----------
        x_train: ndarray(N,D)
            The input training data for all GPs
        y_train: ndarray(N,T)
            The target training data for all GPs. The ndarray can be of dtype=object,
            if the curves have different lengths
        x_test: ndarray(*,D)
            The current test data for the GPs, where * is the number of test points
        sampleSet : ndarray(S,H)
            Set of all GP hyperparameter samples (S, H), with S as the number of samples and H the number of
            GP hyperparameters. This option should only be used, if the GP-hyperparameter samples already exist
        """

        self.X = x_train
        self.ys = y_train
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.Y = None
        
        self.hyper_configs = hyper_configs
        self.chain_length = chain_length
        self.burnin_steps = burnin_steps
        
        self.invChol=invChol
        self.horse = horse
        self.samenoise = samenoise
        
        self.uPrior = UniformPrior(minv=0, maxv=10)
        self.lnPrior = LognormalPrior(sigma=0.1, mean=0.0)
        self.hPrior = HorseshoePrior()

        if x_train is not None:
            self.C_samples = np.zeros(
                (self.hyper_configs, self.x_train.shape[0], self.x_train.shape[0]))
            self.mu_samples = np.zeros(
                (self.hyper_configs, self.x_train.shape[0], 1))
            self.activated = False

        self.lg = lg

    def actualize(self):
        self.C_samples = np.zeros(
            (self.hyper_configs, self.x_train.shape[0], self.x_train.shape[0]))
        self.mu_samples = np.zeros(
            (self.hyper_configs, self.x_train.shape[0], 1))
        self.activated = False


    
    
    def train(self, X=None, Y=None, do_optimize=True):
        """
        Estimates the GP hyperparameter by integrating out the marginal
        loglikelihood over the GP hyperparameters   
        
        Parameters
        ----------
        x_train: ndarray(N,D)
            The input training data for all GPs
        y_train: ndarray(T,N)
            The target training data for all GPs. The ndarray can be of dtype=object,
            if the curves have different lengths.   
        """
        if VU_PRINT >= 0:
            print "in freeze_thaw_model, train(), X = ", X
            print "Y = ", Y

        if X is not None:
            self.X = X
            self.x_train = X
        if Y is not None:
            self.ys = Y
            self.y_train = Y
            self.Y = np.zeros((len(Y), 1))
            for i in xrange(self.Y.shape[0]):
                self.Y[i, :] = Y[i][-1]

        

        self.m_const = self.get_mconst()

        if do_optimize:
            sampleSet = self.create_configs(x_train=self.X, y_train=self.ys, hyper_configs=self.hyper_configs, chain_length=self.chain_length, burnin_steps=self.burnin_steps)
            self.samples = sampleSet
            self.printAlphaBeta()
            # for ide, gpHyper in enumerate(self.samples):
            #     print "## GP_hypers[", ide, "].likelihood = ", self.marginal_likelihood(gpHyper,show=True)

    
    def predict(self, xprime=None, option='asympt', conf_nr=0, from_step=None, further_steps=1, full_cov=False):
        """
        Predict using one of thre options: (1) predicion of the asymtote given a new configuration,
        (2) prediction of a new step of an old configuration, (3) prediction of steps of a curve of 
        a completely new configuration

        Parameters
        ----------
        xprime: ndarray(N,D)
            The new configuration(s)
        option: string
            The prediction type: 'asympt', 'old', 'new'
        conf_nr: integer
            The index of an old configuration of which a new step is predicted
        from_step: integer
            The step from which the prediction begins for an old configuration.
            If none is given, it is assumend one is predicting from the last step
        further_steps: integer
            How many steps must be predicted from 'from_step'/last step onwards

        Results
        -------
        return: ndarray(N, steps), ndarray(N, steps)
            Mean and variance of the predictions
        """
        if option == 'asympt':
            if not full_cov:
                mu, std2, _ = self.pred_asympt_all(xprime)
            else:
                mu, std2, _, cov = self.pred_asympt_all(xprime, full_cov=full_cov)
        elif option == 'old': 
            if from_step is None:
                mu, std2, _ = self.pred_old_all(
                    conf_nr=conf_nr + 1, steps=further_steps)
            else:
                mu, std2, _ = self.pred_old_all(
                    conf_nr=conf_nr + 1, steps=further_steps, fro=from_step)
        elif option == 'new':
            mu, std2 = self.pred_new_all(
                steps=further_steps, xprime=xprime, asy=False)

        if type(mu) != np.ndarray:
            mu = np.array([[mu]])
        elif len(mu.shape)==1:
            mu = mu[:,None]

        if not full_cov:
            return mu, std2
        else:
            return mu, cov



    def setGpHypers(self, sample):
        """
        Sets the gp hyperparameters

        Parameters
        ----------
        sample: ndarray(Number_GP_hyperparameters, 1)
            One sample from the collection of all samples of GP hyperparameters
        """
        # self.m_const = self.get_mconst()


        flex = self.X.shape[-1]
        self.theta_d = np.zeros(flex)
        self.theta_d = sample[:flex]

        if not self.samenoise:
            self.theta0, self.alpha, self.beta, self.noiseHyper, self.noiseCurve = sample[flex:]
        else:
            self.theta0, self.alpha, self.beta, noise = sample[flex:]
            self.noiseHyper = self.noiseCurve = noise

    def create_configs(self, x_train, y_train, hyper_configs=40, chain_length=200, burnin_steps=200):
        """
        MCMC sampling of the GP hyperparameters

        Parameters
        ----------
        x_train: ndarray(N, D)
            The input training data.
        y_train: ndarray(N, dtype=object)
            All training curves. Their number of steps can diverge
        hyper_configs: integer
            The number of walkers
        chain_length: integer
            The number of chain steps 
        burnin_steps: integer
            The number of MCMC burning steps

        Results
        -------
        samples: ndarray(hyper_configs, number_gp_hypers)
            The desired number of samples for all GP hyperparameters
        """

        # number of length scales
        flex = x_train.shape[-1]

        if not self.samenoise:
            #theta0, noiseHyper, noiseCurve, alpha, beta, m_const
            fix = 5
        else:
            #theta0, noise, alpha, beta, m_const
            fix = 4

        #pdl = PredLik(x_train, y_train, invChol=self.invChol,
        #              horse=self.horse, samenoise=self.samenoise)
        if hyper_configs < 2*(fix+flex):
            hyper_configs = 2*(fix+flex)
            self.hyper_configs = hyper_configs
            self.actualize()

        samples = np.zeros((hyper_configs, fix + flex))

        sampler = emcee.EnsembleSampler(
            hyper_configs, fix + flex, self.marginal_likelihood)

        # sample length scales for GP over configs
        #uPrior = UniformPrior(minv=0, maxv=10)
        p0a = self.uPrior.sample_from_prior(n_samples=(hyper_configs, flex))

        # sample amplitude for GP over configs and alpha and beta for GP over
        # curve
        #lnPrior = LognormalPrior(sigma=0.1, mean=0.0)
        p0b = self.lnPrior.sample_from_prior(n_samples=(hyper_configs, 3))

        p0 = np.append(p0a, p0b, axis=1)

        #hPrior = HorseshoePrior()

        if not self.samenoise:
            if not self.horse:
                p0d = self.lnPrior.sample_from_prior(n_samples=(hyper_configs, 2))
            else:
                p0d = np.abs(self.hPrior.sample_from_prior(
                    n_samples=(hyper_configs, 2)))
        else:
            if not self.horse:
                p0d = self.lnPrior.sample_from_prior(n_samples=(hyper_configs, 1))
            else:
                p0d = np.abs(self.hPrior.sample_from_prior(
                    n_samples=(hyper_configs, 1)))

        p0 = np.append(p0, p0d, axis=1)

        p0, _, _ = sampler.run_mcmc(p0, burnin_steps)

        pos, prob, state = sampler.run_mcmc(p0, chain_length)

        p0 = pos

        samples = sampler.chain[:, -1]

        if VU_PRINT >= 2:
            print "in GP.create_configs, after the mcmc runs, p0 = ", p0
            print "while samples = ", samples

        return np.exp(samples)

    def marginal_likelihood(self, theta, show=False):
        """
        Calculates the marginal_likelikood for both the GP over hyperparameters and the GP over the training curves

        Parameters
        ----------
        theta: all GP hyperparameters

        Results
        -------
        marginal likelihood: float
            the resulting marginal likelihood
        """

        x = self.x_train
        y = self.y_train

        # if VU_PRINT >= 0:
        #     print "In freeze_thaw_model, marginal_likelihood , x = ", x
        #     print "y = ", y
            
        flex = self.x_train.shape[-1]

        theta_d = np.zeros(flex)
        theta_d = theta[:flex]
        if not self.samenoise:
            theta0, alpha, beta, noiseHyper, noiseCurve = theta[flex:]
        else:

            theta0, alpha, beta, noise = theta[flex:]
            noiseHyper = noiseCurve = noise

        self.theta_d = np.exp(theta_d)

        self.noiseHyper = exp(noiseHyper)

        self.noiseCurve = exp(noiseCurve)

        self.theta0 = np.exp(theta0)
        self.alpha = np.exp(alpha)
        self.beta = np.exp(beta)

        # self.m_const = self.get_mconst()
        #Wait why the fuck does this one get commented out ???



        y_vec = self.getYvector(y)
        self.y_vec = y_vec
        
        O = self.getOmicron(y)

        kx = self.kernel_hyper(x, x)

        if kx is None:
            print " kx is None !"
            return -np.inf

        if self.lg:
            Lambda, gamma = self.lambdaGamma(self.m_const)      #Default

           

        else:
            Lambda, gamma = self.gammaLambda(self.m_const)

        if Lambda is None or gamma is None:
            print " Lambda/gamma is None !"
            return -np.inf

        kx_inv = self.invers(kx)
        if kx_inv is None:
            print " kx_inv is None !"
            return -np.inf

        kx_inv_plus_L = kx_inv + Lambda

        kx_inv_plus_L_inv = self.invers(kx_inv_plus_L)
        if kx_inv_plus_L_inv is None:
            print " kx_inv_plus_L_inv is None !"
            return -np.inf

        kt = self.getKt(y)

        if kt is None:
            print " kt is None !"
            return -np.inf

        kt_inv = self.invers(kt)
        if kt_inv is None:
            print " kt_inv is None !"
            return -np.inf


        y_minus_Om = y_vec - np.dot(O, self.m_const)


        #kt = kt / 1000.
 
        logP = -(1 / 2.) * np.dot(y_minus_Om.T, np.dot(kt_inv, y_minus_Om)) + (1 / 2.) * np.dot(gamma.T, np.dot(kx_inv_plus_L_inv, gamma))\
               - (1 / 2.) * (self.nplog(np.linalg.det(kx_inv_plus_L)) + self.nplog(np.linalg.det(kx)
                                                                                   ) + self.nplog(np.linalg.det(kt)))

        #SO what about the likelihood of the current alpha, beta number ? Do they "fit" the dataset ??
        #I think this is where its lacking.
        #also, why does logP include det(kt)

        if logP is None or str(logP) == str(np.nan):
            print " logP is None or nan!"
            return -np.inf

        
        lp = logP + np.sum(self.uPrior.lnprob(theta_d)) + np.sum(self.lnPrior.lnprob(np.array([theta0, alpha, beta]))) + self.hPrior.lnprob(np.array([self.noiseHyper]))

        if lp is None or str(lp) == str(np.nan):
            print " lp is None !"
            return -np.inf

        # if VU_PRINT >= -5:
        #     print "In freeze_thaw_model, marginal_likelihood = ", lp
        #     # print "y = ", y
        if show:
            print "lp = ", lp
            print "logP = ", logP
            print "y_minus_Om = ", y_minus_Om

        return lp

    def get_mconst(self):
        #Why the fuck is the mean like this ?? no way !

        m_const = np.zeros((len(self.y_train), 1))
        if VU_PRINT>=-2:
            print "in GP.get_mconst(), step0 m_const = ", m_const

        for i in xrange(self.y_train.shape[0]):
            mean_i = np.mean(self.y_train[i], axis=0)
            m_const[i, :] = mean_i

        if VU_PRINT>=-2:
            print "in GP.get_mconst(), step1 m_const = ", m_const

        overallMean = np.mean(m_const,axis=0)

        for i in range(m_const.shape[0]):
            m_const[i, :] = overallMean

        if VU_PRINT>=-2:
            print "in GP.get_mconst(), m_const = ", m_const

        return m_const

    def pred_asympt(self, xprime, full_cov=False, show=False):
        """
        Given new configuration xprime, it predicts the probability distribution of
        the new asymptotic mean, with mean and covariance of the distribution
        
        Parameters
        ----------
        xprime: ndarray(number_configurations, D)
            The new configurations, of which the mean and the std2 are being predicted
        
        Returns
        -------
        mean: ndarray(len(xprime))
            predicted means for each one of the test configurations
        std2: ndarray(len(xprime))
            predicted std2s for each one of the test configurations
        C: ndarray(N,N)
            The covariance of the posterior distribution. It is used several times in the BO framework
        mu: ndarray(N,1)
            The mean of the posterior distribution. It is used several times in the BO framework
        """
        
         
        if xprime is not None:
            self.xprime = xprime

        theta_d = np.ones(self.X.shape[-1])

        kx_star = self.kernel_hyper(self.X, self.xprime, show=show)

        if kx_star is None:
            if show: print('kx_star is None')
            return None

        kx = self.kernel_hyper(self.X, self.X)
        if kx is None:
            if show: print('kx is None')
            return None

        m_const = self.m_const


        if len(xprime.shape) > 1:
            m_xstar = self.xprime.mean(axis=1).reshape(-1, 1)
        else:
            m_xstar = self.xprime

        m_xstar = np.full(m_xstar.shape, m_const[0], dtype=float)

        # if VU_PRINT >=-1:
        #     print "in GP.pred_asympt, m_xstar = ", m_xstar

        
        kx_inv = self.invers(kx)
        if kx_inv is None:
            if show: print('kx_inv is None')
            return None

        m_const = self.m_const
        
        if self.lg:
            Lambda, gamma = self.lambdaGamma(self.m_const)
            # Lambda2, gamma2 = self.gammaLambda(self.m_const)

            # if VU_PRINT >= -5:
            #     print "in pred_asympt, Lambda1 = ", Lambda, "\n=========== Lambda2 = ", Lambda2, "\ngamma = ", gamma,  "\n============= gamma2 = ", gamma2

        else:
            Lambda, gamma = self.gammaLambda(self.m_const)
        
        if Lambda is None or gamma is None:
            if show: print('Lambda is None or gamma is None')
            return None


        C_inv = kx_inv + Lambda

        C = self.invers(C_inv)
        if C is None:
            if show: print('C is None')
            return None

        self.C = C

        mu = self.m_const + np.dot(C, gamma)

        self.mu = mu



        mean = m_xstar + np.dot(kx_star.T, np.dot(kx_inv, mu - self.m_const))
        
        #Now calculate the covariance
        kstar_star = self.kernel_hyper(self.xprime, self.xprime)
        if kstar_star is None:
            if show: print('kstar_star is None')
            return None

        Lambda_inv = self.invers(Lambda)
        if Lambda_inv is None:
            if show: print('Lambda_inv is None')
            return None

        kx_lamdainv = kx + Lambda_inv

        kx_lamdainv_inv = self.invers(kx_lamdainv)

        if kx_lamdainv_inv is None:
            if show: print('kx_lamdainv_inv is None')
            return None

        cov= kstar_star - np.dot(kx_star.T, np.dot(kx_lamdainv_inv, kx_star))
        
        std2 = np.diagonal(cov).reshape(-1, 1)

        if not full_cov:
            return mean, std2, C, mu
        else:
            return mean, std2, C, mu, cov

    def pred_asympt_all(self, xprime, full_cov=False):
        """
        Predicts mean and std2 for new configurations xprime. The prediction is averaged for
        all GP hyperparameter samples. They are integrated out.

        Parameters
        ----------
        xprime: ndarray(*, D)
            the new configurations for which the mean and the std2 are being predicted

        Returns
        -------
        mean: ndarray(*,1)
            predicted mean for every new configuration in xprime
        std2: ndarray(*,1)
            predicted std2 for every new configuration in xprime
        divby: integer
            number of GP hyperparameter samples which deliver acceptable results for mean
            and std2. 
        """
        if VU_PRINT>=-3:
            print "in GP.pred_asympt_all, xprime = ", xprime

        if not full_cov:
            samples_val = []
            C_valid = []
            mu_val = []
            means_val = []
            std2s_val = []

            divby = self.samples.shape[0]

            if VU_PRINT >= 3:
                print "in freeze_thaw_model, pred_asympt_all, divby = ", divby
                print " samples shape: ", self.samples.shape
                print "samples : ", self.samples

            for i in xrange(self.samples.shape[0]):
                self.setGpHypers(self.samples[i])
                
                pre = self.pred_asympt(xprime)
                
                if pre is not None:
                    mean_one, std2_one, C, mu = pre
                    
                    means_val.append(mean_one.flatten())
                    
                    std2s_val.append(std2_one.flatten())
                    C_valid.append(C)
                    mu_val.append(mu)
                    samples_val.append(self.samples[i])
                else:
                    divby -= 1
                    

            mean_temp = np.zeros((divby, xprime.shape[0]))
            std2_temp = np.zeros((divby, xprime.shape[0]))

            if(divby < self.samples.shape[0]):
                self.C_samples = np.zeros(
                    (divby, self.C_samples.shape[1], self.C_samples.shape[2]))
                self.mu_samples = np.zeros(
                    (divby, self.mu_samples.shape[1], self.mu_samples.shape[2]))
                self.samples = np.zeros((divby, self.samples.shape[1]))

            
            for j in xrange(divby):
                mean_temp[j, :] = means_val[j]
                std2_temp[j, :] = std2s_val[j]
                self.C_samples[j, ::] = C_valid[j]
                self.mu_samples[j, ::] = mu_val[j]
                self.samples[j, ::] = samples_val[j]

            mean = np.mean(mean_temp, axis=0)
            std2 = np.mean(std2_temp, axis=0) + np.mean(mean_temp**2, axis=0)
            std2 -= mean**2

            self.activated = True
            self.asy_mean = mean
            self.asy_std2 = std2

            if VU_PRINT>=-3:
                print "result : mean = ", mean, " std2 = ", std2

            return mean, std2, divby
        
        else:
            samples_val = []
            C_valid = []
            mu_val = []
            means_val = []
            std2s_val = []
            cov_val = []

            divby = self.samples.shape[0]

            for i in xrange(self.samples.shape[0]):
                self.setGpHypers(self.samples[i])
                
                pre = self.pred_asympt(xprime, full_cov=full_cov, show=False)
                
                if pre is not None:
                    mean_one, std2_one, C, mu, cov = pre

                    means_val.append(mean_one.flatten())
                    
                    std2s_val.append(std2_one.flatten())
                    C_valid.append(C)
                    mu_val.append(mu)
                    samples_val.append(self.samples[i])
                    cov_val.append(cov)
                else:
                    divby -= 1
                    # print 'bad: ', divby

            mean_temp = np.zeros((divby, xprime.shape[0]))
            std2_temp = np.zeros((divby, xprime.shape[0]))
            
            cov_temp = np.zeros((divby, cov_val[0].shape[0], cov_val[0].shape[1]))

            if(divby < self.samples.shape[0]):
                self.C_samples = np.zeros(
                    (divby, self.C_samples.shape[1], self.C_samples.shape[2]))
                self.mu_samples = np.zeros(
                    (divby, self.mu_samples.shape[1], self.mu_samples.shape[2]))
                self.samples = np.zeros((divby, self.samples.shape[1]))

            for j in xrange(divby):
                mean_temp[j, :] = means_val[j]
                std2_temp[j, :] = std2s_val[j]
                self.C_samples[j, ::] = C_valid[j]
                self.mu_samples[j, ::] = mu_val[j]
                self.samples[j, ::] = samples_val[j]
                cov_temp[j,::] = cov_val[j] 

            mean = np.mean(mean_temp, axis=0)
            std2 = np.mean(std2_temp, axis=0) + np.mean(mean_temp**2, axis=0)
            std2 -= mean**2
            cov = np.mean(cov_temp, axis=0)

            self.activated = True
            self.asy_mean = mean
            self.asy_std2 = std2

            if VU_PRINT>=-3:
                print "result : mean = ", mean, " std2 = ", std2
                
            return mean, std2, divby, cov

    def printAlphaBeta(self):
        flex = self.X.shape[-1]

        alphasSum = 0
        betasSum = 0
        divby = self.samples.shape[0]
        for i in range(divby):
            alphasSum += self.samples[i][flex+1]
            betasSum += self.samples[i][flex+2]

        print "                                 ========================================================================== "
        print "                                 FYI: mean of self.beta = ", betasSum/divby , ", mean of self.alpha =", alphasSum/divby
        print "                                 ========================================================================== "


    def pred_old(self, t, tprime, yn, mu_n=None, Cnn=None):
        # if VU_PRINT >= -5:
        #     print "                                           FYI: self.beta = ",self.beta, ", self.alpha =",self.alpha

        yn = yn.reshape(-1, 1)

        ktn = self.kernel_curve(t, t)
        if ktn is None:
            return None


        ktn_inv = self.invers(ktn)
        if ktn_inv is None:
            return None

        ktn_star = self.kernel_curve(t, tprime)
        if ktn_star is None:
            return None

        

        

        Omega = np.ones((tprime.shape[0], 1)) - np.dot(ktn_star.T,
                                                       np.dot(ktn_inv, np.ones((t.shape[0], 1))))

        # Exactly why:
        if yn.shape[0] > ktn_inv.shape[0]:
            yn = yn[:ktn_inv.shape[0]]

        if VU_PRINT >= -4:
            print "                             FYI: ktn in pred_old is: ", ktn
            print "                             FYI: ktn_star in pred_old is: ", ktn_star
            print "                             FYI: yn = ", yn
            print "                             FYI: mu_n = ", mu_n


        mean = np.dot(ktn_star.T, np.dot(ktn_inv, yn)) + Omega*mu_n

        ktn_star_star = self.kernel_curve(tprime, tprime)

        if ktn_star_star is None:
            return None

        cov = ktn_star_star - \
            np.dot(ktn_star.T, np.dot(ktn_inv, ktn_star)) + \
            np.dot(Omega, np.dot(Cnn, Omega.T))
        std2 = np.diagonal(cov).reshape(-1, 1)

        return mean, std2

    def pred_old_all(self, conf_nr, steps, fro=None):
        #Here conf_nr is from 1 onwards. That's mu_n = mu[conf_nr - 1, 0] in the for-loop

        # conf_nr = self.basket
        
        if self.activated:

            means_val = []
            std2s_val = []
            divby = self.samples.shape[0]

            yn = self.y_train[conf_nr - 1]
            if fro is None:
                t = np.arange(1, yn.shape[0] + 1)
                tprime = np.arange(yn.shape[0] + 1, yn.shape[0] + 1 + steps)
            else:
                t = np.arange(1, fro)
                tprime = np.arange(fro, fro + steps)

            if VU_PRINT >= -5:
                print "      :::::::       PREDICTING OLD CONFIG, x = ", self.X[conf_nr-1], "- #", conf_nr-1  #, " fro = ", fro, " is activated ?", self.activated
                # print "      :::::::            t = ", t, " , tprime = ", tprime
                # print "      :::::::            mu_samples[0] = ", self.mu_samples[0,::]
                # print "      :::::::            C_samples[0] = ", self.mu_samples[0,::]

            #Added
            sum_mu = 0

            for i in xrange(self.samples.shape[0]):
                
                self.setGpHypers(self.samples[i])

                mu = self.mu_samples[i, ::]

                mu_n = mu[conf_nr - 1, 0]
                
                sum_mu += mu_n
                if VU_PRINT>=-4:
                    # print "     ::::Current mu = ", mu, " so mu_n = ", mu_n
                    print " so current mu_samples[gp_hyper_id].shape = ", mu.shape

                C = self.C_samples[i, ::]
                Cnn = C[conf_nr - 1, conf_nr - 1]

                pre = self.pred_old(t, tprime, yn, mu_n, Cnn)

                if pre is not None:
                    mean_one, std2_one = pre
                    means_val.append(mean_one.flatten())
                    std2s_val.append(std2_one.flatten())
                else:
                    divby -= 1

            if VU_PRINT>=-5:
                print "         ** So the mean asymptotic value is ", sum_mu/self.samples.shape[0]


            mean_temp = np.zeros((divby, steps))
            
            std2_temp = np.zeros((divby, steps))

            

            for j in xrange(divby):
                mean_temp[j, :] = means_val[j]
                std2_temp[j, :] = std2s_val[j]

            if VU_PRINT>=-4:
                print "     :::::: After calculating all configs, mean_temp = ", mean_temp
                print "     :::::: After calculating all configs, std2_temp = ", std2_temp

            mean = np.mean(mean_temp, axis=0)
            std2 = np.mean(std2_temp, axis=0) + np.mean(mean_temp**2, axis=0)
            std2 -= mean**2

            if VU_PRINT >= -5:
                print "            : Result : ", self.X[conf_nr-1], ",last=", self.y_train[conf_nr-1], " =>  mean = ", mean, " std2 = ", std2


            return mean, std2, divby

        else:
            raise Exception


    def pred_new(self, step, asy_mean, y=None):
        
        if VU_PRINT >= -4:
            print "===========  in GP.pred_new, asy_mean = ", asy_mean, " while self.asy_mean = ", self.asy_mean

        if y is not None:
            y_now = y

        if asy_mean is not None:
            self.asy_mean = asy_mean
        fro = 1
        t = np.arange(fro, (fro + 1))
        tprime = np.arange((fro + 1), (fro + 1) + step)
        k_xstar_x = self.kernel_curve(tprime, t)
        k_x_x = self.kernel_curve(t, t)

        chol = self.calc_chol(
            k_x_x + self.noiseCurve * np.eye(k_x_x.shape[0]))

        if VU_PRINT>=-4:
            print "in GP.pred_new, k_x_x = ", k_x_x
            print "in GP.pred_new, chol = ", chol

        # Exactly why:x
        # y_now = np.array([1.])
        # y_now = np.array([100.])
        #WTF IS THIS ???
        y_now = self.y_train
        #????

        sol = np.linalg.solve(chol, y_now)

        if VU_PRINT>=-4:
            print "in GP.pred_new, first sol = ", sol

        sol = np.linalg.solve(chol.T, sol)

        if VU_PRINT>=-4:
            print "in GP.pred_new, second sol = ", sol

        k_xstar_xstar = self.kernel_curve(tprime, tprime)
        k_x_xstar = k_xstar_x.T
        mean = self.asy_mean + np.dot(k_xstar_x, sol)
        solv = np.linalg.solve(chol, k_x_xstar)
        solv = np.dot(solv.T, solv)
        cov = k_xstar_xstar - solv
        std2 = np.diagonal(cov).reshape(-1, 1)
        if VU_PRINT >= -4:
            print "=========> return mean = ", mean, " , std2 = ", std2
        return mean, std2


    def pred_new_new(self, step, asy_mean, asy_std2):
        
        # if VU_PRINT >= -4:
        #     print "===========  in GP.pred_new, asy_mean = ", asy_mean, " while self.asy_mean = ", self.asy_mean

        
        fro = 1
        ts = np.arange(fro, (fro + 1))
        k_ts_ts = self.kernel_curve(ts,ts)

        # if VU_PRINT >= -4:
        #     print "===========  in GP.pred_new, return mean = ", asy_mean, " std2 = ", asy_std2 + k_ts_ts

        return asy_mean, asy_std2 + k_ts_ts

    def pred_new_all(self, steps=13, xprime=None, y=None, asy=False):
        """
        Params
        ------
        asy: Whether the asymptotic has already been calculated or not.
        """
        # if VU_PRINT >=-4:
        #     print "in GP.pred_new_all(), xprime = ", xprime #, " steps = ", steps, "self.asy_mean = ", self.asy_mean



        if xprime is not None:
            self.x_test = xprime

        # Not redundant here. The  PredictiveHyper object is already created. In case kx has already been calculate
        # it's not going to be calculated a second time.
        if asy is False:
            # asy_mean, std2star, _ = self.pred_hyper2(xprime)
            asy_mean, std2star, _ = self.pred_asympt_all(xprime[np.newaxis,:])
            # if VU_PRINT>=-4:
            #     print "in GP.pred_new_all, after calling pred_asympt_all(), asy_mean = ", asy_mean, " std2 = ", std2star
        else:
            asy_mean = self.asy_mean

        if type(asy_mean) is np.ndarray:
            asy_mean = asy_mean[0]


        mean_temp = np.zeros((self.samples.shape[0], steps))
        std2_temp = np.zeros((self.samples.shape[0], steps))

        for i in xrange(self.samples.shape[0]):
            self.setGpHypers(self.samples[i])

            #mean_one, std2_one = self.pred_new(
            #    steps, asy_mean[0], y)
            mean_one, std2_one = self.pred_new_new(
                steps, asy_mean, std2star)

            mean_temp[i, :] = mean_one.flatten()
            std2_temp[i, :] = std2_one.flatten()

        
        mean = np.mean(mean_temp, axis=0)
        std2 = np.mean(std2_temp, axis=0) + np.mean(mean_temp**2, axis=0)
        std2 -= mean**2
        if VU_PRINT >= -5:
            # print "in GP.pred_new_all(), after calculating all the GP hypers, mean_temp = ", mean_temp
            # print "in GP.pred_new_all(), after calculating all the GP hypers, std2_temp = ", std2_temp
            print ":::::    PREDICTING NEW CONFIG:   ", xprime , "    =>>  mean = ", mean, " std2 = ", std2
        return mean, std2


    def getYvector(self, y):
        """
        Transform the y_train from type ndarray(N, dtype=object) to ndarray(T, 1).
        That's necessary for doing matrices operations

        Returns
        -------
        y_vec: ndarray(T,1)
            An array containing all loss measurements of all training curves. They need
            to be stacked in the same order as in the configurations array x_train
        """
        y_vec = np.array([y[0]])
        for i in xrange(1, y.shape[0]):
            y_vec = np.append(y_vec, y[i])
        return y_vec.reshape(-1, 1)

    def getOmicron(self, y):
        """
        Caculates the matrix O = blockdiag(1_1, 1_2,...,1_N), a block-diagonal matrix, where each block is a vector of ones
        corresponding to the number of observations in its corresponding training curve

        Parameters
        ----------
        y: ndarray(N, dtype=object)
            All training curves stacked together

        Returns
        -------
        O: ndarray(T,N)
            Matrix O is used in several computations in the BO framework, specially in the marginal likelihood
        """
        O = block_diag(np.ones((y[0].shape[0], 1)))

        for i in xrange(1, y.shape[0]):
            O = block_diag(O, np.ones((y[i].shape[0], 1)))
        return O

    def kernel_hyper(self, x, xprime, show=False):
        """
        Calculates the kernel for the GP over configuration hyperparameters

        Parameters
        ----------
        x: ndarray
            Configurations of hyperparameters, each one of shape D
        xprime: ndarray
            Configurations of hyperparameters. They could be the same or different than x, 
            depending on which covariace is being built

        Returns
        -------
        ndarray
            The covariance of x and xprime
        """

        if len(xprime.shape)==1:
            xprime = xprime.reshape(1,len(xprime))

        if len(x.shape)==1:
            x = x.reshape(1,len(x))

        if show: 
            print 'in kernel_hyper xprime: {:s} and x: {:s}'.format(xprime.shape, x.shape)
        try:
            r2 = np.sum(((x[:, np.newaxis] - xprime)**2) /
                self.theta_d**2, axis=-1)
            if show:
                print 'in kernel_hyper r2: {:s}'.format(r2.shape)
            fiveR2 = 5 * r2
            result = self.theta0 *(1 + np.sqrt(fiveR2) + fiveR2/3.)*np.exp(-np.sqrt(fiveR2))
            if show: print 'in kernel_hyper result1: {:s}'.format(result.shape)
            if result.shape[1] > 1:
                toadd = np.eye(N=result.shape[0], M=result.shape[1])
                if show: print 'in kernel_hyper toadd: {:s} noiseHyper: {}'.format(toadd.shape, self.noiseHyper) 
                result = result +  toadd*self.noiseHyper
            if show:
                print 'in kernel_hyper result2: {:s}'.format(result.shape)
            return result
        except:
            return None

    def kernel_curve(self, t, tprime, alpha=1., beta=1.):
        """
        Calculates the kernel for the GP over training curves

        Parameters
        ----------
        t: ndarray
            learning curve steps
        tprime: ndarray
            learning curve steps. They could be the same or different than t, depending on which covariace is being built

        Returns
        -------
        ndarray
            The covariance of t and tprime
        """

        # if VU_PRINT>=-5:
        #     print "in kernel_curve, t = ", t, " t[:, np.newaxis] = ", t[:, np.newaxis]
        #     print " and tprime = ", tprime


        try:
            result = np.power(self.beta, self.alpha) / \
                np.power(((t[:, np.newaxis] + tprime) + self.beta), self.alpha)

            result = result + \
                np.eye(N=result.shape[0], M=result.shape[1]) * self.noiseCurve

            return result
        except:
            return None

    def lambdaGamma(self, m_const):
        # I think this method no longer can work, since the the chol decomposition is
        #like precomputed based on the t I guess.

        """
        Difference here is that the cholesky decomposition is calculated just once for the whole Kt and thereafter
        we solve the linear system for each Ktn.
        """
        Kt = self.getKt(self.ys)
        
        self.Kt_chol = self.calc_chol(Kt)
        if self.Kt_chol is None:
            return None, None
        dim = self.ys.shape[0]
        # if VU_PRINT>=-1:
        #     print "in GP.lambdaGamma, self.ys.shape=", self.ys.shape

        Lambda = np.zeros((dim, dim))
        gamma = np.zeros((dim, 1))
        index = 0
        for i, yn in enumerate(self.ys):
            lent = yn.shape[0]
            ktn_chol = self.Kt_chol[index:index + lent, index:index + lent]
            
            index += lent
            ktn_inv = self.inverse_chol(K=None, Chl=ktn_chol)
            if ktn_inv is None:
                return None, None
            one_n = np.ones((ktn_inv.shape[0], 1))
            
            Lambda[i, i] = np.dot(one_n.T, np.dot(ktn_inv, one_n))
            gamma[i, 0] = np.dot(one_n.T, np.dot(ktn_inv, yn - m_const[i]))

        return Lambda, gamma

    def gammaLambda(self, m_const):
        '''
        Calculates Lambda according to the following: Lamda = transpose(O)*inverse(Kt)*O
        = diag(l1, l2,..., ln) =, where ln = transpose(1n)*inverse(Ktn)*1n
        Calculates gamma according to the following: gamma = transpose(O)*inverse(Kt)*(y - Om),
        where each gamma element gamma_n = transpose(1n)*inverse(Ktn)*(y_n -m_n)
        
        Parameters
        ----------
        m_const: float
            the infered mean of f, used in the joint distribution of f and y.        
        
        Returns
        -------
        gamma: ndarray(N, 1)
            gamma is used in several calculations in the BO framework
        Lambda: ndarray(N, N)
                Lamda is used in several calculations in the BO framework
        '''
        dim = self.ys.shape[0]
        Lambda = np.zeros((dim, dim))
        gamma = np.zeros((dim, 1))
        index = 0

        for yn in self.ys:
            yn = yn.reshape(-1,1)
            t = np.arange(1, yn.shape[0]+1)
            
            ktn = self.kernel_curve(t, t)
            if ktn == None:
                return None

            ktn_inv = self.invers(ktn)
            if ktn_inv == None:
                return None
            one_n = np.ones((ktn.shape[0], 1))
            onenT_ktnInv = np.dot(one_n.T, ktn_inv)

            Lambda[index, index] = np.dot(onenT_ktnInv, one_n)
            gamma[index, 0] = np.dot(onenT_ktnInv, yn - m_const[index])            

            index+=1
        
        return Lambda, gamma

    def getKtn(self, yn):
        t = np.arange(1, yn.shape[0] + 1)
        ktn = self.kernel_curve(t, t, 1., 1.)
        # ktn = self.kernel_curve(t, t, 1., 1.)
        
        #why is alpha and beta hard-coded here ???
        # => SO this is fucking where its wrong ?? Hmm but in kernel_curve it uses self.beta and self.gamma right ?


        return ktn

    def getKt(self, y):
        """
        Caculates the blockdiagonal covariance matrix Kt. Each element of the diagonal corresponds
        to a covariance matrix Ktn

        Parameters
        ----------
        y: ndarray(N, dtype=object)
            All training curves stacked together

        Returns
        -------
        """
        
        ktn = self.getKtn(y[0])
        O = block_diag(ktn)

        for i in xrange(1, y.shape[0]):
            ktn = self.getKtn(y[i])
            O = block_diag(O, ktn)

        return O

    def invers(self, K):
        if self.invChol:
            invers = self.inverse_chol(K)
        else:
            try:
                invers = np.linalg.inv(K)
            except:
                invers = None

        return invers

    # def inversing(self, chol):
    #     """
    #     One can use this function for calculating the inverse of K once one has already the
    #     cholesky decompostion

    #     :param chol: the cholesky decomposition of K
    #     :return: the inverse of K
    #     """
    #     inve = 0
    #     error_k = 1e-25
    #     once = False
    #     while(True):
    #         try:
    #             if once is True:
    #                 choly = chol + error_k * np.eye(chol.shape[0])
    #             else:
    #                 choly = chol
    #                 once = True

    #             inve = solve(choly.T, solve(choly, np.eye(choly.shape[0])))
    #             break
    #         except np.linalg.LinAlgError:
    #             error_k *= 10
    #     return inve

    def inverse_chol(self, K=None, Chl=None):
        """ 
        One can use this function for calculating the inverse of K through cholesky decomposition
        
        Parameters
        ----------
        K: ndarray
            covariance K
        Chl: ndarray
            cholesky decomposition of K

        Returns
        -------
        ndarray 
            the inverse of K
        """
        if Chl is not None:
            chol = Chl
        else:
            chol = self.calc_chol(K)

        if chol is None:
            return None

        inve = 0
        error_k = 1e-25
        while(True):
            try:
                choly = chol + error_k * np.eye(chol.shape[0])
                inve = solve(choly.T, solve(choly, np.eye(choly.shape[0])))
                break
            except np.linalg.LinAlgError:
                error_k *= 10
        return inve

    def calc_chol(self, K):
        """
        Calculates the cholesky decomposition of the positive-definite matrix K

        Parameters
        ----------
        K: ndarray
            Its dimensions depend on the inputs x for the inputs. len(K.shape)==2

        Returns
        -------
        chol: ndarray(K.shape[0], K.shape[1])
            The cholesky decomposition of K
        """
        
        
        error_k = 1e-25
        chol = None
        once = False
        index = 0
        found = True
        while(index < 100):
            try:
                if once is True:
                    Ky = K + error_k * np.eye(K.shape[0])
                else:
                    Ky = K + error_k * np.eye(K.shape[0])
                    once = True
                chol = np.linalg.cholesky(Ky)
                found = True
                break
            except np.linalg.LinAlgError:
                error_k *= 10
                found = False
            
            index += 1
        if found:
            return chol
        else:
            return None

    def nplog(self, val, minval=0.0000000001):
        return np.log(val.clip(min=minval)).reshape(-1, 1)
    
