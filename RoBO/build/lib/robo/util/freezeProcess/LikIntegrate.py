# encoding=utf8
__author__ = "Tulio Paiva"
__email__ = "paivat@cs.uni-freiburg.de"

import numpy as np
import random
import matplotlib.pyplot as pl
from numpy.linalg import inv
import emcee
import time
import scipy.stats as sps
from robo.util.freezeProcess.predictiveLikelihood import PredLik


class LikIntegrate(object):
	'''
	Sampling of GP hyperparameter samples from the log likelihood through the GP MCMC
	
	Parameters
	----------
	y_train: ndarray(N, dtype=object)
		All training curves all together. Each training curve can have a different number of steps.
	invChol: boolean
		Use the cholesky decomposition for calculating the covariance matrix inversion
	horse: boolean
		Use the horseshoe prior for sampling the noise values
	samenoise: boolean
		Assume noise of GPs over learning curve and over configurations are the same
	'''
	
	def __init__(self,
				 y_train,
				 invChol = True,
				 horse=True,
				 samenoise=True):
		
		self.y_train = y_train
		self.invChol = invChol
		self.horse = horse
		self.samenoise = samenoise
	
	def samples_norm(self, n_samples):
		"""
		Samples from the lognorm distribution
		Parameters
		----------
		n_samples: scalar | tuple
			The shape of the samples from the lognormal distribution
		Returns
		-------
		ndarray(n_samples)
			The samples from the lognorm
		"""
		return np.random.lognormal(mean=0.,
							   sigma=1,
							   size=n_samples)
	
	def samples_noise(self, n_samples):
		"""
		Samples noise from the lognorm distribution for the kernel_hyper and kernel_curve
		Parameters
		----------
		n_samples: scalar | tuple
			The shape of the samples from the lognormal distribution
		Returns
		-------
		ndarray(n_samples)
			The noise samples from the lognorm
		"""
		return np.random.lognormal(mean=0.,
							   sigma=1,
							   size=n_samples)
	
	def samples_horse(self, n_samples, scale=0.1, rng=None):
		if rng is None:
			rng = np.random.RandomState(42)
		
		lamda = np.abs(rng.standard_cauchy(size=n_samples))
		#p0 = np.log(np.abs(rng.randn() * lamda * scale))
		p0 = np.abs(np.log(np.abs(rng.randn() * lamda * scale)))
		return p0
	
	
	def samples_uniform(self, n_samples):
		"""
		Samples values between 0 and 10 from the uniform distribution
		Parameters
		----------
		n_samples: scalar | tuple
			The shape of the samples from the uniform distribution
		Returns
		-------
		ndarray(n_samples)
			The samples from the uniform distribution
		"""
		return np.log(np.random.uniform(0, 10, n_samples))
	
	def sampleMconst(self, n_samples=(1,1)):
		"""
		Samples values between y_min and y_max from the uniform distribution
		Parameters
		----------
		n_samples: scalar | tuple
			The shape of the samples from the uniform distribution
		Returns
		-------
		ndarray(n_samples)
			The samples from the uniform distribution between y_min and y_max
		"""
		y = self.getYvector()
		return np.log(np.random.uniform(np.min(y), np.max(y), n_samples))
		

	def getYvector(self):
		"""
		Transform the y_train from type ndarray(N, dtype=object) to ndarray(T, 1).
		That's necessary for doing matrices operations
		
		Returns
		-------
		y_vec: ndarray(T,1)
			An array containing all loss measurements of all training curves. They need
			to be stacked in the same order as in the configurations array x_train
		"""
		y_vec = np.array([self.y_train[0]])
		for i in xrange(1, self.y_train.shape[0]):
			y_vec = np.append(y_vec, self.y_train[i])
		return y_vec.reshape(-1, 1)


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
		
		#number of length scales
		flex = x_train.shape[-1]

		if not self.samenoise:
			#theta0, noiseHyper, noiseCurve, alpha, beta, m_const
			fix = 5 
		else:
			#theta0, noise, alpha, beta, m_const
			fix = 4
		
		pdl = PredLik(x_train, y_train, invChol=self.invChol, horse=self.horse, samenoise=self.samenoise)
		
		samples = np.zeros((hyper_configs, fix+flex))
		
		sampler = emcee.EnsembleSampler(hyper_configs, fix+flex, pdl.marginal_likelihood)
		
		#sample length scales for GP over configs
		p0a = self.samples_uniform((hyper_configs, flex))
		
		#sample amplitude for GP over configs and alpha e beta for GP over curve
		p0b = self.samples_norm((hyper_configs, 3))
		
		p0 = np.append(p0a, p0b, axis=1)
		
		if not self.samenoise:
			if not self.horse:
				p0d = self.samples_noise((hyper_configs, 2))
			else:
				p0d = self.samples_horse((hyper_configs, 2))
		else:
			if not self.horse:
				p0d = self.samples_noise((hyper_configs, 1))
			else:
				p0d = self.samples_horse((hyper_configs, 1))
		
		
		p0 = np.append(p0, p0d, axis=1)
		
		p0, _, _ = sampler.run_mcmc(p0, burnin_steps)
		
		
		pos, prob, state = sampler.run_mcmc(p0, chain_length)
		
		p0 = pos
		
		samples = sampler.chain[:, -1]
		
		return np.exp(samples)

if __name__ == '__main__':
		
	y1 = np.random.rand(3,1)
	y2 = np.random.rand(2,1)
	y3 = np.random.rand(4,1)
	y = np.array([y1,y2,y3], dtype=object)
	
	x = np.array([[2, 3, 3.2, 1.5, 2.3, 2.7, 3.6], [4, 5, 5.1, 5.2, 5.7, 5.72, 5.8], [5, 8, 8.1, 8.32, 8.4, 8.46, 8.53]])/10.
	
	#likint = LikIntegrate()
	#samples = likint.create_configs(x,y)
	#print 'samples.shape: ', samples.shape
	#print
	#print 'samples: '
	#print samples
	
