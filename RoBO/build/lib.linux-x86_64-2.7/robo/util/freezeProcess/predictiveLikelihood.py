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
import os
from scipy.optimize import minimize
from numpy.linalg import solve
from math import exp
from scipy.linalg import block_diag



class PredLik(object):
	"""
	Class for the marginal likelihood of the GPs for the whole BO framework.
	
	Parameters
	----------
	x_train: ndarray(N,D)
		The input training data for all GPs
	y_train: ndarray(T,N)
		The target training data for all GPs
	x_test: ndarray(*,D)
		The current test data for the GPs
	theta_d = ndarray(D)
		Hyperparameters of the GP over hyperparameter configurations
	theta0: float
		Hyperparameter of the GP over hyperparameter configurations
	alpha: float
		Hyperparameter of the GP over training curves
	beta: float
		Hyperparameter of the GP over training curves

	"""
	def __init__(self,
	             x_train=None,
	             y_train=None,
	             x_test=None,
	             invChol=True,
	             horse=True,
	             samenoise=False):
		
		self.x_train = x_train
		self.y_train = y_train
		self.y = y_train
		self.x_test = x_test
		self.tetha_d = None
		self.theta0 = None
		self.alpha = None
		self.beta = None
		self.invChol = invChol
		self.horse = horse
		self.samenoise = samenoise
		self.m_const = None
		
	def inverse(self, chol):
		''' 
		Once one already has the cholesky of K, one can use this function for calculating the inverse of K
		
		:param chol: the cholesky decomposition of K
		:return: the inverse of K
		'''
		
		inve = 0
		error_k = 1e-25
		while(True):
			try:
				choly = chol + error_k*np.eye(chol.shape[0])
				inve = solve(choly.T, solve(choly, np.eye(choly.shape[0])))
				break
			except np.linalg.LinAlgError:
				error_k*=10
		return inve
	
	def invers(self, K):
		if self.invChol:
			invers = self.inverse_chol(K)
		else:
			try:
				invers = np.linalg.inv(K)
			except:
				invers=None
		
		return invers
	
	def inverse_chol(self, K):
		''' 
		Once one already has the cholesky of K, one can use this function for calculating the inverse of K
		
		:param chol: the cholesky decomposition of K
		:return: the inverse of K
		'''
		
		chol = self.calc_chol(K)
		if chol==None:
			return None
		
		inve = 0
		error_k = 1e-25
		while(True):
			try:
				choly = chol + error_k*np.eye(chol.shape[0])
				inve = solve(choly.T, solve(choly, np.eye(choly.shape[0])))
				break
			except np.linalg.LinAlgError:
				error_k*=10
		return inve

	def inversing(self, chol):
		inve = 0
		error_k = 1e-25
		once = False
		while(True):
			try:
				if once == True:
					choly = chol + error_k*np.eye(chol.shape[0])
				else:
					choly = chol
					once = True
				
				inve = solve(choly.T, solve(choly, np.eye(choly.shape[0])))
				break
			except np.linalg.LinAlgError:
				error_k*=10
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
		index = 0
		found = True
		while(index < 100):
			try:
				Ky = K + error_k*np.eye(K.shape[0])
				chol = np.linalg.cholesky(Ky)
				found = True
				break
			except np.linalg.LinAlgError:
				error_k*=10
				found = False
			index+=1
			
		if found:
			return chol
		else:
			return None

	def calc_chol3(self, K):
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
		#print 'K: ', K
		error_k = 1e-25
		chol = None
		once = False
		index = 0
		found = True
		while(index < 100):
			try:
				#print 'chol index: ', index
				if once == True:
					#print 'once == True'
					Ky = K + error_k*np.eye(K.shape[0])
				else:
					#print 'once == False'
					#Ky = K
					Ky = K + error_k*np.eye(K.shape[0])
					once = True
				chol = np.linalg.cholesky(Ky)
				#print 'chol: ', chol
				found = True
				break
			except np.linalg.LinAlgError:
				#print 'except'
				error_k*=10
				found = False
			#print 'index: ', index
			index+=1
		if found:
			#print 'it is found'
			return chol
		else:
			#print 'not found'
			return None
	

	
	def get_mconst(self):
		m_const = np.zeros((len(self.y_train), 1))
		for i in xrange(self.y_train.shape[0]):
			mean_i = np.mean(self.y_train[i], axis=0)
			m_const[i,:] = mean_i
		
		return m_const
	
	def kernel_curve(self, t, tprime, alpha, beta):
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

		try:
			result = np.power(self.beta, self.alpha)/np.power(((t[:,np.newaxis] + tprime) + self.beta), self.alpha)
			#print 'result1 in kernel_curve: ', result
			#result = result + np.eye(result.shape)*self.noiseCurve
			result = result + np.eye(M=result.shape[0], N=result.shape[1])*self.noiseCurve
			#print 'result2 in kernel_curve: ', result
			return result
		except:
			return None
		
	def kernel_hyper(self, x, xprime, theta_d, theta0):
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
		#print 'x.shape: ', x.shape
		#print 'xprime.shape: ', xprime.shape
		#print 'theta_d.shape: ', theta_d.shape
		#print 'theta0: ', theta0
		try:
			r2 = np.sum(((x[:, np.newaxis] - xprime)**2)/self.theta_d**2, axis=-1)
			#print 'r2: ', r2
			fiveR2 = 5*r2
			result = self.theta0*(1 + np.sqrt(fiveR2) + (5/3.)*fiveR2)*np.exp(-np.sqrt(fiveR2))
			#print 'result1: ', result
			result = result + np.eye(M=result.shape[0], N=result.shape[1])*self.noiseHyper
			#print 'result2: ', result
			return result
		except:
			return None

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
			#print 'in getKt() y[i]: ', y[i]
			ktn = self.getKtn(y[i])
			#print 'in getKt() ktn: ', ktn
			#print
			O = block_diag(O, ktn)
		return O
	
	def getKtn(self, yn):
		t = np.arange(1, yn.shape[0]+1)
		#not yet using the optimized parameters here
		#print 't range: ', t
		ktn = self.kernel_curve(t, t, 1., 1.)
		#It's already returning None when necessary
		return ktn

	def calc_Lambda(self, y):
		'''
		Calculates Lambda according to the following: Lamda = transpose(O)*inverse(Kt)*O
		= diag(l1, l2,..., ln)=, where ln = transpose(1n)*inverse(Ktn)*1n
		
		Parameters
		----------
		y: ndarray(T,1)
			Vector with all training curves stacked together, in the same order as in the configurations array x_train
		
		Returns
		-------
		Lambda: ndarray(N, N)
				Lamda is used in several calculations in the BO framework
		'''
		dim = y.shape[0]
		Lambda = np.zeros((dim, dim))
		index = 0
		for yn in y:
			t = np.arange(1, yn.shape[0]+1)
			
			ktn = self.kernel_curve(t, t, 1.0, 1.0)
			if ktn == None:
				return None
				
			ktn_inv = self.invers(ktn)
			if ktn_inv==None:
				return None
			one_n = np.ones((ktn.shape[0], 1))
			Lambda[index, index] = np.dot(one_n.T, np.dot(ktn_inv, one_n))
			
			index+=1
		
		return Lambda
		
 	
	
	def lambdaGamma(self, y, m_const):
		dim = y.shape[0]
		Lambda = np.zeros((dim, dim))
		gamma = np.zeros((dim, 1))
		for i, yn in enumerate(y):
			t = np.arange(1, yn.shape[0]+1)
			ktn = self.kernel_curve(t, t, 1., 1.)
			if ktn == None:
				return None, None
			ktn_inv = self.invers(ktn)
			if ktn_inv == None:
				return None, None
			one_n = np.ones((ktn.shape[0], 1))
			Lambda[i, i] = np.dot(one_n.T, np.dot(ktn_inv, one_n))
			gamma[i, 0] = np.dot(one_n.T, np.dot(ktn_inv, yn - m_const[i]))
		
		return Lambda, gamma

	def lambdaGamma2(self, m_const):
		"""
		Difference here is that the cholesky decomposition is calculated just once for the whole Kt and thereafter
		we solve the linear system for each Ktn.
		"""
		Kt = self.getKt(self.y)
		#print 'Kt.shape: ', Kt.shape
		self.Kt_chol = self.calc_chol3(Kt)
		if self.Kt_chol == None:
			return None, None
		dim = self.y.shape[0]
		Lambda = np.zeros((dim, dim))
		gamma = np.zeros((dim, 1))
		index = 0
		for i, yn in enumerate(self.y):
			lent = yn.shape[0]
			ktn_chol = self.Kt_chol[index:index+lent, index:index+lent]
			#print 'ktn_chol.shape: ', ktn_chol.shape
			index+=lent
			ktn_inv = self.inversing(ktn_chol)
			if ktn_inv == None:
				return None, None
			one_n = np.ones((ktn_inv.shape[0], 1))
			#print 'one_n.shape: ', one_n.shape
			Lambda[i, i] = np.dot(one_n.T, np.dot(ktn_inv, one_n))
			gamma[i, 0] = np.dot(one_n.T, np.dot(ktn_inv, yn - m_const[i]))
		
		return Lambda, gamma
		
	def calc_gamma(self, y, m_const):
		'''
        Calculates gamma according to the following: gamma = transpose(O)*inverse(Kt)*(y - Om),
		where each gamma element gamma_n = transpose(1n)*inverse(Ktn)*(y_n -m_n)
		
		Parameters
		----------
		y: ndarray(T,1)
			Vector with all training curves stacked together, in the same order as in the configurations array x_train
		m_const: float
			the infered mean of f, used in the joint distribution of f and y.
		
		Returns
		-------
		gamma: ndarray(N, 1)
			gamma is used in several calculations in the BO framework
		'''
		dim = y.shape[0]
		gamma = np.zeros((dim, 1))
		index = 0
		for i, yn in enumerate(y):
			t = np.arange(1, yn.shape[0]+1)
			
			ktn = self.kernel_curve(t, t, 1.0, 1.0)
			if ktn == None:
				return None
	
			ktn_inv = self.invers(ktn)
			if ktn_inv == None:
				return None
			one_n = np.ones((ktn.shape[0], 1))
			gamma[index, 0] = np.dot(one_n.T, np.dot(ktn_inv, yn - m_const[i]))

			index+=1
	
		
		return gamma 
	
	def predict_asy(self, x, xprime, y):
		'''
		Given new configuration xprime, it predicts the probability distribution of
		the new asymptotic mean, with mean and covariance of the distribution
		
		Parameters
		----------
		xprime: ndarray(number_configurations, D)
			The new configurations, of which the mean an the std2 are being predicted
		
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
		'''
		theta_d = np.ones(x.shape[-1])
		kx_star = self.kernel_hyper(x, xprime, theta_d, 1.0)
		kx = self.kernel_hyper(x, x, theta_d, 1.0)

		m_xstar = xprime.mean(axis=1).reshape(-1, 1)
		m_xstar = np.zeros(m_xstar.shape)
		m_const = self.get_mconst(y)
		m_const = np.zeros(m_const.shape)
		cholx = self.calc_chol(kx)
		kx_inv = self.inverse(cholx)
		Lambda = self.calc_Lambda(y)
		C_inv = kx_inv + Lambda
		C_inv_chol = self.calc_chol(C_inv)
		C = self.inverse(C_inv_chol)
		gamma = self.calc_gamma(y, m_const)

		mu = np.dot(C, gamma)
		
		mean = m_xstar + np.dot(kx_star.T, np.dot(kx_inv, mu))

		kstar_star = self.kernel_hyper(xprime, xprime, theta_d, 1.0)
		Lambda_chol = self.calc_chol(Lambda)
		Lambda_inv = self.inverse(Lambda_chol)
		kx_lamdainv = kx + Lambda_inv
		kx_lamdainv_chol = self.calc_chol(kx_lamdainv)
		kx_lamdainv_inv = self.inverse(kx_lamdainv_chol)
		cov= kstar_star - np.dot(kx_star.T, np.dot(kx_lamdainv_inv, kx_star))
	
	def predict_new_point1(self, t, tprime, yn, mu_n=None, Cnn=None):
		ktn = self.kernel_curve(t, t, 1.0, 1.0)
		ktn_chol = self.calc_chol(ktn)
		ktn_inv = self.inverse(ktn_chol)
		ktn_star = self.kernel_curve(t, tprime, 1.0, 1.0)
		Omega = np.ones((tprime.shape[0], 1)) - np.dot(ktn_star.T, np.dot(ktn_inv, np.ones((t.shape[0], 1))))
		mean = np.dot(ktn_star.T, np.dot(ktn_inv, yn)) + np.dot(Omega, mu_n)
		ktn_star_star = self.kernel_curve(tprime, tprime, 1.0, 1.0)
		cov = ktn_star_star - np.dot(ktn_star.T, np.dot(ktn_inv, ktn_star)) + np.dot(Omega, np.dot(Cnn, Omega.T))
	
		
	def predict_new_point2(self, one_step_pro_config, x, xprime, y):
		mean_star, Sigma_star_star = self.predict_asy(x, xprime, y)
	
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
	
	def nplog(self, val, minval=0.0000000001):
		return np.log(val.clip(min=minval)).reshape(-1, 1)
		
	
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
	
	def marginal_likelihood(self, theta):
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
		
		x=self.x_train
		y=self.y_train
		
		flex = self.x_train.shape[-1]

		theta_d = np.zeros(flex)
		theta_d = theta[:flex]
		if not self.samenoise:
			theta0, alpha, beta, noiseHyper, noiseCurve = theta[flex:]
		else:

			theta0, alpha, beta, noise = theta[flex:]
			noiseHyper = noiseCurve = noise

		theta_d = np.exp(theta_d)

		
		self.noiseHyper = exp(noiseHyper)
		
		self.noiseCurve = exp(noiseCurve)
		
		self.theta_d =theta_d

		self.theta0 = np.exp(theta0)
		self.alpha = np.exp(alpha)
		self.beta= np.exp(beta)

		self.m_const = self.get_mconst()
		
		y_vec = self.getYvector(y)
		self.y_vec = y_vec
		#print 'y_vec: ', y_vec.shape
		O = self.getOmicron(y)
		#print 'O: ', O.shape
		kx = self.kernel_hyper(x, x, theta_d, theta0)
		
		if kx == None:
			print 'failed: kx'
			return -np.inf
		#print 'kx: ', kx.shape
		
		#Lambda, gamma = self.lambdaGamma(y, self.m_const)
		Lambda, gamma = self.lambdaGamma2(self.m_const)
		if Lambda == None or gamma == None:
			#print 'failed: lambda or gamma'
			return -np.inf
			

		kx_inv = self.invers(kx)
		if kx_inv==None:
			print 'failed: kx_inv'
			return -np.inf
		#print 'kx_inv: ', kx_inv.shape
		
		kx_inv_plus_L = kx_inv + Lambda
		#print 'kx_inv_plus_L: ', kx_inv_plus_L.shape
		
		kx_inv_plus_L_inv = self.invers(kx_inv_plus_L)
		if kx_inv_plus_L_inv == None:
			print 'failed: kx_inv_plus_L_inv'
			return -np.inf
			
		kt = self.getKt(y)
		
		if kt == None:
			print 'failed: kt'
			return -np.inf
			

		kt_inv = self.invers(kt)
		if kt_inv == None:
			#print 'failed: kt_inv'
			return -np.inf
		
		#print 'y_vec: ', y_vec.shape
		#print 'O: ', O.shape
		#print 'm_const: ', self.m_const.shape
		#print 'kt_inv: ', kt_inv.shape
		#y_minus_Om = y_vec - O*self.m_const
		y_minus_Om = y_vec - np.dot(O, self.m_const)
		
		#print 'np.dot(y_minus_Om.T, np.dot(kt_inv, y_minus_Om)): ', np.dot(y_minus_Om.T, np.dot(kt_inv, y_minus_Om))
		#print 'np.dot(gamma.T, np.dot(kx_inv_plus_L_inv, gamma)): ', np.dot(gamma.T, np.dot(kx_inv_plus_L_inv, gamma))
		#print 'self.nplog(np.linalg.det(kx)): ', self.nplog(np.linalg.det(kx))
		kt = kt/1000.
		#print 'self.nplog(np.linalg.det(kt)): ', self.nplog(np.linalg.det(kt))
		#print 'self.nplog(np.linalg.det(kx_inv_plus_L)): ', self.nplog(np.linalg.det(kx_inv_plus_L))
		logP = -(1/2.)*np.dot(y_minus_Om.T, np.dot(kt_inv, y_minus_Om)) + (1/2.)*np.dot(gamma.T, np.dot(kx_inv_plus_L_inv, gamma))\
		       - (1/2.)*(self.nplog(np.linalg.det(kx_inv_plus_L)) + self.nplog(np.linalg.det(kx)) + self.nplog(np.linalg.det(kt))) # + const #* Where does const come from?
		
		if logP == None or str(logP) == str(np.nan):
			print 'failed: logP' 
			return -np.inf
		#print 'logP: ', logP
		#print 'self.prob_uniform(theta_d): ', self.prob_uniform(theta_d)
		#print 'self.prob_norm(np.array([theta0, alpha, beta])): ', self.prob_norm(np.array([theta0, alpha, beta]))
		#print 'self.prob_horse(np.array([self.noiseHyper, self.noiseCurve])): ', self.prob_horse(np.array([self.noiseHyper, self.noiseCurve]))

		if not self.horse:
			lp = logP + self.prob_uniform(theta_d) + self.prob_norm(np.array([theta0, alpha, beta])) + self.prob_noise(np.array([self.noiseHyper, self.noiseCurve]))# + self.prob_uniform_mconst(m_const)
		else:
			lp = logP + self.prob_uniform(theta_d) + self.prob_norm(np.array([theta0, alpha, beta])) + self.prob_horse(np.array([self.noiseHyper, self.noiseCurve]))

		if lp == None or str(lp) == str(np.nan):
			print 'failed: lp'
			return -np.inf
		
		#print 'lp: ', lp
		return lp
	
	
	def prob_norm(self, theta):
		"""
		Calculates the log probability of samples extracted from the lognormal distribution
		
		Parameters
		----------
		theta: the GP hyperparameters which were drawn from the lognormal distribution
		
		Returns
		-------
		log probability: float
			The sum of the log probabilities of all different samples extracted from the lognorm
		"""
		std = np.zeros_like(theta)
		std[:] = 1.
		probs = sps.lognorm.logpdf(theta, std, loc=np.zeros_like(theta))
		#probs = np.log(sps.lognorm.logpdf(theta, std, loc=np.zeros_like(theta)))
		return np.sum(probs)
	
	def prob_horse(self, theta, scale=0.1):
		if np.any(theta == 0.0):
			#return np.inf
			return -np.inf
		
		#return np.log(np.log(1 + 3.0 * (scale / np.exp(theta)) ** 2))
		return np.sum(np.log(np.log(1 + 3.0 * (scale / np.exp(theta)) ** 2)))

	def prob_noise(self, theta):
		"""
		Calculates the log probability of noise samples extracted from the lognormal distribution
		
		Parameters
		----------
		theta: the GP noise hyperparameters which were drawn from the lognormal distribution
		
		Returns
		-------
		log probability: float
			The sum of the log probabilities of all different noise samples
		"""
		std = np.zeros_like(theta)
		std[:] = 1.
		probs = sps.lognorm.logpdf(theta, std, loc=np.zeros_like(theta))
		return np.sum(probs)

#I'm not sure about this probs[:] = 0.1. I concluded that from some theory, but I should verify it	
	def prob_uniform(self, theta):
		"""
		Calculates the uniform probability of samples extracted from the uniform distribution between 0 and 10
		
		Parameters
		----------
		theta: the GP hyperparameters which were drawn from the uniform distribution between 0 and 10
		
		Returns
		-------
		uniform probability: float
			The sum of the log probabilities of all different samples extracted from the uniform distribution
		"""
		if np.any(theta < 0) or np.any(theta>10):
			return -np.inf
		else:
			probs = np.zeros_like(theta)
			probs[:] = 0.1
			return np.sum(np.log(probs))
	
	def prob_uniform_mconst(self, theta):
		"""
		Calculates the uniform probability of samples extracted from the uniform distribution between y_min and y_max
		
		Parameters
		----------
		theta: the GP hyperparameters which were drawn from the uniform distribution between y_min and y_max
		
		Returns
		-------
		uniform probability: float
			The sum of the log probabilities of all different samples extracted from the uniform distribution
		"""
		mini = np.min(self.y_vec)
		maxi = np.max(self.y_vec)
		if np.any(theta < mini) or np.any(theta>maxi):
			return -np.inf
		else:
			probs = np.zeros_like(theta)
			probs[:] = 1./(maxi-mini)
			return np.sum(np.log(probs))
	
if __name__ == '__main__':	

	###Test calc_lambda. ok
	y1 = np.random.rand(3,1)
	y2 = np.random.rand(2,1)
	y3 = np.random.rand(4,1)
	y = np.array([y1,y2,y3], dtype=object)
	#Lambda = calc_Lambda(y)
	#print Lambda
	
	###Test get_mconst. ok
	#m = get_mconst(y)
	#print m
	
	###Test calc_gamma. ok
	#gamma = calc_gamma(y, m)
	#print gamma
	
	###Test predict_newpoint1. It works. ok
	config_n = 3
	t=np.arange(10)/10.
	tprime = np.arange(10,13)/10.
	yn = np.arange(10).reshape(-1, 1)/10.
	C = np.random.randn(4,4)
	mu = np.arange(4).reshape(-1, 1)
	#predict_new_point1(t, tprime, yn, mu[config_n-1, 0], C[config_n-1, config_n-1])
	
	###Test predict_newpoint2
	x = np.array([[2,3], [4,5], [5,8]])/10.
	xprime = np.array([[3,4]])/10. # problem
	xprime = np.array([3,4])/10. # problem
	#one_step_pro_config = np.array([[4]])
	#predict_new_point2(one_step_pro_config, x, xprime, y)
	
	###Test getYVector(y). Ok
	
	#y_vec = getYvector(y)
	#print 'y_vec.shape: ', y_vec.shape
	#print 
	#print 'y_vec: ', y_vec
	#print
	
	###Test getOmicron(y). Ok
	#O = getOmicron(y)
	#print 'O.shape: ', O.shape
	#print
	#print 'O: ', O
	
	###Test marginal_likelihood.
	x = np.array([[2,3, 3.2, 1.5, 2.3, 2.7, 3.6], [4,5, 5.1, 5.2, 5.7, 5.72, 5.8], [5,8, 8.1, 8.32, 8.4, 8.46, 8.53]])/10.
	theta_d = np.ones(x.shape[-1])[np.newaxis, :]
	print theta_d
	theta0 = 1.
	alpha = 1.
	beta = 1.
	theta_d = tuple(map(tuple, theta_d))[0]
	print theta_d
	pl = PredLik(x,y)
	theta = theta_d + (theta0, alpha, beta)
	ml = pl.marginal_likelihood(theta)
	#print ml
	
