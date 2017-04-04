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
from scipy.linalg import block_diag

"""
Based on equation 19 from the freeze-thawn paper
"""

class PredictiveHyper(object):
	"""
	Class for the GP over hyperparameters. The Posterior Predictive Distribution.
	
	Parameters
	----------
	x_train: ndarray(N,D)
		The input training data for all GPs
	y_train: ndarray(T,N)
		The target training data for all GPs
	x_test: ndarray(*,D)
		The current test data for the GPs
	alpha: float
		Hyperparameter of the GP over training curves
	beta: float
		Hyperparameter of the GP over training curves
	theta0: float
		Hyperparameter of the GP over hyperparameter configurations
	thetad = ndarray(D)
		Hyperparameters of the GP over hyperparameter configurations
	"""
	def __init__(self,
				 x_train = None,
				 y_train = None,
				 x_test = None,
				 alpha = 1.0,
				 beta = 1.0,
				 theta0 = 1.0,
				 thetad = None,
				 invChol=True,
				 samenoise = False,
				 kx=None,
				 kx_inv=None):
	 
		 self.x = x_train
		 print 'y_train.shape: ', y_train.shape
		 self.y = y_train
		 self.xprime = x_test
		 self.alpha = alpha
		 self.beta = beta
		 self.theta0 = theta0
		 if thetad == None or self.x.shape[-1] != thetad.shape[0]:
			 self.thetad = np.ones(self.x.shape[-1])
		 else:
			self.thetad = thetad
		 self.C = None
		 self.mu = None
		 self.m_const = None
		 self.invChol = invChol
		 self.samenoise = samenoise
		 self.kx = None
		 self.kx_inv = None
		 self.Kt_chol = None
		 
	def setGpHypers(self, sample):
		"""
		Sets the gp hyperparameters
		
		Parameters
		----------
		sample: ndarray(Number_GP_hyperparameters, 1)
				One sample from the collection of all samples of GP hyperparameters
		"""
		self.m_const = self.get_mconst()
		flex = self.x.shape[-1]
		self.thetad = np.zeros(flex)
		self.thetad = sample[:flex]

		if not self.samenoise:
			self.theta0, self.alpha, self.beta, self.noiseHyper, self.noiseCurve = sample[flex:]
		else:
			self.theta0, self.alpha, self.beta, noise = sample[flex:]
			self.noiseHyper = self.noiseCurve = noise

	def inverse(self, chol):
		''' 
		Once one already has the cholesky of K, one can use this function for calculating the inverse of K
		
		:param chol: the cholesky decomposition of K
		:return: the inverse of K
		'''
		return solve(chol.T, solve(chol, np.eye(chol.shape[0])))
	
	def calc_chol(self, K):
		error_k = 1e-25
		chol = None
		while(True):
			try:
				Ky = K + error_k*np.eye(K.shape[0]) 
				chol = np.linalg.cholesky(Ky)
				break
			except np.linalg.LinAlgError:
				error_k*=10
		return chol

	def invers(self, K):
		if self.invChol:
			invers = self.inverse_chol(K)
		else:
			try:
				invers = np.linalg.inv(K)
			except:
				invers=None
		
		return invers

#change also here with the once and in all 	
	def inverse_chol(self, K):
		''' 
		Once one already has the cholesky of K, one can use this function for calculating the inverse of K
		
		:param chol: the cholesky decomposition of K
		:return: the inverse of K
		'''
		
		chol = self.calc_chol2(K)
		if chol==None:
			return None
		
		#print 'chol: ', chol
		inve = 0
		error_k = 1e-25
		while(True):
			try:
				choly = chol + error_k*np.eye(chol.shape[0])
				#print 'choly.shape: ', choly.shape
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
		

	def calc_chol2(self, K):
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
				#print 'chol index: ', index
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
		error_k = 1e-25
		chol = None
		once = False
		index = 0
		found = True
		while(index < 100):
			try:
				#print 'chol index: ', index
				if once == True:
					Ky = K + error_k*np.eye(K.shape[0])
				else:
					Ky = K
					once = True
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
	
	def add_xtrain(self, xtrain):
		if self.x == None:
			self.x = xtrain
		else:
			self.x = np.append(self.x, xtrain)
			
	def add_ytrain(self, ytrain):
		if self.y == None:
			self.y = ytrain
		else:
			self.y = np.append(self.y, ytrain)
			
	def add_xtest(self, xtest):
		if self.xprime == None:
			self.xprime = xtest
		else:
			self.xprime = np.append(self.xprime, xtest)
	
	
	def get_mconst(self):
		m_const = np.zeros((len(self.y), 1))
		for i in xrange(self.y.shape[0]):
			mean_i = np.mean(self.y[i], axis=0)
			m_const[i,:] = mean_i
		
		return m_const
	
	def kernel_curve(self, t, tprime):
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

		cov = np.power(self.beta, self.alpha)/np.power(((t[:,np.newaxis] + tprime) + self.beta), self.alpha)
		cov = cov + np.eye(cov.shape[0])*self.noiseCurve
		return cov
		
	def kernel_curve2(self, t, tprime):
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
			result = result + np.eye(result.shape[0])*self.noiseCurve
			return result
		except:
			return None
		
	def kernel_hyper(self, x, xprime):
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

		r2 = np.sum(((x[:, np.newaxis] - xprime)**2)/self.thetad, axis=-1)
		fiveR2 = 5*r2
		cov = self.theta0*(1 + np.sqrt(fiveR2) + (5/3.)*fiveR2)*np.exp(-np.sqrt(fiveR2))
		cov = cov + np.eye(cov.shape[0])*self.noiseHyper
		return cov
		
	def kernel_hyper2(self, x, xprime):
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
		try:	
			r2 = np.sum(((x[:, np.newaxis] - xprime)**2)/self.thetad**2, axis=-1)
			fiveR2 = 5*r2
			result = self.theta0*(1 + np.sqrt(fiveR2) + (5/3.)*fiveR2)*np.exp(-np.sqrt(fiveR2))
			result = result + np.eye(N=result.shape[0], M=result.shape[1])*self.noiseHyper
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
		
		for i in xrange(1, len(y)):
			ktn = self.getKtn(y[i])
			O = block_diag(O, ktn)
		return O
	
	def getKtn(self, yn):
		t = np.arange(1, yn.shape[0]+1)
		#not yet using the optimized parameters here
		ktn = self.kernel_curve2(t, t)
		#It's already returning None when necessary
		return ktn
		

	def calc_Lambda(self):
		'''
		Calculates Lambda according to the following: Lamda = transpose(O)*inverse(Kt)*O
		= diag(l1, l2,..., ln)=, where ln = transpose(1n)*inverse(Ktn)*1n
		
		Returns
		-------
		Lambda: ndarray(N, N)
				Lamda is used in several calculations in the BO framework
		'''
		dim = self.y.shape[0]
		Lambda = np.zeros((dim, dim))
		index = 0
		for yn in self.y:
			t = np.arange(1, yn.shape[0]+1)
			#not yet using the optimized parameters here
			ktn = self.kernel_curve2(t, t)
			if ktn == None:
				return None
			ktn_inv = self.invers(ktn)
			if ktn_inv == None:
				return None
			one_n = np.ones((ktn.shape[0], 1))
			Lambda[index, index] = np.dot(one_n.T, np.dot(ktn_inv, one_n))
			index+=1
		
		return Lambda
	
	def lambdaGamma(self, m_const):
		dim = self.y.shape[0]
		Lambda = np.zeros((dim, dim))
		gamma = np.zeros((dim, 1))
		for i, yn in enumerate(self.y):
			t = np.arange(1, yn.shape[0]+1)
			ktn = self.kernel_curve2(t, t)
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
		
		
	def calc_gamma(self, m_const):
		'''
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
		'''
		dim = self.y.shape[0]
		gamma = np.zeros((dim, 1))
		index = 0
		for i, yn in enumerate(self.y):
			t = np.arange(1, yn.shape[0]+1)
			#not yet using the optimized parameters here
			ktn = self.kernel_curve2(t, t)
			if ktn == None:
				return None
			ktn_inv = self.invers(ktn)
			if ktn_inv == None:
				return None
			one_n = np.ones((ktn.shape[0], 1))
			gamma[index, 0] = np.dot(one_n.T, np.dot(ktn_inv, yn - m_const[i]))
			index+=1
	
		return gamma    	
	

	def predict_asy(self, xprime=None):
		'''
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
		'''
		
		if xprime != None:
			self.xprime = xprime

		theta_d = np.ones(self.x.shape[-1])

		kx_star = self.kernel_hyper2(self.x, self.xprime)
		
		if kx_star == None:
			return None
		
		kx = self.kernel_hyper2(self.x, self.x)
		if kx == None:
			return None
			
		if len(xprime.shape) > 1:
			m_xstar = self.xprime.mean(axis=1).reshape(-1, 1)
		else:
			m_xstar = self.xprime

		m_xstar = np.zeros(m_xstar.shape)

		m_const = self.m_const
		
		
		kx_inv = self.invers(kx)
		if kx_inv == None:
			return None
		m_const = self.m_const
		#print 'm_const.shape: ', m_const.shape
		Lambda, gamma = self.lambdaGamma2(m_const)
		#print 'Lambda.shape: ', Lambda
		if Lambda == None or gamma == None:
			return None
		
		

		C_inv = kx_inv + Lambda
		
		C = self.invers(C_inv)
		if C == None:
			return None
		
		self.C = C
		
		
		mu = self.m_const + np.dot(C, gamma)
		self.mu = mu
		
		mean = m_xstar + np.dot(kx_star.T, np.dot(kx_inv, mu - self.m_const))
		
		#Now calculate the covariance
		kstar_star = self.kernel_hyper2(self.xprime, self.xprime)
		if kstar_star == None:
			return None
		
		Lambda_inv = self.invers(Lambda)
		if Lambda_inv == None:
			return None
		
		kx_lamdainv = kx + Lambda_inv
		

		kx_lamdainv_inv = self.invers(kx_lamdainv)
		if kx_lamdainv_inv == None:
			return None

		cov= kstar_star - np.dot(kx_star.T, np.dot(kx_lamdainv_inv, kx_star))
		std2 = np.diagonal(cov).reshape(-1, 1)

		return mean, std2, C, mu
	
if __name__ == '__main__':	
	
	y1 = np.ones((3,1))
	y2 = np.ones((2,1))
	y3 = np.ones((4,1))
	y = np.array([y1,y2,y3], dtype=object)
	x = np.array([[2,3], [4,5], [5,8]])
	xprime = np.array([[3,4], [7,8], [5,6], [9,9]])
	
	ph = PredictiveHyper(x, y, xprime)
	###Test calc_lambda. ok
	#y1 = np.ones((3,1))
	#y2 = np.ones((2,1))
	#y3 = np.ones((4,1))
	#y = np.array([y1,y2,y3], dtype=object)
	Lambda = ph.calc_Lambda()
	print Lambda
	
	###Test get_mconst. ok
	m = ph.get_mconst()
	print m
	
	###Test calc_gamma. ok
	gamma = ph.calc_gamma(m)
	#print gamma
	
	###Test predict_asy(x, xprime, y)
	mean, cov, C, mu = ph.predict_asy(xprime = xprime)
	std2 = np.diagonal(cov)
	print 'mean.shape: ', mean.shape
	print 'cov.shape: ', cov.shape
	print 'C.shape: ', C.shape
	print 'mu.shape: ', mu.shape
	print 'std2: ' , std2
	
	###Test kernel_hyper. It's working. ok.
	#kern = ph.kernel_hyper(x, xprime, 1.0, 1.0)
	#print kern
	#print
	#x = np.array([[2,3], [4,5], [5,8]])
	#xprime = np.array([[3,4]])
	#kern = ph.kernel_hyper(x, xprime, 1.0, 1.0)
	#print kern
	#print
	#x = np.array([[2,3], [4,5], [5,8]])
	#xprime = np.array([[3,4], [2,3]])
	#kern = ph.kernel_hyper(x, xprime, 1.0, 1.0)
	#print kern
	
