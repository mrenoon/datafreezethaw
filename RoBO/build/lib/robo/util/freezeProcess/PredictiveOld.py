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

'''
Based on the equation 20 from freeze-thawn paper
'''

class PredictiveOld(object):
	
	def __init__(self,
				 x_train = None,
				 y_train = None,
				 x_test = None,
				 alpha = None,
				 beta = None,
				 theta0 = None,
				 theta_d = None,
				 invChol = True,
				 samenoise = False):
	 
		 self.x = x_train
		 self.y = y_train
		 self.xprime = x_test
		 self.alpha = alpha
		 self.beta = beta
		 self.theta0 = theta0
		 #if theta_d == None or self.x.shape[-1] != len(theta_d):
			 #self.theta_d = np.ones(self.x.shape[-1])
		 #else:
			#self.theta_d = theta_d
		 self.theta_d = theta_d
		 self.invChol = invChol
		 self.samenoise = samenoise
	
	def setGpHypers(self, sample):
		self.m_const = 0.
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
			#print index
		if found:
			return chol
		else:
			return None
				
	def calc_chol2(self, K):
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
	
	
	def get_mconst(self):
		m_const = np.zeros((len(self.y), 1))
		for i in xrange(self.y.shape[0]):
			mean_i = np.mean(self.y[i], axis=0)
			m_const[i,:] = mean_i
		
		return m_const
	
	def kernel_curve(self, t, tprime):
		try:
			result = np.power(self.beta, self.alpha)/np.power(((t[:,np.newaxis] + tprime) + self.beta), self.alpha)
			result = result + np.eye(N=result.shape[0], M=result.shape[1])*self.noiseCurve
			return result
		except:
			return None
		
	def kernel_hyper(self, x, xprime):
		try:
			r2 = np.sum(((x[:, np.newaxis] - xprime)**2)/self.thetad**2, axis=-1)
			fiveR2 = 5*r2
			result = self.theta0*(1 + np.sqrt(fiveR2) + (5/3.)*fiveR2)*np.exp(-np.sqrt(fiveR2))
			result = result + np.eye(result.shape[0])*self.noiseHyper
			return result
		except:
			return None
	
	def calc_Lambda(self):
		'''
		y is an ndarray object with dtype=object, with all learning curves already running
		'''
		dim = self.y.shape[0]
		Lambda = np.zeros((dim, dim))
		index = 0
		for yn in self.y:
			t = np.arange(1, yn.shape[0]+1)
			#not yet using the optimized parameters here
			ktn = self.kernel_curve(t, t)
			chol_ktn = self.calc_chol(ktn)
			ktn_inv = self.inverse(chol_ktn)
			one_n = np.ones((ktn.shape[0], 1))
			Lambda[index, index] = np.dot(one_n.T, np.dot(ktn_inv, one_n))
			index+=1
		
		return Lambda
		
	def calc_gamma(self, m_const):
		'''
		y is an ndarray object with dtype=object, with all learning curves already running
		'''
		dim = self.y.shape[0]
		gamma = np.zeros((dim, 1))
		index = 0
		for yn in self.y:
			t = np.arange(1, yn.shape[0]+1)
			#not yet using the optimized parameters here
			ktn = self.kernel_curve(t, t)
			chol_ktn = self.calc_chol(ktn)
			ktn_inv = self.inverse(chol_ktn)
			one_n = np.ones((ktn.shape[0], 1))
			gamma[index, 0] = np.dot(one_n.T, np.dot(ktn_inv, yn - m_const[index]))
			index+=1
	
		return gamma    	
	
	def predict_asy(self, xprime=None):
		'''
		Given new configuration xprime, it predicts the probability distribution of
		the new asymptotic mean, with mean and covariance of the distribution
		
		:param x: all configurations without the new ones
		:param xprime: new configurations
		:param y: all training curves
		:type y: ndarray(dtype=object)
		:return: mean of the new configurations
		:return: covariance of the new configurations
		'''
		if xprime!=None:
			self.xprime = xprime
		theta_d = np.ones(self.x.shape[-1])
		kx_star = self.kernel_hyper(self.x, self.xprime)
		kx = self.kernel_hyper(self.x, self.x)
		
		m_xstar = self.xprime.mean(axis=1).reshape(-1, 1)
		m_const = self.get_mconst(self.y)
		cholx = self.calc_chol(kx)
		kx_inv = self.inverse(cholx)
		Lambda = self.calc_Lambda(self.y)
		C_inv = kx_inv + Lambda
		C_inv_chol = self.calc_chol(C_inv)
		C = self.inverse(C_inv_chol)
		gamma = self.calc_gamma(m_const)

		mu = np.dot(C, gamma)
		
		mean = m_xstar + np.dot(kx_star.T, np.dot(kx_inv, mu))
		
		#Now calculate the covariance
		kstar_star = self.kernel_hyper(self.xprime, self.xprime)
		Lambda_chol = self.calc_chol(Lambda)
		Lambda_inv = self.inverse(Lambda_chol)
		kx_lamdainv = kx + Lambda_inv
		kx_lamdainv_chol = self.calc_chol(kx_lamdainv)
		kx_lamdainv_inv = self.inverse(kx_lamdainv_chol)
		cov= kstar_star - np.dot(kx_star.T, np.dot(kx_lamdainv_inv, kx_star))

	def predict_new_point1(self, t, tprime, yn, mu_n=None, Cnn=None):
		yn = yn.reshape(-1,1)

		ktn = self.kernel_curve(t, t)
		if ktn == None:
			return None
		#print 'ktn: ', ktn.shape
		
		ktn_inv = self.invers(ktn)
		if ktn_inv == None:
			return None
		
		ktn_star = self.kernel_curve(t, tprime)
		if ktn_star == None:
			return None
		
		Omega = np.ones((tprime.shape[0], 1)) - np.dot(ktn_star.T, np.dot(ktn_inv, np.ones((t.shape[0], 1))))
		
		#Exactly why:
		if yn.shape[0] > ktn_inv.shape[0]:
			yn = yn[:ktn_inv.shape[0]]
		
		mean = np.dot(ktn_star.T, np.dot(ktn_inv, yn)) + np.dot(Omega, mu_n)

		
		#covariance
		ktn_star_star = self.kernel_curve(tprime, tprime)
		#print 'ktn.shape: ', ktn.shape
		#print 'ktn_star.shape: ', ktn_star.shape
		#print 'ktn_star_star.shape: ', ktn_star_star.shape
		if ktn_star_star == None:
			return None
		
		cov = ktn_star_star - np.dot(ktn_star.T, np.dot(ktn_inv, ktn_star)) + np.dot(Omega, np.dot(Cnn, Omega.T))
		std2 = np.diagonal(cov).reshape(-1,1)
		#print cov.shape
		return mean, std2
	
	
if __name__ == '__main__':
	###Test calc_lambda. ok
	
	y1 = np.ones((3,1))
	y2 = np.ones((2,1))
	y3 = np.ones((4,1))
	y = np.array([y1,y2,y3], dtype=object)
	po = PredictiveOld(y_train = y)
	
	#Lambda = po.calc_Lambda()
	#print Lambda
	
	###Test get_mconst. ok
	#m = po.get_mconst()
	#print m
	
	###Test calc_gamma. ok
	#gamma = po.calc_gamma(m)
	#print gamma
	
	###Test predict_newpoint1. It works. ok
	
	config_n = 3
	t=np.arange(10)
	tprime = np.arange(10,13)
	yn = np.ones((10,1))
	C = np.random.randn(4,4)
	mu = np.arange(4).reshape(-1, 1)
	print 'mu.shape :' , mu.shape
	print 'C.shape: ', C.shape
	print 'yn.shape: ', yn.shape
	print 't.shape: ', t.shape
	print 'tprime.shape: ', tprime.shape
	mean, cov = po.predict_new_point1(t, tprime, yn, mu[config_n-1, 0], C[config_n-1, config_n-1])
	#print 'mean: ', mean.shape
	#print 'cov.shape: ', cov.shape

	
