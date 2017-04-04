# encoding=utf8
__author__ = "Tulio Paiva"
__email__ = "paivat@cs.uni-freiburg.de"

import numpy as np
from copy import deepcopy
import logging
import time
import os
import cPickle as pickle
from robo.initial_design.init_random_uniform import init_random_uniform
from robo.task.base_task import BaseTask
from robo.solver.base_solver import BaseSolver
from robo.incumbent.best_observation import BestObservation
# from robo.models.freeze_thaw_model import FreezeThawGP
from robo.models.freeze_thaw_model_fixed import FreezeThawGP

from robo.maximizers.direct import Direct
from robo.acquisition.ei import EI
# from robo.acquisition.information_gain_mc_freeze import InformationGainMC
from robo.acquisition.information_gain_mc_freeze import InformationGainMC

from scipy.stats import norm

logger = logging.getLogger(__name__)
logging.basicConfig(filename='Vu.log', level=logging.INFO)

VU_PRINT = 1
SAVE_FILE = "/data/ml/Vu_Notes/curveDatas/freezeData.pkl"
#	0 = minimal
# 	1 = only details
# 	2 = even more details


class FreezeThawBO(BaseSolver):

	def __init__(self,
		         acquisition_func,
		         freeze_thaw_model,
		         maximize_func,
		         task,
		         initial_design=None,
		         init_points=5,
		         incumbent_estimation=None,
		         basketOld_X=None,
		         basketOld_Y=None,
		         first_steps=3,
		         save_dir=None,
		         num_save=1,
		         train_intervall=1,
		         n_restarts=1,
		         pkl=False,
		         max_epochs=500, 
		         stop_epochs=False,
		         nr_epochs_inits=5,
		         nr_epochs_further=10):
		"""
		Class for the Freeze Thaw Bayesian Optimization by Swersky et al.

		Parameters
		----------
		basketOld_X: ndarray(N,D)
		    Basket with the old configurations, with N as the number of configurations
		    and D as the number of dimensions
		basketOld_Y: ndarray(N,S)
		    Basket with the learning curves of the old configurations, with N as the number
		    of configurations and S as the number of steps
		first_steps: integer
		    Initial number of steps of the learning curves
		acquisition_func: BaseAcquisitionFunction Object
		    The acquisition function which will be maximized.
		freeze_thaw_model: ModelObject
		    Freeze thaw model that models our current
		    belief of the objective function.
		task: TaskObject
		    Task object that contains the objective function and additional
		    meta information such as the lower and upper bound of the search
		    space.
		maximize_func: MaximizerObject
		    Optimization method that is used to maximize the acquisition
		    function
		nr_epochs_init: integer 
		    Number of epochs for executing the task at the initial configurations
		nr_epochs_further: integer 
		    Number of epochs for executing the task at further configurations		
		"""
		
		super(FreezeThawBO, self).__init__(acquisition_func, freeze_thaw_model, maximize_func, task)

		self.start_time = time.time()

		if initial_design == None:
			self.initial_design = init_random_uniform
		else:
			self.initial_design = initial_design

		self.X = None
		self.Y = None
		self.ys = None

		self.freezeModel = self.model

		self.task = task

		self.model_untrained = True

		if incumbent_estimation is None:
			self.estimator = BestObservation(self.model, self.task.X_lower, self.task.X_upper)
		else:
			self.estimator = incumbent_estimation

		self.incumbent = None
		self.incumbents = []
		self.incumbent_value = None
		self.incumbent_values = []
		self.init_points = init_points

		self.basketOld_X = basketOld_X
		self.basketOld_Y = basketOld_Y
		self.basket_files = list()
		self.basket_indices = list()
		self.directory = "temp_configs_" + self.task.base_name

		self.first_steps = first_steps
		self.nr_epochs_inits = nr_epochs_inits
		self.nr_epochs_further = nr_epochs_further

		self.time_func_eval = None
		self.time_overhead = None
		self.train_intervall = train_intervall

		self.num_save = num_save
		self.time_start = None
		self.pkl = pkl
		self.stop_epochs = stop_epochs
		self.max_epochs = max_epochs
		self.all_configs = dict()
		self.total_nr_confs = 0

		self.n_restarts = n_restarts
		self.runtime = []

	def printBasket(self):
		print "::::																	::::"
		print "::::::::					The Current Basket of the BO: 					::::"
		for cId, cand in enumerate(self.basketOld_X):
			print "::::::::		Candidate #", cId, ": ", cand,"\n:::::::: 					   	value = ", self.basketOld_Y[cId]
		print "::::::::																	::::"

	def run(self, num_iterations=10, X=None, Y=None):
		"""
		Bayesian Optimization iterations

		Parameters
		----------
		num_iterations: int
		    How many times the BO loop has to be executed
		X: np.ndarray(N,D)
		    Initial points that are already evaluated
		Y: np.ndarray(N,1)
		    Function values of the already evaluated points

		Returns
		-------
		np.ndarray(1,D)
		    Incumbent
		np.ndarray(1,1)
		    (Estimated) function value of the incumbent
		"""

		init = init_random_uniform(self.task.X_lower, self.task.X_upper, self.init_points)
		if VU_PRINT >= 0 :
			print "Initial init = ", init

		self.basketOld_X = deepcopy(init)
		val_losses_all = []
		nr_epochs = 0

		self.create_file_paths()

		self.time_start = time.time()

		ys = np.zeros(self.init_points, dtype=object)
		self.time_func_eval = np.zeros([self.init_points])
		self.time_overhead = np.zeros([self.init_points])
		##change: the task function should send the model file_path back and it should be stored here
		for i in xrange(self.init_points):
			#ys[i] = self.task.f(np.arange(1, 1 + self.first_steps), x=init[i, :]) ##change: that's not general
			#ys[i] = self.task.objective(nr_epochs=None, save_file_new=self.basket_files, save_file_old=None)
			self.task.set_save_modus(is_old=False, file_old=None, file_new=self.basket_files[i])
			self.task.set_epochs(self.nr_epochs_inits)
			#print 'init[i,:]: ', init[i,:]
			#_, ys[i] = self.task.objective_function(x=init[i,:][np.newaxis,:])
			conf_now = init[i,:]
			logger.info("Evaluate: %s" % conf_now[np.newaxis,:])
			start_time = time.time()
			_, val_losses = self.task.evaluate(x=conf_now[np.newaxis,:])

			if VU_PRINT >= 0:
				print "In initing points, val_losses = ", val_losses

			self.time_func_eval[i] = time.time() - start_time
			self.time_overhead[i] = 0.0

			logger.info("Configuration achieved a performance "
				"of %f in %f seconds" %
				(val_losses[-1], self.time_func_eval[i]))
			#storing configuration, learning curve, activity, index in the basketOld
			ys[i] = np.asarray(val_losses)
			self.all_configs[self.total_nr_confs] = [conf_now, ys[i], True, self.total_nr_confs]
			self.total_nr_confs+=1
			val_losses_all = val_losses_all + val_losses
			nr_epochs+=len(val_losses)


			#+++++++++++++++++++++++++


			if self.save_dir is not None and (i) % self.num_save == 0:
				self.save_json(i, learning_curve=str(val_losses), information_gain=0.,
					entropy=0, phantasized_entropy=0.)
				self.save_iteration(i)
			

		self.basketOld_Y = deepcopy(ys)

		Y = np.zeros((len(ys), 1))
		for i in xrange(Y.shape[0]):
			Y[i, :] = ys[i][-1]
		
		#WTF IS THIS Y ??? OHHHH I see its the ultimate loss

		if VU_PRINT >= 1:
			print "ys after getting init points: ", ys
			print "Y after gettting init points: ", Y

		self.X = deepcopy(self.basketOld_X)
		self.ys =  deepcopy(self.basketOld_Y)
		self.freezeModel.X = self.freezeModel.x_train = self.X
		self.freezeModel.ys = self.freezeModel.y_train = self.ys
		

		self.freezeModel.Y = Y
		self.freezeModel.actualize()

		###############################From here onwards the iteration with for loop###########################################
		#######################################################################################################################

		logAllIters = []
		for k in range(self.init_points, num_iterations):
			
			if self.stop_epochs and nr_epochs >= self.max_epochs:
				print 'Maximal number of epochs'
				break

			logger.info("Start iteration %d ... ", k)
			print '######################iteration nr: {:d} #################################'.format(k)

			if VU_PRINT >=1:
				print "freezeModel.X.shape is :", self.freezeModel.X.shape


			start_time = time.time()
			#res = self.choose_next(X=self.basketOld_X, Y=self.basketOld_Y, do_optimize=True)
			#here just choose the next candidates, which will be stored in basketNew
			res = self.choose_next_ei(X=self.X, Y=self.ys, do_optimize=True)

			#SO this X and Y must be the full X and Y


			# res[0] = [0.5,0.5,0.5,0.5,0.5,0.5,0.5]


			#res = res[0]
			# 	print 'res: {:s}'.format(res)
			igTime = time.time()
			# ig = InformationGainMC(model=self.freezeModel, X_lower=self.task.X_lower, X_upper=self.task.X_upper, sampling_acquisition=EI)
			ig = InformationGainMC(model=self.freezeModel, X_lower=self.task.X_lower, X_upper=self.task.X_upper)

			ig.update(self.freezeModel, calc_repr=True)
			if VU_PRINT >= 0:
				print ":::::		Time taken for InforGainMC:  ", time.time()-igTime


			H = ig.compute()
			zb = deepcopy(ig.zb)
			lmb = deepcopy(ig.lmb)

			# if VU_PRINT >= 2:
			# 	print "::::			in FreezeThawBO, zb of the first InformationGainMC = ", zb
			# 	print "::::			while lmb = ", lmb


			print 'H: {}'.format(H)
			# Fantasize over the old and the new configurations
			nr_old = self.init_points
			fant_old = np.zeros(nr_old)

			for i in xrange(nr_old):
				conf_index = self.basket_indices[i]

				fv = self.freezeModel.predict(option='old', conf_nr=conf_index)
				fant_old[i] = fv[0]
			
			nr_new = res.shape[0]

			fant_new = np.zeros(nr_new)
			for j in xrange(nr_new):
				m, v = self.freezeModel.predict(xprime=res[j,:], option='new')
				fant_new[j] = m

			# if VU_PRINT >= 0:
			# 	print "So fantasized loss of new models: ", fant_new

			Hfant = np.zeros(nr_old + nr_new)

			for i in xrange(nr_old):
				freezeModel = deepcopy(self.freezeModel)

				conf_index = self.basket_indices[i]

				y_i = freezeModel.ys[conf_index]

				y_i = np.append(y_i, np.array([fant_old[i]]), axis=0)

				freezeModel.ys[conf_index] = y_i
				
				freezeModel.train(X=freezeModel.X, Y=freezeModel.ys, do_optimize=False)

				ig1 = InformationGainMC(model=freezeModel, X_lower=self.task.X_lower, X_upper=self.task.X_upper, sampling_acquisition=EI)
				ig1.actualize(zb, lmb)
				ig1.update(freezeModel)
				H1 = ig1.compute()
				Hfant[i] = H1

			
			# print 'freezeModel.X: {}'.format(freezeModel.X)
			# print 'res: {:s}'.format(res)

			for k in xrange(nr_new):

				freezeModel = deepcopy(self.freezeModel)

				newX = np.append(freezeModel.X, res[k,:][np.newaxis,:], axis=0)
				ysNew = np.zeros(len(freezeModel.ys) + 1, dtype=object)
				for i in xrange(len(freezeModel.ys)): ##improve: do not use loop here, but some expansion
					ysNew[i] = freezeModel.ys[i]

				ysNew[-1] = np.array([fant_new[k]])

				freezeModel.train(X=newX, Y=ysNew, do_optimize=False)

				freezeModel.C_samples = np.zeros(
				            (freezeModel.C_samples.shape[0], freezeModel.C_samples.shape[1] + 1, freezeModel.C_samples.shape[2] + 1))
				freezeModel.mu_samples = np.zeros(
				    (freezeModel.mu_samples.shape[0], freezeModel.mu_samples.shape[1] + 1, 1))

				ig1 = InformationGainMC(model=freezeModel, X_lower=self.task.X_lower, X_upper=self.task.X_upper, sampling_acquisition=EI)
				ig1.actualize(zb, lmb)
				ig1.update(freezeModel)
				H1 = ig1.compute()
				Hfant[-(nr_new - k)] = H1 #the why of the initial -
				# print 'Hfant: {}'.format(Hfant)
			
			print 'Hfant: {}'.format(Hfant)

			# Comparison of the different values
			infoGain = -(Hfant - H)
			# winner = np.argmax(infoGain)
			winner = np.argmin(infoGain)

			print 'the winner is index: {:d}'.format(winner)

			time_overhead = time.time() - start_time
			self.time_overhead = np.append(self.time_overhead,
				np.array([time_overhead]))

			logger.info("Optimization overhead was %f seconds" %
				(self.time_overhead[-1]))

			thisIter = {
				"time_overhead": time_overhead,
			}
			#run an old configuration and actualize basket
			if winner <= ((len(Hfant) - 1) - nr_new):
				print('###################### run old config ######################')
				# run corresponding configuration for more one step
				##change: the task function should send the model file_path back and it should be stored here
				#ytplus1 = self.task.f(t=len(self.basketOld_Y[winner]) + 1, x=self.basketOld_X[winner])
				#ytplus1 = self.task.f(t=len(self.basketOld_Y[winner]) + 1, x=self.basketOld_X[winner], save_file_old=self.basket_files[winner])
				self.task.set_save_modus(is_old=True, file_old=self.basket_files[winner], file_new=None)
				self.task.set_epochs(self.nr_epochs_further)

				if self.pkl:
					with open(self.basket_files[winner],'rb') as f:
						weights_final = pickle.load(f)
					W,b = weights_final
					self.task.set_weights(W=W, b=b)
				else:
					self.task.set_weights(self.basket_files[winner])
				#ytplus1 = self.task.objective_function(x=self.basketOld_X[winner])
				conf_to_run = self.basketOld_X[winner]

				logger.info("Evaluate candidate %s" % (str(conf_to_run[np.newaxis,:])))
				start_time = time.time()

				_, val_losses= self.task.evaluate(x=conf_to_run[np.newaxis,:])

				time_func_eval = time.time() - start_time
				thisIter["time_func_eval"] = time_func_eval
				self.time_func_eval = np.append(self.time_func_eval, np.array([time_func_eval]))

				logger.info("Configuration achieved a performance of %f " %(val_losses[-1]))
				logger.info("Evaluation of this configuration took %f seconds" %(self.time_func_eval[-1]))

				ytplus1 = np.asarray(val_losses)
				val_losses_all = val_losses_all + val_losses
				nr_epochs+=len(val_losses)
				self.basketOld_Y[winner] = np.append(self.basketOld_Y[winner], ytplus1)
				
				index_now = self.basket_indices[winner]

				thisIter["config_id"] = index_now
				thisIter["old"] = True

				#self.all_configs[index_now] = [conf_to_run, self.basketOld_Y[winner], True, winner]
				self.all_configs[index_now][1] = self.basketOld_Y[winner]
				
				self.ys[index_now] = self.basketOld_Y[winner]		#new


			#else run the new proposed configuration and actualize
			else:
				print('###################### run new config #' + str(self.total_nr_confs) +  ' ######################')
				thisIter["config_id"] = self.total_nr_confs
				thisIter["old"] = False

				winner = winner - nr_old
				##change: the task function should send the model file_path back and it should be saved here
				#ytplus1 = self.task.f(t=1, x=res[winner])
				if self.pkl:
					file_path = "config_" + str(self.total_nr_confs) + ".pkl"
				else:
					file_path = "config_" + str(self.total_nr_confs)

				file_path = os.path.join(self.directory, file_path)
				#ytplus1 = self.task.f(t=1, x=res[winner], save_file_new=file_path)
				self.task.set_save_modus(is_old=False, file_old=None, file_new=file_path)
				self.task.set_epochs(self.nr_epochs_further)
				#ytplus1 = self.task.objective_function(x=res[winner])

				logger.info("Evaluate candidate %s" % (str(res[winner][np.newaxis,:])))
				start_time = time.time()

				_, val_losses = self.task.evaluate(x=res[winner][np.newaxis,:])
				ytplus1 = np.asarray(val_losses)
				val_losses_all = val_losses_all + val_losses

				time_func_eval = time.time() - start_time
				thisIter["time_func_eval"] = time_func_eval

				self.time_func_eval = np.append(self.time_func_eval, np.array([time_func_eval]))

				logger.info("Configuration achieved a performance of %f " %(val_losses[-1]))
				logger.info("Evaluation of this configuration took %f seconds" %(self.time_func_eval[-1]))

				nr_epochs+=len(val_losses)
				replace = get_min_ei(freezeModel, self.basketOld_X, self.basketOld_Y)
				self.basketOld_X[replace] = res[winner]
				if type(ytplus1) != np.ndarray:
					self.basketOld_Y[replace] = np.array([ytplus1])
				else:
					self.basketOld_Y[replace] = ytplus1
				
				self.basket_files[replace] = file_path

				#deactivate the configuration which is being replaced
				self.all_configs[self.basket_indices[replace]][2] = False
				self.all_configs[self.basket_indices[replace]][3] = -1
				#actualize the indices table
				
				self.basket_indices[replace]=self.total_nr_confs
				#add new configuration with learning curve and index
				conf_to_run = res[winner]
				if VU_PRINT >= 3 :
					print "In freeze_thaw_bo, run, after running new: "
					print "all_configs = ", self.all_configs
					print "total_nr_confs = ", self.total_nr_confs, "length of all_configs = " , len(self.all_configs), " winner = " , winner , " length of basketOld_Y = ", len(self.basketOld_Y)

				# self.all_configs[self.total_nr_confs] = [conf_to_run, self.basketOld_Y[winner], True, replace] 			#FUCKING WRONG MAN !
				self.all_configs[self.total_nr_confs] = [conf_to_run, self.basketOld_Y[replace], True, replace]
				
				if VU_PRINT >= 3:
					print "Before a new model added, ys = ", self.ys
					print "Before a new model added, X = ", self.X

				
				#new
				self.X = np.append(self.X, [conf_to_run], axis=0)
				newY = self.basketOld_Y[replace]
				newArray = np.zeros(1,dtype=object)
				newArray[0] = np.asarray(newY)

				# newArray = np.array([ np.asarray(newY) ],dtype=object)
				# print "new array : ", newArray
				# print "ys.shape: ", self.ys.shape, " newArray shape = ", newArray.shape
				
				# newYs = np.zeros(total_nr_confs+1,dtype=object)
				# for i in range(newYs.shape())
				self.ys = np.append(self.ys, newArray , axis=0 )

				if VU_PRINT >= 3:
					print "After a new model added, ys = ", self.ys
					print "After a new model added, X = ", self.X


				if VU_PRINT >= 3 :
					print "In freeze_thaw_bo, run, after setting all_configs: "
					print "all_configs = ", self.all_configs
					print "total_nr_confs = ", self.total_nr_confs, "length of all_configs = " , len(self.all_configs), " winner = " , winner , " length of basketOld_Y = ", len(self.basketOld_Y)

				self.total_nr_confs+=1

			logAllIters.append(thisIter)
			pickle.dump(logAllIters, open(SAVE_FILE,"wb"))
			
			if VU_PRINT >= 1:
				self.printBasket()

			Y = getY(self.basketOld_Y)
			self.incumbent, self.incumbent_value = estimate_incumbent(Y, self.X)
			self.incumbents.append(self.incumbent)
			self.incumbent_values.append(self.incumbent_value)

			print "::::::::::::::::::::::   CURRENT INCUMBENT: ", self.incumbent, ", value = ", self.incumbent_value


			if self.save_dir is not None and (k) % self.num_save == 0:
				self.save_json(k, learning_curve=str(val_losses), information_gain=str(infoGain),
					entropy=str(H), phantasized_entropy=str(Hfant))
				self.save_iteration(k)

			with open('val_losses_all.pkl', 'wb') as f:
				pickle.dump(val_losses_all, f)

			with open('all_configs.pkl','wb') as c:
				pickle.dump(self.all_configs, c)

			if self.save_dir:
				self.save_json(i)

		return self.incumbent, self.incumbent_value


	def choose_next(self, X=None, Y=None, do_optimize=True):
		"""
		Recommend a new configuration by maximizing the acquisition function

		Parameters
		----------
		num_iterations: int
		    Number of loops for choosing a new configuration
		X: np.ndarray(N,D)
		    Initial points that are already evaluated
		Y: np.ndarray(N,1)
		    Function values of the already evaluated points

		Returns
		-------
		np.ndarray(1,D)
		    Suggested point
		"""

		initial_design = init_random_uniform

		if X is None and Y is None:
			x = initial_design(self.task.X_lower, self.task.X_upper, N=1)

		elif X.shape[0] == 1:
			x = initial_design(self.task.X_lower, self.task.X_upper, N=1)
		else:
			try:
				self.freezeModel.train(X, Y, do_optimize=do_optimize)
			except:
				raise

			model_untrained = False

			self.acquisition_func.update(self.freezeModel)

			x = self.maximize_func.maximize()

		return x


	def choose_next_ei(self, X=None, Y=None, do_optimize=True, N=10, M=3):
		"""
		Recommend a new configuration by maximizing the acquisition function

		Parameters
		----------
		num_iterations: int
		    Number of loops for choosing a new configuration
		X: np.ndarray(N,D)
		    Initial points that are already evaluated
		Y: np.ndarray(N,1)
		    Function values of the already evaluated points
		N: integer
		    Number of configuration samples
		M: integer
		    Chosen configurations
		Returns
		-------
		np.ndarray(1,D)
		    Suggested point
		"""

		initial_design = init_random_uniform

		self.freezeModel.train(X, Y, do_optimize=do_optimize)
		self.freezeModel.actualize()

		x = initial_design(self.task.X_lower, self.task.X_upper, N=N)

		ei_list = np.zeros(x.shape[0])

		for i in xrange(N):
			ei_list[i] = compute_ei(X=x[i,:], model=self.freezeModel, ys=Y, basketOld_X=X)


		sort = np.argsort(ei_list)

		highest_confs = x[sort][-M:]

		if VU_PRINT > 10 :
			print "In BO.choose_next_ei, before returning, the best configurations are :", highest_confs
		return highest_confs

	def create_file_paths(self):
		if not os.path.exists(self.directory):
			os.makedirs(self.directory)

		for index in xrange(self.init_points):
			if self.pkl:
				file_path = "config_" + str(index) + ".pkl"
			else:
				file_path = "config_" + str(index)
			self.basket_files.append(os.path.join(self.directory, file_path))
			self.basket_indices.append(index)#that's exactly true here, thereafter there are changes

	def get_json_data(self, it):
		jsonData = dict()
		jsonData = {
		"optimization_overhead":None if self.time_overhead is None else self.time_overhead[it],
		"runtime":None if self.time_start is None else time.time() - self.time_start,
		"incumbent":None if self.incumbent is None else self.incumbent.tolist(),
		"incumbent_fval":None if self.incumbent_value is None else self.incumbent_value.tolist(),
		"time_func_eval": self.time_func_eval[it],
		"iteration":it
		}
		return jsonData


	def save_json(self, it, **kwargs):
		base_solver_data =self.get_json_data(it)
		base_model_data = self.model.get_json_data()
		base_task_data = self.task.get_json_data()
		base_acquisition_data = self.acquisition_func.get_json_data()

		data = {'Solver': base_solver_data,
		'Model': base_model_data,
		'Task':base_task_data,
		'Acquisiton':base_acquisition_data
		}

		if kwargs is not None:
			for key, value in kwargs.items():
				data[key] = str(value)

		json.dump(data, self.output_file_json)
		self.output_file_json.write('\n')  #Json more readable. Drop it?


def f(t, a=0.1, b=0.1, x=None):
	k=1e3
	if x is not None:
		a, b = x
	return k*a*np.exp(-b*t)

def getY(ys):
    Y = np.zeros((len(ys),1))
    for i in xrange(Y.shape[0]):
        Y[i,:] = ys[i][-1]
    return Y

def estimate_incumbent(Y, basketOld_X):
    best = np.argmin(Y)
    incumbent = basketOld_X[best]
    incumbent_value = Y[best]

    return incumbent[np.newaxis, :], incumbent_value[:, np.newaxis] 

def compute_ei(X, model, ys, basketOld_X, par=0.0):
    m, v = model.predict(X[None,:])

    Y = getY(ys)

    _, eta = estimate_incumbent(Y, basketOld_X)
    # eta = eta[0,0]

    s = np.sqrt(v)

    z = (eta - m - par) / s

    f = s * ( z * norm.cdf(z) +  norm.pdf(z))

    return f

def get_min_ei(model, basketOld_X, basketOld_Y):
    nr = basketOld_X.shape[0]
    eiList = np.zeros(nr)
    for i in xrange(nr):
        val = compute_ei(basketOld_X[i], model, basketOld_Y, basketOld_X)
        # print'val in get_min_ei: ', val
        eiList[i] = val[0][0]

    minIndex = np.argmin(eiList)


    if VU_PRINT >= 1:
    	print "::::			Finding the min EI in the current basket to be replaced by the new config  		::::"
    	print "::::			EI list is :", eiList
    	print ":::: winner = ", minIndex, " : ", eiList[minIndex]
    	
    return minIndex
