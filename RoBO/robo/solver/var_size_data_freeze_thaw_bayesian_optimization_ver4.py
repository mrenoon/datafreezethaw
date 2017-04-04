# encoding=utf8
#   author of the original piece of code
# __author__ = "Tulio Paiva"
# __email__ = "paivat@cs.uni-freiburg.de"

#  modified for DataFreezeThaw BO implementation
__author__ = "Nguyen Hoang Vu"
__email__ = "vunguyenvyvp@gmail.com"



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
from robo.models.freeze_thaw_model import FreezeThawGP
from robo.maximizers.direct_new import Direct
from robo.acquisition.ei import EI
from robo.acquisition.information_gain_mc_data_freeze_ver2 import InformationGainMC, get_training_time
from scipy.stats import norm

logger = logging.getLogger(__name__)
logging.basicConfig(filename='Vu.log', level=logging.INFO)

VU_PRINT = 2
#   0 = everything
#   1 = only details
#   2 = even more details


class VarSizeDataFreezeThawBO(BaseSolver):

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
                 initial_size=0.02,
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



        super(VarSizeDataFreezeThawBO, self).__init__(acquisition_func, freeze_thaw_model, maximize_func, task)

        self.start_time = time.time()

        if initial_design == None:
            self.initial_design = init_random_uniform
        else:
            self.initial_design = initial_design

        self.X = None
        self.Y = None
        self.ys = None
        self.ss = None


        self.freezeModel = self.model

        self.task = task

        self.model_untrained = True

        # Well this is 
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
        self.basketOld_S = None


        self.basket_files = list()
        self.basket_indices = list()
        self.directory = "temp_configs_" + self.task.base_name

        self.initial_size = initial_size
        # self.nr_epochs_inits = nr_epochs_inits
        # self.nr_epochs_further = nr_epochs_further

        self.time_func_eval = None
        self.time_overhead = None
        # self.train_intervall = train_intervall

        # self.num_save = num_save
        self.time_start = None
        # self.pkl = pkl                    ??????
        # self.stop_epochs = stop_epochs    ??????

        # self.max_epochs = max_epochs
        
        self.all_configs = dict()
        self.total_nr_confs = 0

        # self.n_restarts = n_restarts      ??????
        self.runtime = []   

    def printBasket(self):
        # print "::::                                                                   ::::"
        print "::::                 The Current Basket of the BO:                   ::::"
        for cId, cand in enumerate(self.basketOld_X):
            print "::::     Candidate #", cId, ": ", cand,"=>>>  value = ", self.basketOld_Y[cId]
        # print "::::                                                                   ::::"
        print ""

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

        init = init_random_uniform(self.task.X_lower[:-1], self.task.X_upper[:-1], self.init_points)
        if VU_PRINT >= 0 :
            print "Initial init = ", init

        self.basketOld_X = deepcopy(init)
        self.basketOld_S = np.zeros(self.init_points, dtype=object)

        for i in range(self.init_points):
            self.basketOld_S[i] = np.asarray([0.01])

        # self.basketOld_S = np.array( [[0.02]]*self.init_points )

        # val_losses_all = []
        # nr_epochs = 0                 ????

        self.create_file_paths()

        self.time_start = time.time()



        ys = np.zeros(self.init_points, dtype=object)
        self.time_func_eval = np.zeros([self.init_points])
        self.time_overhead = np.zeros([self.init_points])

        #change: the task function should send the model file_path back and it should be stored here
        for i in xrange(self.init_points):
            #ys[i] = self.task.f(np.arange(1, 1 + self.first_steps), x=init[i, :]) ##change: that's not general
            #ys[i] = self.task.objective(nr_epochs=None, save_file_new=self.basket_files, save_file_old=None)
            self.task.set_save_modus(is_old=False, file_old=None, file_new=self.basket_files[i])
            
            # self.task.set_epochs(self.nr_epochs_inits)
            #print 'init[i,:]: ', init[i,:]
            #_, ys[i] = self.task.objective_function(x=init[i,:][np.newaxis,:])

            #Need to append the dataset-size to the config to pass to the task
            conf_now = init[i,:]
            input_now = np.append(conf_now, self.basketOld_S[i])

            logger.info("Evaluate: %s with initial dataset size" % input_now[np.newaxis,:])
            start_time = time.time()

            val_losses = self.task.evaluate(x=input_now[np.newaxis,:])  #actually there is only one training loss

            if VU_PRINT >= 0:
                print "In initing points, val_losses = ", val_losses

            self.time_func_eval[i] = time.time() - start_time
            self.time_overhead[i] = 0.0

            logger.info("Configuration achieved a performance "
                "of %f in %f seconds" %
                (val_losses[-1], self.time_func_eval[i]))
            #storing configuration, learning curve, activity, index in the basketOld
            ys[i] = np.asarray(val_losses)
            
            self.all_configs[self.total_nr_confs] = [conf_now, ys[i], True, self.total_nr_confs, input_now[-1] ]

            self.total_nr_confs+=1
            # val_losses_all = val_losses_all + val_losses
            # nr_epochs+=len(val_losses)




            if self.save_dir is not None and (i) % self.num_save == 0:
                self.save_json(i, learning_curve=str(val_losses), information_gain=0.,
                    entropy=0, phantasized_entropy=0.)
                self.save_iteration(i)
            

        self.basketOld_Y = deepcopy(ys)
        self.X = deepcopy(init)
        self.ys = deepcopy(ys)
        self.ss = deepcopy(self.basketOld_S)

        # Y = np.zeros((len(ys), 1))
        # for i in xrange(Y.shape[0]):
        #   Y[i, :] = ys[i][-1]


        self.Y = np.zeros((len(ys), 1))
        for i in xrange(self.Y.shape[0]):
            self.Y[i, :] = ys[i][-1]
        

        if VU_PRINT >= 1:
            print "ys after getting init points: ", ys
            print "Y after gettting init points: ", self.Y
 


        self.freezeModel.X = self.freezeModel.x_train = self.basketOld_X
        self.freezeModel.ys = self.freezeModel.y_train = self.basketOld_Y
        self.freezeModel.ss = self.freezeModel.s_train = self.basketOld_S

        self.freezeModel.Y = self.Y
        self.freezeModel.actualize()


        self.ig = InformationGainMC(self.freezeModel, self.task.X_lower, self.task.X_upper)
        #So the sampling_acquisition will be logEI
        self.maximizer = Direct(self.ig, np.asarray([0.0]), np.asarray([1.0]), n_func_evals=30, verbose=False)
        # print "self.task.X_lower.shape = ", self.task.X_lower.shape, " our lower = ", np.asarray([0.0]), " shape = ", np.asarray([0.0]).shape
        # self.maximizer = Direct(self.ig, self.task.X_lower, self.task.X_upper, n_func_evals=200)


        ###############################From here onwards are the real iterations ###########################################
        #######################################################################################################################

        for k in range(self.init_points, num_iterations):
            
            # if self.stop_epochs and nr_epochs >= self.max_epochs:
            #   print 'Maximal number of epochs'
            #   break

            logger.info("Start iteration %d ... ", k)
            print '######################iteration nr: {:d} #################################'.format(k)
            start_time = time.time()
            #res = self.choose_next(X=self.basketOld_X, Y=self.basketOld_Y, do_optimize=True)
            
            #here just choose the next candidate
            # newConfig = self.choose_next_ig(X=self.X, Y=self.ys, S=self.ss, do_optimize=True)
            # newConfigIG = self.ig.compute(newConfig)

            

            newConfigs = self.choose_next_ei(X=self.X, Y=self.ys, S=self.ss, do_optimize=True)
            nr_new = len(newConfigs)
            IG_new = np.zeros(nr_new)

            time_overhead = time.time() - start_time
            start_time_2 = time.time()

            for i in xrange(nr_new):
                IG_new[i] = self.ig.computeNew(np.array([ newConfigs[i] ]), time_overhead)


            # print "IG_new = ", IG_new


            winnerNew = np.argmax(IG_new)
            newConfig = np.asarray( [newConfigs[winnerNew]] )

            
            nr_old = self.init_points
            asympt_old = np.zeros(nr_old)
            for i in range(nr_old):
                temp , _ = self.freezeModel.predict(self.basketOld_X[i][None,:])
                asympt_old[i] = temp[0,0]

            IG_old = np.zeros(nr_old)
            newSize = np.zeros(nr_old)
            for i in xrange(nr_old):
                #find out how much IG/time does training on this old yield ?
                config_id = self.basket_indices[i]
                self.ig.config_nr = config_id
                self.ig.overhead = time_overhead

                temp = self.maximizer.maximize()

                # print "|||||||||        For old candidate #", i, ", config number #", config_id, " last = ", self.basketOld_Y[i], "=> asympt= ", asympt_old[i]
                
                newSize[i] = temp[0][0]
                IG_old[i] = self.ig.compute(temp)[0][0]
                # print "     |||||||||||||            Last size = ", self.basketOld_S[i],  " new size = ", temp[0][0]
                print "     |||||||||||||            "

                # self.ig.computeOldModel(config_id, time_overhead)




            winnerOld = np.argmax(IG_old)


            if VU_PRINT >= 0:
                # print ""
                print "$$$$$$$$  IG/unit time of old configs: ", IG_old

            if VU_PRINT >= 0:
                print "$$$$$$$$  IG/unit time of new configs = ", IG_new
                print ""

            oldWins = True
            if IG_old[winnerOld] < IG_new[winnerNew] or IG_old[winnerOld]<0:            #important: if the old IG are all negative, meaning its all freaking huge dataset size, not supposed to be !
                print "Winner: new config number ", winnerNew
                oldWins = False
            else:
                print "Winner: old config number ", winnerOld

            time_overhead = time.time() - start_time
            
            # self.time_overhead = np.append(self.time_overhead,
            #   np.array([time_overhead]))
            logger.info("Optimization overhead was %f seconds" %
                (self.time_overhead[-1]))
            if VU_PRINT >= -10:
                print "%%%%%%%% TIME FOR CALCULATING IG = ", time.time() - start_time_2, ", TIME OVERHEAD = ", time_overhead


            #run an old configuration and actualize basket
            if oldWins:
                print('###################### run old config ######################')
                


                self.task.set_save_modus(is_old=True, file_old=self.basket_files[winnerOld], file_new=None)
                
                stplus1 = np.asarray(newSize[winnerOld])
                self.basketOld_S[winnerOld] = np.append(self.basketOld_S[winnerOld], stplus1) 


                conf_to_run = self.basketOld_X[winnerOld]


                # if VU_PRINT >= -5:
                #   print "basketOld_X = ", basketOld_X
                #   print "conf_to_run = ", conf_to_run


                input_to_run = np.append(conf_to_run, np.asarray([stplus1]), axis=0)


                if VU_PRINT >= 2:
                    print "input_to_run[np.newaxis,:] = ", input_to_run[np.newaxis,:]

                logger.info("Evaluate candidate %s" % (str(input_to_run[np.newaxis,:])))
                
                start_time = time.time()

                val_losses= self.task.evaluate(x=input_to_run[np.newaxis,:])

                time_func_eval = time.time() - start_time

                self.time_func_eval = np.append(self.time_func_eval, np.array([time_func_eval]))
                logger.info("Configuration achieved a performance of %f " %(val_losses[-1]))
                logger.info("Evaluation of this configuration took %f seconds" %(self.time_func_eval[-1]))
                
                if VU_PRINT >= -10:
                    print "%%%%%%%% TIME FOR EVALUATION = ", time_func_eval, " s = ", stplus1 * 500, " TOTAL=", time_overhead + time_func_eval, " (estimated=", self.ig.get_training_time_old(stplus1,time_overhead, self.basketOld_S[winnerOld][-2])
                    # print "%%%%%%%% TOTAL TIME = ", time_overhead + time_func_eval
                    # print "%%%%%%%% TIME ESTIMATED = ", self.ig.get_training_time(stplus1)

                ytplus1 = np.asarray(val_losses)
                self.basketOld_Y[winnerOld] = np.append(self.basketOld_Y[winnerOld], ytplus1)

                index_now = self.basket_indices[winnerOld]
                #self.all_configs[index_now] = [conf_to_run, self.basketOld_Y[winner], True, winner]
                self.all_configs[index_now][1] = self.basketOld_Y[winnerOld]
                self.all_configs[index_now][-1] = self.basketOld_S[winnerOld]


                self.ys[index_now] = self.basketOld_Y[winnerOld]        #new
                self.ss[index_now] = self.basketOld_S[winnerOld]

            #else run the new proposed configuration and actualize
            else:
                print('###################### run new config #' + str(self.total_nr_confs) +  ' ######################')
                

                # if self.pkl:
                #   file_path = "config_" + str(self.total_nr_confs) + ".pkl"
                # else:
                #   file_path = "config_" + str(self.total_nr_confs)


                file_path = "config_" + str(self.total_nr_confs)

                file_path = os.path.join(self.directory, file_path)
                

                self.task.set_save_modus(is_old=False, file_old=None, file_new=file_path)
                # self.task.set_epochs(self.nr_epochs_further)
                #ytplus1 = self.task.objective_function(x=res[winner])

                logger.info("Evaluate candidate %s" % (str(newConfig)) )
                start_time = time.time()

                val_losses = self.task.evaluate(x=newConfig)
                ytplus1 = np.asarray(val_losses)
                stplus1 = np.asarray(newConfig[0,-1])

                # val_losses_all = val_losses_all + val_losses

                time_func_eval = time.time() - start_time
                self.time_func_eval = np.append(self.time_func_eval, np.array([time_func_eval]))
                logger.info("Configuration achieved a performance of %f " %(val_losses[-1]))
                logger.info("Evaluation of this configuration took %f seconds" %(self.time_func_eval[-1]))

                if VU_PRINT >= -10:
                    print "%%%%%%%% TIME FOR EVALUATION = ", time_func_eval, " s = ", stplus1 * 500, " TOTAL=", time_overhead + time_func_eval, " (estimated=",self.ig.get_training_time(stplus1, time_overhead)

                    # print "%%%%%%%% TIME FOR EVALUATION = ", time_func_eval, " s = ", stplus1 * 500
                    # print "%%%%%%%% TOTAL TIME = ", time_overhead + time_func_eval
                    # print "%%%%%%%% TIME ESTIMATED = ", self.ig.get_training_time(stplus1)

                # nr_epochs+=len(val_losses)

                

                # print " Asymptotic values of old configs: ", asympt_old

                # replace = np.argmin(IG_old)               option 1
                replace = np.argmax(asympt_old)            #option 2
                
                # print " The worst is ", replace
                #Actually what should be replaced ??? Maybe the worst in terms of info gain ?? 
                #I reckon so. Or maybe not, the worst asymptotic mean ?


                if VU_PRINT >= 3:
                    print "replace = ", replace
                    print "basketOld_X[replace] = ", self.basketOld_X[replace]
                    print "newConfig[0,:-1] = ", newConfig[0,:-1]

                self.basketOld_X[replace] = newConfig[0,:-1]
                
                self.basketOld_S[replace] = np.asarray([stplus1])

                if VU_PRINT >= 3:
                    print "So after setting new basketOld_S, basketOld_S[replace] = ", np.asarray([stplus1])
                    print "stplus1 = ", stplus1

                #this might be wrong in dimensions

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
                conf_to_run = newConfig[:,:-1]

                if VU_PRINT >= 3 :
                    print "In freeze_thaw_bo, run, after running new: "
                    print "all_configs = ", self.all_configs
                    print "total_nr_confs = ", self.total_nr_confs, "length of all_configs = " , len(self.all_configs), " winner = " , winner , " length of basketOld_Y = ", len(self.basketOld_Y)

                # self.all_configs[self.total_nr_confs] = [conf_to_run, self.basketOld_Y[winner], True, replace]            #FUCKING WRONG MAN !
                self.all_configs[self.total_nr_confs] = [conf_to_run, self.basketOld_Y[replace], True, replace, self.basketOld_S[replace]]
                

                if VU_PRINT >= 3:
                    print "np.asarray(conf_to_run) = ", np.asarray(conf_to_run)
                    print "np.asarray(conf_to_run).shape = ", np.asarray(conf_to_run).shape
                    print "X.shape = ", self.X.shape

                self.X = np.append(self.X, np.asarray(conf_to_run), axis=0)

                newY = self.basketOld_Y[replace]
                newArray = np.zeros(1,dtype=object)
                newArray[0] = np.asarray(newY)


                newS = self.basketOld_S[replace]
                newSArray = np.zeros(1,dtype=object)
                newSArray[0] = np.asarray(newS)

                if VU_PRINT >= 3:
                    print "newY = ", newY
                    print "newArray = ", newArray
                    print "newS = ", newS
                    print "newSArray = ", newSArray

                self.ys = np.append(self.ys, newArray , axis=0)
                self.ss = np.append(self.ss, newSArray, axis=0)

                

                self.total_nr_confs+=1

            if VU_PRINT >= 3 :
                print "In freeze_thaw_bo, After an interation: "
                # print "all_configs = ", self.all_configs
                # print "total_nr_confs = ", self.total_nr_confs, "length of all_configs = " , len(self.all_configs)
                print "self.X = ", self.X
                print "self.ys = ", self.ys
                print "self.ss = ", self.ss



            if VU_PRINT >= 1:
                self.printBasket()

            Y = getY(self.ys)
            # if VU_PRINT >=0:
            #   print "Y after an iteration: ", Y

            self.incumbent, self.incumbent_value = estimate_incumbent(Y, self.X)
            self.incumbents.append(self.incumbent)
            self.incumbent_values.append(self.incumbent_value)

            print "::::::::::::::::::::::   CURRENT INCUMBENT: ", self.incumbent, ", value = ", self.incumbent_value


            if self.save_dir is not None and (k) % self.num_save == 0:
                self.save_json(k, learning_curve=str(val_losses), information_gain=str(infoGain),
                    entropy=str(H), phantasized_entropy=str(Hfant))
                self.save_iteration(k)

            # with open('val_losses_all.pkl', 'wb') as f:
            #   pickle.dump(val_losses_all, f)

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


    def choose_next_ei(self, X=None, Y=None, S=None, do_optimize=True, N=200, M=10):
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

        startTime = time.time()
        self.freezeModel.train(X, Y, S, do_optimize=do_optimize)
        self.freezeModel.actualize()
        train_time = time.time() - startTime

        # print "After training freezeModel, its samples = ", self.freezeModel.samples

        startTime = time.time()

        self.ig.update(self.freezeModel,calc_repr=True,true_update=True)
        ig_time  = time.time() - startTime


        startTime = time.time()
        x = initial_design(self.task.X_lower, self.task.X_upper, N=N)

        ei_list = np.zeros(x.shape[0])

        for i in xrange(N):
            ei_list[i] = compute_ei(X=x[i,:], model=self.freezeModel, ys=Y, S=S, basketX = self.X)


        sort = np.argsort(ei_list)

        highest_confs = x[sort][-M:]


        if VU_PRINT >= 0:
            # print "The best new configs are :", highest_confs
            print "$$$$$$$$ Time taken for training GP = ", train_time, ", time for IG_update = ", ig_time, " while ei_compute took ", time.time() - startTime
        
        return highest_confs

    def choose_next_ig(self, X=None, Y=None, S=None, do_optimize=True, N=100, M=3):
        """
        Recommend a new configuration by maximizing the Information Gain acquisition function

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




        # initial_design = init_random_uniform

        #Obviously, train the GP model first
        self.freezeModel.train(X, Y, S, do_optimize=do_optimize)
        self.freezeModel.actualize()

        start_time_ig = time.time()
        self.ig.update(self.freezeModel,calc_repr=True,true_update=True)
        ig_time = time.time() - start_time_ig
        #Here, it will sample for representer points
        
        start_time = time.time()
        new_x = self.maximizer.maximize()

        if VU_PRINT >= 0:
            print "The best new config us :", new_x
            print "$$$$$$$$ Time taken for Direct = ", time.time() - start_time, " while IG_update took ",ig_time
        
        return new_x

    def create_file_paths(self):
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

        for index in xrange(self.init_points):
            # if self.pkl:
            #   file_path = "config_" + str(index) + ".pkl"
            # else:
            #   file_path = "config_" + str(index)

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
    # print "In estimate_incumbent, Y=", Y
    # print "X = ", basketOld_X

    best = np.argmin(Y)
    incumbent = basketOld_X[best]
    incumbent_value = Y[best]

    return incumbent[np.newaxis, :], incumbent_value[:, np.newaxis] 
    # return incumbent_value[:, np.newaxis]





def compute_ei(X, model, ys, basketX, S, par=0.0):
    m, v = model.predict(X[None,:],option="new")
    # m, v = model.predict(X[None,:-1])

    Y = getY(ys)

    _, eta = estimate_incumbent(Y, basketX)
    # eta = eta[0,0]

    s = np.sqrt(v)

    z = (eta - m - par) / s

    f = s * ( z * norm.cdf(z) +  norm.pdf(z))

    datasize = X[-1]
    # print "Dataset size of new config = ", datasize

    timeTaken = get_training_time(datasize)

    return f / timeTaken

def get_min_ei(model, basketOld_X, basketOld_Y):
    nr = basketOld_X.shape[0]
    eiList = np.zeros(nr)
    for i in xrange(nr):
        val = compute_ei(basketOld_X[i], model, basketOld_Y, basketOld_X)
        # print'val in get_min_ei: ', val
        eiList[i] = val[0][0]
    if VU_PRINT >= 2:
        print "::::         Finding the min EI in the current basket to be replaced by the new config       ::::"
        print "::::         EI list is :", eiList
        
    minIndex = np.argmin(eiList)
    return minIndex
