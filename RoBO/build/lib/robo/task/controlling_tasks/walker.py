import os
import numpy as np

try:
    from pymatbridge import Matlab
except ImportError:
    raise ImportError("For this task you have to install either MATLAB(R) and matpybridge or octave and oc2py")

from robo.task.base_task import BaseTask


class Walker(BaseTask):
    
    def __init__(self, matlab_code_directory,num_steps = 10, matlab_executable='/usr/bin/matlab'):
        '''       
        This task is based on the Walker simulation which was originally used
        for experiments in [1].
        The idea of using it was inspired by [2].
        
        The simulation code examples can be found in 
        http://web.eecs.umich.edu/~grizzle/biped_book_web/
        [WGCCM_three_link_walker_example]
        

        [1] E. Westervelt and J. Grizzle. Feedback Control of Dynamic
        Bipedal Robot Locomotion. Control and Automation Series.
        CRC PressINC, 2007. ISBN 9781420053722.
        
        [2] Bobak Shahriari, Ziyu Wang, Matthew W. Hoffman, 
        Alexandre Bouchard-Cote and Nando de Freitas. An Entropy Search 
        Portfolio for Bayesian Optimization. University of Oxford, 2014.
        "http://arxiv.org/abs/1406.4625"
        
        '''
        self.matlab_code_directory = matlab_code_directory
        self.matlab_executable = matlab_executable
        
        self.num_steps = num_steps
        
        X_lower = np.array([-3,-3,-3,-3,-3,-3,-3,-3])
        X_upper = np.array([3,3,3,3,3,3,3,3])        
        super(Walker, self).__init__(X_lower, X_upper)
        
    def objective_function(self, x):
        '''
        matlabpath: we need the path to matlab since we need to run it.
        
        IMPORTANT: walker_simulation.m must be included in the 
        WGCCM_three_link_walker_example file in order to work. 
        So simply modify the path below.
        '''
		mlab = Matlab(executable= matlabpath)
		mlab.start()
		output = mlab.run_func(os.path.join(self.matlab_code_directory, 'walker_simulation.m'), 
						   {'arg1': x[:, 0],'arg2': x[:, 1],'arg3': x[:, 2],
						   'arg4': x[:, 3],'arg5':x[:, 4],'arg6': x[:, 5],
						   'arg7': x[:, 6],'arg8': x[:, 7],'arg9': self.num_steps})
						   

		answer = output['result']
		simul_output = answer['speed']
		mlab.stop()
        return simul_output
        
        
    def objective_function_test(self, x):
       return self.objective_function(x)
