import numpy as np

from robo.task.base_task import BaseTask

class ExpDecay(BaseTask):

    def __init__(self, do_scaling=False):

        X_lower = np.array([0.01, 0.01])

        X_upper = np.array([0.1, 0.1])

        super(ExpDecay, self).__init__(X_lower, X_upper, opt=None, fopt=None, do_scaling=do_scaling)

    def objective_function(self, x):
        """
        x[:,0]: alpha
        x[:,1]: beta
        x[:,2]: time step
        """
        a = x[0]
        b = x[1]
        t = x[2]
        k = 1e3
        x_input = np.array([a,b])
        y = self.f(t=t, x=x_input)
        return y[:, np.newaxis]

    def objective_function_test(self, x):

        return self.objective_function(x)

    def f(self, t, a=0.1, b=0.1, x=None):
        #k=1e3
        if x is not None:
            a, b = x

        return a*np.exp(-b*t)
    def set_save_modus(self, is_old=True, file_old=None, file_new=None):
        self.save_old = is_old
        self.save = True
        if self.save_old:
            self.file_name = file_old
        else:
            self.file_name = file_new