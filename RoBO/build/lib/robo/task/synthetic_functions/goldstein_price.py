'''
Created on 14.07.2015

@author: Aaron Klein
'''

import numpy as np

from robo.task.base_task import BaseTask


class GoldsteinPrice(BaseTask):
    '''
    classdocs
    '''

    def __init__(self):
        X_lower = np.array([-2, -2])
        X_upper = np.array([2, 2])
        opt = np.array([[0, -1]])
        fopt = 3
        super(GoldsteinPrice, self).__init__(X_lower, X_upper, opt, fopt)

    def objective_function(self, x):
        fval = np.array(1 + (x[:,
                               0] + x[:,
                                      1] + 1) ** 2 * (19 - 14 * x[:,
                                                                  0] + 3 * x[:,
                                                                             0] ** 2 - 14 * x[:,
                                                                                              1] + 6 * x[:,
                                                                                                         0] * x[:,
                                                                                                                1] + 3 * x[:,
                                                                                                                           1] ** 2)) * (30 + (2 * x[:,
                                                                                                                                                    0] - 3 * x[:,
                                                                                                                                                               1]) ** 2 * (18 - 32 * x[:,
                                                                                                                                                                                       0] + 12 * x[:,
                                                                                                                                                                                                   0] ** 2 + 48 * x[:,
                                                                                                                                                                                                                    1] - 36 * x[:,
                                                                                                                                                                                                                                0] * x[:,
                                                                                                                                                                                                                                       1] + 27 * x[:,
                                                                                                                                                                                                                                                   1] ** 2))
        return fval[:, np.newaxis]

    def objective_function_test(self, x):
        return self.objective_function(x)
