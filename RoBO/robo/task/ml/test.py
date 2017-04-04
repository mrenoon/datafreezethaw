from var_size_data_freeze_convnet_cifar import *
import numpy as np

task = VarSizeDataConvNetCifar(num_epochs=100,file_name="test1")
# x = np.array([[0.05314127, 0.30185222 , 0.16204598 , 0.19431632 , 0.36325013 ,   0.25049316 ,  0.53116273,0.01]])

# x = np.array([[0.001,0.002,0.001,0.001,0.001,0.001,0.001,0.01]])
x = np.array([[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.01]])
print "evaluate test x = ", task.evaluate(x)
