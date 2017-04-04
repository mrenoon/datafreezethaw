import numpy as np
import time
import theano
import theano.tensor as T
import lasagne
import os, subprocess
from lasagne.regularization import regularize_layer_params, l2, l1


from robo.task.base_task import BaseTask

ROOT_ML_DIR = "/data/ml/"
EXAMPLES_LINK = "/data/ml/RoBo/examples/"



class ConvNetCifar(BaseTask):

    # def __init__(self, train, train_targets,
    #              valid, valid_targets,
    #              test, test_targets,
    #              n_classes, num_epochs=500,
    #              save=False, file_name=None):
    def __init__(self, save=False, file_name=None, num_epochs=500, train_range=5):
        '''
  

        Parameters
        ----------
        train : (N, D) numpy array
            Training matrix where N are the number of data points
            and D are the number of features
        train_targets : (N) numpy array
            Labels for the training data
        valid : (K, D) numpy array
            Validation data
        valid_targets : (K) numpy array
            Validation labels
        test : (L, D) numpy array
            Test data
        test_targets : (L) numpy array
            Test labels
        n_classes: int
            Number of classes in the dataset
        '''
        #commented out
        # self.X_train = train
        # self.y_train = train_targets
        # self.X_val = valid
        # self.y_val = valid_targets
        # self.X_test = test
        # self.y_test = test_targets
        self.num_epochs = num_epochs
        # self.save = save
        self.file_name = file_name
        self.base_name = "ConvNetCifar"
        self.is_old = False
        self.filename_to_epochs = dict()
        self.train_range = train_range
        # 1 Dim Learning Rate:
        # 2 Dim L2 regularization: 0 to 1
        # 3 Dim Batch size: 20 to 2000
        # 4 Dim Dropout rate: 0 to 0.75
        # 5 Dim L1 regularization: 0.1 to 20
        # 6 Dim Epochs Number: 1 to 100
        # X_lower = np.array([np.log(1e-6), 0.0, 20, 0, 0.1, 1])
        
        X_lower = np.array([0.00001, 0.00001, 0.00001 , 0.00001 ,0.00001 ,0.00001 ,0.00001])
        
        self.params = X_lower

        #X_lower = np.array([np.log(1e-6), 0.0, 1000, 0, 0.1, 1])
        #X_upper = np.array([np.log(1e-1), 1.0, 2000, 0.75, 20, 100])
        #X_upper = np.array([np.log(1e-1), 1.0, 2000, 0.75, 20, 10])
        # X_upper = np.array([np.log(1e-1), 1.0, 2000, 0.75, 20, 7])
        X_upper = np.array([0.01, 0.02, 0.1 , 0.1 ,0.1 ,0.1 ,0.1])

        #X_upper = np.array([np.log(1e-1), 1.0, 2000, 0.75, 20, 3])
        super(ConvNetCifar, self).__init__(X_lower, X_upper)


    def set_weights(self, old_file_name):   # FREEZE            **********************************TODO
        #actually dont need to do anything since the config is already in filename.cfg
        #while the data is at filename.data
        pass
        # file_name = old_file_name + '.npz'
        # with np.load(file_name) as f:
        #     param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        # lasagne.layers.set_all_param_values(self.network, param_values)

    def set_epochs(self, n_epochs):     # FREEZE            **********************************TODO
        self.num_epochs = n_epochs

    def set_save_modus(self, is_old=True, file_old=None, file_new=None):    # FREEZE            **********************************TODO
        self.is_old = is_old
        self.save = True
        if self.is_old:
            self.file_name = file_old
        else:
            self.file_name = file_new


    def objective_function(self, x):
        # well, depending on the self.is_old, load the file and config from existing one or creating a new one


        print 'in objective_function x: ', x
        # learning_rate = np.float32(np.exp(x[0, 0]))
        # l2_reg = np.float32(x[0, 1])
        # batch_size = np.int32(x[0, 2])
        # print 'in objective_function batch_size: ', batch_size
        # dropout_rate = np.int32(x[0, 3])
        # l1_reg = np.int32(x[0, 4])
        # epochs_number = np.int32(x[0, 5])
        # best_validation_loss = np.inf
        
        #we dont need this one since we will always run a fixed number of epochs
        # self.num_epochs = epochs_number
        # print("filename = " + self.file_name)

        # dir_path = os.path.dirname(os.path.realpath(__file__))
        
        dir_path = ROOT_ML_DIR + "RoBO/examples/"

        # print("dir_path = " + dir_path)
        
        os.chdir(ROOT_ML_DIR + "/cuda-convnet2")
        dataPath = ROOT_ML_DIR + "/data/cifar-10-py-colmajor"
        save_file = os.path.join(dir_path,self.file_name + ".data")
        layersCfg = ROOT_ML_DIR + "/Spearmint-EI/examples/convnetcifar/layers-80sec.cfg"
        layersParams = os.path.join(dir_path, self.file_name + ".cfg")
        # testFreq = self.num_epochs * 5
        testFreq = str(self.train_range)

        layersParamsTemplatePath = ROOT_ML_DIR + "/Spearmint-EI/examples/convnetcifar/layer-params-template.cfg"

        #Have to rewrite the params to the config file, not sure if its there or not, even for old configs
        

        template = open(layersParamsTemplatePath,"r").read()
        epsW = x[0][0]
        epsB = x[0][1]
        wc1 = x[0][2]
        wc2 = x[0][3]
        wc3 = x[0][4]
        wc4 = x[0][5]
        wc5 = x[0][6]
        open(layersParams,"w").write(template % ( epsW, epsB, wc1, epsW, epsB, wc2, epsW, epsB, wc3, epsW, epsB, wc4, epsW, epsB, wc5 ))

        if not self.is_old:
            #its a new model, we need to write the layersParams file
            if os.path.exists(save_file):
                temp = subprocess.check_output("rm -rf " + save_file, shell=True)
                # temp = subprocess.check_output("rm " + layersParams, shell=True)

            
            self.filename_to_epochs[self.file_name] = self.num_epochs

            command = "python convnet.py --data-provider cifar --test-range 6 --train-range 1-" + str(self.train_range) + " --data-path " + dataPath + " --inner-size 24 --save-file " + save_file +  " --gpu 0 --layer-def " + layersCfg + " --layer-params " + layersParams + " --epochs " + str(self.num_epochs) + " --test-freq " + testFreq
        else:
            self.filename_to_epochs[self.file_name] += self.num_epochs
            # if the model is already run, we need to load file
            command = "python convnet.py --data-provider cifar --test-range 6 --train-range 1-" + str(self.train_range) + " --data-path " + dataPath + " --inner-size 24 --save-file " + save_file +  " --gpu 0 --layer-def " + layersCfg + " --layer-params " + layersParams + " --epochs " + str(self.filename_to_epochs[self.file_name]) + " --test-freq " + testFreq + " --load-file " + save_file

        #so I guess it will only tests at the end right, so there will only one result string

        output = subprocess.check_output(command, shell=True)
        # print("+++++       Command = _" + command)
        # print("+++++       ml.convnet_cifar_freeze, In objective function, output from command: ")
        # print(output)

        losses_strings = output.split("Test output=")[1:]

        def get_val_loss(s):
            AveragesId = s.find("Averages-")
            output = s[:AveragesId]
            resultString = output.split(" ")[3][:-1]
            return float( resultString )

        val_losses = map(get_val_loss, losses_strings)
        print ":::::        ml.convnet_cifar_freeze, In objective function, after training, filtered, got these val_lossess:    ", val_losses
        

        best_validation_loss = 0 # well we dont even use this value for now

        return np.array([[best_validation_loss]]), val_losses
        
    def objective_function_test(self, x):        
        self.objective_function(x)
        return self.test_error