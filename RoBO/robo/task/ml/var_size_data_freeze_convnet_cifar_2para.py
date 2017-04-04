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



class VarSizeDataConvNetCifar(BaseTask):

    # def __init__(self, train, train_targets,
    #              valid, valid_targets,
    #              test, test_targets,
    #              n_classes, num_epochs=500,
    #              save=False, file_name=None):
    def __init__(self, save=False, file_name=None, num_epochs=500, train_range=5):
        
        # the last dimension of input is always the data-size
        # currently, its the [7]

        self.num_epochs = num_epochs
        # self.save = save
        self.file_name = file_name
        self.base_name = "VarSizeFreezeConvNetCifar"
        self.is_old = False
        # self.filename_to_epochs = dict()
        # self.train_range = train_range
        
        # 1 Dim Learning Rate:
        # 2 Dim L2 regularization: 0 to 1
        # 3 Dim Batch size: 20 to 2000
        # 4 Dim Dropout rate: 0 to 0.75
        # 5 Dim L1 regularization: 0.1 to 20
        # 6 Dim Epochs Number: 1 to 100
        # X_lower = np.array([np.log(1e-6), 0.0, 20, 0, 0.1, 1])
        
        X_lower = np.array([0.00001, 0.00001, 10])
        
        self.params = X_lower

        #X_lower = np.array([np.log(1e-6), 0.0, 1000, 0, 0.1, 1])
        #X_upper = np.array([np.log(1e-1), 1.0, 2000, 0.75, 20, 100])
        #X_upper = np.array([np.log(1e-1), 1.0, 2000, 0.75, 20, 10])
        # X_upper = np.array([np.log(1e-1), 1.0, 2000, 0.75, 20, 7])
        X_upper = np.array([0.01, 0.02, 500])

        #X_upper = np.array([np.log(1e-1), 1.0, 2000, 0.75, 20, 3])
        super(VarSizeDataConvNetCifar, self).__init__(X_lower, X_upper)


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

        print 'in objective_function x: ', x
        #well now if its even an old model, we don't need to care about the past data 
        #and just learn from fresh.


        # print("filename = " + self.file_name)

        # dir_path = os.path.dirname(os.path.realpath(__file__))
        

        # return [ np.random.rand() ]


        dir_path = ROOT_ML_DIR + "RoBO/examples/"

        os.chdir(ROOT_ML_DIR + "/cuda-convnet2-vu")
        dataPath = ROOT_ML_DIR + "/data/vu-cifar-10"
        save_file = os.path.join(dir_path,self.file_name + ".data")
        layersCfg = ROOT_ML_DIR + "/Spearmint-EI/examples/convnetcifar/layers-80sec.cfg"
        layersParams = os.path.join(dir_path, self.file_name + ".cfg")
        # testFreq = self.num_epochs * 5
        # testFreq = str(self.train_range)


        layersParamsTemplatePath = ROOT_ML_DIR + "RoBO/examples/dataFreeze/layer-params-template_2para.cfg"




        
        #write the layersParams file
        # if os.path.exists(save_file):
        #     subprocess.call("rm -rf " + save_file, shell=True)
        #     subprocess.call("rm " + layersParams, shell=True)

        template = open(layersParamsTemplatePath,"r").read()
        epsW = x[0][0]
        epsB = x[0][1]
        open(layersParams,"w").write(template % ( epsW, epsB, epsW, epsB, epsW, epsB, epsW, epsB, epsW, epsB ))



        dataSize = x[0][2]

        num_epochs = int( self.num_epochs * 500 / dataSize )
        dataSize = int(dataSize+0.00000000001)
        dataSize = min(dataSize, 500)
        testRange = str( max(500 + dataSize/5,502) )
        testRange = "600"
        testFreq = str(4 * dataSize)     #Lets just make it 3 * dataSize for now

        if not self.is_old:
            #its a new model, we need to write the layersParams file
            if os.path.exists(save_file):
                temp = subprocess.check_output("rm -rf " + save_file, shell=True)
                # temp = subprocess.check_output("rm " + layersParams, shell=True)

            
            # self.filename_to_epochs[self.file_name] = self.num_epochs
            command = "python convnet.py --data-provider cifar --test-range 501-" + testRange + " --train-range 1-" + str(dataSize) + " --data-path " + dataPath + " --inner-size 24 --save-file " + save_file +  " --gpu 0 --layer-def " + layersCfg + " --layer-params " + layersParams + " --epochs " + str(num_epochs) + " --test-freq " + testFreq
            # command = "python convnet.py --data-provider cifar --test-range 6 --train-range 1-" + str(self.train_range) + " --data-path " + dataPath + " --inner-size 24 --save-file " + save_file +  " --gpu 0 --layer-def " + layersCfg + " --layer-params " + layersParams + " --epochs " + str(self.num_epochs) + " --test-freq " + testFreq
        else:
            # self.filename_to_epochs[self.file_name] += self.num_epochs
            # if the model is already run, we need to load file
            command = "python convnet.py --data-provider cifar --test-range 501-" + testRange + "  --train-range 1-" + str(dataSize) + " --data-path " + dataPath + " --inner-size 24 --save-file " + save_file +  " --gpu 0 --layer-def " + layersCfg + " --layer-params " + layersParams + " --epochs " + str(num_epochs) + " --test-freq " + testFreq + " --load-file " + save_file



        # command = "python convnet.py --data-provider cifar --test-range 501-" + testRange + " --train-range 1-" + str(dataSize) + " --data-path " + dataPath + " --inner-size 24 --save-file " + save_file +  " --gpu 0 --layer-def " + layersCfg + " --layer-params " + layersParams + " --epochs " + str(num_epochs) + " --test-freq " + testFreq
        
        output = subprocess.check_output(command, shell=True)
        # print("+++++       Command = _" + command)
        # print("+++++       ml.convnet_cifar_freeze, In objective function, output from command: ")
        # print(output)
        open(layersParams+".log","a").write(command+"\n"+output+"\n\n\n\n")

        losses_strings = output.split("STOPPING TRAINING")[1]

        # def get_val_loss(s):
        #     AveragesId = s.find("Averages-")
        #     output = s[:AveragesId]
        #     resultString = output.split(" ")[3][:-1]
        #     return float( resultString )

        stoppingStringId = losses_strings.find("logprob")
        val_loss = losses_strings[stoppingStringId:].split(", ")[1]

        print ":::::        ml.convnet_cifar_freeze, In objective function, after training, filtered, got this val_loss:    ", val_loss
        
        return [float(val_loss),]
        
    def objective_function_test(self, x):        
        self.objective_function(x)
        return self.test_error