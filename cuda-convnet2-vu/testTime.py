import subprocess, os
import time


def test(datasize):
    os.system("rm -rf /data/ml/RoBO/examples/temp_configs_VarSizeFreezeConvNetCifar/config_3.data")
    start = time.time()

    # cfgFile = "cuda-convnet2-vu/layers/cifar/layer-params-80sec-spearminted.cfg"
    
    cfgFile = "RoBO/examples/temp_configs_VarSizeFreezeConvNetCifar/config_3.cfg"

    testFreqMultiple = 4
    testRange = str(max(500 + datasize/5,502))
    testRange = str(600)
    command = "python convnet.py --data-provider cifar --test-range 501-" + testRange + " --train-range 1-" + str(datasize) + " --data-path /data/ml//data/vu-cifar-10 --inner-size 24 --save-file /data/ml/RoBO/examples/temp_configs_VarSizeFreezeConvNetCifar/config_3.data --gpu 0 --layer-def /data/ml//Spearmint-EI/examples/convnetcifar/layers-80sec.cfg --layer-params /data/ml/" + cfgFile +  " --epochs 500 --test-freq " + str(datasize*testFreqMultiple)
    print command
    temp = subprocess.check_output(command, shell=True)

    num_iters = len(temp.split("==Test output")) - 1 

    losses_strings = temp.split("STOPPING TRAINING")[1]

    
    stoppingStringId = losses_strings.find("logprob")
    val_loss = losses_strings[stoppingStringId:].split(", ")[1]

    # print ":::::        ml.var_size_data_convnet_cifar_freeze, In objective function, got this val_loss:    ", val_loss
    
    return [float(val_loss),]
    print datasize, "\t\t", time.time()-start, "\t\t", val_loss, "\t\t", num_iters, " tests"



for i in range(1,50):
    test(i*10)

# test(2)
# test(3)
# test(4)
# test(5)