import subprocess, os
import time, pickle
from statistics import mean, stdev
from math import sqrt
import pprint
import matplotlib.pyplot as plt


def train(datasize,fileName):
    os.system("rm -rf /data/ml/RoBO/examples/temp_configs_VarSizeFreezeConvNetCifar/" + fileName + ".data")
    start = time.time()

    # cfgFile = "cuda-convnet2-vu/layers/cifar/layer-params-80sec-spearminted.cfg"
    
    cfgFile = "Vu_Notes/curveDatas/configs/" + fileName + ".cfg"

    testFreqMultiple = 4
    # testRange = str(max(500 + datasize/5,502))
    testRange = str(600)
    command = "python convnet.py --data-provider cifar --test-range 501-" + testRange + " --train-range 1-" + str(datasize) + " --data-path /data/ml//data/vu-cifar-10 --inner-size 24 --save-file /data/ml/RoBO/examples/temp_configs_VarSizeFreezeConvNetCifar/" + fileName + ".data --gpu 0 --layer-def /data/ml//Spearmint-EI/examples/convnetcifar/layers-80sec.cfg --layer-params /data/ml/" + cfgFile +  " --epochs 500 --test-freq " + str(datasize*testFreqMultiple)
    # print(command)
    temp = subprocess.check_output(command, shell=True)

    # print("Type of output = ", type(temp))

    num_iters = len(temp.decode().split("==Test output")) - 1 

    losses_strings = temp.decode().split("STOPPING TRAINING")[1]

    stoppingStringId = losses_strings.find("logprob")
    val_loss = losses_strings[stoppingStringId:].split(", ")[1]
    # print ":::::        ml.var_size_data_convnet_cifar_freeze, In objective function, got this val_loss:    ", val_loss
    
    # return [float(val_loss),]
    time_taken = time.time()-start

    print(datasize, "\t\t", time_taken, "\t\t", val_loss, "\t\t", num_iters, " tests")

    return float(val_loss), time_taken

AVERAGE_RANGE  = 6
def runningAverage(losses):
    sum = 0
    # RANGE = 6
    for loss in losses[-AVERAGE_RANGE:]:
        sum += loss
    return sum / AVERAGE_RANGE

def getLossOverData():
    allData = []


    for j in range(10,97):

        test = {
            "datasizes": [],
            "losses": [],
            "timeTakens": []    
        }

        for i in range(1,50):
            datasize = i*10
            test["datasizes"].append(datasize)
            newLoss, newTime = train(datasize, "config_" + str(j))
            test["losses"].append(newLoss)
            test["timeTakens"].append(newTime)
            averageLoss = runningAverage(test["losses"]) 
            if newLoss > averageLoss and len(test["losses"]) >= AVERAGE_RANGE:
                print("newLoss = ", newLoss, " average loss = ", averageLoss)
                break

        allData.append(test)


        pickle.dump(allData, open("/data/ml/Vu_Notes/curveDatas/data3.pkl","wb"))
        print("Done config ", j , ", test = ", test)

    print("Done, allData = ", allData)

def getLossAtFullSize(startIndex, endIndex, saveFile):
    allData = {}
    for i in range(startIndex, endIndex):
        newLoss, newTime = train(500, "config_" + str(i))
        allData[str(i)] = {
            "loss": newLoss,
            "timeTaken": newTime
        }

        pickle.dump(allData, open("/data/ml/Vu_Notes/curveDatas/" + saveFile,"wb"))
        print("Done config ", i)

    print("Done, allData = ", allData)

def effectSize(c0, c1):
    return  (mean(c0) - mean(c1)) / (sqrt((stdev(c0) ** 2 + stdev(c1) ** 2) / 2))

def getLossesOverSmallSize(startIndex, endIndex, datasize, fileName):
    lossesOverSmallSize = {}

    for configId in range(startIndex, endIndex):

        # config20 = []
        # config21 = []
        losses = []
        for i in range(10):
            loss, _ = train(datasize, "config_" + str(configId))
            losses.append(loss)

        # print("Done config20, losses = ", config20)

        # for i in range(10):
        #   loss, _ = train(20,"config_21")
        #   config21.append(loss)

        # print("Done config21, losses = ", config21)

        lossesOverSmallSize[str(configId)] = losses

        print("Done small size loss for config " + str(configId) + ", losses = " + str(losses) )
        
        pickle.dump(lossesOverSmallSize, open("/data/ml/Vu_Notes/curveDatas/" + fileName ,"wb"))

# getLossAtFullSize(30, 50, "lossesAtFullSize30_50.pkl")
# getLossesOverSmallSize(30, 50, 20, "lossesOverSize20_index30_50.pkl")

getLossesOverSmallSize(30, 50, 40, "lossesOverSize40_index30_50.pkl")

print("##############################       Done size 40")                                                  #DONE
getLossesOverSmallSize(30, 50, 50, "lossesOverSize50_index30_50.pkl")

print("##############################       Done size 50")
getLossesOverSmallSize(30, 50, 60, "lossesOverSize60_index30_50.pkl")


print("##############################       Done size 60")
getLossesOverSmallSize(30, 50, 70, "lossesOverSize70_index30_50.pkl")


print("##############################       Done size 70")

getLossesOverSmallSize(30, 50, 80, "lossesOverSize80_index30_50.pkl")




# getLossAtFullSize(30, 50, "lossesAtFullSize30_50_run2.pkl")

# getLossAtFullSize(50, 60, "lossesAtFullSize50_60.pkl")
# getLossesOverSmallSize(50, 60, 30, "lossesOverSize30_index50_60.pkl")
# getLossAtFullSize(50, 60, "lossesAtFullSize50_60_run2.pkl")
# getLossOverData()
# train(160,"config_0")
# test(2)
# test(3)
# test(4)
# test(5)