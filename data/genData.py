import os
import cPickle
import pprint
import numpy as np

pp = pprint.PrettyPrinter(indent=4)
ORIGINAL_DATA_PATH = "cifar-10-py-colmajor"
OUTPUT_DATA_PATH = "vu-cifar-10"

def pickle(filename, data):
    with open(filename, "w") as fo:
        cPickle.dump(data, fo, protocol=cPickle.HIGHEST_PROTOCOL)

def unpickle(filename):
    fo = open(filename, 'r')
    contents = cPickle.load(fo)
    fo.close()
    return contents

def printDataToObjects():


    # print("::::::         batches.meta        ::::::")
    # pp.pprint(unpickle(ORIGINAL_DATA_PATH+"/batches.meta"))

    print("::::::         vu_batches.meta        ::::::")
    pp.pprint(unpickle(OUTPUT_DATA_PATH+"/batches.meta"))

    # print("::::::         data_batch_1        ::::::")
    # pp.pprint(unpickle(ORIGINAL_DATA_PATH+"/data_batch_1"))
    # print "Data shape = ", unpickle(ORIGINAL_DATA_PATH + "/data_batch_1")['data'].shape


    print("::::::         vu_data_batch_1        ::::::")
    pp.pprint(unpickle(OUTPUT_DATA_PATH + "/data_batch_1"))
    print "Data shape = ", unpickle(OUTPUT_DATA_PATH + "/data_batch_1")['data'].shape

    # print("::::::           data_batch_6        ::::::")
    # pp.pprint(unpickle(ORIGINAL_DATA_PATH + "/data_batch_6"))

def genVuCifar10():
    for i in range(1,7):
        dataBatch = unpickle(ORIGINAL_DATA_PATH + "/data_batch_" + str(i))
        # allData = np.array(dataBatch['data'])
        allData = dataBatch['data']

        for j in range(100):
            #index of this new smaller batch
            bIndex = (i-1)*100 + (j+1)
            outputDataFile = OUTPUT_DATA_PATH + "/data_batch_" + str(bIndex)
            print "Current batch index = ", bIndex

            newBatch = dict()
            newBatch['batch_label'] = "Batch #" + str(bIndex)
            newBatch['data'] = allData[:,j*100:(j+1)*100]

            newBatch['filenames'] = dataBatch['filenames'][j*100:(j+1)*100]
            newBatch['labels'] = dataBatch['labels'][j*100:(j+1)*100]

            # pp.pprint(newBatch)

            pickle(outputDataFile,newBatch)

def genVuCifarBatchsMeta():
    batchMeta = unpickle(ORIGINAL_DATA_PATH + "/batches.meta")
    # print "data_mean length = ", len(batchMeta['data_mean'])
    batchMeta['num_cases_per_batch'] = 100
    pickle(OUTPUT_DATA_PATH + "/batches.meta", batchMeta)


genVuCifar10()
# genVuCifarBatchsMeta()
printDataToObjects()