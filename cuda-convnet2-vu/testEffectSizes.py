from statistics import mean, stdev
from math import sqrt
import pickle
import pprint
import matplotlib.pyplot as plt
from PlotLossOverData import train



def effectSize(c0, c1):
	return  (mean(c0) - mean(c1)) / (sqrt((stdev(c0) ** 2 + stdev(c1) ** 2) / 2))

lossesOverSmallSize = {}

for configId in range(10):

	# config20 = []
	# config21 = []
	losses = []
	for i in range(10):
		loss, _ = train(20,"config_" + str(configId))
		losses.append(loss)

	# print("Done config20, losses = ", config20)

	# for i in range(10):
	# 	loss, _ = train(20,"config_21")
	# 	config21.append(loss)

	# print("Done config21, losses = ", config21)

	lossesOverSmallSize[str(configId)] = losses

	print("Done config " + str(configId) + ", losses = " + str(losses) )
	
	pickle.dump(lossesOverSmallSize, open("/data/ml/Vu_Notes/curveDatas/lossesOverSmallSize.pkl","wb"))

# print(effectSize(config20,config21))