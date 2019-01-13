import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

x = np.array([95,85,80,70,60])

x = x.reshape(5,1)

y = np.array([85,95,70,65,70])

y = y.reshape(5,1)

xmean = np.mean(x)
ymean = np.mean(y)
xval = x-xmean
yval = y-ymean
dem  = np.sum(np.power(xval,2))
b1 = np.sum(xval*yval)/dem
b0= ymean-b1*(xmean)

Y = b0+(b1*x)
e1 = np.sum(np.power(y-ymean,2))
e2 = np.sum(np.power((y-Y),2))
R = 1-(e2/e1)
RMSE = np.sqrt((np.sum(np.power(y-Y,2)))/b1)
print(RMSE)

head = pd.read_csv("D:/data set/python/headbrain.csv")
print(head)

