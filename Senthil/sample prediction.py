# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 14:30:15 2018

@author: Sai Aditya Garlapati
"""

import pandas as pd
import numpy as np

head_brain=pd.read_csv("headbrain.csv")

x=np.array(head_brain['Head Size(cm^3)']).reshape(237,1)
y=np.array(head_brain['Brain Weight(grams)']).reshape(237,1)

head_mean=np.mean(head_brain['Head Size(cm^3)'])
brain_mean=np.mean(head_brain['Brain Weight(grams)'])

y2=y-brain_mean
x2=x-head_mean

b1=sum(x2*y2)/sum(x2*x2)

b0= brain_mean - b1*head_mean

print(b1)
print(b0)


Yp = b0+(b1*x)
e1 = np.sum(np.power(y-brain_mean,2))
e2 = np.sum(np.power((y-Yp),2))
R = 1-(e2/e1)
RMSE = np.sqrt((np.sum(np.power(y-Yp,2)))/237)
print(RMSE)
print(e1)
print(e2)
print(R)

#import seaborn as sns
#sns.set(color_codes=True)
#sns.regplot(x=x1, y=y1)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

reg =LinearRegression()
reg =reg.fit(x,y)
y_pred = reg.predict(x)

rmse=np.sqrt(mean_squared_error(y,y_pred))
r2=reg.score(x,y)

print(rmse)
print(r2)
