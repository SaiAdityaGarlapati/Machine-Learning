# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 10:43:58 2018

@author: senthilkumar
"""

import numpy as np
from matplotlib import pyplot as plt

X=np.array([95,85,80,70,60])
m=len(X)
X=X.reshape(m,1)

Y=np.array([85,95,70,65,70])
Y=Y.reshape(m,1)

x_mean=np.mean(X)
y_mean=np.mean(Y)

num=0
den=0
for i in range(m):
    num+=((X[i]-x_mean)*(Y[i]-y_mean))
    den+=((X[i]-x_mean)**2)

b1=num/den

b0=y_mean-b1*x_mean  

y_pred1=b0+b1*X  

err=Y-y_pred1

avg_err=sum(((err)**2))/m
rmse1=np.sqrt(avg_err)

er1=sum((Y-y_mean)**2)
er2=sum((Y-y_pred1)**2)

R2=(1-(er2/er1))

plt.scatter(X,Y, c='#ef5423', label='Scatter Plot')
            
max_x = np.max(X)+10
min_x = np.min(X)-10
# Calculating line values x and y
x = np.linspace(min_x, max_x, 1000)
y = b0 + b1 * x
plt.plot(x, y, color='#58b970', label='Regression Line')

#
#ss_tot = 0
#ss_res = 0
#for i in range(m):
#    y_pred = b0 + b1 * X[i]
#    ss_tot += (Y[i] - y_mean) ** 2
#    ss_res += (Y[i] - y_pred) ** 2
#r2 = 1 - (ss_res/ss_tot)
#print("R2 Score")
#print(r2)

#rmse = 0
#for i in range(m):
#    y_pred = b0 + b1 * X[i]
#    rmse += (Y[i] - y_pred) ** 2
#rmse = np.sqrt(rmse/m)
#print("RMSE")
#print(rmse)
