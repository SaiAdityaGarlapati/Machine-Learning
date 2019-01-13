# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np

x=np.array([95,85,80,70,60])
y=np.array([85,95,70,65,70])

print(x)
print(y)

x1=np.reshape(x,(5,1))
y1=np.reshape(y,(5,1))

print(x1)
print(y1)

x0=np.mean(x)
y0=np.mean(y)

y2=y-y0
x2=x-x0
b1=sum(x2*y2)/sum(x2*x2)

print(b1)

b0=77-0.6438*78

print(b0)