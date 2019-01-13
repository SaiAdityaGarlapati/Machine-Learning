# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 14:33:56 2018

@author: senthil kumar
"""
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

traindata=pd.read_csv('train.csv')
testdata=pd.read_csv('test.csv')

print(traindata['Loan_Status'].value_counts())

traindata1=traindata.dropna()
testdata1=testdata.dropna()

traindata1.drop(['Loan_ID'],axis=1,inplace=True)
testdata1.drop(['Loan_ID'],axis=1,inplace=True)

#df=traindata.drop(['Loan_ID'],axis=1)
#TT=traindata1.iloc[:,1:14]

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
LE=LabelEncoder()
traindata1.iloc[:,0]=LE.fit_transform(traindata1['Gender'])
traindata1.iloc[:,1]=LE.fit_transform(traindata1['Married'])
traindata1.iloc[:,2]=LE.fit_transform(traindata1['Dependents'])
traindata1.iloc[:,3]=LE.fit_transform(traindata1['Education'])
traindata1.iloc[:,4]=LE.fit_transform(traindata1['Self_Employed'])
traindata1.iloc[:,10]=LE.fit_transform(traindata1['Property_Area'])
traindata1.iloc[:,11]=LE.fit_transform(traindata1['Loan_Status'])

testdata1.iloc[:,0]=LE.fit_transform(testdata1['Gender'])
testdata1.iloc[:,1]=LE.fit_transform(testdata1['Married'])
testdata1.iloc[:,2]=LE.fit_transform(testdata1['Dependents'])
testdata1.iloc[:,3]=LE.fit_transform(testdata1['Education'])
testdata1.iloc[:,4]=LE.fit_transform(testdata1['Self_Employed'])
testdata1.iloc[:,10]=LE.fit_transform(testdata1['Property_Area'])


out=traindata1['Loan_Status']
X=traindata1.drop(['Loan_Status'],axis=1)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

Xtrain, Xval , Ytrain, Yval = train_test_split(X,out,test_size=0.2,random_state=60)

lreg=LogisticRegression()
lreg.fit(Xtrain,Ytrain)
res=lreg.predict(Xval)
#res_test=lreg.predict(testdata1)

acc=accuracy_score(Yval,res)

print('Intial accuracy with dropping all NA value is :', acc)


##ANN
#inp_size=11
#hid_size=6
#out_size=1
#
#
#w1=np.random.randn(inp_size,hid_size)
#w2=np.random.randn(hid_size,out_size)
#
#
#
#def sigmoid(z):
#    sig=1/(1+np.exp(-z))
#    return sig
#
#def forward(x):
#    z1=np.dot(x,w1)
#    y1=sigmoid(z1)
#    z2=np.dot(y1,w2)
#    out=sigmoid(z2)
#    return out,y1
#
#def backward(x,y,w1,w2,y1,out):
#    out_err=y-out
#    out_delta=out_err*(out*(1-out))
#    
#    y1_err=out_delta.dot(w2.T)
#    #y1_err=np.dot(out_delta,w2.T)
#    y1_delta=y1_err*(y1*(1-y1))
#    
#    #w1+=x.T.dot(y1_delta)
#    #w2+=y1.T.dot(out_delta)
#    w1+=np.dot(x.T,y1_delta)
#    w2+=np.dot(y1.T,out_delta)
#    
#    return w1,w2
#
#def train(x,y):
#    [out,y1]=forward(x)
#    [W1,W2]=backward(x,y,w1,w2,y1,out)
#    return W1,W2,out
#        
#def predict(tes):
#    tes=tes/np.amax(tes)
#    [out,y1]=forward(tes)
#    pred=out*100
#    return pred
#
#loss=np.zeros([100,1],dtype=float)
#for i in range(99):
#    [w1,w2,out]=train(Xtrain,Ytrain)
#    loss[i]=np.mean(np.square(Ytrain-out))
#    print(loss[i])
#
#
#plt.plot(loss)
#plt.show()
#
#
#pred=predict(Xval)
#print(pred)
#






