# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 10:25:43 2018

@author: Sai Aditya Garlapati
"""


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error

import pandas as pd
import numpy as np

loan=pd.read_csv('train.csv')

k=loan[(loan.Loan_status=='Y']).count
loan.dropna()

income=loan['ApplicantIncome'].values
amount=loan['LoanAmount'].values
term=loan['Loan_Amount_Term'].values
#dependents=loan['Dependents'].values
status=loan['Loan_Status'].values


x=np.array([income,amount,term])
y=np.array(status)

reg=LogisticRegression()

reg=reg.fit(x,y)

y_pred=reg.predict(x)

rmse=np.sqrt(mean_squared_error(y,y_pred))
r2=reg=reg.score(x,y)
print('rmse',+rmse)
print('r2',+r2)