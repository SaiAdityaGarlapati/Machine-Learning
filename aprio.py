# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 06:45:10 2018

@author: senthilku
"""
import numpy as np
import pandas as pd
from apyori import apriori
data1=pd.read_csv('store_data.csv')
store_data = pd.read_csv('store_data.csv', header=None)
data1.head()


records = []  
for i in range(0, 7501):  
    records.append([str(store_data.values[i,j]) for j in range(0, 20) 
    if str(store_data.values[i,j]) !='nan'])


association_rules = apriori(records, min_support=0.005, 
                            min_confidence=0.2, min_lift=3, 
                            min_length=2)  


association_results = list(association_rules) 

print(len(association_results))  
print(association_results[2]) 


for item in association_results:

    # first index of the inner list
    # Contains base item and add item
    pair = item[0] 
    items = [x for x in pair]
    print("Rule: " + items[0] + " -> " + items[1])

    #second index of the inner list
    print("Support: " + str(item[1]))

    #third index of the list located at 0th
    #of the third index of the inner list

    print("Confidence: " + str(item[2][0][2]))
    print("Lift: " + str(item[2][0][3]))
    print("=====================================")


