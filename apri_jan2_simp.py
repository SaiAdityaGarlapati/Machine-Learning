# -*- coding: utf-8 -*-
"""
Created on Mon Dec 31 19:06:26 2018

@author: senthilku
"""

import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

data=pd.read_excel('Online Retail.xlsx')

data['Description'] = data['Description'].str.strip()
data.isnull().sum()

bask= (data[data['Country'] =="France"]
            .groupby(['InvoiceNo', 'Description'])['Quantity']
            .sum().unstack())

basket1 = (data[data['Country'] =="France"]
            .groupby(['InvoiceNo', 'Description'])['Quantity']
            .sum().unstack().fillna(0))

def encode_units(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1
basket_sets = basket1.applymap(encode_units)

basket_sets.drop('POSTAGE', inplace=True, axis=1)

frequent_itemsets = apriori(basket_sets, min_support=0.07, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="lift")


top=rules[ (rules['lift'] >= 6) &
       (rules['confidence'] >= 0.8) ]