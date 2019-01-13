import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error

data = pd.read_csv('headbrain.csv')
print(data.shape)
data.head()

gender=data['Gender'].values
age = data['Age Range'].values
size = data['Head Size(cm^3)'].values
weight = data['Brain Weight(grams)'].values

# Ploting the scores as scatter plot
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(age, size, weight, color='#ef1234')
plt.show()

X = np.array([age, size,weight]).T
Y = np.array(gender)

# Model Intialization
reg = LogisticRegression()
# Data Fitting
reg = reg.fit(X, Y)
# Y Prediction
Y_pred = reg.predict(X)

# Model Evaluation
rmse = np.sqrt(mean_squared_error(Y, Y_pred))
r2 = reg.score(X, Y)

print("RMSE")
print(rmse)
print("R2 Score")
print(r2)