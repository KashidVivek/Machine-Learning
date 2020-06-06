import pandas as pd 
from matplotlib import pyplot as plt 
import numpy as np 
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import r2_score

df = pd.read_csv('/home/vivek/ml/Regression/FuelConsumptionCo2.csv')

c_df = df.copy()[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY', 'FUELCONSUMPTION_COMB','CO2EMISSIONS']]

msk = np.random.rand(len(df)) < 0.8
train = c_df[msk]
test = c_df[~msk]

model = linear_model.LinearRegression()
x = np.asanyarray(train[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY']])
y = np.asanyarray(train[['CO2EMISSIONS']])
model.fit(x,y)

print('Coefficients:', model.coef_)

y_hat = model.predict(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY']])
x = np.asanyarray(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY']])
y = np.asanyarray(test[['CO2EMISSIONS']])


print('Variance Score:', model.score(x,y))