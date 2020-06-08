'''

Quadratic or cublic regressions
polynomial regression

'''
import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np 
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from sklearn.metrics import r2_score

df = pd.read_csv('/home/vivek/ml/Regression/FuelConsumptionCo2.csv')

cdf = df.copy([['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']])

# plt.scatter(cdf.ENGINESIZE,cdf.CO2EMISSIONS,color='blue')
# plt.xlabel('ENGINESIZE')
# plt.ylabel('CO2EMISSIONS')
# plt.show() 

msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]

train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])

test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])


poly = PolynomialFeatures(degree=3)
train_x_poly = poly.fit_transform(train_x)
print(train_x_poly)

clf = linear_model.LinearRegression()
train_y = clf.fit(train_x_poly,train_y)
print('Coefficients: ',clf.coef_)
print('Intercept: ',clf.intercept_)

plt.scatter(train.ENGINESIZE,train.CO2EMISSIONS,color='blue')
XX = np.arange(0.0,10.0,0.1)
yy = clf.intercept_[0] + clf.coef_[0][1] * XX + clf.coef_[0][2] * np.power(XX,2) + clf.coef_[0][3] * np.power(XX,3)
plt.plot(XX,yy,'-r')
plt.xlabel('ENGINESIZE')
plt.ylabel('CO2EMISSIONS')
plt.show()

test_x_poly = poly.fit_transform(test_y)
test_y_ = clf.predict(test_x_poly)

print('Mean Absolute Error:', np.mean(np.absolute(test_y_- test_y)))
print('MSE:', np.mean((test_y_- test_y)**2))
print('R2 Score:', r2_score(test_y_,test_y))