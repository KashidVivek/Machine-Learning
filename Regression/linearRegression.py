import pandas as pd 
from matplotlib import pyplot as plt 
import numpy as np 
from sklearn import linear_model
from sklearn.metrics import r2_score

df = pd.read_csv('/home/vivek/ml/Regression/FuelConsumptionCo2.csv')
print(df.head())
print(df.describe())

c_df = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
print(c_df.head(8))

# viz = c_df[['CYLINDERS','ENGINESIZE', 'CO2EMISSIONS', 'FUELCONSUMPTION_COMB']]
# viz.hist()
# plt.show()

# plt.scatter(c_df.FUELCONSUMPTION_COMB, c_df.CO2EMISSIONS, color = 'blue')
# plt.xlabel('FUELCONSUMPTION_COMB')
# plt.ylabel('CO2EMISSIONS')
# plt.show()

# plt.scatter(c_df.CYLINDERS,c_df.CO2EMISSIONS, color='red')
# plt.xlabel('CYLINDERS')
# plt.ylabel('CO2EMISSIONS')
# plt.show()

msk = np.random.rand(len(df)) < 0.8
train = c_df[msk]
test = c_df[~msk]

# plt.scatter(train.ENGINESIZE,train.CO2EMISSIONS, color='red')
# plt.xlabel('CYLINDERS')
# plt.ylabel('CO2EMISSIONS')
# plt.show()

reg = linear_model.LinearRegression()
train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])
reg.fit(train_x,train_y)

print('Regression Coeff: ', reg.coef_)
print('Bias: ', reg.intercept_)

plt.scatter(train.ENGINESIZE,train.CO2EMISSIONS, color='blue')
plt.plot(train_x,reg.coef_[0][0] * train_x + reg.intercept_[0], '-r')
plt.xlabel('ENGINESIZE')
plt.ylabel('CO2EMISSIONS')
plt.show()

test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])
test_y_hat = reg.predict(test_x)

print('Mean Absolute Error:', np.mean(np.absolute(test_y_hat- test_y)))
print('MSE:', np.mean((test_y_hat- test_y)**2))
print('R2 Score:', r2_score(test_y_hat,test_y))















