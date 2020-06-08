import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np 
import pandas as pd 
from scipy.optimize import curve_fit
# x = np.arange(-5.0, 5.0, 0.1)


def sigmoid(x, Beta_1, Beta_2):
     y = 1 / (1 + np.exp(-Beta_1*(x-Beta_2)))
     return y

# y = 1-4/(1+np.power(3,x-2))#1 * (x**3) + 1*(x**2) + 1*x + 3
# y_noise = 20 * np.random.normal(size=x.size)
# ydata = y + y_noise

# # plt.plot(x,ydata,'bo')
# plt.plot(x,y)
# plt.xlabel('Dependent Variable')
# plt.ylabel('Independent Variable')
# plt.show()

df = pd.read_csv('/home/vivek/ml/Regression/china_gdp.csv')

print(df.head(3))
plt.figure(figsize = (8,5))
x_data, y_data = (df['Year'].values,df['Value'].values)
# plt.plot(x_data,y_data,'ro')
# plt.ylabel('GDP')
# plt.xlabel('Year')
# plt.show()

beta1 = 0.10
beta2 = 1990.0

y = 1/ (1 + np.exp(-beta1*(x_data-beta2)))
y_pred = y

plt.plot(x_data,y_pred*15000000000000)
plt.plot(x_data,y_data,'ro')
plt.show()

xdata = x_data/max(x_data)
ydata = y_data/max(y_data)


popt, pcov = curve_fit(sigmoid,xdata,ydata)
x = np.linspace(1960, 2015, 55)
x = x/max(x)
plt.figure(figsize=(8,5))
y = sigmoid(x, *popt)
plt.plot(xdata, ydata, 'ro', label='data')
plt.plot(x,y, linewidth=3.0, label='fit')
plt.legend(loc='best')
plt.ylabel('GDP')
plt.xlabel('Year')
plt.show()


