import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import matplotlib.ticker as ticker
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

df = pd.read_csv('/home/vivek/ml/Classification/teleCust1000t.csv')

# print(df['custcat']).value_counts()
# df.hist(column='income', bins=50)

print(df.columns)

X = df[['region', 'tenure', 'age', 'marital', 'address', 'income', 'ed','employ', 'retire', 'gender', 'reside']]
Y = df['custcat'].values 

X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.2, random_state = 4)
print('Train set:',X_train.shape,Y_train.shape)
print('Test set:',X_test.shape,Y_test.shape)

k = 6
clf = KNeighborsClassifier(n_neighbors = k).fit(X_train,Y_train)
y_hat = clf.predict(X_test)

print("Train set Accuracy: ", metrics.accuracy_score(Y_train, clf.predict(X_train)))
print("Test set Accuracy: ", metrics.accuracy_score(Y_test, y_hat))

Ks = 10
mean_acc = np.zeros((Ks-1))
std_acc = np.zeros((Ks-1))
ConfustionMx = [];
for n in range(1,Ks):
    
    #Train Model and Predict  
    neigh = KNeighborsClassifier(n_neighbors = n).fit(X_train,Y_train)
    yhat=neigh.predict(X_test)
    mean_acc[n-1] = metrics.accuracy_score(Y_test, yhat)

    
    std_acc[n-1]=np.std(yhat==Y_test)/np.sqrt(yhat.shape[0])

print(mean_acc)
plt.plot(range(1,Ks),mean_acc,'g')
plt.fill_between(range(1,Ks),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10)
plt.legend(('Accuracy ', '+/- 3xstd'))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Nabors (K)')
plt.tight_layout()
plt.show()



