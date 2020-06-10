import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
from sklearn import preprocessing 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import log_loss

churn_df = pd.read_csv('/home/vivek/ml/Classification/ChurnData.csv')

churn_df = churn_df.copy()[['tenure','age','address','income','ed','employ','equip','callcard','wireless','churn']]
churn_df['churn'] = churn_df['churn'].astype('int')

X = np.asanyarray(churn_df[['tenure','age','address','income','ed','employ','equip']])
print(X[0:5])
Y = np.asanyarray(churn_df['churn'])

X = preprocessing.StandardScaler().fit(X).transform(X)
X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size = 0.2,random_state = 4)

LR = LogisticRegression(C=0.01,solver='liblinear').fit(X_train,y_train)

yhat = LR.predict(X_test)
yhat_prob = LR.predict_proba(X_test)
print('jaccard_similarity_score:',jaccard_similarity_score(y_test, yhat))
print('log_loss',log_loss(y_test, yhat_prob))










