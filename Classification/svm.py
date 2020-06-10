import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
from sklearn import preprocessing 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import f1_score
from sklearn.metrics import jaccard_similarity_score

cancer_df = pd.read_csv('/home/vivek/ml/Classification/cell_samples.csv')

print(cancer_df.head())
ax = cancer_df[cancer_df['Class'] ==4][0:50].plot(kind='scatter',x='Clump',y='UnifSize',color='DarkBlue',label="malignant")
cancer_df[cancer_df['Class'] ==2][0:50].plot(kind='scatter',x='Clump',y='UnifSize',color='Yellow',label="benign",ax=ax)
plt.show()

cancer_df = cancer_df[pd.to_numeric(cancer_df['BareNuc'],errors='coerce').notnull()]
cancer_df['BareNuc'] = cancer_df['BareNuc'].astype('int')

features_df = cancer_df[['Clump', 'UnifSize', 'UnifShape', 'MargAdh', 'SingEpiSize', 'BareNuc', 'BlandChrom', 'NormNucl', 'Mit']]
X = np.asanyarray(features_df)

cancer_df['Class'] = cancer_df['Class'].astype('int')
y = np.asanyarray(cancer_df['Class'])
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)

clf = svm.SVC(kernel='rbf')
clf.fit(X_train,y_train)

yhat = clf.predict(X_test)
print('f1_score',f1_score(y_test, yhat, average='weighted') )
print('jaccard_similarity_score',jaccard_similarity_score(y_test, yhat))

