import numpy as np 
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.externals.six import StringIO
import 	pydotplus
import matplotlib.image as mpimg
from sklearn import tree

df = pd.read_csv('/home/vivek/ml/Classification/drug200.csv',delimiter=',')

x = df[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values
le_sex = preprocessing.LabelEncoder()
le_sex.fit(['F','M'])
x[:,1] = le_sex.transform(x[:,1])

le_bp = preprocessing.LabelEncoder()
le_bp.fit(['LOW','NORMAL','HIGH'])
x[:,2] = le_bp.transform(x[:,2])

le_chol = preprocessing.LabelEncoder()
le_chol.fit(['NORMAL','HIGH'])
x[:,3] = le_chol.transform(x[:,3])

y = df['Drug']

X_trainset, X_testset, y_trainset, y_testset = train_test_split(x, y, test_size=0.3, random_state=3)

tree_ = DecisionTreeClassifier(criterion='entropy',max_depth=4)
tree_.fit(X_trainset,y_trainset)

predTree = tree_.predict(X_testset)
print (predTree [0:5])
print (y_testset [0:5])

print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_testset, predTree))


dot_data = StringIO()
filename = "drugtree.png"
featureNames = df.columns[0:5]
targetNames = df["Drug"].unique().tolist()
out=tree.export_graphviz(tree_,feature_names=featureNames, out_file=dot_data, class_names= np.unique(y_trainset), filled=True,  special_characters=True,rotate=False)  
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png(filename)
img = mpimg.imread(filename)
plt.figure(figsize=(100, 200))
plt.imshow(img,interpolation='nearest')






