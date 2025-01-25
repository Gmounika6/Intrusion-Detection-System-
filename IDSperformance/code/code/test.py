import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import normalize
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

accuracy = []
precision = []
recall = []
fscore = []

labels = ['Benign' 'Bot' 'Brute Force -Web' 'Brute Force -XSS' 'Infilteration' 'SQL Injection']

dataset = pd.read_csv("test.csv")
dataset.fillna(0, inplace = True)
unique, count = np.unique(dataset['Label'], return_counts=True);
print(unique)
print(count)

dataset.drop(['Timestamp'], axis = 1,inplace=True)
dataset.drop(['Flow Byts/s'], axis = 1,inplace=True)
dataset.drop(['Flow Pkts/s'], axis = 1,inplace=True)
le = LabelEncoder()
dataset['Label'] = pd.Series(le.fit_transform(dataset['Label'].astype(str)))
print(dataset)
print(dataset.shape)

dataset = dataset.values
X = dataset[:,0:dataset.shape[1]-1]
Y = dataset[:,dataset.shape[1]-1]

indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X = X[indices]
Y = Y[indices]
print(X)
print(Y)

sm = SMOTE(random_state=42)
X, Y = sm.fit_resample(X, Y)
for i in range(0,6):
    total = sum(Y == i)
    print(total)


def calculateMetrics(y_test,predict,name):
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100
    a = accuracy_score(y_test,predict)*100
    accuracy.append(a)    
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    print(name+" Accuracy  : "+str(a))
    print(name+" Precision : "+str(p))
    print(name+" Recall    : "+str(r))
    print(name+" FSCORE    : "+str(f))

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
knn_cls = KNeighborsClassifier(n_neighbors = 6) 
knn_cls.fit(X_train, y_train)
predict = knn_cls.predict(X_test)
calculateMetrics(y_test,predict,"KNN Algorithm")


rf_cls = RandomForestClassifier(n_estimators=2,criterion="entropy",max_features="sqrt") 
rf_cls.fit(X_train, y_train)
predict = rf_cls.predict(X_test)
calculateMetrics(y_test,predict,"Random Forest Algorithm")

dt_cls = DecisionTreeClassifier(max_depth=2,criterion="entropy") 
dt_cls.fit(X_train, y_train)
predict = dt_cls.predict(X_test)
calculateMetrics(y_test,predict,"Decision Tree Algorithm")

ada = AdaBoostClassifier()
ada.fit(X_train, y_train)
predict = ada.predict(X_test)
calculateMetrics(y_test,predict,"AdaBoost Algorithm")


gb = GradientBoostingClassifier()
gb.fit(X_train, y_train)
predict = gb.predict(X_test)
calculateMetrics(y_test,predict,"Gradient Boosting Algorithm")


lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)
predict = lda.predict(X_test)
calculateMetrics(y_test,predict,"LDA Algorithm")

