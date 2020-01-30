# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 05:36:09 2019

@author: Shiv
"""
#Use Preprocessing to help in converting non numerical data into numerical data
import pandas as pd
from sklearn import tree
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
txt=pd.read_csv(r"D:/Code/Kaggle/car_dataset.txt")
print(txt)
print(txt.head())
'''class_=["unacc", "acc", "good", "vgood"]
buying=["vhigh", "high", "med", "low"]
maint=["vhigh", "high", "med", "low"]
lug_boot=["small", "med", "big"]
safety=["low", "med", "high"]
doors=[2,3,4,"5more"]
persons=[2,4,"more"]
clf=preprocessing.OneHotEncoder(categories=[buying,maint,lug_boot,safety,doors,persons])
print(clf)
X=[["low","low","5more",2,"small","high","unacc"],["low","low","5more",2,"med","low","unacc"]]
clf.fit(X)
arr=clf.transform([["low","low","5more","more","small","med","acc"]]).toarray()
print(arr)'''
le=preprocessing.LabelEncoder()
buying=le.fit_transform(list(txt["buying"]))
maint=le.fit_transform(list(txt["maint"]))
dcor=le.fit_transform(list(txt["dcor"]))
persons=le.fit_transform(list(txt["persons"]))
lug_boot=le.fit_transform(list(txt["lug_boot"]))
safety=le.fit_transform(list(txt["safety"]))
cla=le.fit_transform(list(txt["class_"]))

#predict="class"
X=list(zip(buying,maint,dcor,persons,lug_boot,safety))
Y=list(cla)

x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.1)

'''classifier=tree.DecisionTreeClassifier()

classifier=classifier.fit(x_train,y_train)
pred=classifier.predict(x_test)

print(accuracy_score(y_test,pred))'''

model = KNeighborsClassifier(n_neighbors=7)
model.fit(x_train,y_train)
predict=model.predict(x_test)
acc=model.score(x_test,y_test)
print(acc)
predicted=model.predict(x_test)
names=["unacc","acc","good", "vgood"]
for x in range(len(x_test)):
    print("Predicted: ",names[predicted[x]],"Data: ", x_test[x],"Actual: ",names[y_test[x]])
    n=model.kneighbors([x_test[x]],7)
    print(f"N: {n}")