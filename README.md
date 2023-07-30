# -Classifying-Iris-Flowers-with-Machine-Learning
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
iris=pd.read_csv(r"C:\Users\diksh\Desktop\IRIS.csv")
iris
iris.shape
iris.dtypes
iris.isna().sum()
iris.describe()
iris.head(150)
iris.tail(100)
n = len(iris[iris['species']=='Iris-versicolor'])
print("no of versicolor in dataset:",n)
n = len(iris[iris['species']=='Iris-virginica'])
print("no of versicolor in dataset:",n)
n = len(iris[iris['species']=='Iris-setosa'])
print("no of versicolor in dataset:",n)
plt.figure(1)
plt.boxplot([iris['sepal_length']])
plt.figure(2)
plt.boxplot([iris['sepal_width']])
plt.show()
iris.hist()
plt.show
import seaborn as sns
sns.pairplot(iris,hue='species')
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
train,test=train_test_split(iris,test_size=0.25)
print(train.shape)
print(test.shape)
train_x=train[['sepal_length','sepal_width','petal_length','petal_width']]
train_y=train.species
test_y=test.species
train_x.head()
test_y.head()
model=LogisticRegression()
model.fit(train_x,train_y)
prediction=model.predict(test_X)
print('Accuracy:',metrics.accuracy_score(prediction,test_y))
