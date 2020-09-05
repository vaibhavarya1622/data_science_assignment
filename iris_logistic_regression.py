# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 19:16:10 2020

@author: vaibhav arya
Scholar Number:171112099
"""
#==================================================================
#Classifying iris species
#=================================================================

#import the dependencies
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

#===================================================================
#setting dimensions for plot
sns.set(rc={'figure.figsize':(11.7,8.27)})
#===================================================================


#========================================================
#reading the csv file
#======================================================
data=pd.read_csv('iris.csv')

#====================================================
#Creating the copy
#==============================================
iris=data.copy()

#=================================================
#Structure of the dataset
#==================================================
iris.info()
#Data does not contain NULL values

#============================================================
#Summarizing the data
#================================================================
iris.describe()

#======================================================
#Plot the relation of each feature with species
#==========================================================
plt.xlabel('Features')
plt.ylabel('Species')
plt.scatter(iris['SepalLengthCm'],iris['Species'],c='blue',label='Sepal_Length')
plt.scatter(iris['SepalWidthCm'],iris['Species'],c='green',label='sepal_width')
plt.scatter(iris['PetalLengthCm'],iris['Species'],c='red',label='petal_length')
plt.scatter(iris['PetalWidthCm'],iris['Species'],c='black',label='petal_width')
plt.legend(loc=4,prop={'size':8})
plt.show()

#====================================================
#Model Building
#===================================================

#====================================================
#Separating input and output features
#=================================================
X=iris.drop(['Species'],axis=1,inplace=False)
Y=iris['Species']

#===================================================
#Splitting the data into train and test
#==================================================
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=2)
print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)

#======================================================
#Train the model
#===========================================================
model=LogisticRegression()
model.fit(x_train,y_train)

#============================================================
#Test the model
#=========================================================
predictions=model.predict(x_test)
#Check precision
print(classification_report(y_test,predictions))
print(accuracy_score(y_test,predictions))


