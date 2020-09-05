# -*- coding: utf-8 -*-
"""
Created on Sat Sep  5 13:03:12 2020

@author:vaibhav arya
scholar No.171112099
"""
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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

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
#Mapping values to int values
mapping = {
    'Iris-setosa' : 1,
    'Iris-versicolor' : 2,
    'Iris-virginica' : 3
}

#====================================================
#Separating input and output features
#=================================================
X=iris.drop(['Species'],axis=1,inplace=False)
Y=iris['Species'].replace(mapping)

#===================================================
#Splitting the data into train and test
#==================================================
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=2)
print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)

#===============================================================
#Baseline model
#===========================================================
base_pred=np.mean(y_test)
print(base_pred)

#Repeating same value till length of test data
base_pred=np.repeat(base_pred,len(y_test))


#=============================================================
#finding the RMSE
#============================================================
base_root_mean_square_error=np.sqrt(mean_squared_error(y_test,base_pred))
print(base_root_mean_square_error)

#========================================================
#Linear Regression
#=========================================================
#setting intercept as true
lgr=LinearRegression(fit_intercept=True)

#Model
model_lin1=lgr.fit(x_train,y_train)

#Predicting model on test set
iris_predictions_lin1=lgr.predict(x_test)

#Computing MSE and RMSE
lin_mse1=mean_squared_error(y_test,iris_predictions_lin1)
lin_rmse1=np.sqrt(lin_mse1)
print(lin_rmse1)

#R squared value
r2_lin_test1=model_lin1.score(x_test,y_test)
r2_lin_train1=model_lin1.score(x_train,y_train)
print(r2_lin_test1,r2_lin_train1)

#Regression diagnostics -Residual plot analysis
residuals=y_test-iris_predictions_lin1
sns.regplot(x=iris_predictions_lin1,y=residuals,scatter=True,fit_reg=False)
residuals.describe()



